"""Episode-macro evaluation with explicit recorded-state or ideal feedback replay.

Recorded-state evaluation reads each current state from the dataset. Feedback
replay feeds each accepted joint command into the next step and recomputes the
current EEF with FK. The latter assumes perfect execution and excludes servo
dynamics, contacts, sensing delays, and physical robot protection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import sha256
from .data import dataset_fingerprint, load_episodes, split_episodes
from .kinematics import Kinematics, quat_to_matrix, rotation_error
from .policy import MimicIK


def _numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def _pose_xyzw(position, rotation):
    """Convert one FK rotation to XYZW, including rotations near pi."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if not np.isfinite(position).all():
        raise FloatingPointError("FK returned a nonfinite position")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise FloatingPointError("FK returned an invalid rotation matrix")
    candidates = np.asarray(
        (
            1 + np.trace(matrix),
            1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
            1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
            1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
        )
    )
    which = int(candidates.argmax())
    scale = 2 * np.sqrt(max(0.0, candidates[which]))
    if scale <= 1e-12:
        raise FloatingPointError("FK rotation cannot be converted to a quaternion")
    if which == 0:
        q = [
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            scale / 4,
        ]
    elif which == 1:
        q = [
            scale / 4,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        ]
    elif which == 2:
        q = [
            (matrix[0, 1] + matrix[1, 0]) / scale,
            scale / 4,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        ]
    else:
        q = [
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            scale / 4,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return np.concatenate((np.asarray(position).reshape(3), q)).astype(np.float32)


def select_episodes(episodes, split="val", manifest=None):
    """Never silently evaluate training data when a validation split is absent."""
    if split not in ("val", "test", "all"):
        raise ValueError("split must be val, test, or all")
    if manifest is None:
        if split != "all":
            raise ValueError(
                "A split manifest is required for val/test; supply --split-manifest or explicitly use --split all"
            )
        return list(episodes), None
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    _, _, validated = split_episodes(episodes, manifest=manifest)
    if split == "all":
        return list(episodes), validated
    chosen = set(validated[split])
    selected = [ep for ep in episodes if ep.name in chosen]
    if not selected:
        raise ValueError(f"The {split} split is empty")
    return selected, validated


def _command_failure(command, current, max_step_rad):
    command = np.asarray(command, dtype=np.float32)
    if command.shape != current.shape:
        raise ValueError("Policy command shape does not match joint coordinates")
    if not np.isfinite(command).all():
        return "nonfinite_joint_command"
    if (
        max_step_rad is not None
        and np.max(
            np.abs(command.astype(np.float64) - np.asarray(current, dtype=np.float64))
        )
        > max_step_rad
    ):
        return "max_step_rad_exceeded"
    return None


def _error_arrays(commands, target_pose, target_q, kinematics):
    position, rotation = kinematics.pose(torch.as_tensor(commands, dtype=torch.float32))
    position = torch.as_tensor(_numpy(position), dtype=torch.float32)
    rotation = torch.as_tensor(_numpy(rotation), dtype=torch.float32)
    target = torch.as_tensor(target_pose, dtype=torch.float32)
    finite = torch.isfinite(position).all(-1) & torch.isfinite(rotation).all(-1).all(-1)
    position_error = torch.linalg.vector_norm(position - target[:, :3], dim=-1) * 1000
    rotation_degrees = (
        rotation_error(rotation, quat_to_matrix(target[:, 3:])) * 180 / torch.pi
    )
    joint_mae = np.mean(
        np.abs(commands.astype(np.float64) - np.asarray(target_q, dtype=np.float64)),
        axis=-1,
    )
    finite &= torch.isfinite(position_error) & torch.isfinite(rotation_degrees)
    return _numpy(position_error), _numpy(rotation_degrees), joint_mae, _numpy(finite)


def _recorded_commands(policy, episode, batch_size, max_step_rad):
    commands = []
    frames = episode.frames
    history = int(policy.obs_steps)
    failure = None
    for start in range(0, len(episode), batch_size):
        end = min(len(episode), start + batch_size)
        indices = np.maximum(
            0, np.arange(start, end)[:, None] + np.arange(-history + 1, 1)[None]
        )
        try:
            delta = policy.predict_delta(frames[indices])
        except FloatingPointError:
            # The public policy rejects a nonfinite batch as a whole. Retry its
            # rows only on failure to retain the exact valid prefix.
            delta_rows = []
            for row in indices:
                try:
                    delta_rows.append(policy.predict_delta(frames[row]))
                except FloatingPointError:
                    break
            delta = np.asarray(delta_rows, dtype=np.float32).reshape(-1, policy.dof)
            failure = "nonfinite_model_prediction"
        delta = np.asarray(delta, dtype=np.float32)
        if delta.ndim != 2 or delta.shape[1] != policy.dof or len(delta) > end - start:
            raise ValueError("Policy returned an unexpected batch shape")
        for local, dq in enumerate(delta):
            current = episode.q[start + local]
            command = current + dq
            reason = _command_failure(command, current, max_step_rad)
            if reason:
                return np.asarray(commands, dtype=np.float32).reshape(
                    -1, policy.dof
                ), reason
            commands.append(command)
        if failure:
            break
        if len(delta) != end - start:
            raise ValueError(
                "Policy returned an incomplete batch without reporting failure"
            )
    return np.asarray(commands, dtype=np.float32).reshape(-1, policy.dof), failure


def _feedback_commands(policy, episode, kinematics, max_step_rad):
    commands = []
    current = episode.q[0].copy()
    failure = None
    for target in episode.target_eef:
        position, rotation = kinematics.pose(
            torch.as_tensor(current[None], dtype=torch.float32)
        )
        try:
            current_pose = _pose_xyzw(_numpy(position)[0], _numpy(rotation)[0])
            command = policy.step(current, current_pose, target)
        except FloatingPointError:
            failure = "nonfinite_model_or_fk_prediction"
            break
        reason = _command_failure(command, current, max_step_rad)
        if reason:
            failure = reason
            break
        commands.append(command.copy())
        current = command
    return np.asarray(commands, dtype=np.float32).reshape(-1, policy.dof), failure


def _mean_defined(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def evaluate_episodes(
    policy, episodes, kinematics, protocol="recorded", batch_size=256, max_step_rad=None
):
    """Return aggregate and per-episode metrics, retaining failed episodes.

    Error means are conditional on each episode's finite accepted prefix.
    Success rates include every planned frame; unexecuted suffixes count as
    failures. Joint-step d95 is the per-episode 95th percentile of the L2 norm
    between consecutive accepted commands, averaged across defined episodes.
    """
    if protocol not in ("recorded", "feedback"):
        raise ValueError("protocol must be recorded or feedback")
    if batch_size < 1 or not episodes:
        raise ValueError("Evaluation requires episodes and a positive batch size")
    if max_step_rad is not None and (
        not np.isfinite(max_step_rad) or max_step_rad <= 0
    ):
        raise ValueError("max_step_rad must be finite and positive")
    if any(ep.q.shape[1] != policy.dof for ep in episodes):
        raise ValueError("Data joint count does not match checkpoint")
    if len(kinematics.joint_names) != policy.dof:
        raise ValueError("URDF joint count does not match checkpoint")
    reports = []
    for episode in episodes:
        policy.reset()
        if protocol == "recorded":
            commands, failure = _recorded_commands(
                policy, episode, batch_size, max_step_rad
            )
        else:
            commands, failure = _feedback_commands(
                policy, episode, kinematics, max_step_rad
            )
        position = rotation = joint = np.empty(0)
        if len(commands):
            position, rotation, joint, finite = _error_arrays(
                commands,
                episode.target_eef[: len(commands)],
                episode.target_q[: len(commands)],
                kinematics,
            )
            if not finite.all():
                first = int(np.flatnonzero(~finite)[0])
                commands, position, rotation, joint = (
                    commands[:first],
                    position[:first],
                    rotation[:first],
                    joint[:first],
                )
                failure = "nonfinite_fk_metric"
        valid = len(commands)
        steps = np.linalg.norm(np.diff(commands.astype(np.float64), axis=0), axis=-1)
        success_count = int((position < 5.0).sum())
        report = {
            "episode": episode.name,
            "planned_frames": len(episode),
            "valid_frames": valid,
            "valid_prefix_frames": valid,
            "unexecuted_frames": len(episode) - valid,
            "completed": valid == len(episode) and failure is None,
            "failure_reason": failure,
            "failure_frame": valid if failure else None,
            "mean_position_mm": float(position.mean()) if valid else None,
            "mean_rotation_deg": float(rotation.mean()) if valid else None,
            "joint_mae_rad": float(joint.mean()) if valid else None,
            "success_frames_5mm": success_count,
            "sr_5mm_percent": 100 * success_count / len(episode),
            "joint_step_d95_rad_per_frame": float(np.quantile(steps, 0.95))
            if len(steps)
            else None,
        }
        reports.append(report)
    means = [
        row["mean_position_mm"]
        for row in reports
        if row["mean_position_mm"] is not None
    ]
    aggregate = {
        "episodes": len(reports),
        "completed_episodes": sum(row["completed"] for row in reports),
        "failed_episodes": sum(not row["completed"] for row in reports),
        "planned_frames": sum(row["planned_frames"] for row in reports),
        "valid_frames": sum(row["valid_frames"] for row in reports),
        "episodes_with_valid_prefix": len(means),
        "episodes_with_joint_steps": sum(
            row["joint_step_d95_rad_per_frame"] is not None for row in reports
        ),
        "mean_position_mm": _mean_defined([row["mean_position_mm"] for row in reports]),
        "p95_episode_mean_position_mm": float(np.quantile(means, 0.95))
        if means
        else None,
        "mean_rotation_deg": _mean_defined(
            [row["mean_rotation_deg"] for row in reports]
        ),
        "joint_mae_rad": _mean_defined([row["joint_mae_rad"] for row in reports]),
        "sr_5mm_percent": float(np.mean([row["sr_5mm_percent"] for row in reports])),
        "joint_step_d95_rad_per_frame": _mean_defined(
            [row["joint_step_d95_rad_per_frame"] for row in reports]
        ),
    }
    aggregate["valid_frame_fraction"] = (
        aggregate["valid_frames"] / aggregate["planned_frames"]
    )
    return {
        "protocol": protocol,
        "aggregate": aggregate,
        "episodes": reports,
        "definitions": {
            "aggregation": "episode macro; error means use valid prefixes, and report their episode/frame coverage",
            "position_p95": "linear-interpolated 95th percentile of defined episode mean position errors, not pooled frame errors",
            "success": "position error < 5 mm; every planned frame is in the denominator; failed/unexecuted suffixes count as unsuccessful",
            "joint_mae": "mean absolute command-minus-demonstration joint error over coordinates, radians, valid prefixes",
            "joint_step_d95": "mean of defined episode q95(||q_cmd[t]-q_cmd[t-1]||_2), radians/frame, valid prefixes; initial q to first command excluded",
            "max_step_rad": "optional per-joint max(abs(q_command-q_current)) threshold, rejected commands end that episode",
            "feedback_assumption": "q_next=q_command, current EEF=FK(q_next); ideal kinematic feedback replay, not physical closed-loop validation",
        },
        "max_step_rad": max_step_rad,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test", "all"), default="val")
    parser.add_argument("--split-manifest")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--ee-link", required=True)
    parser.add_argument("--base-link")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--protocol", choices=("recorded", "feedback"), default="recorded"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-step-rad", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    episodes = load_episodes(args.data)
    manifest = args.split_manifest
    if manifest is None:
        candidate = Path(args.data) / "splits.json"
        if candidate.is_file():
            manifest = candidate
    selected, split = select_episodes(episodes, args.split, manifest)
    policy = MimicIK.from_pretrained(args.checkpoint, device=args.device)
    task = policy.model.config["task"]
    for configured, supplied, name in (
        (task.get("end_link_name"), args.ee_link, "TCP/end link"),
        (task.get("base_link_name"), args.base_link, "base link"),
    ):
        if configured is not None and configured != supplied:
            raise ValueError(
                f"Checkpoint {name} differs from the supplied URDF convention"
            )
    urdf_hash = sha256(args.urdf)
    if policy.metadata.get("urdf_sha256") not in (None, urdf_hash):
        raise ValueError(
            "URDF SHA256 differs from the checkpoint's declared kinematics"
        )
    kinematics = Kinematics(
        args.urdf, args.ee_link, device=args.device, base_link=args.base_link
    )
    if task.get("joint_names") is not None and list(task["joint_names"]) != list(
        kinematics.joint_names
    ):
        raise ValueError("Checkpoint joint order differs from the supplied URDF")
    result = evaluate_episodes(
        policy, selected, kinematics, args.protocol, args.batch_size, args.max_step_rad
    )
    result["metadata"] = {
        "checkpoint_sha256": policy.metadata["sha256"],
        "data_sha256": dataset_fingerprint(episodes),
        "selected_data_sha256": dataset_fingerprint(selected),
        "split": args.split,
        "split_manifest": split,
        "urdf_sha256": urdf_hash,
        "end_link": args.ee_link,
        "base_link": args.base_link,
        "joint_names": kinematics.joint_names,
        "device": args.device,
        "units": {"joint": "rad", "position": "mm", "rotation": "degree"},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2, allow_nan=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
