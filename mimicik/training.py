"""Portable supervised MimicIK training, with one joint AdamW update per step.

This recipe preserves the MIP loss and optional NatureIK FK proxy, while using
an explicit episode split and deterministic per-step minibatches. It is a
community training recipe, not a claim that unavailable private-data paper
experiments can be reproduced from a public download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_bundle
from .data import (
    EpisodeDataset,
    dataset_fingerprint,
    fit_normalization,
    load_episodes,
    split_episodes,
)
from .kinematics import Kinematics, fk_pose_loss
from .model import MimicIKModel
from .normalization import Normalization

TRAINER_VERSION = "mimicik-portable-v1"


def _json_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_torch(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    os.close(fd)
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def isolated_rng(seed, step, stream):
    """Match the controlled pipeline's independent MIP and FK random streams."""
    value = int(
        np.random.SeedSequence([int(seed), int(step), int(stream)]).generate_state(
            1, dtype=np.uint64
        )[0]
        % np.uint64(2**63 - 1)
    )
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(value)
        yield


def mip_primary_loss(model, obs, action, optimization):
    """Two supervised endpoint losses; the second training input is noisy GT.

    This is deliberately different from deployment, which feeds the first
    prediction into the second network call. The original MIP l2 convention
    sums joint coordinates before averaging batch/horizon dimensions.
    """
    midpoint = float(optimization.get("t_two_step", 0.9))
    if not 0 < midpoint < 1:
        raise ValueError("t_two_step must lie strictly between 0 and 1")
    if optimization.get("norm_type", "l2") != "l2":
        raise ValueError("The portable recipe currently supports norm_type=l2 only")
    noise = torch.empty_like(action).normal_()
    intermediate = action + (1 - midpoint) * noise
    condition = model.encoder(obs, None)
    zeros = torch.zeros(len(action), device=action.device, dtype=action.dtype)
    first = model.flow_map.get_velocity(zeros, torch.zeros_like(action), condition)
    second = model.flow_map.get_velocity(
        torch.full_like(zeros, midpoint), intermediate, condition
    )
    loss0 = ((first - action) / midpoint).square().sum(-1)
    loss1 = ((second - action) / (1 - midpoint)).square().sum(-1)
    return float(optimization.get("loss_scale", 1.0)) * (loss0 + loss1).mean()


def fk_proxy_loss(model, batch, normalization, kinematics, task):
    """Original separate interpolation/Euler FK path, not deployed second output."""
    obs, action = batch["obs"], batch["action"]
    s = torch.rand(len(action), device=action.device, dtype=action.dtype)
    noise = torch.randn_like(action)
    intermediate = (1 - s[:, None, None]) * noise + s[:, None, None] * action
    condition = model.encoder(obs, None)
    prediction = model.flow_map(s, torch.ones_like(s), intermediate, condition)
    q_prediction = batch["q_curr_raw"] + normalization.denormalize_action(prediction)
    batch_size, horizon, dof = q_prediction.shape
    position, rotation = kinematics.pose(q_prediction.reshape(-1, dof))
    return fk_pose_loss(
        position.reshape(batch_size, horizon, 3),
        rotation.reshape(batch_size, horizon, 3, 3),
        batch["eef_tgt_raw"],
        task,
    )


@torch.no_grad()
def update_ema(ema, model, rate):
    source = dict(model.named_parameters())
    for name, value in ema.named_parameters():
        value.mul_(rate).add_(source[name], alpha=1 - rate)
    source_buffers = dict(model.named_buffers())
    for name, value in ema.named_buffers():
        value.copy_(source_buffers[name])


def _batch(dataset, indices, device):
    rows = [dataset[int(index)] for index in indices]
    return {key: torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


@torch.no_grad()
def validation_mse(model, dataset, device, batch_size, max_batches=None):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for index, batch in enumerate(
        DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ):
        if max_batches is not None and index >= max_batches:
            break
        obs, action = batch["obs"].to(device), batch["action"].to(device)
        prediction = model.predict_normalized(obs)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite validation prediction")
        total += float((prediction - action).square().sum())
        count += action.numel()
    model.train(was_training)
    if count == 0:
        raise ValueError("Validation must include at least one batch")
    return total / count


def train(
    config,
    data,
    output,
    *,
    device="cpu",
    resume=None,
    split_manifest=None,
    stop_after=None,
):
    """Train and export ``final/``; return a compact run summary.

    ``stop_after`` pauses after an absolute update index for planned pauses and
    reproducibility tests. The scheduler's total budget stays unchanged.
    ``resume`` is a checkpoint produced by this function, loaded weights-only.
    """
    config = deepcopy(config)
    task, optimization = config["task"], config["optimization"]
    if int(task.get("horizon", 1)) != 1:
        raise ValueError(
            "The portable data recipe supports one action per observation (horizon=1)"
        )
    if optimization.get("loss_type", "mip") != "mip":
        raise ValueError("The portable trainer implements loss_type=mip")
    seed = int(optimization.get("seed", 0))
    steps = int(optimization.get("gradient_steps", 300000))
    batch_size = int(optimization.get("batch_size", 512))
    ema_rate = float(optimization.get("ema_rate", 0.995))
    lr = float(optimization.get("lr", 1e-4))
    if steps < 1 or batch_size < 1 or not 0 <= ema_rate < 1 or lr <= 0:
        raise ValueError(
            "Invalid optimizer budget, batch size, EMA rate, or learning rate"
        )
    if stop_after is not None and not 0 < int(stop_after) <= steps:
        raise ValueError("stop_after must be in [1, gradient_steps]")
    training = config.get("training", {})
    save_every = int(training.get("save_every", 5000))
    eval_every = int(training.get("eval_every", 20000))
    log_every = int(training.get("log_every", 100))
    if min(save_every, eval_every, log_every) < 1:
        raise ValueError("save_every/eval_every/log_every must be positive")
    threads = int(training.get("torch_threads", 2))
    if threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}; use a new path or --resume"
        )
    if resume is not None:
        # Resuming is deliberately an in-place operation. A separate output
        # would require an explicit fork policy for logs and provenance.
        if Path(resume).expanduser().resolve() != (output / "resume.pt").resolve():
            raise ValueError(
                "Resume must use OUTPUT/resume.pt from the same output directory"
            )
        allowed = {
            "config.json",
            "splits.json",
            "identity.json",
            "normalizer.json",
            "resume.pt",
            "metrics.jsonl",
            "summary.json",
            "final",
            "paused",
        }
        unknown = {item.name for item in output.iterdir()} - allowed
        if unknown or not (output / "identity.json").is_file():
            raise ValueError(
                "Resume output is not a recognized training directory; refusing to overwrite it"
            )
        for name in ("paused", "final"):
            item = output / name
            if item.exists() and (
                not item.is_dir() or not (item / "metadata.json").is_file()
            ):
                raise ValueError(
                    f"Resume output contains an unrecognized {name} artifact"
                )
    episodes = load_episodes(data)
    if any(ep.q.shape[1] != int(task["act_dim"]) for ep in episodes):
        raise ValueError("Dataset joint count does not match task.act_dim")
    if int(task["obs_dim"]) != int(task["act_dim"]) + 14:
        raise ValueError(
            "Portable observations must contain q + current EEF + target EEF"
        )
    if split_manifest is None:
        candidate = Path(data).expanduser() / "splits.json"
        if candidate.is_file():
            split_manifest = candidate
    train_episodes, val_episodes, splits = split_episodes(
        episodes,
        val_fraction=float(task.get("val_dataset_percentage", 0.1)),
        seed=int(training.get("split_seed", 0)),
        manifest=split_manifest,
    )
    normalization = fit_normalization(train_episodes)
    fingerprint = dataset_fingerprint(episodes)
    identity = {
        "trainer": TRAINER_VERSION,
        "config_sha256": _json_sha(config),
        "data_sha256": fingerprint,
        "splits_sha256": _json_sha(splits),
        "normalization_sha256": _json_sha(normalization.to_dict()),
    }
    train_data = EpisodeDataset(
        train_episodes, normalization, int(task.get("obs_steps", 2))
    )
    val_data = EpisodeDataset(
        val_episodes, normalization, int(task.get("obs_steps", 2))
    )
    if not len(train_data) or not len(val_data):
        raise ValueError(
            "Train and validation each need at least one complete observation window"
        )
    model = MimicIKModel(config).to(device)
    ema = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(optimization.get("weight_decay", 1e-5)),
    )
    fk_weight = float(task.get("fk_loss_weight", 0.0))
    if fk_weight < 0:
        raise ValueError("fk_loss_weight must be nonnegative")
    kinematics = None
    if fk_weight:
        if task.get("fk_update_strategy", "joint") != "joint":
            raise ValueError("The portable recipe supports one joint FK update only")
        if not task.get("urdf_path") or not task.get("end_link_name"):
            raise ValueError("FK requires task.urdf_path and task.end_link_name")
        kinematics = Kinematics(
            task["urdf_path"],
            task["end_link_name"],
            device=device,
            base_link=task.get("base_link_name"),
        )
        if len(kinematics.joint_names) != int(task["act_dim"]):
            raise ValueError("URDF chain joint count does not match task.act_dim")
        expected = task.get("joint_names")
        if expected is not None and list(expected) != list(kinematics.joint_names):
            raise ValueError(
                "Configured joint_names must exactly match URDF chain order"
            )
        identity["urdf_sha256"] = hashlib.sha256(
            Path(task["urdf_path"]).read_bytes()
        ).hexdigest()
        print(
            "FK loss enabled; joint order: " + ", ".join(kinematics.joint_names),
            flush=True,
        )
    start = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state.get("identity") != identity:
            raise ValueError(
                "Resume identity mismatch: config, split, normalization, data, or URDF changed"
            )
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError(
                "Output directory identity does not match the resume checkpoint"
            )
        if _json_sha(state["normalization"]) != identity["normalization_sha256"]:
            raise ValueError(
                "Resume normalization payload does not match its identity hash"
            )
        start = int(state["step"])
        if not 0 <= start < steps:
            raise ValueError(
                "Resume checkpoint must precede the configured total budget"
            )
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary_step = int(json.loads(summary_path.read_text())["steps_completed"])
            if summary_step > start:
                raise ValueError(
                    "Resume checkpoint is older than the output directory's recorded progress"
                )
        metrics_path = output / "metrics.jsonl"
        if metrics_path.is_file():
            previous_step = 0
            for line in metrics_path.read_text().splitlines():
                recorded_step = int(json.loads(line)["step"])
                if recorded_step <= previous_step or recorded_step > start:
                    raise ValueError(
                        "Output metrics contain newer or conflicting progress; refusing to mix resume branches"
                    )
                previous_step = recorded_step
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        normalization = Normalization.from_dict(state["normalization"])
    _atomic_json(output / "config.json", config)
    _atomic_json(output / "splits.json", splits)
    _atomic_json(output / "identity.json", identity)
    _atomic_json(output / "normalizer.json", normalization.to_dict())
    model.train()
    end = steps if stop_after is None else int(stop_after)
    if end <= start:
        raise ValueError("stop_after must be greater than the resumed step")
    log_path = output / "metrics.jsonl"
    started = time.monotonic()
    last_validation = None
    for step in range(start, end):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        indices = rng.integers(len(train_data), size=batch_size)
        batch = _batch(train_data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        current_lr = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        with isolated_rng(seed, step, 0):
            primary = mip_primary_loss(
                model, batch["obs"], batch["action"], optimization
            )
        total = primary
        values = {}
        if kinematics is not None:
            with isolated_rng(seed, step, 1):
                fk, values = fk_proxy_loss(
                    model, batch, normalization, kinematics, task
                )
            total = total + fk_weight * fk
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss at update {step + 1}; previous resume.pt remains intact"
            )
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(optimization.get("grad_clip_norm", 10.0)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        # One device-to-host synchronization, rather than one for each tensor.
        if not torch.stack(
            [torch.isfinite(parameter).all() for parameter in model.parameters()]
        ).all():
            raise FloatingPointError(
                f"Non-finite model at update {step + 1}; previous resume.pt remains intact"
            )
        update_ema(ema, model, ema_rate)
        completed = step + 1
        record = {
            "step": completed,
            "primary_loss": float(primary.detach()),
            "loss": float(total.detach()),
            "grad_norm": float(norm),
            "lr": current_lr,
        }
        record.update({key: float(value) for key, value in values.items()})
        if completed % eval_every == 0 or completed == end:
            last_validation = validation_mse(
                ema,
                val_data,
                device,
                batch_size,
                max_batches=training.get("max_validation_batches"),
            )
            record["validation_normalized_mse"] = last_validation
        if completed % log_every == 0 or completed == end or completed == 1:
            with log_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
        if completed % save_every == 0 or completed == end:
            _atomic_torch(
                output / "resume.pt",
                {
                    "identity": identity,
                    "step": completed,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "normalization": normalization.to_dict(),
                },
            )
    done = end == steps
    summary = {
        **identity,
        "steps_completed": end,
        "total_steps": steps,
        "completed": done,
        "train_episodes": len(train_episodes),
        "validation_episodes": len(val_episodes),
        "train_frames": sum(len(ep) for ep in train_episodes),
        "validation_frames": sum(len(ep) for ep in val_episodes),
        "train_windows": len(train_data),
        "validation_windows": len(val_data),
        "validation_normalized_mse": last_validation,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "checkpoint_rule": "final_update_ema" if done else "paused_update_ema",
    }
    save_bundle(
        output / ("final" if done else "paused"),
        ema,
        normalization,
        config,
        metadata=summary,
        overwrite=True,
    )
    _atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume", help="This trainer's resume.pt, loaded weights-only"
    )
    parser.add_argument(
        "--split-manifest", help="Explicit train/val/test episode ID JSON"
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Pause at an absolute update while retaining total schedule",
    )
    parser.add_argument("--urdf", help="Override task.urdf_path for optional FK")
    parser.add_argument(
        "--end-link", help="Override task.end_link_name for optional FK"
    )
    parser.add_argument(
        "--base-link", help="Override task.base_link_name for optional FK"
    )
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    if args.steps is not None:
        config["optimization"]["gradient_steps"] = args.steps
    if args.seed is not None:
        config["optimization"]["seed"] = args.seed
    for flag, key in (
        (args.urdf, "urdf_path"),
        (args.end_link, "end_link_name"),
        (args.base_link, "base_link_name"),
    ):
        if flag is not None:
            config["task"][key] = flag
    summary = train(
        config,
        args.data,
        args.output,
        device=args.device,
        resume=args.resume,
        split_manifest=args.split_manifest,
        stop_after=args.stop_after,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
