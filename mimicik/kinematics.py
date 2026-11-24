"""Explicit URDF FK and versioned pose losses.

The loss equations preserve NatureIK's joint-update v1/v2 formulations. This
module deliberately does not infer base transforms or silently disable FK.
"""

from __future__ import annotations

from pathlib import Path

import torch


def quat_to_matrix(quaternion, normalize=True):
    """Convert XYZW quaternions, optionally retaining historical v1 magnitude."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if not torch.isfinite(quaternion).all() or (norm <= 1e-12).any():
        raise ValueError("Quaternion targets must be finite and nonzero")
    q = quaternion / norm if normalize else quaternion
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_error(rotation, target):
    """Target-relative SO(3) geodesic angle in radians, with no acos floor."""
    relative = rotation.transpose(-1, -2) @ target
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(0.5 * torch.linalg.vector_norm(skew, dim=-1), cosine)


class Kinematics:
    """A serial URDF chain; joint order is available as ``joint_names``."""

    def __init__(self, urdf, end_link, device="cpu", base_link=None):
        try:
            import pytorch_kinematics as pk
        except ImportError as exc:
            raise ImportError("URDF FK requires pip install 'mimicik[fk]'") from exc
        content = Path(urdf).expanduser().read_bytes()
        kwargs = {"root_link_name": base_link} if base_link else {}
        self.chain = pk.build_serial_chain_from_urdf(content, end_link, **kwargs).to(
            device=device, dtype=torch.float32
        )
        self.joint_names = self.chain.get_joint_parameter_names()
        self.device = torch.device(device)

    def matrix(self, q):
        q = torch.as_tensor(q, device=self.device, dtype=torch.float32)
        if q.shape[-1] != len(self.joint_names):
            raise ValueError(
                f"FK expects joints {self.joint_names}, got {q.shape[-1]} coordinates"
            )
        return self.chain.forward_kinematics(q).get_matrix()

    def pose(self, q):
        matrix = self.matrix(q)
        return matrix[..., :3, 3], matrix[..., :3, :3]


def fk_pose_loss(position, rotation, target_pose, task):
    """Return scalar FK loss and diagnostics in physical units."""
    version = task.get("fk_loss_version", "legacy_mixed_units_v1")
    if version not in ("legacy_mixed_units_v1", "tolerance_normalized_v2"):
        raise ValueError(f"Unknown FK loss version: {version}")
    target_rotation = quat_to_matrix(
        target_pose[..., 3:], normalize=version != "legacy_mixed_units_v1"
    )
    error = torch.linalg.vector_norm(position - target_pose[..., :3], dim=-1)
    angle = rotation_error(rotation, target_rotation)
    if version == "legacy_mixed_units_v1":
        position_loss = (position - target_pose[..., :3]).square().mean()
        relative = rotation.transpose(-1, -2) @ target_rotation
        cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        rotation_loss = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6)).mean()
    else:
        pos_scale = float(task.get("fk_pos_scale_m", 0.005))
        rot_scale = float(task.get("fk_rot_scale_rad", 0.017453292519943295))
        if pos_scale <= 0 or rot_scale <= 0:
            raise ValueError("FK tolerance scales must be positive")
        position_loss = (error / pos_scale).square().mean()
        rotation_loss = (angle / rot_scale).square().mean()
    pw, rw = (
        float(task.get("fk_pos_weight", 1.0)),
        float(task.get("fk_rot_weight", 0.1)),
    )
    if pw < 0 or rw < 0:
        raise ValueError("FK component weights must be nonnegative")
    total = pw * position_loss + rw * rotation_loss
    return total, {
        "fk_loss": total.detach(),
        "fk_position_loss": position_loss.detach(),
        "fk_rotation_loss": rotation_loss.detach(),
        "position_error_m": error.mean().detach(),
        "rotation_error_rad": angle.mean().detach(),
    }
