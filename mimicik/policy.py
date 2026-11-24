"""Stateful and stateless EEF-to-joint inference for one robot stream."""

from collections import deque

import numpy as np
import torch

from .checkpoint import load_bundle


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _canonical_pose(value):
    """Reject invalid zero quaternions and select their w >= 0 hemisphere.

    Callers supply unit quaternions. As in the research implementation, slight
    recording deviations are preserved rather than silently renormalized.
    """
    value = np.array(value, dtype=np.float32, copy=True)
    if value.shape[-1] != 7 or not np.isfinite(value).all():
        raise ValueError("EEF poses must contain finite xyz + XYZW quaternion")
    quaternion = value[..., 3:7]
    if np.any(np.all(quaternion == 0, axis=-1)):
        raise ValueError("EEF quaternion must be nonzero; supply unit XYZW quaternions")
    quaternion *= np.where(quaternion[..., 3:4] < 0, -1.0, 1.0)
    return value


class MimicIK:
    """A learned local execution adapter, with one history buffer per instance.

    Inputs use radians, metres and XYZW quaternions, with both EEF poses in the
    training robot's base/TCP convention. The model does not apply servo limits,
    collision checks or filters. Keep the robot's normal protection layer active.
    """

    def __init__(self, model, normalization, metadata=None):
        if (normalization.obs_dim, normalization.act_dim) != (model.obs_dim, model.dof):
            raise ValueError("Model and normalizer dimensions differ")
        self.model = model.eval().requires_grad_(False)
        self.normalization = normalization
        self.metadata = metadata or {}
        self.dof, self.obs_dim = model.dof, model.obs_dim
        self.obs_steps = model.obs_steps
        self.device = next(model.parameters()).device
        self._history = deque(maxlen=self.obs_steps)

    @classmethod
    def from_pretrained(cls, bundle_dir, device="cpu"):
        """Load a downloaded local bundle; this method performs no network access."""
        return cls(*load_bundle(bundle_dir, device))

    def reset(self):
        """Clear observation history at episode boundaries or after resets."""
        self._history.clear()

    @torch.inference_mode()
    def predict_delta(self, history, *, endpoint="final"):
        """Decode raw history without changing the stateful buffer.

        Input is [history, obs_dim] or [batch, history, obs_dim], ordered oldest
        to newest. Output is [dof] or [batch, dof] in radians. Every raw frame is
        [q_current, EEF_current_xyz_xyzw, EEF_target_xyz_xyzw].
        """
        history = np.array(history, dtype=np.float32, copy=True)
        unbatched = history.ndim == 2
        if unbatched:
            history = history[None]
        if (
            history.ndim != 3
            or history.shape[1:] != (self.obs_steps, self.obs_dim)
            or len(history) == 0
        ):
            raise ValueError(
                f"Expected [batch,{self.obs_steps},{self.obs_dim}] or [{self.obs_steps},{self.obs_dim}]"
            )
        if not np.isfinite(history).all():
            raise ValueError("History must be finite")
        for start in (self.dof, self.dof + 7):
            history[..., start : start + 7] = _canonical_pose(
                history[..., start : start + 7]
            )
        normalized = self.normalization.normalize_obs(history)
        obs = torch.from_numpy(normalized).to(self.device)
        action = self.model.predict_normalized(obs, endpoint=endpoint)
        delta = self.normalization.denormalize_action(action[:, 0])
        if not torch.isfinite(delta).all():
            raise FloatingPointError("MimicIK predicted a nonfinite joint delta")
        delta = delta.cpu().numpy()
        return delta[0] if unbatched else delta

    def step(self, q, eef_current, eef_target):
        """Append measured state and return q_command = q + delta_q.

        Feed measured q on the next call for real feedback. Feeding the command
        instead assumes perfect execution. Use one instance per arm or stream.
        """
        q = _finite_array(q, (self.dof,), "q")
        current = _canonical_pose(_finite_array(eef_current, (7,), "eef_current"))
        target = _canonical_pose(_finite_array(eef_target, (7,), "eef_target"))
        frame = np.concatenate((q, current, target)).astype(np.float32)
        candidate = list(self._history)
        if not candidate:
            candidate = [frame.copy() for _ in range(self.obs_steps)]
        else:
            candidate = (candidate + [frame])[-self.obs_steps :]
        delta = self.predict_delta(np.stack(candidate))
        command = q + delta
        if not np.isfinite(command).all():
            raise FloatingPointError("Joint command overflow")
        self._history.clear()
        self._history.extend(candidate)
        return command
