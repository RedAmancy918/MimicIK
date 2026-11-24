"""Portable affine normalization with NumPy and differentiable Torch operations."""

from copy import deepcopy

import numpy as np
import torch


def _validate_stats(value, width=None):
    kind = value.get("type")
    if kind == "IdentityNormalizer":
        return width
    if kind == "MinMaxNormalizer":
        arrays = {
            key: np.asarray(value[key], dtype=np.float32)
            for key in ("min", "max", "range")
        }
        shape = arrays["min"].shape
        if (
            len(shape) != 1
            or not shape[0]
            or any(
                x.shape != shape or not np.isfinite(x).all() for x in arrays.values()
            )
        ):
            raise ValueError("Invalid normalization statistics")
        if width is not None and width != shape[0]:
            raise ValueError("Normalization width does not match its slice")
        if np.any(arrays["range"] <= 0) or np.any(arrays["max"] < arrays["min"]):
            raise ValueError("Normalization ranges must be positive and min <= max")
        expected = arrays["max"] - arrays["min"]
        expected[expected == 0] = 1.0
        if not np.allclose(expected, arrays["range"], rtol=1e-6, atol=1e-7):
            raise ValueError("Normalization range is inconsistent with min/max")
        return shape[0]
    if kind == "CompositeNormalizer":
        children, slices = value["normalizers"], value["dim_slices"]
        if not children or len(children) != len(slices):
            raise ValueError("Invalid composite normalizer")
        offset = 0
        for child, (start, end) in zip(children, slices, strict=True):
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start != offset
                or end <= start
            ):
                raise ValueError("Normalizer slices must cover consecutive dimensions")
            _validate_stats(child, end - start)
            offset = end
        if width is not None and width != offset:
            raise ValueError("Composite normalizer width mismatch")
        return offset
    raise ValueError(f"Unsupported normalization type: {kind!r}")


def _apply(value, stats, inverse=False):
    if stats["type"] == "IdentityNormalizer":
        return value
    if stats["type"] == "CompositeNormalizer":
        parts = [
            _apply(value[..., start:end], child, inverse)
            for child, (start, end) in zip(
                stats["normalizers"], stats["dim_slices"], strict=True
            )
        ]
        return (
            torch.cat(parts, dim=-1)
            if isinstance(value, torch.Tensor)
            else np.concatenate(parts, axis=-1)
        )
    if isinstance(value, torch.Tensor):
        minimum = torch.as_tensor(stats["min"], device=value.device, dtype=value.dtype)
        span = torch.as_tensor(stats["range"], device=value.device, dtype=value.dtype)
    else:
        minimum = np.asarray(stats["min"], dtype=np.float32)
        span = np.asarray(stats["range"], dtype=np.float32)
    return (
        (value + 1) / 2 * span + minimum
        if inverse
        else (value - minimum) / span * 2 - 1
    )


class Normalization:
    """Read the legacy JSON schema without importing or unpickling legacy classes."""

    def __init__(self, data: dict):
        self._data = deepcopy(data)
        if "obs_state" in self._data:
            self._data = {
                "obs": {"state": self._data["obs_state"]},
                "action": self._data["action"],
            }
        self.obs_dim = _validate_stats(self._data["obs"]["state"])
        self.act_dim = _validate_stats(self._data["action"])
        if self.obs_dim is None or self.act_dim is None:
            raise ValueError("Top-level normalizers must specify feature dimensions")

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data)

    def to_dict(self) -> dict:
        return deepcopy(self._data)

    def _transform(self, value, stats, width, inverse=False):
        if isinstance(value, torch.Tensor):
            value = value.float()
            finite = bool(torch.isfinite(value).all())
        else:
            value = np.asarray(value, dtype=np.float32)
            finite = bool(np.isfinite(value).all())
        if value.ndim < 1 or value.shape[-1] != width or not finite:
            raise ValueError(f"Expected finite input with last dimension {width}")
        return _apply(value, stats, inverse)

    def normalize_obs(self, value):
        return self._transform(value, self._data["obs"]["state"], self.obs_dim)

    def normalize_action(self, value):
        return self._transform(value, self._data["action"], self.act_dim)

    def denormalize_action(self, value):
        return self._transform(value, self._data["action"], self.act_dim, inverse=True)
