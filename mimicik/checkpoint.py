"""Portable bundles containing JSON and safetensors, never executable pickle."""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model import MimicIKModel
from .normalization import Normalization

FORMAT = "mimicik.safetensors.v1"
FILES = ("config.json", "normalizer.json", "model.safetensors")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _check_replaceable(path, overwrite):
    if path.is_symlink():
        raise FileExistsError(f"Refusing to replace a symlink: {path}")
    if not path.exists():
        return
    if not overwrite or not path.is_dir():
        raise FileExistsError(
            f"Refusing to replace {path}; use overwrite=True for a bundle"
        )
    expected_files = {*FILES, "metadata.json"}
    if {item.name for item in path.iterdir()} != expected_files:
        raise FileExistsError(
            f"Refusing to replace {path}: contains files outside a MimicIK bundle"
        )
    if any(
        (path / name).is_symlink() or not (path / name).is_file()
        for name in expected_files
    ):
        raise FileExistsError(
            f"Refusing to replace {path}: expected regular bundle files"
        )
    try:
        metadata = json.loads((path / "metadata.json").read_text())
    except (OSError, ValueError) as exc:
        raise FileExistsError(
            f"Refusing to replace {path}: invalid bundle metadata"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("format") != FORMAT:
        raise FileExistsError(f"Refusing to replace {path}: not a MimicIK bundle")
    manifest = metadata.get("sha256")
    if not isinstance(manifest, dict) or set(manifest) != set(FILES):
        raise FileExistsError(f"Refusing to replace {path}: invalid integrity manifest")
    if any(manifest[name] != sha256(path / name) for name in FILES):
        raise FileExistsError(
            f"Refusing to replace {path}: existing bundle has changed"
        )


def save_bundle(path, model, normalization, config, metadata=None, *, overwrite=False):
    """Export inference weights to a verified bundle directory.

    Pass the EMA model when exporting training results. Only tensors, supplied
    configuration/metadata, and normalization statistics are serialized. Keep
    private information out of the supplied JSON fields before sharing. An
    existing bundle is replaced only when ``overwrite=True``; unrelated files
    and directories are rejected.
    """
    path = Path(path)
    if model.config != config:
        raise ValueError("Export configuration differs from the model configuration")
    if (normalization.obs_dim, normalization.act_dim) != (model.obs_dim, model.dof):
        raise ValueError("Model and normalizer dimensions differ")
    _check_replaceable(path, overwrite)
    meta = deepcopy(metadata or {})
    if {"format", "sha256"} & meta.keys():
        raise ValueError("metadata cannot override the format or integrity manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    backup = None
    try:
        _write_json(staging / "config.json", config)
        _write_json(staging / "normalizer.json", normalization.to_dict())
        tensors = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Cannot export nonfinite model tensors")
        save_file(tensors, str(staging / "model.safetensors"))
        meta.update(
            format=FORMAT, sha256={name: sha256(staging / name) for name in FILES}
        )
        _write_json(staging / "metadata.json", meta)
        _check_replaceable(path, overwrite)
        if path.exists():
            backup = path.with_name(f".{path.name}-backup-{uuid.uuid4().hex}")
            os.replace(path, backup)
        try:
            os.replace(staging, path)
        except BaseException:
            if backup is not None:
                os.replace(backup, path)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return path


def load_bundle(path, device="cpu"):
    """Load a local bundle and verify file integrity before parsing weights.

    Hashes detect accidental corruption; authenticate the download separately
    using the release's published checksum when obtaining a bundle remotely.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Expected a bundle directory containing metadata.json")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("format") != FORMAT:
        raise ValueError("Unsupported bundle format")
    manifest = metadata.get("sha256", {})
    if set(manifest) != set(FILES):
        raise ValueError(
            "Bundle integrity manifest must list config, normalizer and tensors"
        )
    for name in FILES:
        expected = manifest[name]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or sha256(path / name) != expected
        ):
            raise ValueError(f"Bundle SHA256 mismatch: {name}")
    config = json.loads((path / "config.json").read_text())
    normalization = Normalization.from_dict(
        json.loads((path / "normalizer.json").read_text())
    )
    model = MimicIKModel(config)
    if (normalization.obs_dim, normalization.act_dim) != (model.obs_dim, model.dof):
        raise ValueError("Model and normalizer dimensions differ")
    weights = load_file(str(path / "model.safetensors"), device="cpu")
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Bundle contains nonfinite model tensors")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, normalization, metadata
