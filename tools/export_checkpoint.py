"""One-time conversion of explicitly trusted research checkpoints to safe bundles.

This tool is separate from the inference package. Pickle can execute arbitrary
code, so conversion requires --trust-pickle and must only be used on checkpoints
you produced or obtained from a trusted source. A checksum is not an authenticity
guarantee unless obtained through a trusted channel.
"""

import argparse
import json
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mimicik.checkpoint import save_bundle, sha256
from mimicik.model import MimicIKModel
from mimicik.normalization import Normalization

FIELDS = {
    "task": (
        "act_dim",
        "obs_dim",
        "obs_steps",
        "horizon",
        "env_name",
        "obs_type",
        "env_type",
        "action_type",
        "abs_action",
    ),
    "network": (
        "network_type",
        "emb_dim",
        "num_layers",
        "n_heads",
        "dropout",
        "encoder_type",
        "num_encoder_layers",
        "encoder_dropout",
        "timestep_emb_type",
        "timestep_emb_dim",
        "disable_time_embedding",
    ),
    "optimization": ("loss_type", "t_two_step"),
}


@contextmanager
def legacy_normalizer_classes():
    """Resolve only the old normalizer class names without installing old MIP.

    These attribute containers do not make pickle safe. The explicit trust gate
    still applies to the entire file; arbitrary other pickle objects may execute.
    """
    names = ("mip", "mip.dataset_utils")
    original = {name: sys.modules.get(name) for name in names}
    root = types.ModuleType("mip")
    root.__path__ = []
    module = types.ModuleType("mip.dataset_utils")
    for name in (
        "CompositeNormalizer",
        "MinMaxNormalizer",
        "IdentityNormalizer",
        "EmptyNormalizer",
    ):
        setattr(module, name, type(name, (), {"__module__": "mip.dataset_utils"}))
    root.dataset_utils = module
    sys.modules.update({"mip": root, "mip.dataset_utils": module})
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def normalizer_to_dict(data):
    def convert(value):
        if isinstance(value, dict):
            return value
        kind = type(value).__name__
        if kind == "CompositeNormalizer":
            return {
                "type": kind,
                "dim_slices": [list(x) for x in value.dim_slices],
                "normalizers": [convert(x) for x in value.normalizers],
            }
        if kind == "MinMaxNormalizer":
            return {
                "type": kind,
                **{
                    key: np.asarray(getattr(value, key)).tolist()
                    for key in ("min", "max", "range")
                },
            }
        if kind in ("IdentityNormalizer", "EmptyNormalizer"):
            return {"type": "IdentityNormalizer"}
        raise ValueError(f"Unsupported legacy normalizer class {kind}")

    obs = data["obs_state"] if "obs_state" in data else data["obs"]["state"]
    return {"obs": {"state": convert(obs)}, "action": convert(data["action"])}


def load_trusted_checkpoint(path, *, trust_pickle=False, expected_sha256=None):
    if not trust_pickle:
        raise ValueError(
            "Legacy pickle is disabled: supply --trust-pickle only for a trusted checkpoint"
        )
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Source checkpoint SHA256 mismatch")
    with legacy_normalizer_classes():
        return torch.load(path, map_location="cpu", weights_only=False)


def convert_checkpoint(
    source,
    destination,
    *,
    trust_pickle=False,
    expected_sha256=None,
    normalizer_path=None,
    overwrite=False,
):
    source = Path(source)
    bundle = load_trusted_checkpoint(
        source, trust_pickle=trust_pickle, expected_sha256=expected_sha256
    )
    if not isinstance(bundle, dict) or not isinstance(bundle.get("config"), dict):
        raise TypeError(
            "Expected a research bundle containing a plain configuration dictionary"
        )
    if bundle.get("framework") == "controlled_ik_v1":
        if bundle.get("learner") not in ("mip_fk", "mip_no_fk"):
            raise ValueError("Only controlled MIP/MIP+FK checkpoints can be exported")
        state = bundle["model_state"]
        framework = "controlled_ik_v1"
    elif (
        "framework" not in bundle
        and {"flow_map_ema", "encoder_ema", "normalizer"} <= bundle.keys()
    ):
        state = bundle
        framework = "legacy_train_ik_bundle"
    else:
        raise ValueError("Unsupported research checkpoint schema; no implicit fallback")
    if not {"flow_map_ema", "encoder_ema"} <= state.keys():
        raise ValueError("EMA flow map and encoder weights are required")
    original_config = bundle["config"]
    config = {
        section: {
            key: original_config[section][key]
            for key in fields
            if key in original_config[section]
        }
        for section, fields in FIELDS.items()
    }
    normalizer_data = normalizer_to_dict(bundle["normalizer"])
    normalization = Normalization.from_dict(normalizer_data)
    if normalizer_path is not None:
        sidecar = Normalization.from_dict(json.loads(Path(normalizer_path).read_text()))
        if normalization.to_dict() != sidecar.to_dict():
            raise ValueError("Normalizer sidecar differs from the source checkpoint")
    model = MimicIKModel(config)
    model.flow_map.load_state_dict(state["flow_map_ema"], strict=True)
    model.encoder.load_state_dict(state["encoder_ema"], strict=True)
    model.eval()
    metadata = {
        "source_checkpoint": {"filename": source.name, "sha256": sha256(source)},
        "source_framework": framework,
        "weights": "EMA",
        "seed": int(original_config["optimization"].get("seed", 0)),
        "learner": bundle.get("learner", "mip"),
        "joint_unit": "radian",
        "position_unit": "metre",
        "quaternion_order": "xyzw",
        "quaternion_preprocessing": "flip all quaternion components when w < 0; no renormalization",
        "inference": "two complete normalized delta estimates; second input is first prediction",
    }
    run_id = original_config.get("training", {}).get("run_id")
    if run_id is not None:
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError(
                "Unexpected run identifier; refusing to export path-like metadata"
            )
        metadata["run_id"] = run_id
    training_state = bundle.get("training_state", {})
    if "n_gradient_step" in training_state:
        metadata["training_step"] = int(training_state["n_gradient_step"])
        metadata["checkpoint_status"] = str(training_state.get("status", "unknown"))
    if framework == "controlled_ik_v1":
        metadata["checkpoint_rule"] = (
            "final_300000_adamw_updates_ema"
            if metadata.get("training_step") == 300000
            and training_state.get("status") == "completed"
            else "research_checkpoint"
        )
    save_bundle(
        destination, model, normalization, config, metadata, overwrite=overwrite
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow code execution by a legacy checkpoint you explicitly trust",
    )
    parser.add_argument(
        "--source-sha256",
        help="Verify the trusted original file checksum before unpickling",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        help="Optional JSON sidecar to cross-check exact statistics",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing safe bundle"
    )
    args = parser.parse_args(argv)
    if not args.trust_pickle:
        parser.error(
            "--trust-pickle is required; inference does not need this conversion tool"
        )
    result = convert_checkpoint(
        args.source,
        args.destination,
        trust_pickle=args.trust_pickle,
        expected_sha256=args.source_sha256,
        normalizer_path=args.normalizer,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
