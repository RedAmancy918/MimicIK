"""Numerical contracts for the compact release model."""

import ast
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from mimicik.model import MimicIKModel


def tiny_config():
    return {
        "task": {
            "act_dim": 6,
            "obs_dim": 20,
            "obs_steps": 2,
            "horizon": 1,
            "obs_type": "state",
            "env_name": "ik_delta_joint",
        },
        "network": {
            "network_type": "sudeepdit",
            "encoder_type": "mlp",
            "emb_dim": 16,
            "num_layers": 1,
            "n_heads": 2,
            "num_encoder_layers": 2,
            "dropout": 0.0,
            "encoder_dropout": 0.0,
            "timestep_emb_type": "positional",
        },
        "optimization": {"loss_type": "mip", "t_two_step": 0.9},
    }


def test_decoder_uses_predicted_first_endpoint_without_residual_sum():
    model = MimicIKModel(tiny_config()).eval()
    calls = []

    def velocity(time, action, condition):
        calls.append((time.clone(), action.clone()))
        return torch.full_like(action, 2 if len(calls) == 1 else 7)

    model.flow_map.get_velocity = velocity
    output = model.predict_normalized(torch.zeros(3, 2, 20))
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][1], torch.zeros(3, 1, 6))
    torch.testing.assert_close(calls[1][1], torch.full((3, 1, 6), 2.0))
    torch.testing.assert_close(calls[1][0], torch.full((3,), 0.9))
    torch.testing.assert_close(output, torch.full((3, 1, 6), 7.0))


def test_model_rejects_unhandled_checkpoint_types():
    for section, key, value in (
        ("network", "encoder_type", "identity"),
        ("task", "obs_dim", 24),
        ("task", "horizon", 2),
        ("optimization", "loss_type", "regression"),
    ):
        config = deepcopy(tiny_config())
        config[section][key] = value
        with pytest.raises(ValueError):
            MimicIKModel(config)


def _source_definitions(path, names, namespace):
    text = path.read_text()
    nodes = [
        node
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    code = ast.Module(body=nodes, type_ignores=[])
    exec(compile(code, str(path), "exec"), namespace)  # noqa: S102 - explicitly supplied trusted reference checkout


@pytest.mark.skipif(
    not os.environ.get("MIMICIK_REFERENCE_SOURCE"),
    reason="Set MIMICIK_REFERENCE_SOURCE to the original research checkout for parity audit",
)
def test_upstream_state_dict_and_predictions_match():
    """Compare against locally supplied upstream code; never download or vendor it."""
    source = Path(os.environ["MIMICIK_REFERENCE_SOURCE"]) / "mip" / "mip"
    scope = {
        "torch": torch,
        "nn": torch.nn,
        "np": np,
        "copy": __import__("copy"),
        "F": torch.nn.functional,
    }
    _source_definitions(source / "embeddings.py", {"PositionalEmbedding"}, scope)
    scope["SUPPORTED_TIMESTEP_EMBEDDING"] = {"positional": scope["PositionalEmbedding"]}
    _source_definitions(source / "networks/base.py", {"BaseNetwork"}, scope)
    network_text = (source / "networks/sudeepdit.py").read_text()
    names = {
        node.name
        for node in ast.parse(network_text).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name != "test_sudeepdit"
    }
    _source_definitions(source / "networks/sudeepdit.py", names, scope)
    _source_definitions(source / "torch_utils.py", {"at_least_ndim"}, scope)
    _source_definitions(
        source / "encoders.py", {"get_mask", "BaseEncoder", "Mlp", "MLPEncoder"}, scope
    )
    model = MimicIKModel(tiny_config()).eval()
    reference_net = scope["SudeepDiT"](
        6, 1, 16, 2, d_model=16, n_heads=2, depth=1, dropout=0.0
    ).eval()
    reference_encoder = scope["MLPEncoder"](20, 16, 2, [16, 16], dropout=0.0).eval()
    reference_net.load_state_dict(model.flow_map.net.state_dict(), strict=True)
    reference_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(47)
    obs = torch.randn(4, 2, 20, generator=generator)
    zeros, t0, t1 = torch.zeros(4, 1, 6), torch.zeros(4), torch.full((4,), 0.9)
    with torch.no_grad():
        cond = reference_encoder(obs, None)
        first = reference_net(zeros, t0, t0, cond)[0]
        final = reference_net(first, t1, t1, cond)[0]
        torch.testing.assert_close(model.encoder(obs), cond, rtol=0, atol=0)
        torch.testing.assert_close(
            model.predict_normalized(obs, "first"), first, rtol=0, atol=0
        )
        torch.testing.assert_close(model.predict_normalized(obs), final, rtol=0, atol=0)
