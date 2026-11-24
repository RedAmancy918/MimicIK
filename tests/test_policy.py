"""Inference boundaries: safe bundles, normalization, pose and history semantics."""

import numpy as np
import pytest
import torch

from mimicik.checkpoint import load_bundle, save_bundle
from mimicik.model import MimicIKModel
from mimicik.normalization import Normalization
from mimicik.policy import MimicIK


def config():
    return {
        "task": {"act_dim": 6, "obs_dim": 20, "obs_steps": 2, "horizon": 1},
        "network": {
            "emb_dim": 16,
            "num_layers": 1,
            "n_heads": 2,
            "num_encoder_layers": 2,
            "dropout": 0.0,
        },
        "optimization": {"t_two_step": 0.9},
    }


def minmax(n, lo=-1.0, hi=1.0):
    return {
        "type": "MinMaxNormalizer",
        "min": [lo] * n,
        "max": [hi] * n,
        "range": [hi - lo] * n,
    }


def normalization():
    return Normalization.from_dict(
        {
            "obs": {
                "state": {
                    "type": "CompositeNormalizer",
                    "dim_slices": [[0, 6], [6, 9], [9, 13], [13, 16], [16, 20]],
                    "normalizers": [
                        minmax(6),
                        minmax(3),
                        {"type": "IdentityNormalizer"},
                        minmax(3),
                        {"type": "IdentityNormalizer"},
                    ],
                }
            },
            "action": minmax(6, -0.2, 0.2),
        }
    )


def pose():
    return np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], dtype=np.float32)


def test_numpy_torch_normalization_agree_and_preserve_gradient():
    normalizer = normalization()
    action = np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(2, 6)
    tensor = torch.tensor(action, requires_grad=True)
    normalized = normalizer.normalize_action(tensor)
    np.testing.assert_allclose(
        normalized.detach().numpy(), normalizer.normalize_action(action), atol=1e-7
    )
    recovered = normalizer.denormalize_action(normalized)
    recovered.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
    np.testing.assert_allclose(recovered.detach().numpy(), action, atol=1e-7)
    assert normalized.max() > 1  # Extrapolation is not silently clipped.


def test_bundle_roundtrip_and_corruption_rejection(tmp_path):
    model = MimicIKModel(config()).eval()
    path = tmp_path / "bundle"
    save_bundle(path, model, normalization(), model.config, {"robot": "test"})
    loaded, norm, metadata = load_bundle(path)
    assert metadata["robot"] == "test"
    assert norm.to_dict() == normalization().to_dict()
    obs = torch.randn(2, 2, 20)
    torch.testing.assert_close(model(obs), loaded(obs), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_bundle(path, model, norm, model.config)
    (path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        load_bundle(path)


def test_bundle_overwrite_preserves_unrelated_files(tmp_path):
    model = MimicIKModel(config()).eval()
    path = tmp_path / "unrelated"
    path.mkdir()
    (path / "metadata.json").write_text('{"project": "keep"}')
    (path / "notes.txt").write_text("must survive")
    with pytest.raises(FileExistsError):
        save_bundle(path, model, normalization(), model.config, overwrite=True)
    assert (path / "notes.txt").read_text() == "must survive"
    clean = tmp_path / "clean"
    save_bundle(clean, model, normalization(), model.config)
    save_bundle(
        clean, model, normalization(), model.config, {"revision": 2}, overwrite=True
    )
    assert load_bundle(clean)[2]["revision"] == 2


def test_stateless_quaternion_sign_and_batch_predictions():
    policy = MimicIK(MimicIKModel(config()), normalization())
    frame = np.concatenate([np.zeros(6, dtype=np.float32), pose(), pose()])
    history = np.stack([frame, frame])
    opposite = history.copy()
    opposite[:, 9:13] *= -1
    opposite[:, 16:20] *= -1
    np.testing.assert_array_equal(
        policy.predict_delta(history), policy.predict_delta(opposite)
    )
    batch = policy.predict_delta(np.stack([history, opposite]))
    assert batch.shape == (2, 6)
    # Batched CPU kernels can differ by a few float32 ULPs across rows.
    np.testing.assert_allclose(batch[0], batch[1], rtol=1e-6, atol=1e-7)
    assert len(policy._history) == 0
    invalid = history.copy()
    invalid[:, 9:13] = 0
    with pytest.raises(ValueError, match="nonzero"):
        policy.predict_delta(invalid)


def test_stateful_history_is_per_instance_and_failure_does_not_mutate():
    policy = MimicIK(MimicIKModel(config()), normalization())
    independent = MimicIK(MimicIKModel(config()), normalization())
    observed = []

    def predict(history, **kwargs):
        observed.append(history.copy())
        return np.full(6, 0.1, dtype=np.float32)

    policy.predict_delta = predict
    q = np.zeros(6, dtype=np.float32)
    np.testing.assert_allclose(policy.step(q, pose(), pose()), q + 0.1)
    np.testing.assert_array_equal(observed[0][0], observed[0][1])
    policy.step(q + 0.2, pose(), pose())
    np.testing.assert_array_equal(observed[1][0, :6], q)
    np.testing.assert_allclose(observed[1][1, :6], q + 0.2)
    assert len(independent._history) == 0
    before = np.stack(policy._history)
    with pytest.raises(ValueError):
        policy.step(q, np.zeros(7), pose())
    np.testing.assert_array_equal(np.stack(policy._history), before)
    policy.reset()
    assert not policy._history


def test_legacy_conversion_requires_trust_and_strips_local_paths(tmp_path):
    from tools.export_checkpoint import convert_checkpoint, load_trusted_checkpoint

    with pytest.raises(ValueError, match="disabled"):
        load_trusted_checkpoint(tmp_path / "missing.pt")
    original = config()
    original["task"]["dataset_path"] = "/private/training/data"
    original["log"] = {"log_dir": "/private/training/logs"}
    model = MimicIKModel(original)
    source = tmp_path / "trusted.pt"
    torch.save(
        {
            "framework": "controlled_ik_v1",
            "learner": "mip_fk",
            "config": original,
            "normalizer": normalization().to_dict(),
            "model_state": {
                "flow_map_ema": model.flow_map.state_dict(),
                "encoder_ema": model.encoder.state_dict(),
            },
            "training_state": {"n_gradient_step": 300000, "status": "completed"},
        },
        source,
    )
    destination = tmp_path / "export"
    convert_checkpoint(source, destination, trust_pickle=True)
    assert "/private" not in (destination / "config.json").read_text()
    assert "log" not in (destination / "config.json").read_text()
    restored, _, metadata = load_bundle(destination)
    assert metadata["training_step"] == 300000
    model.eval()
    obs = torch.randn(2, 2, 20)
    torch.testing.assert_close(model(obs), restored(obs), rtol=0, atol=0)
