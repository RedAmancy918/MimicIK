from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mimicik.checkpoint import load_bundle
from mimicik.kinematics import Kinematics, fk_pose_loss, quat_to_matrix, rotation_error
from mimicik.training import train


def config():
    result = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "smoke.yaml").read_text()
    )
    result["optimization"]["gradient_steps"] = 4
    result["training"].update(log_every=2, save_every=2, eval_every=2)
    return result


def write_dataset(path):
    path.mkdir()
    rng = np.random.default_rng(12)
    for i in range(4):
        q = rng.normal(size=(8, 6)).astype(np.float32) * 0.1
        dq = rng.normal(size=(8, 6)).astype(np.float32) * 0.01
        pose = np.zeros((8, 7), dtype=np.float32)
        pose[:, :3] = rng.normal(size=(8, 3)).astype(np.float32) * 0.1
        pose[:, 6] = 1
        np.savez(
            path / f"episode_{i}.npz", q=q, eef=pose, target_q=q + dq, target_eef=pose
        )


def test_interrupted_resume_matches_uninterrupted_ema_exactly(tmp_path):
    data = tmp_path / "data"
    write_dataset(data)
    cfg = config()
    whole = train(cfg, data, tmp_path / "whole")
    paused = train(cfg, data, tmp_path / "resumed", stop_after=2)
    assert not paused["completed"]
    resumed = train(
        cfg, data, tmp_path / "resumed", resume=tmp_path / "resumed" / "resume.pt"
    )
    assert whole["steps_completed"] == resumed["steps_completed"] == 4
    left = load_bundle(tmp_path / "whole" / "final")[0]
    right = load_bundle(tmp_path / "resumed" / "final")[0]
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
    assert whole["validation_normalized_mse"] == resumed["validation_normalized_mse"]


def test_resume_rejects_changed_budget(tmp_path):
    data = tmp_path / "data"
    write_dataset(data)
    cfg = config()
    train(cfg, data, tmp_path / "out", stop_after=2)
    cfg["optimization"]["gradient_steps"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        train(cfg, data, tmp_path / "out", resume=tmp_path / "out" / "resume.pt")


def test_requested_fk_never_silently_disables(tmp_path):
    data = tmp_path / "data"
    write_dataset(data)
    cfg = config()
    cfg["task"]["fk_loss_weight"] = 0.1
    with pytest.raises(ValueError, match="requires"):
        train(cfg, data, tmp_path / "out")


def test_resume_cannot_overwrite_another_run_or_unknown_directory(tmp_path):
    data = tmp_path / "data"
    write_dataset(data)
    cfg = config()
    train(cfg, data, tmp_path / "a", stop_after=2)
    other = tmp_path / "b"
    other.mkdir()
    sentinel = other / "config.json"
    sentinel.write_text('{"unrelated": true}')
    before = sentinel.read_bytes()
    with pytest.raises(ValueError, match="same output directory"):
        train(cfg, data, other, resume=tmp_path / "a" / "resume.pt")
    assert sentinel.read_bytes() == before
    assert list(other.iterdir()) == [sentinel]


def test_resume_rejects_tampered_normalizer_payload(tmp_path):
    data = tmp_path / "data"
    write_dataset(data)
    cfg = config()
    output = tmp_path / "out"
    train(cfg, data, output, stop_after=2)
    state = torch.load(output / "resume.pt", weights_only=True)
    state["normalization"]["action"]["min"][0] += 0.1
    torch.save(state, output / "resume.pt")
    with pytest.raises(ValueError, match="normalization payload"):
        train(cfg, data, output, resume=output / "resume.pt")


def test_fk_versions_and_differentiability(tmp_path):
    pytest.importorskip("pytorch_kinematics")
    urdf = tmp_path / "one_joint.urdf"
    urdf.write_text("""<robot name="test"><link name="base"/><link name="arm"/><link name="tcp"/>
      <joint name="joint1" type="revolute"><parent link="base"/><child link="arm"/>
      <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="2"/></joint>
      <joint name="tool" type="fixed"><parent link="arm"/><child link="tcp"/>
      <origin xyz="1 0 0"/></joint></robot>""")
    fk = Kinematics(urdf, "tcp")
    q = torch.tensor([[0.2]], requires_grad=True)
    position, rotation = fk.pose(q)
    target = torch.tensor([[1.0, 0, 0, 0, 0, 0, 1.0]])
    total, _info = fk_pose_loss(
        position, rotation, target, {"fk_loss_version": "tolerance_normalized_v2"}
    )
    total.backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    identity = torch.eye(3)[None]
    equal_position = torch.zeros((1, 3))
    target = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1.0]])
    v1, _ = fk_pose_loss(
        equal_position, identity, target, {"fk_loss_version": "legacy_mixed_units_v1"}
    )
    v2, _ = fk_pose_loss(
        equal_position, identity, target, {"fk_loss_version": "tolerance_normalized_v2"}
    )
    assert v1 > 0  # Historical clamp floor is retained intentionally.
    assert v2 == 0
    torch.testing.assert_close(
        rotation_error(identity, quat_to_matrix(-target[:, 3:])), torch.zeros(1)
    )
