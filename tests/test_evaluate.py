import numpy as np
import pytest
import torch

from mimicik.data import Episode
from mimicik.evaluate import _pose_xyzw, evaluate_episodes, select_episodes
from mimicik.kinematics import quat_to_matrix


class LinearKinematics:
    joint_names = ["joint1"]

    def pose(self, q):
        q = torch.as_tensor(q, dtype=torch.float32)
        position = torch.cat((q, torch.zeros(len(q), 2)), dim=-1)
        return position, torch.eye(3).repeat(len(q), 1, 1)


class ToyPolicy:
    dof, obs_steps = 1, 2

    def __init__(self, gain=0.5, fail_target=None):
        self.gain, self.fail_target = gain, fail_target
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_delta(self, history):
        history = np.asarray(history)
        target = history[..., -1, 8]
        current = history[..., -1, 1]
        if self.fail_target is not None and np.any(target == self.fail_target):
            raise FloatingPointError("synthetic model failure")
        return np.asarray(self.gain * (target - current))[..., None]

    def step(self, q, current, target):
        frame = np.concatenate((q, current, target))
        return q + self.predict_delta(np.repeat(frame[None], self.obs_steps, axis=0))


def episode(name="test", targets=None, quaternion_sign=1):
    targets = [1.0, 1.0, 1.0, 1.0] if targets is None else targets
    q = np.zeros((len(targets), 1), np.float32)
    current = np.zeros((len(targets), 7), np.float32)
    current[:, 6] = quaternion_sign
    goal = current.copy()
    goal[:, 0] = targets
    return Episode(name, q, current, np.asarray(targets, np.float32)[:, None], goal)


def test_recorded_and_feedback_are_distinct_protocols():
    ep = episode()
    recorded = evaluate_episodes(
        ToyPolicy(), [ep], LinearKinematics(), "recorded", batch_size=3
    )
    feedback = evaluate_episodes(ToyPolicy(), [ep], LinearKinematics(), "feedback")
    assert recorded["aggregate"]["mean_position_mm"] == 500
    # Commands .5,.75,.875,.9375: ideal feedback progressively approaches x=1.
    assert feedback["aggregate"]["mean_position_mm"] == pytest.approx(234.375)
    assert recorded["aggregate"]["joint_step_d95_rad_per_frame"] == 0
    assert feedback["aggregate"]["joint_step_d95_rad_per_frame"] > 0


def test_failed_suffix_stays_in_success_denominator_and_resets_each_episode():
    policy = ToyPolicy(gain=1, fail_target=3)
    result = evaluate_episodes(
        policy, [episode("a", [1, 2, 3, 4]), episode("b", [3, 3])], LinearKinematics()
    )
    assert policy.resets == 2
    assert result["aggregate"]["episodes"] == 2
    assert result["aggregate"]["failed_episodes"] == 2
    assert result["aggregate"]["planned_frames"] == 6
    assert result["aggregate"]["valid_frames"] == 2
    assert result["aggregate"]["sr_5mm_percent"] == 25  # macro: mean(50%, 0%)
    assert result["aggregate"]["episodes_with_valid_prefix"] == 1
    assert result["episodes"][0]["failure_frame"] == 2
    assert result["episodes"][1]["mean_position_mm"] is None


def test_optional_protection_counts_rejected_commands_as_failure():
    result = evaluate_episodes(
        ToyPolicy(gain=1),
        [episode(targets=[0.1, 1])],
        LinearKinematics(),
        protocol="feedback",
        max_step_rad=0.2,
    )
    assert result["aggregate"]["valid_frames"] == 1
    assert result["aggregate"]["sr_5mm_percent"] == 50
    assert result["episodes"][0]["failure_reason"] == "max_step_rad_exceeded"


def test_rotation_is_quaternion_sign_invariant_and_pi_conversion_is_stable():
    positive = evaluate_episodes(
        ToyPolicy(gain=1), [episode(quaternion_sign=1)], LinearKinematics()
    )
    negative = evaluate_episodes(
        ToyPolicy(gain=1), [episode(quaternion_sign=-1)], LinearKinematics()
    )
    assert (
        positive["aggregate"]["mean_rotation_deg"]
        == negative["aggregate"]["mean_rotation_deg"]
        == 0
    )
    for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.1, 0.2, 0.3, 0.9]):
        rotation = quat_to_matrix(torch.tensor(q, dtype=torch.float32))
        pose = _pose_xyzw(np.zeros(3), rotation.numpy())
        torch.testing.assert_close(
            quat_to_matrix(torch.tensor(pose[3:])), rotation, atol=1e-6, rtol=1e-6
        )


def test_no_implicit_train_evaluation_and_no_group_leakage():
    episodes = [episode("train"), episode("val"), episode("test")]
    with pytest.raises(ValueError, match="split manifest"):
        select_episodes(episodes)
    assert len(select_episodes(episodes, "all")[0]) == 3
    manifest = {"train": ["train"], "val": ["val"], "test": ["test"]}
    selected, _ = select_episodes(episodes, "test", manifest)
    assert [ep.name for ep in selected] == ["test"]
    episodes[0].group_id = episodes[1].group_id = "same_parent"
    with pytest.raises(ValueError, match="leaks"):
        select_episodes(episodes, "val", manifest)
