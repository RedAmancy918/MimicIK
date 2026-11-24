import numpy as np
import pytest

from mimicik.data import (
    Episode,
    EpisodeDataset,
    fit_normalization,
    load_episodes,
    split_episodes,
)


def make_episode(name, value=0.0, n=5, group=None):
    q = np.full((n, 6), value, dtype=np.float32)
    q[:, 0] += np.arange(n, dtype=np.float32)
    pose = np.zeros((n, 7), dtype=np.float32)
    pose[:, 6] = 1
    return Episode(name, q, pose, q + 0.1, pose.copy(), group_id=group)


def test_history_never_crosses_episode_and_training_excludes_start():
    episodes = [make_episode("a"), make_episode("b", 100)]
    normalization = fit_normalization(episodes)
    dataset = EpisodeDataset(episodes, normalization, obs_steps=2)
    assert len(dataset) == 8
    expected = normalization.normalize_obs(episodes[1].frames[:2])
    np.testing.assert_array_equal(dataset[4]["obs"].numpy(), expected)
    padded = EpisodeDataset(episodes, normalization, obs_steps=2, pad_start=True)
    np.testing.assert_array_equal(
        padded[5]["obs"][0].numpy(), padded[5]["obs"][1].numpy()
    )
    assert padded[5]["q_curr_raw"][0, 0] == 100


def test_validation_outliers_do_not_change_training_normalization():
    train_ep, val_ep = make_episode("train"), make_episode("val", 10000)
    train, val, _ = split_episodes(
        [train_ep, val_ep], manifest={"train": ["train"], "val": ["val"]}
    )
    normalizer = fit_normalization(train)
    assert normalizer.to_dict()["obs"]["state"]["normalizers"][0]["max"][0] == 4
    assert normalizer.normalize_obs(val[0].frames)[0, 0] > 1
    # Identity quaternion coordinates must survive, rather than become min/max.
    np.testing.assert_array_equal(
        normalizer.normalize_obs(train[0].frames)[:, 9:13], train[0].eef[:, 3:]
    )


def test_group_split_and_manifest_reject_leakage():
    episodes = [
        make_episode("a-left", group="a"),
        make_episode("a-right", group="a"),
        make_episode("b-left", group="b"),
        make_episode("b-right", group="b"),
    ]
    train, val, _split = split_episodes(episodes, val_fraction=0.5, seed=4)
    assert {ep.group_id for ep in train}.isdisjoint({ep.group_id for ep in val})
    with pytest.raises(ValueError, match="leaks"):
        split_episodes(
            episodes,
            manifest={"train": ["a-left", "b-left"], "val": ["a-right", "b-right"]},
        )
    with pytest.raises(ValueError, match="more than one"):
        split_episodes(episodes, manifest={"train": ["a-left"], "val": ["a-left"]})


def test_npz_roundtrip_requires_explicit_targets(tmp_path):
    ep = make_episode("example")
    path = tmp_path / "example.npz"
    np.savez(path, q=ep.q, eef=ep.eef, target_q=ep.target_q, target_eef=ep.target_eef)
    loaded = load_episodes(tmp_path)[0]
    np.testing.assert_array_equal(loaded.target_q, ep.target_q)
    np.savez(path, q=ep.q, eef=ep.eef)
    with pytest.raises(ValueError, match="missing arrays"):
        load_episodes(tmp_path)


def test_parquet_rows_are_same_row_pairs(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ep = make_episode("example")
    pq.write_table(
        pa.table(
            {
                "episode_index": [17] * len(ep),
                "frame_index": list(range(len(ep))),
                "observation.state": ep.q.tolist(),
                "observation.state_quat": ep.eef.tolist(),
                "action": ep.target_q.tolist(),
                "action_quat": ep.target_eef.tolist(),
            }
        ),
        tmp_path / "episode_17.parquet",
    )
    loaded = load_episodes(tmp_path)[0]
    assert loaded.name == "17"
    assert len(loaded) == len(ep)
    np.testing.assert_array_equal(loaded.target_q, ep.target_q)


def test_quaternion_sign_is_canonical_without_changing_magnitude():
    ep = make_episode("a")
    ep.eef[:, 6] = -2
    loaded = Episode("b", ep.q, ep.eef, ep.target_q, ep.target_eef)
    np.testing.assert_array_equal(loaded.eef[:, 6], np.full(len(ep), 2))
    ep.eef[:, 3:] = 0
    with pytest.raises(ValueError, match="nonzero"):
        Episode("c", ep.q, ep.eef, ep.target_q, ep.target_eef)
