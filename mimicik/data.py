"""Episode-preserving data loading for the portable MimicIK training recipe.

The four arrays in an episode describe paired observations and commands at the
same row. In particular, this module never invents a target by shifting EEF
observations. Dataset-specific conversion belongs in an explicit preparation
step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import Normalization


@dataclass
class Episode:
    name: str
    q: np.ndarray
    eef: np.ndarray
    target_q: np.ndarray
    target_eef: np.ndarray
    group_id: str | None = None

    def __post_init__(self):
        for name in ("q", "eef", "target_q", "target_eef"):
            value = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
                raise ValueError(
                    f"{self.name}: {name} must be a finite, nonempty 2D array"
                )
            setattr(self, name, value)
        n, dof = self.q.shape
        if dof < 1 or self.target_q.shape != (n, dof):
            raise ValueError(
                f"{self.name}: q and target_q must have identical [T, DOF] shapes"
            )
        if self.eef.shape != (n, 7) or self.target_eef.shape != (n, 7):
            raise ValueError(
                f"{self.name}: EEF arrays must have shape [T, 7] (xyz + xyzw)"
            )
        for value in (self.eef, self.target_eef):
            if np.any(np.linalg.norm(value[:, 3:], axis=1) <= 1e-12):
                raise ValueError(f"{self.name}: EEF quaternions must be nonzero")
            # Preserve the AIRBOT training convention: choose w >= 0 without
            # changing the magnitude of the recorded quaternion.
            value[value[:, 6] < 0, 3:] *= -1

    def __len__(self):
        return len(self.q)

    @property
    def frames(self):
        return np.concatenate((self.q, self.eef, self.target_eef), axis=-1)


def _scalar_text(value, default):
    return str(np.asarray(value).item()) if value is not None else default


def load_episodes(path: str | Path) -> list[Episode]:
    """Read NPZ episodes or canonical Parquet episodes; never unpickle data.

    NPZ: ``q``, ``eef``, ``target_q``, ``target_eef`` and optional scalar
    ``episode_id``/``group_id``. Each file is one episode.

    Parquet: ``observation.state``, ``observation.state_quat``, ``action``,
    ``action_quat``. If an episode ID column exists, each file is split by ID.
    Rows retain their stored order; a frame_index column, when present, must
    increase within each episode.
    """
    path = Path(path).expanduser()
    if path.is_file():
        files = [path]
    else:
        npz_files = sorted(path.rglob("*.npz"))
        parquet_files = sorted(path.rglob("*.parquet"))
        if npz_files and parquet_files:
            raise ValueError(
                "Data directory mixes NPZ and Parquet; choose one prepared dataset"
            )
        files = npz_files or parquet_files
    if not files:
        raise FileNotFoundError(f"No NPZ or Parquet episodes found at {path}")
    episodes = []
    for file in files:
        if file.suffix == ".npz":
            with np.load(file, allow_pickle=False) as arrays:
                missing = {"q", "eef", "target_q", "target_eef"} - set(arrays.files)
                if missing:
                    raise ValueError(f"{file}: missing arrays {sorted(missing)}")
                name = _scalar_text(arrays.get("episode_id"), file.stem)
                group = _scalar_text(arrays.get("group_id"), name)
                episodes.append(
                    Episode(
                        name,
                        *(arrays[k] for k in ("q", "eef", "target_q", "target_eef")),
                        group_id=group,
                    )
                )
        elif file.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "Parquet loading requires pip install 'mimicik[train]'"
                ) from exc
            columns = (
                "observation.state",
                "observation.state_quat",
                "action",
                "action_quat",
            )
            schema = pq.ParquetFile(file).schema_arrow.names
            missing = set(columns) - set(schema)
            if missing:
                raise ValueError(
                    f"{file}: missing canonical columns {sorted(missing)}; prepare the data first"
                )
            extra = [
                k
                for k in ("episode_index", "episode_id", "frame_index", "group_id")
                if k in schema
            ]
            values = pq.read_table(file, columns=list(columns) + extra).to_pydict()
            ids = values.get(
                "episode_index",
                values.get("episode_id", [file.stem] * len(values[columns[0]])),
            )
            for identity in dict.fromkeys(ids):
                indices = np.flatnonzero(np.asarray(ids) == identity)
                if "frame_index" in values:
                    frames = np.asarray(values["frame_index"])[indices]
                    if len(frames) > 1 and np.any(np.diff(frames) <= 0):
                        raise ValueError(
                            f"{file}: frame_index must increase within episode {identity}"
                        )
                name = str(identity)
                groups = (
                    {str(values["group_id"][i]) for i in indices}
                    if "group_id" in values
                    else {name}
                )
                if len(groups) != 1:
                    raise ValueError(
                        f"{file}: episode {identity} has multiple group IDs"
                    )
                arrays = [
                    np.asarray([values[k][i] for i in indices], dtype=np.float32)
                    for k in columns
                ]
                episodes.append(Episode(name, *arrays, group_id=groups.pop()))
        else:
            raise ValueError(f"Unsupported data file: {file}")
    names = [ep.name for ep in episodes]
    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate episode IDs across files; merge or rename episodes before loading"
        )
    dofs = {ep.q.shape[1] for ep in episodes}
    if len(dofs) != 1:
        raise ValueError(
            "This loader requires one DOF count; mixed embodiments need an explicit masked adapter"
        )
    return episodes


def split_episodes(episodes, val_fraction=0.1, seed=0, manifest=None):
    """Split entire episodes/groups, or enforce a supplied train/val/test map."""
    by_name = {ep.name: ep for ep in episodes}
    if manifest is not None:
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text())
        split = {
            key: [str(x) for x in manifest.get(key, [])]
            for key in ("train", "val", "test")
        }
        selected = [x for names in split.values() for x in names]
        if len(selected) != len(set(selected)):
            raise ValueError("An episode occurs in more than one split")
        unknown = set(selected) - set(by_name)
        omitted = set(by_name) - set(selected)
        if unknown or omitted:
            raise ValueError(
                f"Split manifest does not match dataset: unknown={sorted(unknown)[:5]}, omitted={sorted(omitted)[:5]}"
            )
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        groups = sorted({ep.group_id or ep.name for ep in episodes})
        if len(groups) < 2:
            raise ValueError(
                "At least two independent episodes/groups are required for train/val splitting"
            )
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(groups))
        n_val = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
        validation = {groups[i] for i in order[:n_val]}
        split = {"train": [], "val": [], "test": []}
        for ep in episodes:
            split["val" if (ep.group_id or ep.name) in validation else "train"].append(
                ep.name
            )
    if not split["train"] or not split["val"]:
        raise ValueError("Both train and val must contain episodes")
    group_splits = {}
    for key, names in split.items():
        for name in names:
            group = by_name[name].group_id or name
            if group in group_splits and group_splits[group] != key:
                raise ValueError(f"Group {group} leaks across train/val/test splits")
            group_splits[group] = key
    return (
        [by_name[x] for x in split["train"]],
        [by_name[x] for x in split["val"]],
        split,
    )


def _minmax(values):
    low, high = values.min(axis=0), values.max(axis=0)
    span = high - low
    span[span == 0] = 1
    return {
        "type": "MinMaxNormalizer",
        "min": low.tolist(),
        "max": high.tolist(),
        "range": span.tolist(),
    }


def fit_normalization(episodes):
    """Fit only the episodes passed here; callers must pass the training split."""
    if not episodes:
        raise ValueError("Cannot fit normalization to no episodes")
    dof = episodes[0].q.shape[1]

    # Extrema can be combined without concatenating a large dataset in memory.
    def extrema(key):
        return np.stack(
            [v for ep in episodes for v in (key(ep).min(axis=0), key(ep).max(axis=0))]
        )

    q_norm = _minmax(extrema(lambda ep: ep.q))
    current_norm = _minmax(extrema(lambda ep: ep.eef[:, :3]))
    target_norm = _minmax(extrema(lambda ep: ep.target_eef[:, :3]))
    action_norm = _minmax(extrema(lambda ep: ep.target_q - ep.q))
    obs_norm = {
        "type": "CompositeNormalizer",
        "dim_slices": [
            [0, dof],
            [dof, dof + 3],
            [dof + 3, dof + 7],
            [dof + 7, dof + 10],
            [dof + 10, dof + 14],
        ],
        "normalizers": [
            q_norm,
            current_norm,
            {"type": "IdentityNormalizer"},
            target_norm,
            {"type": "IdentityNormalizer"},
        ],
    }
    return Normalization.from_dict({"obs": {"state": obs_norm}, "action": action_norm})


def dataset_fingerprint(episodes):
    """Hash exact loaded float32 arrays and identity for resume verification."""
    digest = hashlib.sha256()
    for ep in episodes:
        digest.update(
            json.dumps([ep.name, ep.group_id], separators=(",", ":")).encode()
        )
        for key in ("q", "eef", "target_q", "target_eef"):
            value = np.ascontiguousarray(getattr(ep, key), dtype="<f4")
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


class EpisodeDataset(Dataset):
    """Supervised commands with complete history, never crossing episodes.

    Training excludes the first H-1 rows, matching the source dataset. Set
    ``pad_start=True`` only for an evaluation protocol that includes starts.
    """

    def __init__(self, episodes, normalization, obs_steps=2, pad_start=False):
        if int(obs_steps) < 1:
            raise ValueError("obs_steps must be positive")
        self.episodes = list(episodes)
        self.normalization = normalization
        self.obs_steps = int(obs_steps)
        self.pad_start = bool(pad_start)
        self.start = 0 if self.pad_start else self.obs_steps - 1
        self.ends = np.cumsum([max(0, len(ep) - self.start) for ep in episodes])
        self._frames = [ep.frames for ep in episodes]

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        which = int(np.searchsorted(self.ends, index, side="right"))
        row = int(index - (self.ends[which - 1] if which else 0)) + self.start
        ep = self.episodes[which]
        indices = np.maximum(0, np.arange(row - self.obs_steps + 1, row + 1))
        obs = self.normalization.normalize_obs(self._frames[which][indices])
        action = self.normalization.normalize_action(
            (ep.target_q[row] - ep.q[row])[None]
        )
        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32),
            "action": torch.as_tensor(action, dtype=torch.float32),
            "q_curr_raw": torch.from_numpy(ep.q[row : row + 1]),
            "eef_tgt_raw": torch.from_numpy(ep.target_eef[row : row + 1]),
        }
