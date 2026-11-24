# Data and coordinate conventions

MimicIK learns from paired robot states and joint targets. Each row must describe
one current state and the command or next state chosen as its supervised target.
The loader does not infer time alignment or create target poses by shifting rows.

## One NPZ file per episode

Store the following arrays using `numpy.savez` or `numpy.savez_compressed`:

- `q`: `[T, DOF]`, current joint angles in radians.
- `eef`: `[T, 7]`, current EEF position and XYZW quaternion.
- `target_q`: `[T, DOF]`, corresponding absolute joint targets in radians.
- `target_eef`: `[T, 7]`, corresponding target EEF position and XYZW quaternion.
- Optional `episode_id`: scalar string, unique across the dataset. The default is the filename without its extension.
- Optional `group_id`: scalar string connecting recordings that must stay in one split, such as the two arms of one bimanual demonstration. The default is the episode ID.

Positions use metres. EEF arrays are ordered `[x, y, z, qx, qy, qz, qw]`.
Arrays must be finite and have matching lengths; object arrays are not supported.
Use one joint order, DOF count, base frame and TCP convention throughout a dataset.
Gripper values are not arm joints and must be removed during preparation.

```python
import numpy as np

# q, eef, target_q and target_eef are already aligned, measured/prepared arrays.
np.savez_compressed(
    "data/my_robot/episode_000.npz",
    q=q.astype(np.float32),
    eef=eef.astype(np.float32),
    target_q=target_q.astype(np.float32),
    target_eef=target_eef.astype(np.float32),
    episode_id="episode_000",
    group_id="recording_000",
)
```

The joint-increment label is `target_q - q`; no angle wrapping is applied.
Use joint coordinates consistent with the recorded controller and its URDF.
If poses are reconstructed with FK, record the URDF hash, base and end links,
joint order, sign/offset corrections, and any tool transform used.

## Canonical Parquet input

Prepared Parquet data may use the original AIRBOT field names:

- `observation.state`: current `q`, length `DOF`.
- `observation.state_quat`: current `eef`, length 7.
- `action`: absolute `target_q`, length `DOF`.
- `action_quat`: target `target_eef`, length 7.

An `episode_index` or `episode_id` column separates episodes within a file.
Otherwise the file stem is its episode ID. When present, `frame_index` must
increase within an episode; the loader preserves stored order. Episode IDs
must be unique across files, so merge shards belonging to the same episode
before loading. A `group_id` column can preserve parent-recording membership.

This is an explicit schema, not a generic LeRobot adapter. Raw datasets using
other joint layouts, gripper fields, relative actions or missing EEF columns
must be converted first. Keep NPZ and Parquet datasets in separate directories;
the loader prefers NPZ when both are present.

## Fixed train, validation and test splits

Place `splits.json` in the dataset directory, listing episode IDs:

```json
{
  "train": ["episode_000", "episode_001"],
  "val": ["episode_002"],
  "test": ["episode_003"]
}
```

Every loaded episode must appear exactly once. Both training and validation
must be nonempty. Episodes sharing a `group_id` must stay in the same partition.
For a bimanual recording, split the parent recording before separating its arms.
Do not generate train/validation splits by randomly assigning individual frames.

The portable loader can also create a seeded group-level train/validation split
when no manifest is supplied. Save and reuse that split for comparisons; it is
not the paper's frozen split and does not create an independent test set.

## History and normalization

Each raw observation is
`[q_current, eef_current_xyz_xyzw, eef_target_xyz_xyzw]`, with dimension
`DOF + 14`. History is ordered oldest to newest and stays within an episode.
Training excludes the first `H-1` rows of each episode so every training window
contains a full recorded history. Evaluation and the stateful API repeat the
first observation to initialize history and retain those initial frames. A
model's required history length is in its configuration.

Normalization is fitted on the training split only and stored in each model
bundle. Joint coordinates, EEF positions and joint increments use componentwise
min/max normalization without clipping. Quaternions are not min/max normalized.
For AIRBOT compatibility, quaternion signs are selected with `qw >= 0`, without
changing their magnitude. New input data and inference poses should use unit
quaternions; validate and document any correction during data preparation.
Never refit a released model's normalizer on evaluation data.

## AIRBOT data

The AIRBOT recordings use leader-follower teleoperation. Current joints/current
EEF and target joints/target EEF are paired within the same recorded row; the
targets are **not shifted by one frame**. Recorded interface poses and commands
provide supervision without an extra numerical-IK labeling step.

The NoStatic dataset used in later controlled experiments contains 8,949
single-arm episodes and 2,135,868 frames. The frozen split contains 8,048 training
episodes and 901 validation episodes. It covers the same task families in both
partitions. This is an
episode-held-out development split, not an unseen-task benchmark.

These recordings are not included in this public repository. If you have access
to them, preserve their original manifest and metadata. Publicly released model
weights are available independently of the training data.

## Public OpenArm folding data

The [LeRobot high_quality_folding dataset](https://huggingface.co/datasets/lerobot/high_quality_folding)
is an available source of paired bimanual motion. The examined revision is
`c9eb858d4b84e520edecbda84a3534c3c1e78436` (Apache-2.0): 1,200 episodes,
3,254,196 frames, 30 Hz, and a folding task. Its raw state has 16 dimensions:
seven right joints, right gripper, seven left joints, left gripper. Joint positions
are recorded in degrees for the preparation used by MimicIK.

The joint-only preparation does not require the videos. A selective download is:

```bash
python -m pip install huggingface_hub
hf download lerobot/high_quality_folding \
  --repo-type dataset \
  --revision c9eb858d4b84e520edecbda84a3534c3c1e78436 \
  --include 'data/*' 'meta/*' \
  --local-dir data/openarm_hq_raw
```

The raw download is **not** directly compatible with the canonical loader.
Convert it explicitly:

1. Split parent episodes before separating arms. The research preparation used
   seed `20260911` and an 80%/10%/10% train/validation/test partition.
2. Read right joints from `observation.state[0:7]` and left joints from
   `observation.state[8:15]`; remove both grippers and convert degrees to radians.
3. Use adjacent measured states `q_t` and `q_(t+1)`, verifying consecutive frame
   indices and increasing timestamps. Drop each episode's last frame. Raw
   `action` requests may differ from executed states after follower-side limits.
4. Reconstruct current and target EEF poses with the chosen arm's URDF. Record
   the exact base/TCP definition and transformations. The research OpenArmX
   chains used `openarmx_{left,right}_link0` to
   `openarmx_{left,right}_hand_tcp`.
5. Save one NPZ episode per arm and preserve a common parent `group_id`.

The research target URDF describes the author's OpenArmX. The collection robot
had modified upper-arm geometry, so these FK-derived poses describe the chosen
target model rather than measured EEF poses from the collector. A different
URDF defines a different preparation and must be reported as such.

The compact release accepts one fixed robot layout per model. The historical
mixed AIRBOT/OpenArm experiments additionally used padding, active-joint masks,
source balancing and sometimes identity conditioning; ordinary concatenation
of incompatible datasets does not reproduce that setup.
