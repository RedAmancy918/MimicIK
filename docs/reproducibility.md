# Reproducibility and release scope

This repository separates three useful operations: running a released decoder,
training that design on paired motion data, and evaluating trajectories under an
explicit state-feedback protocol. It does not require the full NatureIK research
workspace, a robot SDK, a simulator, or a W&B account for inference.

## What the release reproduces

- **Pretrained inference:** the released bundle contains model tensors,
  architecture settings and the normalizer used by its source checkpoint.
  Loading verifies the bundle's checksums. A release identifies its source
  checkpoint in metadata rather than selecting weights from their filename.
- **Portable training:** the training entry point implements the two-call model
  and optional joint MIP/FK objective with a documented paired-episode format.
  It is a compact training recipe, not a bitwise replay of every historical
  data loader, random stream or checkpoint-selection procedure.
- **Offline evaluation:** recorded-state prediction and ideal kinematic feedback
  replay share an explicit target stream and metric definitions.
- **Dataset-specific results:** repeating a published numerical result also
  requires its exact data, split, kinematic model, preprocessing, weights and
  evaluator. The original AIRBOT recordings are not distributed here.

The public [MimicIK paper](https://arxiv.org/abs/2606.15148) and the
[historical model zoo](https://huggingface.co/vistar918/MimicIK) contain earlier
experiments. This repository provides a curated interface and distinguishes
those assets from later controlled recipes. Do not combine results from
different model families or releases as if they came from one matched experiment.

## Curated AIRBOT checkpoints

The initial release includes three models from the controlled AIRBOT training
series. All use **final 300,000-update EMA weights**, not validation-best
selection. The public bundles contain inference tensors and normalization;
optimizer state and machine-specific paths are omitted.

- `airbot-mip-fk-seed1`: full joint-v1 MIP + FK, seed 1, run
  `airbot-controlled-20260915-mip_fk-s1`. Original `final.pt` SHA256:
  `b68a3ef04b3df550a1679c0e21e7f64305a442099f89d1175e6eccd2d6e21dea`.
- `airbot-mip-fk-seed2`: full joint-v1 MIP + FK, seed 2, run
  `airbot-controlled-20260915-mip_fk-s2`. Original `final.pt` SHA256:
  `1c8a8017d8502a5be8ab217820b6f54e6c72e7d3745497591bf3c54c81d3edcd`.
- `airbot-mip-seed1`: matched MIP without the auxiliary FK term, seed 1, run
  `airbot-controlled-20260915-mip_no_fk-s1`. Original `final.pt` SHA256:
  `b07bd36e4caab08747f7c192c4d23112c3c63df05809936ef187bd6305c58a97`.

These source hashes identify the original training artifacts, not the exported
archives. Bundle metadata separately lists the exported config, normalizer and
safetensors checksums. This selection supports pretrained use and a paired
seed-1 FK comparison; it is not the complete three-seed ablation suite. Both
the FK and no-FK models use the same two-call deployment path.

## Model and inference contract

For a robot with `D` arm joints, the observation is
`[q(D), current_EEF(7), target_EEF(7)]`. EEF fields contain metres and XYZW
quaternions in a fixed base/TCP convention. A short history conditions a
SuDeepDiT action decoder. The standard configuration uses history length 2,
hidden width 256, four attention heads, eight condition-encoder blocks and eight
action-decoder blocks, with a two-layer observation MLP.

The MIP inference path makes two deterministic endpoint predictions:

1. Evaluate the decoder with zero action input and time inputs `(0, 0)`.
2. Evaluate it again using that prediction and time inputs `(0.9, 0.9)`.

Denormalize the final increment and add it to the current joints. Network depth,
the two inference calls, and the number of optimizer updates are independent.
Inference uses evaluation mode; it does not run iterative numerical IK or add
sampling noise. Each stateful policy instance owns its own history buffer.

Released bundles use JSON and safetensors. The inference loader does not
deserialize the original executable-pickle training checkpoints. Use the
download command and its release checksum to authenticate an obtained archive;
the checksums inside a local bundle additionally detect corruption.

## Training objective and reference settings

The supervised action is the normalized demonstrated increment
`a = normalize(target_q - q)`. MIP trains a zero-start endpoint prediction and
a second endpoint prediction from a noisy version of the demonstrated action,
with the stage-dependent weights of the original MIP objective.

Optional FK supervision uses a separate interpolated action proxy with shared
network parameters. It compares the proxy's FK pose with the paired target EEF;
it is not an FK correction loop at deployment. With joint optimization,
`L_total = L_MIP + lambda_FK * L_FK` receives one backward pass, one AdamW update
and one EMA update per iteration.

The later AIRBOT joint-v1 reference recipe uses:

- 300,000 updates, batch size 512, seeds 0/1/2.
- AdamW learning rate `1e-4`, weight decay `1e-5`, cosine decay, clipping norm 10.
- EMA coefficient `0.995`; final-step EMA evaluation.
- `legacy_mixed_units_v1` FK: componentwise position MSE with weight 1,
  clamped-acos rotation angle in radians with weight 0.1, and outer FK weight 0.1.

The v1 loss has mixed physical units and a small angle floor from its clamp.
An explicitly selected `tolerance_normalized_v2` loss changes the objective;
using it is a separate experiment. Historical `legacy_two_step` optimizer runs
also differ: they took separate MIP and FK updates. They are not equivalent to
joint optimization or to the two-call inference procedure.

Training excludes the first `H-1` rows of each episode, using complete recorded
history windows. Evaluation and stateful inference repeat the initial observation
to initialize history and retain all planned frames. Random-number streams and
sampling implementations can still differ from historical runs. Matching
architecture and update count alone does not establish exact numerical
reproduction of historical tables.

The trainer saves `resume.pt` periodically and exports a portable `final/`
bundle only after completing the configured update budget. To pause at an
absolute update without changing that budget, add `--stop-after N`. Continue
with the same config/data/output and `--resume runs/airbot/resume.pt`, omitting
the pause limit. Resume checks configuration, data, split, normalization and
URDF identities. A short `--steps N` run changes the total schedule and is a
software smoke test rather than the beginning of a longer matched run.

When comparing alternatives, fix the data manifest, normalization, architecture,
seeds, update budget, final-weight rule and evaluator. A separately trained
single-pass regressor tests a different training/decoding design. The first
output of an already-trained MIP model only tests within-model refinement.

## Evaluation protocols

**Recorded state.** Each frame receives the demonstration's current joints and
current EEF. Predictions do not replace the next recorded state. This measures
local prediction, including agreement with the demonstrated joint target.

**Feedback.** Initialize from the first recorded state. After each command, set
the next current joints to that predicted command, reconstruct current EEF with
the specified FK, and update the method's own history. Recorded targets remain
unchanged. This assumes perfect actuation and excludes servo, contact and
sensor dynamics; it is not a physical closed-loop trial.

Joint variation and pose accuracy must be interpreted together. A constant
command is smooth but can fail to track. Numerical completion also does not
imply accurate tracking.

## Metric definitions

- Position error: Euclidean distance to the target EEF, in millimetres.
- Rotation error: principal target-relative SO(3) angle, in degrees.
- SR@5: compute each episode's fraction of planned frames with position error
  strictly below 5 mm, then average episodes. Failed or unexecuted suffixes count
  as unsuccessful and remain in the denominator.
- Position P95 (`p95_episode_mean_position_mm`): the linearly interpolated 95th
  percentile of the defined per-episode mean position errors. It is neither a
  pooled frame-error percentile nor the mean of within-episode percentiles.
- Joint-target MAE: mean absolute difference between command and paired joint
  target, without angle wrapping, in radians or explicitly reported milliradians.
- Command variation (`joint_step_d95_rad_per_frame`): compute the 95th
  percentile of `||q_command[t] - q_command[t-1]||_2` within each valid episode
  prefix, then average the defined episode percentiles. This uses the L2 norm
  in radians/frame and excludes the initial-state-to-first-command transition.
  It is distinct from the decoded increment relative to the current state.

Error means use each episode's finite accepted prefix and average defined
episode means equally. Episodes without a valid prefix have no defined error
mean. Report episode and frame coverage alongside these conditional errors;
low error on a short prefix does not establish successful execution. Success
rates instead include all planned frames, with failed suffixes counted as
unsuccessful. In feedback replay, report finite-trajectory completion separately
from tracking success. These metric definitions describe this release's
evaluator; compare historical reports only after matching their aggregation
rules and joint-difference norm.

## Data and experiment provenance

Record the dataset fingerprint and fixed split alongside the training config.
Keep paired arms and derivative views of one recording in one split. Fit
normalization only on training episodes, then reuse it unchanged at evaluation
and deployment. Record the URDF hash, joint order, base/TCP frames and label
alignment; [data.md](data.md) describes the AIRBOT and OpenArm distinctions.

AIRBOT NoStatic has 8,949 episodes, with 8,048 training and 901 validation
episodes. Those 901 episodes are a development split. Public OpenArm folding
offers a separate data source, but reproducing a preparation requires its
pinned dataset revision and target URDF as well as the conversion procedure.
Models trained on a different geometry or newly generated split are new runs.

For a new result, retain the model bundle, resolved configuration, split,
normalizer, source revision and evaluation JSON. Compare retrained models over
multiple seeds; deterministic inference from one checkpoint is not a substitute
for training-seed replication.
