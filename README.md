# MimicIK

**Learned joint-motion decoding from end-effector targets.**

[Paper](https://arxiv.org/abs/2606.15148) · [Checkpoints](https://huggingface.co/vistar918/MimicIK/tree/main/releases/v0.1.0) · [Installation](docs/installation.md) · [Data format](docs/data.md) · [Reproducibility](docs/reproducibility.md) · [中文](README.zh-CN.md)

MimicIK turns the current robot state and a target end-effector (EEF) pose into a local joint command. It learns from paired teleoperation records and uses two deterministic network evaluations to decode a joint increment. A short observation history supplies motion context.

This repository provides a compact implementation for inference, training, and trajectory evaluation, extracted from the original NatureIK research code. It includes a Python API, portable pretrained weights, optional differentiable FK training, and explicit data and evaluation conventions. Each model is trained for a particular robot and coordinate convention; this release does not provide zero-shot control of unseen robots.

## Install

Use Python 3.11 or newer. Install an appropriate [PyTorch build](https://pytorch.org/get-started/locally/) first if you need CUDA.

```bash
git clone https://github.com/RedAmancy918/MimicIK.git
cd MimicIK
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The inference package uses PyTorch, NumPy, and safetensors. Training and URDF-based FK are optional:

```bash
python -m pip install -e '.[train,fk]'
```

## Try a pretrained model

```bash
python -m mimicik.download --model airbot --output checkpoints/airbot
python examples/infer.py --checkpoint checkpoints/airbot
```

`airbot` selects `airbot-mip-fk-seed1`. Run `python -m mimicik.download --list`
to see all three released AIRBOT models and their download names.
Downloads use a pinned Hugging Face revision with a GitHub release mirror and
verify both the archive and bundle checksums. No account or token is required.

The example runs synthetic inputs to demonstrate loading and inference. It does not connect to a robot, establish tracking accuracy, or supply a hardware command sequence. The download command retrieves the curated AIRBOT bundle; its metadata records the source checkpoint and checksums. The larger [historical model collection](https://huggingface.co/vistar918/MimicIK) contains other architectures and training runs and is separate from this quick start.

## Python API

```python
import numpy as np
from mimicik import MimicIK

policy = MimicIK.from_pretrained("checkpoints/airbot", device="cpu")

# One prepared episode; see docs/data.md for these four arrays.
with np.load("data/my_robot/episode_000.npz", allow_pickle=False) as episode:
    q = episode["q"][0]
    eef_current = episode["eef"][0]
    eef_target = episode["target_eef"][0]

q_command = policy.step(q, eef_current, eef_target)
policy.reset()  # Start a new episode or reset the control stream.
```

- Joint angles and output commands use **radians**; positions use **metres**.
- EEF poses are `[x, y, z, qx, qy, qz, qw]` in the model's training base/TCP convention. Use unit quaternions.
- `step` returns an **absolute joint command**, `q + predicted_delta`, and manages history. Use one instance per arm or control stream. Feed the measured state on the next call.
- `predict_delta(history)` accepts raw history `[H, DOF + 14]` or `[B, H, DOF + 14]`, ordered oldest to newest, and returns joint increments. It does not update the stateful history buffer.
- Gripper control is external. The API does not apply collision checks, servo limits, or command filtering; retain the robot controller's protection layer.

## Train on paired motion data

Prepare one NPZ file per episode with `q`, `eef`, `target_q`, and `target_eef`, plus an explicit `splits.json`. Canonical Parquet episodes are also supported. See [data preparation](docs/data.md) for shapes, target alignment, and the public OpenArm data source.

```bash
python -m mimicik.training \
  --config configs/airbot.yaml \
  --data data/my_robot \
  --output runs/airbot \
  --device cuda --seed 0
```

`configs/airbot.yaml` trains MIP without FK. To enable the joint v1 FK objective:

```bash
python -m mimicik.training \
  --config configs/airbot_fk.yaml \
  --data data/my_robot --output runs/airbot_fk \
  --urdf assets/airbot.urdf --end-link end_link \
  --device cuda --seed 0
```

Match the configuration's joint count, normalization, URDF, and EEF link to your robot. FK supervision is a training option; inference does not need a URDF or an IK solver. The portable training recipe preserves the model and objective design, while [reproducibility notes](docs/reproducibility.md) distinguish it from historical experiment pipelines.

## Evaluate trajectories

Recorded-state evaluation predicts each command from the recorded input state:

```bash
python -m mimicik.evaluate \
  --checkpoint runs/airbot/final \
  --data data/my_robot --split val \
  --urdf assets/airbot.urdf --ee-link end_link \
  --output outputs/eval.json
```

Add `--protocol feedback` to feed each predicted joint command into the next step and reconstruct its EEF pose using FK. This is ideal kinematic feedback replay: it assumes perfect execution and excludes physical servo/contact dynamics. Both modes run offline.

AIRBOT training data are not distributed in this repository. The code and pretrained decoder can be used directly, but reproducing AIRBOT dataset-specific results requires the original data and split. See [reproducibility](docs/reproducibility.md) for the exact scope and metric definitions.

## Citation

```bibtex
@misc{yang2026mimicik,
  title = {MimicIK: Real-Time Generative Inverse Kinematics from Teleoperation with FK Consistency},
  author = {Jiahao Yang and Shenhao Yan and Fan Feng and Chengsi Yao and Ge Wang and Zhixin Mai and Yiming Zhao and Yatong Han},
  year = {2026},
  eprint = {2606.15148},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2606.15148}
}
```

## Acknowledgements and license

MimicIK builds on [Minimal Iterative Policy (MIP)](https://simchowitzlabpublic.github.io/much-ado-about-noising-project/) and uses [PyTorch Kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) for optional FK. We thank the maintainers of these projects and [LeRobot](https://github.com/huggingface/lerobot) for making their work available.

Code is released under the [MIT License](LICENSE). See [NOTICE](NOTICE) for attribution. External datasets and assets retain their own licenses.
