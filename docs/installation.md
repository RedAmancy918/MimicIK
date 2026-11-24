# Installation and tested environment

Use Python **3.11 or newer**. CPU installation, tests, a short training run,
inference and packaging passed on Linux with Python 3.11 and 3.12 in
[GitHub Actions](https://github.com/RedAmancy918/MimicIK/actions/runs/45351192674).
Local release verification also passed on macOS with Python 3.12.16.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,fk]'
```

For NVIDIA training, first install the appropriate PyTorch CUDA wheel using
the selector at https://pytorch.org/get-started/locally/. A compatible NVIDIA
driver is required. The package does not install or change GPU drivers. On
Windows, activate with `.venv\Scripts\activate` instead.

- Core inference: `pip install -e .` (PyTorch, NumPy, safetensors).
- Training/data: `pip install -e '.[train]'` (PyArrow and PyYAML).
- FK training/evaluation: `pip install -e '.[train,fk]'` (PyTorch Kinematics).
- Contributor checks: `pip install -e '.[train,fk,dev]'`.

No ROS, simulator, robot SDK, W&B account or original private repository is
required for these commands. All example scripts run offline and send no
hardware commands.

## CPU end-to-end check

```bash
python examples/make_demo_data.py --output data/demo
python -m mimicik.training --config configs/smoke.yaml \
  --data data/demo --output runs/smoke --device cpu --steps 10
python examples/infer.py --checkpoint runs/smoke/final
python -m mimicik.evaluate --checkpoint runs/smoke/final \
  --data data/demo --split test --urdf assets/toy6.urdf \
  --ee-link tcp --output outputs/smoke.json
```

The tiny model and planar data only verify the software pipeline. They are not
the AIRBOT pretrained model or a scientific benchmark. Use a new output
directory for a fresh run. Resume an existing run with its unchanged total
budget and `--resume runs/smoke/resume.pt`.

## Reproducible dependencies

`requirements/verified-cpu.txt` records the versions used for release checks.
Install the package first; then, if reproducing that CPU environment, apply
these constraints with `python -m pip install -e '.[train,fk]' -c
requirements/verified-cpu.txt`. CUDA wheels may have a platform-specific
version suffix. A version record does not imply identical floating-point
results across CPU/GPU architectures.
