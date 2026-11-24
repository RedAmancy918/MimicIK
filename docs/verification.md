# Release verification — 2026-11-24

The compact release was checked in a fresh Python 3.12.16 environment on macOS
arm64 with CPU PyTorch. See `requirements/verified-cpu.txt` for dependency
versions. GPU throughput and physical robot behavior are not measured by these
software checks.

- 28 tests passed, including upstream-network equivalence, episode isolation,
  train-only normalization, quaternion handling, bundle integrity, resume
  equivalence and overwrite protection, and evaluation failure accounting.
- Each of the three curated AIRBOT checkpoints was compared with the original
  SuDeepDiT, state encoder and normalizer on 32 seeded, randomly generated
  training-range history inputs. First output, second output, denormalized joint
  increment and public API output all had maximum absolute difference **0** on
  the checked CPU path.
- MIP, FK v1 and FK v2 losses and their parameter gradients were compared against
  the original implementations on matching inputs and random streams. Maximum
  observed differences were **0**.
- Pausing and resuming the tiny training fixture produced identical final EMA
  weights to uninterrupted training with the same total budget.
- A clean install ran synthetic episode creation, ten training updates,
  inference, recorded-state evaluation and ideal feedback replay successfully.
- All three model archives were checksum-verified, extracted and loaded before
  release. Wheel and source distribution builds succeeded.
- [Linux CI on Python 3.11 and 3.12](https://github.com/RedAmancy918/MimicIK/actions/runs/45351192674)
  passed installation, the standalone test suite, synthetic data creation, a
  short training run, inference, and wheel/source distribution builds.
- Anonymous downloads from the public GitHub release and pinned Hugging Face
  revision passed size and SHA256 checks and bundle loading. The inference
  example ran on the downloaded AIRBOT model in the clean environment.

These are compatibility and software checks. Synthetic fixtures and random
input parity checks do not measure real-robot task success or reproduce the
paper's dataset-specific accuracy.

To run the public tests:

```bash
python -m pip install -e '.[train,fk,dev]'
python -m pytest -q
```

The upstream parity test is optional because it requires the original research
checkout. If available, set `MIMICIK_REFERENCE_SOURCE` to its root. Without it,
that one test is skipped; the standalone public suite still runs.
