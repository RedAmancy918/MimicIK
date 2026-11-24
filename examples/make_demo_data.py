"""Generate a tiny synthetic dataset for testing the complete training pipeline.

This planar six-joint toy is not AIRBOT data and does not reproduce paper scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def toy_fk(q):
    angle = np.cumsum(q, axis=-1)
    x = (0.05 * np.cos(angle)).sum(-1)
    y = (0.05 * np.sin(angle)).sum(-1)
    zero = np.zeros_like(x)
    return np.stack(
        (
            x,
            y,
            zero,
            zero,
            zero,
            np.sin(angle[..., -1] / 2),
            np.cos(angle[..., -1] / 2),
        ),
        axis=-1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 3 or args.frames < 2:
        parser.error("Use at least three episodes and two frames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output must be empty to avoid mixing datasets")
    rng = np.random.default_rng(args.seed)
    ids = []
    for i in range(args.episodes):
        name = f"episode_{i:04d}"
        ids.append(name)
        q = np.cumsum(rng.normal(0, 0.012, (args.frames + 1, 6)), axis=0).astype(
            np.float32
        )
        eef = toy_fk(q).astype(np.float32)
        np.savez_compressed(
            output / f"{name}.npz",
            q=q[:-1],
            eef=eef[:-1],
            target_q=q[1:],
            target_eef=eef[1:],
            episode_id=name,
        )
    splits = {"train": ids[:-2], "val": ids[-2:-1], "test": ids[-1:]}
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    print(f"Wrote {args.episodes} synthetic episodes to {output}")


if __name__ == "__main__":
    main()
