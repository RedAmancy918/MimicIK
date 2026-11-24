"""Run the downloaded decoder on synthetic inputs, without robot hardware."""

import argparse
import json

import numpy as np

from mimicik import MimicIK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    policy = MimicIK.from_pretrained(args.checkpoint, device=args.device)
    # A plumbing example, not a calibrated robot pose or a reachable-task test.
    q = np.zeros(policy.dof, dtype=np.float32)
    current = np.array([0.25, 0, 0.3, 0, 0, 0, 1], dtype=np.float32)
    target = current.copy()
    target[0] += 0.001
    command = policy.step(q, current, target)
    print(
        json.dumps(
            {
                "example": "synthetic inputs; no robot commands sent",
                "q_command_rad": command.tolist(),
                "delta_q_rad": (command - q).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
