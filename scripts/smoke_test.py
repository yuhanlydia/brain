"""Run all deterministic synthetic paths on CPU.

The output is execution evidence for invented data, never a benchmark result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from brain_evidence.experiments.synthetic import run_synthetic_suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    arguments = parser.parse_args(argv)
    try:
        result = run_synthetic_suite(seed=arguments.seed, steps=arguments.steps)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
