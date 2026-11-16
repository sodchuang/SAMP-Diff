#!/usr/bin/env python3
"""Find the highest test_score checkpoint whose run path matches a regex."""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/outputs")
    parser.add_argument("--run-regex", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    run_pattern = re.compile(args.run_regex, re.IGNORECASE)
    score_pattern = re.compile(
        r"epoch=(?P<epoch>\d+)_test_score=(?P<score>[0-9]+(?:\.[0-9]+)?)"
    )
    candidates = []

    if root.exists():
        for checkpoint in root.rglob("*.ckpt"):
            run_dir = checkpoint.parent.parent
            if not run_pattern.search(str(run_dir).replace("\\", "/")):
                continue
            match = score_pattern.search(checkpoint.name)
            if match is None:
                continue
            candidates.append(
                (
                    float(match.group("score")),
                    int(match.group("epoch")),
                    str(checkpoint),
                )
            )

    if not candidates:
        raise SystemExit(2)

    _, _, checkpoint = max(candidates, key=lambda item: (item[0], item[1]))
    print(checkpoint)


if __name__ == "__main__":
    main()
