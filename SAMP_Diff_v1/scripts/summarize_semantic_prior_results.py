#!/usr/bin/env python3
"""Summarize multi-seed semantic-prior experiments from logs.json.txt."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


SEED_SUFFIX = re.compile(r"_seed(\d+)$")


def read_scores(log_path: Path, metric: str):
    scores = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                scores.append((int(row.get("epoch", -1)), float(value)))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_root",
        nargs="?",
        default="data/outputs/semantic_prior_generalization",
    )
    parser.add_argument("--metric", default="test/mean_score")
    parser.add_argument("--expected-seeds", default="42,43,44")
    args = parser.parse_args()

    root = Path(args.output_root)
    expected = {int(seed) for seed in args.expected_seeds.split(",") if seed}
    groups = {}

    for log_path in sorted(root.glob("*/logs.json.txt")):
        match = SEED_SUFFIX.search(log_path.parent.name)
        if match is None:
            continue
        scores = read_scores(log_path, args.metric)
        if not scores:
            continue
        seed = int(match.group(1))
        experiment = SEED_SUFFIX.sub("", log_path.parent.name)
        groups.setdefault(experiment, {})[seed] = {
            "latest": scores[-1],
            "best": max(scores, key=lambda item: item[1]),
        }

    if not groups:
        raise SystemExit(f"No multi-seed metric '{args.metric}' found under {root}")

    print(
        "experiment\tn\tseeds\tmissing\tlatest_mean\tlatest_std\t"
        "best_mean\tbest_std"
    )
    for experiment, runs in sorted(groups.items()):
        seeds = sorted(runs)
        latest = [runs[seed]["latest"][1] for seed in seeds]
        best = [runs[seed]["best"][1] for seed in seeds]
        latest_std = statistics.stdev(latest) if len(latest) > 1 else float("nan")
        best_std = statistics.stdev(best) if len(best) > 1 else float("nan")
        missing = ",".join(map(str, sorted(expected - set(seeds)))) or "-"
        print(
            f"{experiment}\t{len(seeds)}\t{','.join(map(str, seeds))}\t{missing}\t"
            f"{statistics.mean(latest):.4f}\t{latest_std:.4f}\t"
            f"{statistics.mean(best):.4f}\t{best_std:.4f}"
        )


if __name__ == "__main__":
    main()
