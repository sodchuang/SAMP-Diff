#!/usr/bin/env python3
"""Detect the Stack relative-position slice from the actual HDF5 dataset.

robosuite releases have used different concatenation orders for the 23-D
Stack object observation.  This preflight checks the layouts used by the
supported releases and selects the one whose values satisfy the physical
stack geometry at expert release transitions.
"""

import argparse
import sys

import h5py
import numpy as np


GATE_MIN = np.asarray([-0.035, -0.035, -0.080], dtype=np.float32)
GATE_MAX = np.asarray([0.035, 0.035, -0.015], dtype=np.float32)
CANDIDATE_STARTS = (14, 20)


def infer_close_semantics(demos):
    commands = []
    next_widths = []
    for demo in demos.values():
        action = np.asarray(demo["actions"][:, -1], dtype=np.float32)
        qpos = np.asarray(
            demo["obs"]["robot0_gripper_qpos"], dtype=np.float32)
        if len(action) < 2 or len(qpos) < 2:
            continue
        count = min(len(action) - 1, len(qpos) - 1)
        commands.append(action[:count])
        next_widths.append(np.abs(qpos[1:count + 1]).sum(axis=-1))
    if not commands:
        raise RuntimeError("cannot infer gripper semantics from empty demos")
    commands = np.concatenate(commands)
    next_widths = np.concatenate(next_widths)
    low = float(np.quantile(commands, 0.10))
    high = float(np.quantile(commands, 0.90))
    if high - low < 1e-6:
        raise RuntimeError("gripper command has no detectable open/close range")
    threshold = 0.5 * (low + high)
    low_width = float(np.median(next_widths[commands <= threshold]))
    high_width = float(np.median(next_widths[commands > threshold]))
    close_is_greater = high_width < low_width
    return threshold, close_is_greater


def last_release(action, threshold, close_is_greater):
    gripper = np.asarray(action[:, -1], dtype=np.float32)
    closed = gripper > threshold if close_is_greater else gripper < threshold
    previous = np.zeros_like(closed, dtype=bool)
    previous[1:] = closed[:-1]
    indices = np.flatnonzero(previous & ~closed)
    return None if len(indices) == 0 else int(indices[-1])


def coverage_for_start(
        demos, start, threshold, close_is_greater, radius=0):
    found = 0
    total = 0
    values = []
    for demo in demos.values():
        index = last_release(
            np.asarray(demo["actions"]), threshold, close_is_greater)
        if index is None:
            continue
        object_obs = np.asarray(demo["obs"]["object"])
        if object_obs.shape[-1] < start + 3 or index >= len(object_obs):
            continue
        value = object_obs[index, start:start + 3]
        values.append(value)
        total += 1
        window_start = max(0, index - int(radius))
        window_end = min(len(object_obs), index + int(radius) + 1)
        window = object_obs[window_start:window_end, start:start + 3]
        valid = np.all(window >= GATE_MIN, axis=-1)
        valid &= np.all(window <= GATE_MAX, axis=-1)
        if np.any(valid):
            found += 1
    ratio = found / max(1, total)
    median = np.median(values, axis=0) if values else np.full(3, np.nan)
    return ratio, found, total, median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        demos = handle["data"]
        threshold, close_is_greater = infer_close_semantics(demos)
        results = []
        for start in CANDIDATE_STARTS:
            exact = coverage_for_start(
                demos, start, threshold, close_is_greater, radius=0)
            windowed = coverage_for_start(
                demos, start, threshold, close_is_greater,
                radius=args.radius)
            results.append((windowed[0], start, *windowed[1:], exact[0]))

    for ratio, start, found, total, median, exact_ratio in results:
        print(
            "[STACK-GATE] candidate="
            f"[{start},{start + 1},{start + 2}] "
            f"coverage={found}/{total} ({ratio:.3f}) "
            f"exact={exact_ratio:.3f} radius=+/-{args.radius} "
            f"release_median={np.array2string(median, precision=4)}",
            file=sys.stderr,
        )
    ratio, start, found, total, _, _ = max(results)
    if ratio < args.min_coverage:
        raise SystemExit(
            "[ERROR] no supported Stack object layout reaches release-gate "
            f"coverage {args.min_coverage:.3f}; best="
            f"[{start},{start + 1},{start + 2}] {found}/{total} "
            f"({ratio:.3f}). Refusing to train on a guessed semantic slice.")
    print(f"[{start},{start + 1},{start + 2}]")


if __name__ == "__main__":
    main()
