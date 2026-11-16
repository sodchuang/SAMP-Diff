#!/usr/bin/env python3
"""Audit Stack demonstrations without changing or filtering the dataset."""

import argparse

import h5py
import numpy as np

from detect_stack_release_gate import (
    CANDIDATE_STARTS,
    GATE_MAX,
    GATE_MIN,
    infer_close_semantics,
    last_release,
)


def fmt(values):
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        demos = handle["data"]
        threshold, close_is_greater = infer_close_semantics(demos)
        first = demos[sorted(demos.keys(), key=lambda x: int(x.split("_")[-1]))[0]]
        print(f"[AUDIT] demos={len(demos)} keys={sorted(first.keys())}")
        print(
            "[AUDIT] gripper "
            f"threshold={threshold:.6g} close_is_greater={close_is_greater}")

        lengths = []
        reward_success = 0
        reward_available = 0
        done_success = 0
        done_available = 0
        releases = 0
        release_values = {start: [] for start in CANDIDATE_STARTS}
        window_pass = {start: 0 for start in CANDIDATE_STARTS}

        for demo in demos.values():
            actions = np.asarray(demo["actions"])
            lengths.append(len(actions))
            if "rewards" in demo:
                reward_available += 1
                reward_success += int(np.max(np.asarray(demo["rewards"])) > 0)
            if "dones" in demo:
                done_available += 1
                done_success += int(np.any(np.asarray(demo["dones"]) > 0))

            index = last_release(actions, threshold, close_is_greater)
            if index is None:
                continue
            releases += 1
            object_obs = np.asarray(demo["obs"]["object"])
            for start in CANDIDATE_STARTS:
                if object_obs.shape[-1] < start + 3:
                    continue
                release_values[start].append(object_obs[index, start:start + 3])
                lo = max(0, index - args.radius)
                hi = min(len(object_obs), index + args.radius + 1)
                window = object_obs[lo:hi, start:start + 3]
                valid = np.all(window >= GATE_MIN, axis=-1)
                valid &= np.all(window <= GATE_MAX, axis=-1)
                window_pass[start] += int(np.any(valid))

    lengths = np.asarray(lengths)
    print(
        "[AUDIT] length "
        f"min={lengths.min()} median={np.median(lengths):.1f} "
        f"p90={np.quantile(lengths, .9):.1f} max={lengths.max()}")
    print(f"[AUDIT] demos_with_last_release={releases}/{len(lengths)}")
    if reward_available:
        print(
            f"[AUDIT] reward_positive={reward_success}/{reward_available}")
    else:
        print("[AUDIT] rewards=not_stored")
    if done_available:
        print(f"[AUDIT] done_positive={done_success}/{done_available}")
    else:
        print("[AUDIT] dones=not_stored")

    for start in CANDIDATE_STARTS:
        values = np.asarray(release_values[start])
        if len(values) == 0:
            continue
        print(
            f"[AUDIT] candidate=[{start},{start + 1},{start + 2}] "
            f"radius_pass={window_pass[start]}/{releases} "
            f"({window_pass[start] / max(1, releases):.3f})")
        print(
            "        release p01=" + fmt(np.quantile(values, .01, axis=0))
            + " p10=" + fmt(np.quantile(values, .10, axis=0)))
        print(
            "        release median=" + fmt(np.median(values, axis=0))
            + " p90=" + fmt(np.quantile(values, .90, axis=0))
            + " p99=" + fmt(np.quantile(values, .99, axis=0)))


if __name__ == "__main__":
    main()
