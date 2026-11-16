#!/usr/bin/env python3
import argparse
import json
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("root", type=pathlib.Path)
args = parser.parse_args()

columns = [
    "run", "test_mean_score", "noise_std", "noise_seed",
    "latency_ms", "hz", "jerk_mse", "dct_high_energy", "boundary_mse",
]
print("\t".join(columns))
for path in sorted(args.root.rglob("eval_log.json")):
    data = json.loads(path.read_text())
    values = [
        str(path.parent.relative_to(args.root)),
        data.get("test/mean_score", "NA"),
        data.get("eval/obs_noise_std", 0.0),
        data.get("eval/obs_noise_seed", "NA"),
        data.get("profile/latency_ms_mean", "NA"),
        data.get("profile/hz_mean", "NA"),
        data.get("profile/action_jerk_mse_mean", "NA"),
        data.get("profile/dct_high_energy_mean", "NA"),
        data.get("profile/chunk_boundary_delta_mse_mean", "NA"),
    ]
    print("\t".join(map(str, values)))
