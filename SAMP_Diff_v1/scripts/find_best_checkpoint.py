#!/usr/bin/env python3
import argparse
import pathlib
import re


def metric(path: pathlib.Path):
    text = path.name.lower()
    scores = re.findall(r"(?:test_score|test_mean_score)[=_-]([0-9]+(?:\.[0-9]+)?)", text)
    score = max((float(x) for x in scores), default=-1.0)
    epochs = re.findall(r"epoch[=_-]([0-9]+)", text)
    epoch = max((int(x) for x in epochs), default=-1)
    latest_penalty = -1 if path.name == "latest.ckpt" else 0
    return score, epoch, latest_penalty, path.stat().st_mtime


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=pathlib.Path)
parser.add_argument("--include", required=True, help="Case-insensitive regex matched against the full path")
args = parser.parse_args()

if not args.root.exists():
    raise SystemExit(1)
pattern = re.compile(args.include, re.I)
paths = [
    p for p in args.root.rglob("*.ckpt")
    if "checkpoints" in p.parts and pattern.search(str(p).replace("\\", "/"))
]
if not paths:
    raise SystemExit(1)
print(max(paths, key=metric))
