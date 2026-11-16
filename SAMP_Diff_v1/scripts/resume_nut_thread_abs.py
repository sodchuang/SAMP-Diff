#!/usr/bin/env python3
"""Resume independent MimicGen ABS runs to a total epoch target, sequentially."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import dill
import torch
from omegaconf import OmegaConf, open_dict


ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    with path.open('rb') as f:
        return torch.load(f, map_location='cpu', pickle_module=dill)


def saved_epoch(payload):
    value = payload['pickles']['epoch']
    return int(dill.loads(value) if isinstance(value, bytes) else value)


def filename_epoch(path):
    match = re.search(r'epoch[=_-](\d+)', path.name)
    return int(match.group(1)) if match else -1


def find_source(output_root, task, seed):
    sources = []
    for run in sorted(output_root.glob(f'{task}_abs_seed{seed}_*')):
        checkpoints = run / 'checkpoints'
        candidates = sorted(checkpoints.glob('*.ckpt'), key=filename_epoch, reverse=True)
        latest = checkpoints / 'latest.ckpt'
        if latest in candidates:
            candidates.remove(latest)
            candidates.insert(0, latest)
        for path in candidates:
            try:
                payload = load_checkpoint(path)
                cfg = payload['cfg']
                if (int(cfg.action_dim) != 10 or not cfg.task.dataset.abs_action
                        or not cfg.task.env_runner.abs_action
                        or int(cfg.training.seed) != seed
                        or OmegaConf.select(cfg, 'task_semantics.task_name') != task):
                    raise ValueError('checkpoint task, seed, or ABS representation mismatch')
                if not {'model', 'ema_model', 'optimizer'} <= set(payload['state_dicts']):
                    raise ValueError('checkpoint lacks full training state')
                epoch = saved_epoch(payload)
                sources.append((epoch, path, cfg))
                del payload
                # Compare latest with the newest readable fixed-epoch file;
                # a stale latest must not hide a later saved epoch.
                if path != latest:
                    break
            except Exception as exc:
                print(f'[WARN] Cannot resume {path}: {exc}', flush=True)
    return max(sources, key=lambda item: item[0]) if sources else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=['nut_assembly_d0', 'threading_d2'],
                        choices=['nut_assembly_d0', 'threading_d2'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/outputs/mimicgen_abs')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/mimicgen/core')
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    os.chdir(ROOT)
    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    for task in args.tasks:
        source = find_source(output_root, task, args.seed)
        if source is None:
            # Never silently restart a run whose checkpoints cannot be read.
            existing = list(output_root.glob(f'{task}_abs_seed{args.seed}_*/checkpoints/*.ckpt'))
            if existing:
                raise RuntimeError(f'{task}: checkpoints exist but none passed resume validation')
            print(f'[NEW] {task}: no previous checkpoint; starts after earlier task completes', flush=True)
            if args.inspect_only:
                continue
            from datetime import datetime
            run_name = f'{task}_abs_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}'
            env = os.environ.copy()
            env.update(NUM_EPOCHS=str(args.epochs), SEED=str(args.seed), DEVICE=args.device,
                       OUTPUT_ROOT=str(output_root), ROLLOUT_EVERY='null',
                       CHECKPOINT_EVERY='50', WANDB_MODE='disabled')
            subprocess.run(['bash', 'scripts/train_abs_from_delta_dataset.sh', task,
                            str(data_root / f'{task}_abs.hdf5'), run_name], env=env, check=True)
            continue
        epoch, checkpoint, cfg = source
        print(f'[RESUME] {task}: completed_epoch={epoch}, next_epoch={epoch + 1}, '
              f'total_target={args.epochs}\n[CHECKPOINT] {checkpoint}', flush=True)
        if epoch + 1 >= args.epochs:
            print(f'[SKIP] {task}: total target already reached', flush=True)
            continue
        if args.inspect_only:
            continue
        run_dir = checkpoint.parent.parent.resolve()
        dataset = data_root / f'{task}_abs.hdf5'
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        with open_dict(cfg):
            cfg.training.resume = True
            cfg.training.resume_from_path = str(checkpoint.resolve())
            cfg.training.num_epochs = args.epochs
            cfg.training.device = args.device
            cfg.training.rollout_every = None
            cfg.training.checkpoint_every = 50
            cfg.logging.mode = 'disabled'
            cfg.task.dataset_path = str(dataset)
            cfg.task.dataset.dataset_path = str(dataset)
            cfg.task.env_runner.dataset_path = str(dataset)
        # Keep the original composed architecture/prior parameters, not a new preset.
        config_dir = run_dir / 'resume_config'
        config_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dir / 'continue.yaml', resolve=True)
        subprocess.run([sys.executable, '-u', 'train.py', '--config-path', str(config_dir),
                        '--config-name', 'continue', f'hydra.run.dir={run_dir}'], check=True)
    print('[DONE] Requested tasks reached their total epoch targets', flush=True)


if __name__ == '__main__':
    main()
