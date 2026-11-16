#!/usr/bin/env python3
"""Audit demonstrated release/retreat and saved policy settings; no simulation."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def summarize_demo(actions, positions, close_is_greater=True, threshold=0.0):
    if len(actions) != len(positions):
        raise ValueError('actions and EEF observations have different lengths')
    signal = actions[:, -1]
    closed = signal >= threshold if close_is_greater else signal <= threshold
    releases = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1
    result = {'steps': len(actions), 'release_count': len(releases)}
    if not len(releases):
        return result
    # Last release is a command event, not proof that the object was placed.
    idx = int(releases[-1])
    z = positions[idx:, 2] - positions[idx, 2]
    target_z = actions[idx:, 2] - actions[idx, 2]
    result.update(
        release_index=idx, post_release_steps=len(actions) - idx - 1,
        observed_final_dz_m=float(z[-1]), observed_max_dz_m=float(z.max()),
        action_target_final_dz_m=float(target_z[-1]),
        reclosed_after_last_release=bool(closed[idx:].any()),
        final_open=bool(not closed[-1]),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', default='stack_finish_audit.json', type=Path)
    parser.add_argument('--close-is-less', action='store_true')
    parser.add_argument('--close-threshold', default=0.0, type=float)
    args = parser.parse_args()
    demos = {}
    with h5py.File(args.dataset, 'r') as f:
        raw = f['data'].attrs['env_args']
        meta = json.loads(raw)
        delta = meta['env_kwargs']['controller_configs'].get('control_delta')
        if delta is not False:
            raise ValueError('Expected converted absolute dataset: control_delta=False')
        for name, group in f['data'].items():
            actions = group['actions'][:]
            positions = group['obs/robot0_eef_pos'][:]
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f'{name}: expected stored 7D absolute actions')
            if not np.isfinite(actions).all() or not np.isfinite(positions).all():
                raise ValueError(f'{name}: non-finite values')
            demos[name] = summarize_demo(actions, positions,
                                        not args.close_is_less, args.close_threshold)
    released = [v for v in demos.values() if 'release_index' in v]
    summary = {'demos': len(demos), 'demos_with_release_command': len(released),
               'close_is_greater': not args.close_is_less,
               'close_threshold': args.close_threshold}
    for key in ['steps', 'post_release_steps', 'observed_final_dz_m',
                'observed_max_dz_m', 'action_target_final_dz_m']:
        values = [v[key] for v in demos.values() if key in v]
        summary[key + '_p10_p50_p90'] = (
            np.percentile(values, [10, 50, 90]).tolist() if values else None)
    summary['open_and_final_up_2cm'] = sum(
        v['final_open'] and v['observed_final_dz_m'] >= 0.02 for v in released)
    summary['reclosed_after_last_release'] = sum(
        v['reclosed_after_last_release'] for v in released)
    report = {'summary': summary, 'demos': demos}
    if args.checkpoint:
        import torch
        import dill
        from omegaconf import OmegaConf
        # Only load a checkpoint produced by your own training run.
        with args.checkpoint.open('rb') as f:
            payload = torch.load(f, map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        keys = ['horizon', 'n_action_steps', 'action_dim',
                'task.env_runner.max_steps', 'task.dataset.pad_after',
                'policy.action_group_history_params',
                'policy.action_group_timing_params',
                'policy.action_phase_loss_params']
        report['checkpoint_config'] = {}
        for key in keys:
            value = OmegaConf.select(cfg, key)
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            report['checkpoint_config'][key] = value
        print(json.dumps(report['checkpoint_config'], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved:', args.output.resolve())
    print('Release commands and EEF height do not establish object stacking success.')


if __name__ == '__main__':
    main()
