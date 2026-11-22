"""Create explicit WEAK OFFLINE phase labels; never an online task detector."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.common.action_slices import event_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weak-event-labels', action='store_true', required=True)
    parser.add_argument('--align-window', type=int, default=12)
    parser.add_argument('--close-threshold', type=float, default=0.)
    parser.add_argument('--close-is-less', action='store_true')
    args = parser.parse_args()
    import h5py
    import numpy as np
    output = pathlib.Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = {'schema_version': 1, 'phase_source': 'weak_offline_gripper_events',
              'task_id': 0, 'demos': {}}
    with h5py.File(args.dataset, 'r') as file:
        metadata = json.loads(file['data'].attrs['env_args'])
        controller = metadata['env_kwargs']['controller_configs']
        if controller.get('control_delta', True):
            raise ValueError('convert dataset to absolute actions first')
        for key in sorted(file['data'], key=lambda k: int(k.split('_')[-1])):
            actions = file['data'][key]['actions'][:]
            if actions.ndim != 2 or actions.shape[1] not in (7, 14) or not np.isfinite(actions).all():
                raise ValueError(f'invalid stored absolute actions: {key}')
            arms = actions.shape[1] // 7
            dim = arms * 10
            if 'action_dim' in result and result['action_dim'] != dim:
                raise ValueError('mixed action layouts within one dataset')
            result['action_dim'] = dim
            result['demos'][key] = {'length': len(actions), 'arms': {
                f'arm{arm}': event_segments(actions[:, arm * 7 + 6], args.close_threshold,
                                           not args.close_is_less, args.align_window)
                for arm in range(arms)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[OFFLINE ONLY] {len(result["demos"])} demos -> {output}')


if __name__ == '__main__':
    main()
