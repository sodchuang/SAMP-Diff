"""Opt-in fixed-profile training. Online evaluation needs a causal phase provider."""
import argparse
import json
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='can_ph', help='existing task config; transport_ph for dual arms')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--history-mode', choices=['none', 'inherit', 'reset', 'selective'], default='selective')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    import h5py
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    dataset = pathlib.Path(args.dataset).resolve()
    manifest = pathlib.Path(args.manifest).resolve()
    output = pathlib.Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('use a NEW output directory; this launcher never overwrites an experiment')
    with manifest.open(encoding='utf-8') as file:
        labels = json.load(file)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(root / 'config_task/low_dim')):
        cfg = compose(config_name=args.config)
    with h5py.File(dataset, 'r') as file:
        controller = json.loads(file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
        if controller.get('control_delta', True):
            raise ValueError('absolute-action dataset required')
        demo = file['data']['demo_0']
        stored_dim = demo['actions'].shape[-1]
        dim = stored_dim // 7 * 10
        if stored_dim not in (7, 14) or labels['action_dim'] != dim:
            raise ValueError('dataset/manifest action layout mismatch')
        obs_keys = list(cfg.task.dataset.obs_keys)
        obs_dim = sum(demo['obs'][key].shape[-1] for key in obs_keys)
    with open_dict(cfg):
        cfg.action_dim, cfg.obs_dim = dim, obs_dim
        cfg.task.dataset_path = str(dataset)
        cfg.task.abs_action = True
        cfg.task.dataset.abs_action = True
        cfg.task.dataset.dataset_path = str(dataset)
        cfg.task.dataset.rotation_rep = 'rotation_6d'
        cfg.task.dataset.seed = args.seed
        # Keep chunks and observations unchanged. Disable unrelated task-specific tricks.
        for key in ('episode_prefix_enabled', 'event_oversample_enabled'):
            if key in cfg.task.dataset:
                cfg.task.dataset[key] = False
        base = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        cfg.task.dataset = {
            '_target_': 'diffusion_policy.dataset.sliced_robomimic_dataset.SlicedRobomimicDataset',
            'dataset': base, 'manifest_path': str(manifest), 'action_dim': dim,
            'n_obs_steps': cfg.n_obs_steps, 'n_action_steps': cfg.n_action_steps,
            'mode': args.history_mode}
        for key in ('action_group_spectral_params', 'action_group_history_params',
                    'action_group_loss_weights', 'action_group_timing_params', 'action_phase_loss_params'):
            cfg.policy[key] = None
        cfg.policy.history_training_mode = 'aligned'
        cfg.policy.fixed_phase_prior_params = {'enabled': True, 'mode': args.history_mode}
        cfg.training.device = args.device
        cfg.training.seed = args.seed
        cfg.training.num_epochs = args.epochs
        cfg.training.resume = False
        cfg.training.rollout_every = None
        cfg.training.checkpoint_every = 50
        cfg.logging.mode = 'disabled'
        cfg.logging.name = output.name
        if 'task_semantics' in cfg:
            cfg.task_semantics.task_name = output.name
            cfg.task_semantics.control_mode = 'abs_eef_pose'
            cfg.task_semantics.action_representation = 'xyz_rotation6d_gripper'
            # Remove inherited task-specific routing descriptions, not used by this launcher.
            cfg.task_semantics.action_groups = {
                f'arm{arm}_{group}': indices for arm in range(dim // 10)
                for group, indices in zip(('translation', 'rotation', 'gripper'),
                    (list(range(arm*10, arm*10+3)), list(range(arm*10+3, arm*10+9)), [arm*10+9]))}
            cfg.task_semantics.routing = {'spectral': list(range(dim)), 'direct': []}
            cfg.task_semantics.diagnostics = {}
    output.mkdir(parents=True)
    config_dir = output / 'launch_config'
    config_dir.mkdir()
    OmegaConf.save(cfg, config_dir / 'config.yaml')
    print('[EXPERIMENTAL] offline phase labels; online rollout disabled; no automatic phase detector', flush=True)
    if not args.dry_run:
        subprocess.run([sys.executable, str(root / 'train.py'), '--config-path', str(config_dir),
                        '--config-name', 'config', f'hydra.run.dir={output}'], cwd=root, check=True)


if __name__ == '__main__':
    main()
