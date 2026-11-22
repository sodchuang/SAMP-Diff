import ast
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.dataset.sliced_robomimic_dataset import SlicedRobomimicDataset


class LaunchTests(unittest.TestCase):
    def test_single_and_dual_config_composition(self):
        for arms, config in ((1, 'can_ph'), (2, 'transport_ph')):
            with tempfile.TemporaryDirectory() as temporary:
                folder = pathlib.Path(temporary)
                dataset = folder / 'absolute.hdf5'
                manifest = folder / 'phases.json'
                with h5py.File(dataset, 'w') as file:
                    data = file.create_group('data')
                    data.attrs['env_args'] = json.dumps({'env_kwargs': {'controller_configs': {'control_delta': False}}})
                    demo = data.create_group('demo_0')
                    action = np.zeros((8, 7*arms))
                    action[:, 6::7] = np.array([-1, -1, 1, 1, 1, 1, -1, -1])[:, None]
                    demo['actions'] = action
                    obs = demo.create_group('obs')
                    obs['object'] = np.zeros((8, 23))
                    for arm in range(arms):
                        for key, dim in [('eef_pos', 3), ('eef_quat', 4), ('gripper_qpos', 2)]:
                            obs[f'robot{arm}_{key}'] = np.zeros((8, dim))
                subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_action_slices.py'),
                                '--dataset', str(dataset), '--output', str(manifest),
                                '--weak-event-labels'], check=True, capture_output=True)
                output = folder / 'run'
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_fixed_phase_task.py'),
                    '--config', config, '--dataset', str(dataset), '--manifest', str(manifest),
                    '--output', str(output), '--dry-run'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                cfg = OmegaConf.load(output / 'launch_config/config.yaml')
                self.assertEqual(cfg.action_dim, 10*arms)
                self.assertEqual(cfg.obs_dim, 23+9*arms)
                self.assertIsNone(cfg.training.rollout_every)
                self.assertTrue(cfg.policy.fixed_phase_prior_params.enabled)
                self.assertEqual(cfg.task.dataset.dataset.rotation_rep, 'rotation_6d')

    def test_dataset_anchor_and_boundary_mask(self):
        class FakeDataset:
            replay_buffer = types.SimpleNamespace(episode_ends=np.array([8, 16]))
            sampler = types.SimpleNamespace(indices=np.array([[0, 6, 1, 7], [2, 8, 0, 6], [8, 14, 1, 7]]))
            def __len__(self):
                return 3
            def __getitem__(self, index):
                return {'action': torch.zeros(8, 10), 'obs': torch.zeros(8, 4)}
            def get_validation_dataset(self):
                return self
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / 'phases.json'
            segments = [dict(start=0, end=2, phase='transport'), dict(start=2, end=8, phase='align')]
            path.write_text(json.dumps({'schema_version': 1, 'action_dim': 10, 'demos': {
                f'demo_{i}': {'length': 8, 'arms': {'arm0': segments}} for i in range(2)}}))
            dataset = SlicedRobomimicDataset(FakeDataset(), path, 10, 2, 2)
            self.assertFalse(dataset[0]['history_valid_mask'].any())
            self.assertEqual(int(dataset[1]['step']), 3)
            self.assertTrue(dataset[1]['history_valid_mask'][:3].all())
            self.assertFalse(dataset[1]['history_valid_mask'][3:9].any())
            self.assertEqual(int(dataset[2]['episode_id']), 1)
            self.assertFalse(dataset.get_validation_dataset()[2]['history_valid_mask'].any())

    def test_sampler_source_hook_and_legacy_path(self):
        # Execute the real sample method with a tiny mock flow integrator;
        # this isolates the hook without requiring the full CUDA/timm stack.
        tree = ast.parse((ROOT / 'diffusion_policy/model/SAMP/samp_net.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SampNet')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
        function.decorator_list = []
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<sample hook>', 'exec'), namespace)
        core = types.SimpleNamespace(horizon=4, trajectory_dim=10, full_idct=lambda x: x,
                                     spectral_prior=lambda x: x+5)
        flow = types.SimpleNamespace(sample=lambda **kwargs: kwargs['start'])
        reference = torch.ones(2, 4, 10)
        condition = torch.zeros(2, 3)
        actual = namespace['sample'](core, flow, reference, condition)
        torch.testing.assert_close(actual, reference+5)
        actual = namespace['sample'](core, flow, None, condition, source_prior=reference)
        torch.testing.assert_close(actual, reference)


if __name__ == '__main__':
    unittest.main()
