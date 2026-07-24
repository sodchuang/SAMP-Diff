"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import time
import click
import hydra
import torch
import dill
import wandb
import json
import numpy as np
from omegaconf import open_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace

try:
    from scipy.fftpack import dct as scipy_dct
except Exception:
    scipy_dct = None


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _find_action_tensor(result):
    if torch.is_tensor(result):
        return result
    if isinstance(result, dict):
        for key in ('action', 'action_pred', 'actions', 'pred_action'):
            value = result.get(key, None)
            if torch.is_tensor(value):
                return value
    return None


def _as_btd(action):
    if action is None:
        return None
    action = action.detach().float().cpu()
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        return None
    return action


def _mean_square(x):
    if x is None or x.numel() == 0:
        return float('nan')
    return torch.mean(x ** 2).item()


def _jerk_energy(action):
    action = _as_btd(action)
    if action is None or action.shape[1] < 4:
        return float('nan')
    vel = action[:, 1:] - action[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    return _mean_square(jerk)


def _delta_energy(action):
    action = _as_btd(action)
    if action is None or action.shape[1] < 2:
        return float('nan')
    return _mean_square(action[:, 1:] - action[:, :-1])


def _accel_energy(action):
    action = _as_btd(action)
    if action is None or action.shape[1] < 3:
        return float('nan')
    vel = action[:, 1:] - action[:, :-1]
    return _mean_square(vel[:, 1:] - vel[:, :-1])


def _dct_high_energy(action, low_ratio=0.25):
    action = _as_btd(action)
    if action is None:
        return float('nan')
    x = action.numpy()
    if x.shape[1] < 2:
        return float('nan')
    if scipy_dct is not None:
        coeff = scipy_dct(x, axis=1, norm='ortho')
    else:
        coeff = np.fft.rfft(x, axis=1)
    energy = np.abs(coeff) ** 2
    cutoff = max(1, int(energy.shape[1] * low_ratio))
    if cutoff >= energy.shape[1]:
        return float('nan')
    return float(np.mean(energy[:, cutoff:, :]))


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(np.mean(values))


def _finite_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(np.std(values))


class EvalActionProfiler:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = torch.device(device)
        self.predict_action = policy.predict_action
        self.latency_ms = []
        self.jerk = []
        self.delta = []
        self.accel = []
        self.dct_high = []
        self.boundary_delta = []
        self.prev_last_action = None

    def install(self):
        def measured_predict_action(obs_dict):
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            result = self.predict_action(obs_dict)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.latency_ms.append(elapsed_ms)

            action = _as_btd(_find_action_tensor(result))
            if action is not None:
                self.jerk.append(_jerk_energy(action))
                self.delta.append(_delta_energy(action))
                self.accel.append(_accel_energy(action))
                self.dct_high.append(_dct_high_energy(action))
                if self.prev_last_action is not None:
                    n = min(action.shape[0], self.prev_last_action.shape[0])
                    if n > 0:
                        jump = action[:n, 0] - self.prev_last_action[:n]
                        self.boundary_delta.append(_mean_square(jump))
                self.prev_last_action = action[:, -1].clone()
            return result

        self.policy.predict_action = measured_predict_action
        return self

    def summary(self):
        lat = np.asarray(self.latency_ms, dtype=np.float64)
        log = {
            'profile/predict_calls': int(lat.size),
            'profile/action_chunks': int(len(self.jerk)),
        }
        if lat.size > 0:
            log.update({
                'profile/latency_ms_mean': float(np.mean(lat)),
                'profile/latency_ms_std': float(np.std(lat)),
                'profile/latency_ms_min': float(np.min(lat)),
                'profile/latency_ms_max': float(np.max(lat)),
                'profile/hz_mean': float(1000.0 / np.mean(lat)),
            })
        log.update({
            'profile/action_delta_mse_mean': _finite_mean(self.delta),
            'profile/action_delta_mse_std': _finite_std(self.delta),
            'profile/action_accel_mse_mean': _finite_mean(self.accel),
            'profile/action_accel_mse_std': _finite_std(self.accel),
            'profile/action_jerk_mse_mean': _finite_mean(self.jerk),
            'profile/action_jerk_mse_std': _finite_std(self.jerk),
            'profile/dct_high_energy_mean': _finite_mean(self.dct_high),
            'profile/dct_high_energy_std': _finite_std(self.dct_high),
            'profile/chunk_boundary_delta_mse_mean': _finite_mean(self.boundary_delta),
            'profile/chunk_boundary_delta_mse_std': _finite_std(self.boundary_delta),
        })
        return log


class EvalObservationNoise:
    def __init__(self, policy, std, seed=None, keys=('obs',)):
        self.policy = policy
        self.std = float(std)
        self.seed = seed
        self.keys = set(keys)
        self.predict_action = policy.predict_action
        self.generators = {}

    def _generator_for(self, tensor):
        if self.seed is None:
            return None
        device_key = str(tensor.device)
        generator = self.generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(int(self.seed))
            self.generators[device_key] = generator
        return generator

    def _add_noise(self, key, value):
        if (
            key not in self.keys
            or not torch.is_tensor(value)
            or not torch.is_floating_point(value)
            or self.std <= 0.0
        ):
            return value
        generator = self._generator_for(value)
        noise = torch.randn(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            generator=generator,
        )
        return value + noise * self.std

    def install(self):
        def noisy_predict_action(obs_dict):
            if isinstance(obs_dict, dict):
                obs_dict = {
                    key: self._add_noise(key, value)
                    for key, value in obs_dict.items()
                }
            return self.predict_action(obs_dict)

        self.policy.predict_action = noisy_predict_action
        return self

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--overwrite', is_flag=True, help='Overwrite output_dir without asking.')
@click.option('--gripper-latch', is_flag=True, help='Enable eval-only soft gripper close latch.')
@click.option('--gripper-min-close-steps', default=2, type=int)
@click.option(
    '--gripper-max-latch-chunks-after-close',
    default=None,
    type=int,
    help='Disable latch after N policy chunks from the first close. Omit for no phase gate.',
)
@click.option('--gripper-close-threshold', default=0.0, type=float)
@click.option(
    '--gripper-close-value',
    default=None,
    type=float,
    help='Force gripper to this value while latched. Omit for soft latch.',
)
@click.option('--gripper-close-is-greater/--gripper-close-is-less', default=True)
@click.option('--n-envs', default=None, type=int, help='Override task.env_runner.n_envs for safer eval.')
@click.option('--n-test', default=None, type=int, help='Override task.env_runner.n_test.')
@click.option('--n-train', default=None, type=int, help='Override task.env_runner.n_train.')
@click.option('--n-test-vis', default=None, type=int, help='Override task.env_runner.n_test_vis.')
@click.option('--n-train-vis', default=None, type=int, help='Override task.env_runner.n_train_vis.')
@click.option(
    '--profile-actions/--no-profile-actions',
    default=True,
    help='Measure predict_action latency and action smoothness metrics during eval.',
)
@click.option('--obs-noise-std', default=0.0, type=float, help='Add Gaussian noise N(0, std^2) to low-dim obs tensors during eval.')
@click.option('--obs-noise-seed', default=None, type=int, help='Seed for eval observation noise. Omit for stochastic noise.')
def main(
        checkpoint,
        output_dir,
        device,
        overwrite,
        gripper_latch,
        gripper_min_close_steps,
        gripper_max_latch_chunks_after_close,
        gripper_close_threshold,
        gripper_close_value,
        gripper_close_is_greater,
        n_envs,
        n_test,
        n_train,
        n_test_vis,
        n_train_vis,
        profile_actions,
        obs_noise_std,
        obs_noise_seed):
    if os.path.exists(output_dir) and not overwrite:
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    if gripper_latch:
        with open_dict(cfg.policy):
            cfg.policy.action_group_timing_params = {
                'gripper': {
                    'enabled': True,
                    'indices': [cfg.policy.action_dim - 1],
                    'close_threshold': gripper_close_threshold,
                    'close_is_greater': gripper_close_is_greater,
                    'close_value': gripper_close_value,
                    'min_close_steps': gripper_min_close_steps,
                    'max_latch_chunks_after_close': gripper_max_latch_chunks_after_close,
                }
            }
    runner_overrides = {
        'n_envs': n_envs,
        'n_test': n_test,
        'n_train': n_train,
        'n_test_vis': n_test_vis,
        'n_train_vis': n_train_vis,
    }
    with open_dict(cfg.task.env_runner):
        for key, value in runner_overrides.items():
            if value is not None:
                cfg.task.env_runner[key] = value
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    if obs_noise_std > 0.0:
        EvalObservationNoise(policy, obs_noise_std, obs_noise_seed).install()
    profiler = EvalActionProfiler(policy, device).install() if profile_actions else None
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    if profiler is not None:
        runner_log.update(profiler.summary())
    runner_log.update({
        'eval/obs_noise_std': float(obs_noise_std),
        'eval/obs_noise_seed': obs_noise_seed,
    })
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = _json_safe(value)
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
