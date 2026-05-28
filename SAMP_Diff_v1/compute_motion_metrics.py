"""
Compute motion quality metrics for a trained SAMP policy on PushT.

In PushT the agent is fully-actuated (position is set to the commanded
action directly), so executed actions == agent positions.

Metrics
-------
Path Length     : Σ ‖pos[t] - pos[t-1]‖  (pixels, over the episode)
Jerk Cost       : mean squared discrete 3rd-order finite difference of pos
                  (proxy for smoothness; lower = smoother)
Discontinuity   : mean ‖last_action_chunk[i][-1] - action_chunk[i+1][0]‖
                  (re-planning gap; lower = more continuous)

Usage
-----
  cd SAMP_Diff_v1
  python compute_motion_metrics.py \\
      -c outputs/2026-05-21/08-59-47/checkpoints/epoch=3900-test_mean_score=0.971.ckpt \\
      -d cuda:0 --n_episodes 50
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import torch
import dill
import click
import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


# ── metric helpers ────────────────────────────────────────────────────────────

def path_length(traj: np.ndarray) -> float:
    """traj: (T, 2) in pixels"""
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=-1)))


def jerk_cost(traj: np.ndarray, fps: float = 10.0) -> float:
    """Mean squared 3rd-order discrete finite difference (physical jerk proxy).
    traj: (T, 2).  Returns nan when T < 4."""
    if len(traj) < 4:
        return float('nan')
    dt = 1.0 / fps
    j = traj[3:] - 3 * traj[2:-1] + 3 * traj[1:-2] - traj[:-3]
    j /= (dt ** 3)   # units: pixels / s^3
    return float(np.mean(np.sum(j ** 2, axis=-1)))


def discontinuity(chunk_ends: list, chunk_starts: list) -> float:
    """Mean L2 jump between consecutive action-chunk boundaries."""
    if not chunk_ends:
        return float('nan')
    return float(np.mean([
        np.linalg.norm(e - s)
        for e, s in zip(chunk_ends, chunk_starts)
    ]))


# ── main ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to .ckpt file')
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_episodes', default=50, type=int)
@click.option('--max_steps',  default=300, type=int)
@click.option('--fps',        default=10.0, type=float)
@click.option('--start_seed', default=100000, type=int,
              help='First test seed (matches pusht.yaml test_start_seed)')
def main(checkpoint, device, n_episodes, max_steps, fps, start_seed):
    # ── load checkpoint ────────────────────────────────────────────────────
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/motion_metrics_scratch')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    dev = torch.device(device)
    policy.to(dev)
    policy.eval()

    n_obs_steps    = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps

    # ── pygame + env imports ───────────────────────────────────────────────
    import pygame
    if not pygame.get_init():
        pygame.init()

    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

    # ── episode loop ───────────────────────────────────────────────────────
    all_path_lengths    = []
    all_jerk_costs      = []
    all_discontinuities = []
    all_scores          = []

    print(f"Evaluating {n_episodes} episodes (seed {start_seed} … {start_seed+n_episodes-1}) ...")

    for ep in range(n_episodes):
        seed = start_seed + ep

        env = MultiStepWrapper(
            PushTKeypointsEnv(**kp_kwargs),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )
        env.seed(seed)
        obs = env.reset()   # (n_obs_steps, obs_per_step_total)

        policy.reset()

        # obs_per_step_total = Do * 2  (keypoints + visibility mask)
        Do = obs.shape[-1] // 2

        done       = False
        chunk_list = []     # each: (n_action_steps, 2) of executed actions
        ep_rewards = []

        while not done:
            # build batched input (B=1)
            obs_input = obs[:n_obs_steps, :Do].astype(np.float32)  # (To, Do)
            obs_dict = dict_apply(
                {'obs': obs_input[np.newaxis]},                     # (1, To, Do)
                lambda x: torch.from_numpy(x).to(device=dev)
            )

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # action: (1, n_action_steps, 2) → (n_action_steps, 2)
            action = action_dict['action'].detach().cpu().numpy()[0]

            obs, reward, done, _ = env.step(action)
            ep_rewards.append(float(reward))
            chunk_list.append(action)

        env.close()

        # ── per-episode metrics ───────────────────────────────────────────
        # In PushT, agent.position := action at every step → action IS position
        traj = np.concatenate(chunk_list, axis=0)   # (T, 2)

        disc_ends   = [c[-1] for c in chunk_list[:-1]]
        disc_starts = [c[0]  for c in chunk_list[1:]]

        pl    = path_length(traj)
        jc    = jerk_cost(traj, fps=fps)
        disc  = discontinuity(disc_ends, disc_starts)
        score = float(np.max(ep_rewards)) if ep_rewards else 0.0

        all_path_lengths.append(pl)
        all_jerk_costs.append(jc)
        all_discontinuities.append(disc)
        all_scores.append(score)

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(f"  ep {ep+1:3d}/{n_episodes}  "
                  f"score={score:.3f}  path={pl:.1f} px  "
                  f"jerk={jc:.3e}  disc={disc:.3f} px")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'Mean':>12}  {'Std':>12}")
    print("-" * 65)
    print(f"{'Mean Score (coverage)':<28} {np.mean(all_scores):>12.4f}  {np.std(all_scores):>12.4f}")
    print(f"{'Path Length (px)':<28} {np.mean(all_path_lengths):>12.2f}  {np.std(all_path_lengths):>12.2f}")
    print(f"{'Jerk Cost (px/s^3)^2':<28} {np.mean(all_jerk_costs):>12.4e}  {np.std(all_jerk_costs):>12.4e}")
    print(f"{'Discontinuity (px)':<28} {np.nanmean(all_discontinuities):>12.4f}  {np.nanstd(all_discontinuities):>12.4f}")
    print("=" * 65)


if __name__ == '__main__':
    main()

