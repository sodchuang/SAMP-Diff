"""
LeRobot Lowdim Runner
在 LeRobot gym 環境（gym-pusht / gym-aloha / gym-xarm）中執行 SAMP-Diff policy 評估。

依賴：
  pip install gym-pusht          # PushT
  pip install gym-aloha          # ALOHA 雙臂
  pip install gym-xarm           # xArm

支援 gym 環境 ID：
  - gym_pusht/PushT-v0
  - gym_aloha/AlohaTransferCube-v0
  - gym_aloha/AlohaInsertion-v0
  - gym_xarm/XarmLift-v0
"""

from __future__ import annotations

import collections
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply


class LeRobotLowdimRunner(BaseLowdimRunner):
    """
    在 LeRobot gymnasium 環境中評估 lowdim policy。

    每次呼叫 policy.predict_action() 執行 n_action_steps 個動作，
    重複至 max_steps 或 episode done。
    """

    def __init__(
        self,
        output_dir: str,
        gym_env_id: str = "gym_pusht/PushT-v0",
        obs_key: str = "observation.state",
        obs_dim: Optional[int] = None,
        n_train: int = 0,
        n_test: int = 50,
        max_steps: int = 500,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 10,
        tqdm_interval_sec: float = 1.0,
        render_size: int = 96,
        device: str = "cpu",
    ):
        super().__init__(output_dir)
        self.gym_env_id     = gym_env_id
        self.obs_key        = obs_key
        self.obs_dim        = obs_dim   # if set, slice state to first obs_dim dims
        self.n_test         = n_test
        self.max_steps      = max_steps
        self.n_obs_steps    = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps            = fps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.render_size    = render_size
        self.device         = device

    # ──────────────────────────────────────────────────────────────────
    def run(self, policy: BaseLowdimPolicy) -> Dict:
        device = torch.device(self.device)
        policy.to(device)
        policy.eval()

        success_list: List[bool] = []
        reward_list:  List[float] = []

        pbar = tqdm.tqdm(
            total=self.n_test,
            desc=f"LeRobotRunner [{self.gym_env_id}]",
            mininterval=self.tqdm_interval_sec,
            leave=False,
        )

        for ep_idx in range(self.n_test):
            env = _make_env(self.gym_env_id, self.render_size, seed=ep_idx)
            raw_obs, _ = env.reset(seed=ep_idx)
            policy.reset()

            # 初始化觀測佇列（用來 stack n_obs_steps 幀）
            obs_deque: collections.deque = collections.deque(
                [_extract_state(raw_obs, self.obs_key, self.obs_dim)] * self.n_obs_steps,
                maxlen=self.n_obs_steps,
            )

            total_reward = 0.0
            done = False
            step = 0

            while not done and step < self.max_steps:
                # ── 組成 obs_dict ────────────────────────────────────
                obs_seq = np.stack(list(obs_deque), axis=0)          # (T_o, obs_dim)
                obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)  # (1, T_o, D)
                obs_dict = {"obs": obs_tensor}

                # ── policy 推論 ──────────────────────────────────────
                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                actions = result["action"].squeeze(0).cpu().numpy()  # (n_action_steps, Da)

                # ── 執行動作 ─────────────────────────────────────────
                for a_idx in range(min(self.n_action_steps, len(actions))):
                    raw_obs, reward, terminated, truncated, info = env.step(
                        actions[a_idx]
                    )
                    obs_deque.append(_extract_state(raw_obs, self.obs_key, self.obs_dim))
                    total_reward += float(reward)
                    step += 1
                    done = terminated or truncated or step >= self.max_steps
                    if done:
                        break

            success = _extract_success(info, total_reward)
            success_list.append(success)
            reward_list.append(total_reward)
            env.close()
            pbar.update(1)

        pbar.close()

        success_rate = float(np.mean(success_list))
        mean_reward  = float(np.mean(reward_list))

        log = {
            "test/mean_score": success_rate,
            "test/mean_reward": mean_reward,
            "test/n_episodes": self.n_test,
        }
        return log


# ── 工具函式 ───────────────────────────────────────────────────────────

def _make_env(gym_env_id: str, render_size: int, seed: int = 0):
    """載入對應的 gym 模組並建立環境。"""
    pkg = gym_env_id.split("/")[0]   # e.g. "gym_pusht"
    try:
        __import__(pkg)
    except ImportError as e:
        raise ImportError(
            f"請先安裝 '{pkg}'：pip install {pkg.replace('_', '-')}"
        ) from e

    import gymnasium as gym
    env = gym.make(gym_env_id, obs_type="state",
                   render_mode=None)
    return env


def _extract_state(raw_obs, obs_key: str, obs_dim: Optional[int] = None) -> np.ndarray:
    """從 gym 觀測中取出 lowdim state，可選擇截取前 obs_dim 維。"""
    if isinstance(raw_obs, dict):
        val = raw_obs.get(obs_key, raw_obs.get("observation", None))
        if val is None:
            # 嘗試扁平化所有數值
            arrays = [
                np.asarray(v, dtype=np.float32).ravel()
                for v in raw_obs.values()
                if isinstance(v, (np.ndarray, list, float, int))
            ]
            val = np.concatenate(arrays, axis=0)
    else:
        val = raw_obs
    arr = np.asarray(val, dtype=np.float32)
    if obs_dim is not None:
        arr = arr[:obs_dim]
    return arr


def _extract_success(info: dict, total_reward: float) -> bool:
    """從 info dict 判斷 episode 是否成功。"""
    if isinstance(info, dict):
        for key in ("is_success", "success", "task_success"):
            if key in info:
                return bool(info[key])
    # fallback：reward > 0 視為成功
    return total_reward > 0.0
