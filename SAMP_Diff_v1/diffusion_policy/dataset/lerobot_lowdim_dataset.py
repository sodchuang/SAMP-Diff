"""
LeRobot Lowdim Dataset
將 HuggingFace LeRobot 格式的資料集轉換為 SAMP-Diff 訓練格式。

支援任何只包含 lowdim state 的 LeRobot 資料集，例如：
  - lerobot/pusht            (state 5-dim, action 2-dim)
  - lerobot/aloha_sim_transfer_cube_human  (state 14-dim, action 14-dim)
  - lerobot/aloha_sim_insertion_human
  - lerobot/xarm_lift_medium

用法（Hydra config）：
  dataset:
    _target_: diffusion_policy.dataset.lerobot_lowdim_dataset.LeRobotLowdimDataset
    repo_id: lerobot/pusht
    obs_key: observation.state
    action_key: action
    horizon: 16
    pad_before: 1
    pad_after: 7
    val_ratio: 0.02
    seed: 42
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import array_to_stats


class LeRobotLowdimDataset(BaseLowdimDataset):
    """
    將 LeRobot HuggingFace 資料集包裝成 BaseLowdimDataset。

    返回的每筆樣本：
        {
            'obs':    (horizon, obs_dim)   float32
            'action': (horizon, action_dim) float32
        }
    """

    def __init__(
        self,
        repo_id: str = "lerobot/pusht",
        obs_key: str = "observation.state",
        action_key: str = "action",
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        local_files_only: bool = False,
    ):
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as HFDataset
        except ImportError as e:
            raise ImportError(
                "LeRobot is not installed. Run: pip install lerobot"
            ) from e

        # ── 從 HuggingFace 載入資料集 ──────────────────────────────────
        hf_dataset = HFDataset(
            repo_id=repo_id,
            local_files_only=local_files_only,
        )

        # ── 依 episode 建立 ReplayBuffer ──────────────────────────────
        replay_buffer = ReplayBuffer.create_empty_numpy()

        episode_data_index = hf_dataset.episode_data_index
        n_episodes = len(episode_data_index["from"])

        for ep_idx in range(n_episodes):
            start = int(episode_data_index["from"][ep_idx])
            end   = int(episode_data_index["to"][ep_idx])

            obs_frames    = []
            action_frames = []

            for frame_idx in range(start, end):
                item = hf_dataset[frame_idx]

                obs_val    = _to_numpy(item[obs_key])     # (obs_dim,)
                action_val = _to_numpy(item[action_key])  # (action_dim,)

                obs_frames.append(obs_val)
                action_frames.append(action_val)

            episode = {
                "obs":    np.stack(obs_frames,    axis=0).astype(np.float32),
                "action": np.stack(action_frames, axis=0).astype(np.float32),
            }
            replay_buffer.add_episode(episode)

        # ── 訓練 / 驗證切分 ────────────────────────────────────────────
        val_mask   = get_val_mask(n_episodes=replay_buffer.n_episodes,
                                  val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask,
                                     max_n=max_train_episodes, seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler       = sampler
        self.train_mask    = train_mask
        self.horizon       = horizon
        self.pad_before    = pad_before
        self.pad_after     = pad_after

    # ── BaseLowdimDataset 介面 ─────────────────────────────────────────

    def get_validation_dataset(self) -> "LeRobotLowdimDataset":
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action — 以 min/max 縮放至 [-1, 1]
        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = _minmax_normalizer(action_stat)

        # obs — 同上
        obs_stat = array_to_stats(self.replay_buffer["obs"])
        normalizer["obs"] = _minmax_normalizer(obs_stat)

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        return dict_apply(data, torch.from_numpy)


# ── 工具函式 ───────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    """將 tensor / list / ndarray 統一轉成 float32 ndarray。"""
    if isinstance(x, torch.Tensor):
        return x.float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _minmax_normalizer(stat: dict) -> SingleFieldLinearNormalizer:
    """
    將資料縮放至 [-1, 1]：
        x_norm = 2 * (x - min) / (max - min) - 1
    等價於 scale = 2/(max-min)，offset = -1 - scale*min
    """
    dmin  = stat["min"].astype(np.float64)
    dmax  = stat["max"].astype(np.float64)
    delta = dmax - dmin
    # 避免除零（constant dim）
    delta = np.where(delta < 1e-8, 1.0, delta)
    scale  = (2.0 / delta).astype(np.float32)
    offset = (-1.0 - scale * dmin).astype(np.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )
