from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            normalizer_use_full_dataset=False,
            episode_prefix_enabled=False,
            episode_prefix_anchor='first_close',
            episode_prefix_close_threshold=0.0,
            episode_prefix_close_is_greater=True,
            episode_prefix_after=48,
            episode_prefix_min_steps=None,
            episode_prefix_max_steps=None,
            episode_prefix_min_anchor_ratio=0.9,
            anchor_oversample_enabled=False,
            anchor_oversample_before=48,
            anchor_oversample_after=8,
            anchor_oversample_repeats=1,
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "RobomimicReplayLowdimDataset requires h5py to read HDF5 "
                "datasets. Install it in the training environment with "
                "`pip install h5py` or `conda install h5py`."
            ) from e

        replay_buffer = ReplayBuffer.create_empty_numpy()
        prefix_lengths = list()
        prefix_anchor_indices = list()
        full_stats = None
        if normalizer_use_full_dataset:
            full_stats = {'action': None, 'obs': None}
        with h5py.File(dataset_path) as file:
            demos = file['data']
            auto_threshold = _is_auto(episode_prefix_close_threshold)
            auto_direction = _is_auto(episode_prefix_close_is_greater)
            if auto_threshold or auto_direction:
                inferred_threshold, inferred_direction, inference_info = (
                    _infer_gripper_close_semantics(demos)
                )
                if auto_threshold:
                    episode_prefix_close_threshold = inferred_threshold
                else:
                    episode_prefix_close_threshold = float(
                        episode_prefix_close_threshold)
                if auto_direction:
                    episode_prefix_close_is_greater = inferred_direction
                else:
                    episode_prefix_close_is_greater = _as_bool(
                        episode_prefix_close_is_greater)
                print(
                    "[Dataset] inferred gripper close semantics: "
                    f"threshold={episode_prefix_close_threshold:.6g}, "
                    f"close_is_greater={episode_prefix_close_is_greater}, "
                    f"{inference_info}"
                )
            else:
                episode_prefix_close_threshold = float(
                    episode_prefix_close_threshold)
                episode_prefix_close_is_greater = _as_bool(
                    episode_prefix_close_is_greater)

            self.gripper_close_threshold = episode_prefix_close_threshold
            self.gripper_close_is_greater = episode_prefix_close_is_greater
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer)
                if full_stats is not None:
                    for key in ('action', 'obs'):
                        full_stats[key] = _update_running_stats(
                            full_stats[key], episode[key])
                if episode_prefix_enabled:
                    prefix_anchor_indices.append(_find_prefix_anchor_index(
                        actions=episode['action'],
                        anchor=episode_prefix_anchor,
                        close_threshold=episode_prefix_close_threshold,
                        close_is_greater=episode_prefix_close_is_greater,
                    ))
                    episode = _truncate_episode_prefix(
                        episode=episode,
                        anchor=episode_prefix_anchor,
                        close_threshold=episode_prefix_close_threshold,
                        close_is_greater=episode_prefix_close_is_greater,
                        after=episode_prefix_after,
                        min_steps=episode_prefix_min_steps,
                        max_steps=episode_prefix_max_steps)
                    prefix_lengths.append(len(episode['action']))
                replay_buffer.add_episode(episode)

        if episode_prefix_enabled and len(prefix_lengths) > 0:
            found_anchor_indices = [
                index for index in prefix_anchor_indices if index is not None]
            anchor_ratio = len(found_anchor_indices) / len(prefix_anchor_indices)
            anchor_summary = "none"
            if found_anchor_indices:
                anchor_summary = (
                    f"p10={np.quantile(found_anchor_indices, 0.10):.1f}, "
                    f"median={np.median(found_anchor_indices):.1f}, "
                    f"p90={np.quantile(found_anchor_indices, 0.90):.1f}"
                )
            print(
                "[Dataset] episode prefix enabled: "
                f"anchor={episode_prefix_anchor}, after={episode_prefix_after}, "
                f"min={min(prefix_lengths)}, mean={np.mean(prefix_lengths):.1f}, "
                f"max={max(prefix_lengths)}, anchor_ratio={anchor_ratio:.3f}, "
                f"anchor_step({anchor_summary})")
            self.episode_prefix_anchor_ratio = float(anchor_ratio)
            self.episode_prefix_anchor_indices = tuple(found_anchor_indices)
            if (
                episode_prefix_min_anchor_ratio is not None
                and anchor_ratio < float(episode_prefix_min_anchor_ratio)
            ):
                raise ValueError(
                    "episode prefix anchor coverage is too low: "
                    f"anchor={episode_prefix_anchor}, found="
                    f"{len(found_anchor_indices)}/{len(prefix_anchor_indices)} "
                    f"({anchor_ratio:.3f}), required>="
                    f"{float(episode_prefix_min_anchor_ratio):.3f}. "
                    "Refusing to train on fallback prefixes because they may "
                    "teach the wrong ToolHang stage."
                )

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        # A phase-loss window can only see events that fall inside the sampled
        # horizon. ToolHang uses horizon=16, so configuring a 48-step window
        # before first release did not actually emphasize most of the insertion
        # approach. Repeat the relevant sequence indices at the dataset level
        # instead. This keeps the full A->B trajectory available while making
        # alignment / insertion chunks occur often enough to learn.
        if anchor_oversample_enabled:
            if not episode_prefix_enabled:
                raise ValueError(
                    "anchor_oversample_enabled requires "
                    "episode_prefix_enabled so anchor indices are available")
            repeats = int(anchor_oversample_repeats)
            before = int(anchor_oversample_before)
            after = int(anchor_oversample_after)
            if repeats < 1:
                raise ValueError(
                    "anchor_oversample_repeats must be at least 1")
            if before < 0 or after < 0:
                raise ValueError(
                    "anchor_oversample_before/after must be non-negative")

            episode_ends = replay_buffer.episode_ends[:]
            episode_starts = np.concatenate(
                [np.zeros(1, dtype=episode_ends.dtype), episode_ends[:-1]])
            focus_mask = np.zeros(len(sampler.indices), dtype=bool)
            buffer_starts = sampler.indices[:, 0]
            buffer_ends = sampler.indices[:, 1]
            usable_anchors = 0
            for episode_idx, anchor_idx in enumerate(prefix_anchor_indices):
                if (
                    anchor_idx is None
                    or not train_mask[episode_idx]
                    or anchor_idx >= prefix_lengths[episode_idx]
                ):
                    continue
                usable_anchors += 1
                global_anchor = (
                    int(episode_starts[episode_idx]) + int(anchor_idx))
                focus_start = max(
                    int(episode_starts[episode_idx]),
                    global_anchor - before)
                focus_end = min(
                    int(episode_ends[episode_idx]),
                    global_anchor + after + 1)
                focus_mask |= (
                    (buffer_ends > focus_start)
                    & (buffer_starts < focus_end)
                )

            focus_indices = sampler.indices[focus_mask]
            if usable_anchors == 0 or len(focus_indices) == 0:
                raise ValueError(
                    "anchor oversampling found no usable training windows")
            if repeats > 1:
                sampler.indices = np.concatenate(
                    [sampler.indices]
                    + [focus_indices.copy() for _ in range(repeats - 1)],
                    axis=0)
            print(
                "[Dataset] anchor oversampling enabled: "
                f"anchor={episode_prefix_anchor}, before={before}, "
                f"after={after}, repeats={repeats}, "
                f"focus_samples={len(focus_indices)}, "
                f"total_samples={len(sampler.indices)}, "
                f"usable_anchors={usable_anchors}")
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.full_normalizer_stats = None
        if full_stats is not None:
            self.full_normalizer_stats = {
                key: _finalize_running_stats(value)
                for key, value in full_stats.items()
            }
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        if self.full_normalizer_stats is None:
            stat = array_to_stats(self.replay_buffer['action'])
        else:
            stat = self.full_normalizer_stats['action']
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # aggregate obs stats
        if self.full_normalizer_stats is None:
            obs_stat = array_to_stats(self.replay_buffer['obs'])
        else:
            obs_stat = self.full_normalizer_stats['obs']


        normalizer['obs'] = normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def _update_running_stats(accumulator, array):
    """Accumulate full-dataset statistics without retaining a second buffer."""
    values = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
    if accumulator is None:
        accumulator = {
            'count': 0,
            'sum': np.zeros(values.shape[-1], dtype=np.float64),
            'sum_sq': np.zeros(values.shape[-1], dtype=np.float64),
            'min': np.full(values.shape[-1], np.inf, dtype=np.float64),
            'max': np.full(values.shape[-1], -np.inf, dtype=np.float64),
        }
    accumulator['count'] += len(values)
    accumulator['sum'] += values.sum(axis=0, dtype=np.float64)
    accumulator['sum_sq'] += np.square(
        values, dtype=np.float64).sum(axis=0, dtype=np.float64)
    accumulator['min'] = np.minimum(
        accumulator['min'], values.min(axis=0))
    accumulator['max'] = np.maximum(
        accumulator['max'], values.max(axis=0))
    return accumulator


def _finalize_running_stats(accumulator):
    count = int(accumulator['count'])
    if count <= 0:
        raise ValueError("cannot finalize empty normalizer statistics")
    mean = accumulator['sum'] / count
    variance = np.maximum(
        accumulator['sum_sq'] / count - np.square(mean), 0.0)
    return {
        'min': accumulator['min'].astype(np.float32),
        'max': accumulator['max'].astype(np.float32),
        'mean': mean.astype(np.float32),
        'std': np.sqrt(variance).astype(np.float32),
    }
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    data = {
        'obs': obs,
        'action': raw_actions
    }
    return data


def _truncate_episode_prefix(
        episode,
        anchor='first_close',
        close_threshold=0.0,
        close_is_greater=True,
        after=48,
        min_steps=None,
        max_steps=None):
    """Keep only an early prefix of a Robomimic episode.

    This is useful for staged ToolHang training: the first stage should learn
    reach -> close -> lift without seeing later rack / hang behavior.
    """
    actions = episode['action']
    episode_len = len(actions)
    if episode_len <= 0:
        return episode

    if anchor not in ('first_close', 'close', 'first_release', 'release', 'fixed'):
        raise ValueError(
            "episode_prefix_anchor must be one of "
            "('first_close', 'close', 'first_release', 'release', 'fixed'), got "
            f"{anchor!r}")

    if anchor in ('first_close', 'close', 'first_release', 'release'):
        anchor_index = _find_prefix_anchor_index(
            actions=actions,
            anchor=anchor,
            close_threshold=close_threshold,
            close_is_greater=close_is_greater,
        )
        if anchor_index is not None:
            end = int(anchor_index) + int(after) + 1
        else:
            # Conservative fallback: if a demo has no detectable close event,
            # keep an early chunk rather than exposing the whole hang stage.
            fallback = max_steps if max_steps is not None else (after + 1)
            end = int(fallback)
    else:
        end = max_steps if max_steps is not None else episode_len

    if min_steps is not None:
        end = max(end, int(min_steps))
    if max_steps is not None:
        end = min(end, int(max_steps))
    end = min(max(1, end), episode_len)

    return {
        key: value[:end]
        for key, value in episode.items()
    }


def _find_prefix_anchor_index(
        actions,
        anchor='first_close',
        close_threshold=0.0,
        close_is_greater=True):
    if anchor == 'fixed':
        return 0
    if anchor not in ('first_close', 'close', 'first_release', 'release'):
        raise ValueError(f"unsupported episode prefix anchor: {anchor!r}")
    if len(actions) == 0:
        return None

    gripper = np.asarray(actions)[..., -1]
    if close_is_greater:
        close_mask = gripper > close_threshold
    else:
        close_mask = gripper < close_threshold
    if anchor in ('first_release', 'release'):
        prev_close = np.zeros_like(close_mask, dtype=bool)
        prev_close[1:] = close_mask[:-1]
        anchor_idxs = np.flatnonzero(prev_close & (~close_mask))
    else:
        anchor_idxs = np.flatnonzero(close_mask)
    if len(anchor_idxs) == 0:
        return None
    return int(anchor_idxs[0])


def _is_auto(value):
    return isinstance(value, str) and value.strip().lower() == 'auto'


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', '1', 'yes', 'y'):
            return True
        if normalized in ('false', '0', 'no', 'n'):
            return False
        raise ValueError(f"expected a boolean or 'auto', got {value!r}")
    return bool(value)


def _infer_gripper_close_semantics(demos, max_demos=64):
    """Infer the raw gripper command split and which side closes the fingers.

    A close command should produce a smaller gripper opening on the following
    observation.  This uses that physical relation instead of assuming that
    positive or negative commands always mean close across robosuite versions.
    """
    commands = []
    next_widths = []
    demo_count = min(len(demos), int(max_demos))
    for i in range(demo_count):
        demo = demos[f'demo_{i}']
        if 'robot0_gripper_qpos' not in demo['obs']:
            continue
        action = np.asarray(demo['actions'][:], dtype=np.float32)
        qpos = np.asarray(
            demo['obs']['robot0_gripper_qpos'][:], dtype=np.float32)
        n = min(len(action), len(qpos))
        if n < 2:
            continue
        # Raw robomimic dual-arm actions are
        # [robot0(pos3, rot3, grip1), robot1(pos3, rot3, grip1)].
        # The old -1 indexing paired robot1's command with robot0's observed
        # finger width and could infer the wrong close direction.
        if action.shape[-1] == 14:
            robot0_gripper_index = 6
        elif action.shape[-1] == 20:
            # Also support datasets that already store rotation-6D actions.
            robot0_gripper_index = 9
        else:
            robot0_gripper_index = action.shape[-1] - 1
        command = action[:n - 1, robot0_gripper_index].reshape(-1)
        width = np.abs(qpos[1:n]).reshape(n - 1, -1).sum(axis=-1)
        finite = np.isfinite(command) & np.isfinite(width)
        commands.append(command[finite])
        next_widths.append(width[finite])

    if not commands:
        raise ValueError(
            "cannot infer gripper close direction: no aligned gripper actions "
            "and robot0_gripper_qpos observations were found")

    command = np.concatenate(commands)
    width = np.concatenate(next_widths)
    low_value, high_value = np.quantile(command, [0.10, 0.90])
    if not np.isfinite(low_value + high_value) or np.isclose(low_value, high_value):
        raise ValueError(
            "cannot infer gripper close direction: gripper commands have no "
            f"usable spread (p10={low_value}, p90={high_value})")

    threshold = float((low_value + high_value) / 2.0)
    lower = command < threshold
    upper = command > threshold
    if lower.sum() < 16 or upper.sum() < 16:
        raise ValueError(
            "cannot infer gripper close direction: too few command samples on "
            f"one side of threshold={threshold:.6g} "
            f"(lower={lower.sum()}, upper={upper.sum()})")

    lower_width = float(np.median(width[lower]))
    upper_width = float(np.median(width[upper]))
    if np.isclose(lower_width, upper_width, rtol=1e-3, atol=1e-6):
        raise ValueError(
            "cannot infer gripper close direction: observed finger widths are "
            f"indistinguishable (lower={lower_width}, upper={upper_width})")
    close_is_greater = upper_width < lower_width
    info = (
        f"command_p10={low_value:.6g}, command_p90={high_value:.6g}, "
        f"next_width_lower={lower_width:.6g}, "
        f"next_width_upper={upper_width:.6g}, samples={len(command)}"
    )
    return threshold, bool(close_is_greater), info
