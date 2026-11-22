"""Annotate chunks without splitting or mixing demonstration episodes."""
import copy
import json
import numpy as np
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.action_slices import PHASES, validate_segments, phase_at
from diffusion_policy.common.fixed_phase_prior import FixedPhasePrior


class SlicedRobomimicDataset(BaseLowdimDataset):
    def __init__(self, dataset, manifest_path, action_dim, n_obs_steps, n_action_steps,
                 mode='selective', profiles=None):
        self.dataset = dataset
        self.n_obs_steps = int(n_obs_steps)
        self.stride = int(n_action_steps)
        self.engine = FixedPhasePrior(action_dim, self.stride, mode=mode, profiles=profiles)
        with open(manifest_path, encoding='utf-8') as file:
            self.manifest = json.load(file)
        if self.manifest.get('schema_version') != 1:
            raise ValueError('unsupported phase manifest schema')
        if self.manifest['action_dim'] != action_dim:
            raise ValueError('manifest action layout does not match policy')
        ends = np.asarray(dataset.replay_buffer.episode_ends)
        self.ends = ends
        self.starts = np.r_[0, ends[:-1]]
        if len(self.manifest['demos']) != len(ends):
            raise ValueError('manifest demo count mismatch')
        for episode, (start, end) in enumerate(zip(self.starts, ends)):
            demo = self.manifest['demos'][f'demo_{episode}']
            if demo['length'] != int(end - start):
                raise ValueError('manifest length mismatch; prefix truncation is not supported')
            for arm in range(self.engine.n_arms):
                validate_segments(demo['arms'][f'arm{arm}'], demo['length'])
        for key in ('gripper_close_threshold', 'gripper_close_is_greater'):
            if hasattr(dataset, key):
                setattr(self, key, getattr(dataset, key))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        if sample['action'].shape[-1] != self.engine.action_dim:
            raise ValueError('dataset must convert stored axis-angle actions to rotation6d')
        buffer_start, _, sample_start, _ = self.dataset.sampler.indices[index]
        episode = int(np.searchsorted(self.ends, buffer_start, side='right'))
        demo = self.manifest['demos'][f'demo_{episode}']
        step = int(np.clip(buffer_start - sample_start - self.starts[episode]
                           + self.n_obs_steps - 1, 0, demo['length'] - 1))
        phases = []
        valid = torch.zeros(self.engine.action_dim, dtype=torch.bool)
        for arm in range(self.engine.n_arms):
            segments = demo['arms'][f'arm{arm}']
            current = phase_at(segments, step)
            phases.append(current)
            if step < self.stride:
                continue
            previous = phase_at(segments, step - self.stride)
            for group, indices in enumerate(self.engine.indices(arm)):
                spec = self.engine.profiles[PHASES[current]][group]
                keep = self.engine.mode != 'none' and bool(spec[3])
                if current != previous:
                    keep = keep and (self.engine.mode == 'inherit' or
                                     (self.engine.mode == 'selective' and bool(spec[4])))
                valid[indices] = keep
        sample.update(phase_id=torch.tensor(phases), history_valid_mask=valid,
                      task_id=torch.tensor(int(self.manifest.get('task_id', 0))),
                      episode_id=torch.tensor(episode), step=torch.tensor(step))
        return sample

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.dataset = self.dataset.get_validation_dataset()
        return result

    def get_normalizer(self, **kwargs):
        return self.dataset.get_normalizer(**kwargs)

    def get_all_actions(self):
        return self.dataset.get_all_actions()
