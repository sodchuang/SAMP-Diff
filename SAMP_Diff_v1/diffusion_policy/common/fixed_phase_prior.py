"""Fixed semantic profiles; one generator, independent per-arm history ownership.

Phase IDs must come from an explicit caller, not future demonstration actions.
This object has no learned parameters and is not part of the model state_dict.
"""
import math
import torch

from diffusion_policy.common.action_slices import PHASES
from diffusion_policy.common.history_context import HistoryContext, HistoryContextManager


# (low-frequency cutoff, low noise, high noise, reuse, inherit on entry).
# Starting hypotheses, not experimentally optimized or validated constants.
DEFAULT_PROFILES = {
    'grasp': ((8, .30, .20, True, True), (12, .25, .30, True, True), (16, .5, .5, False, False)),
    'transport': ((8, .15, .10, True, True), (12, .12, .10, True, True), (16, .05, .05, True, True)),
    'align': ((12, .12, .08, True, True), (16, .10, .10, True, False), (16, .05, .05, True, True)),
    'release': ((8, .20, .15, True, True), (12, .18, .18, True, True), (16, .5, .5, False, False)),
}


class FixedPhasePrior:
    def __init__(self, action_dim, stride, mode='selective', profiles=None, max_streams=256):
        if action_dim not in (10, 20):
            raise ValueError('fixed phases require absolute xyz/rotation6d/gripper: 10D or 20D')
        if mode not in ('none', 'inherit', 'reset', 'selective'):
            raise ValueError('invalid phase history mode')
        self.action_dim, self.stride, self.mode = action_dim, stride, mode
        if stride < 1:
            raise ValueError('stride must be positive')
        self.n_arms = action_dim // 10
        self.profiles = profiles or DEFAULT_PROFILES
        for phase in PHASES:
            if phase not in self.profiles or len(self.profiles[phase]) != 3:
                raise ValueError('all four profiles must define translation, rotation, gripper')
            for cutoff, low, high, reuse, inherit in self.profiles[phase]:
                if not all(math.isfinite(float(x)) and x >= 0 for x in (cutoff, low, high)):
                    raise ValueError('cutoffs and noise scales must be nonnegative')
        self.cache = HistoryContextManager(mode='inherit', max_streams=max_streams)

    def clear(self, stream_ids=None):
        self.cache.clear(stream_ids)

    def phases(self, phase_id, batch, device):
        phase = torch.as_tensor(phase_id, device=device, dtype=torch.long)
        if phase.shape != (batch, self.n_arms):
            raise ValueError(f'phase_id must have shape ({batch}, {self.n_arms})')
        if ((phase < 0) | (phase >= len(PHASES))).any():
            raise ValueError('phase_id must be 0=grasp, 1=transport, 2=align, 3=release')
        return phase

    def prepare(self, metadata, reference):
        missing = set(('phase_id', 'stream_id', 'task_id', 'episode_id', 'step')) - set(metadata)
        if missing:
            raise ValueError(f'fixed-phase inference needs explicit causal context: {sorted(missing)}; '
                             'the legacy eval runner does not supply a phase detector')
        batch, horizon, dim = reference.shape
        phase = self.phases(metadata['phase_id'], batch, reference.device)
        def values(key):
            value = torch.as_tensor(metadata[key]).reshape(-1).tolist()
            if len(value) != batch:
                raise ValueError(f'{key} batch size mismatch')
            return value
        streams, tasks, episodes, steps = [values(k) for k in ('stream_id', 'task_id', 'episode_id', 'step')]
        contexts = [HistoryContext(int(s), int(t), int(e), 0, int(k))
                    for s, t, e, k in zip(streams, tasks, episodes, steps)]
        decisions = self.cache.plan(contexts, ['actions'], self.stride)
        aligned = torch.zeros_like(reference)
        valid = torch.zeros(batch, dim, dtype=torch.bool, device=reference.device)
        for row, decision in enumerate(decisions):
            state = decision['state']
            if state is None or state['actions'].shape != reference[row].shape:
                continue
            old = state['actions'].to(reference)
            shift = min(self.stride, horizon)
            aligned[row] = torch.cat((old[shift:], old[-1:].expand(shift, -1)), dim=0)
            for arm in range(self.n_arms):
                changed = int(phase[row, arm]) != state['phase'][arm]
                for group, indices in enumerate(self.indices(arm)):
                    spec = self.profiles[PHASES[int(phase[row, arm])]][group]
                    keep = self.mode != 'none' and bool(spec[3])
                    if changed and self.mode == 'reset':
                        keep = False
                    if changed and self.mode == 'selective':
                        keep = keep and bool(spec[4])
                    valid[row, indices] = keep
        return aligned, valid, phase, contexts

    @staticmethod
    def indices(arm):
        offset = arm * 10
        return [list(range(offset, offset + 3)), list(range(offset + 3, offset + 9)), [offset + 9]]

    def source(self, aligned, valid, phase_id, dct):
        batch, horizon, dim = aligned.shape
        phase = self.phases(phase_id, batch, aligned.device)
        if valid.shape != (batch, dim):
            raise ValueError('history_valid_mask must have shape (B, action_dim)')
        coeff = dct(aligned)
        noise = torch.randn_like(coeff)
        result = noise.clone()
        for row in range(batch):
            for arm in range(self.n_arms):
                for group, indices in enumerate(self.indices(arm)):
                    cutoff, low, high, reuse, _ = self.profiles[PHASES[int(phase[row, arm])]][group]
                    scale = torch.full((horizon, 1), float(high), device=aligned.device, dtype=aligned.dtype)
                    scale[:min(int(cutoff), horizon)] = float(low)
                    for index in indices:
                        if reuse and self.mode != 'none' and bool(valid[row, index]):
                            result[row, :, index] = coeff[row, :, index] + scale[:, 0] * noise[row, :, index]
        return result

    def commit(self, contexts, actions, phase):
        states = [{'actions': a.detach().clone(), 'phase': p.tolist()} for a, p in zip(actions, phase)]
        self.cache.commit(contexts, states)
