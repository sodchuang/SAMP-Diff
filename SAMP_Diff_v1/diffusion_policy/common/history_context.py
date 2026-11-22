"""Runtime history ownership and boundary decisions (no model dependencies).

Profiles are reusable across tasks; action tensors are owned by one continuous
stream and episode. No cache is indexed by phase alone.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryContext:
    stream_id: int
    task_id: int
    episode_id: int
    phase_id: int
    step: int
    reset: bool = False


class HistoryContextManager:
    def __init__(self, mode='selective', inherit_groups=None, max_streams=256):
        if mode not in ('inherit', 'reset', 'selective', 'none'):
            raise ValueError('history boundary mode must be inherit/reset/selective/none')
        self.mode = mode
        self.inherit_groups = {str(k): set(v) for k, v in (inherit_groups or {}).items()}
        self.max_streams = int(max_streams)
        if self.max_streams < 1:
            raise ValueError('max_streams must be positive')
        self.records = {}

    def clear(self, stream_ids=None):
        if stream_ids is None:
            self.records.clear()
        else:
            for stream_id in stream_ids:
                self.records.pop(int(stream_id), None)

    def plan(self, contexts, groups, stride):
        if len({c.stream_id for c in contexts}) != len(contexts):
            raise ValueError('duplicate stream_id in the same inference batch')
        if stride < 1:
            raise ValueError('history stride must be positive')
        unknown = set().union(*self.inherit_groups.values()) - set(groups) if self.inherit_groups else set()
        if unknown:
            raise ValueError(f'unknown history groups: {sorted(unknown)}')
        decisions = []
        for c in contexts:
            if min(c.stream_id, c.task_id, c.episode_id, c.phase_id, c.step) < 0:
                raise ValueError('history context identifiers and step must be non-negative')
            record = self.records.get(c.stream_id)
            reason = 'continuous'
            if self.mode == 'none':
                record, reason = None, 'history_disabled'
            elif c.reset:
                record, reason = None, 'explicit_reset'
            elif record is None:
                reason = 'new_stream'
            elif (record['context'].task_id != c.task_id
                  or record['context'].episode_id != c.episode_id):
                record, reason = None, 'task_or_episode_changed'
            elif c.step != record['context'].step + stride:
                record, reason = None, 'noncontiguous_step'
            keep = set(groups) if record is not None else set()
            if record is not None and record['context'].phase_id != c.phase_id:
                reason = 'phase_changed'
                if self.mode == 'reset':
                    keep = set()
                elif self.mode == 'selective':
                    # Missing phase profiles fail closed: no history is inherited.
                    keep = self.inherit_groups.get(str(c.phase_id), set()) & set(groups)
            decisions.append({'state': record['state'] if record else None,
                              'keep_groups': keep, 'reason': reason})
        return decisions

    def commit(self, contexts, states):
        if len(contexts) != len(states):
            raise ValueError('context/state batch sizes differ')
        if len({c.stream_id for c in contexts}) != len(contexts):
            raise ValueError('duplicate stream_id in commit')
        if len(set(self.records) | {c.stream_id for c in contexts}) > self.max_streams:
            raise RuntimeError('history capacity exceeded; release finished streams with reset_streams()')
        for context, state in zip(contexts, states):
            self.records[context.stream_id] = {'context': context, 'state': state}
