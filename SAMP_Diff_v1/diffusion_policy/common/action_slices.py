"""Validated, episode-local action slices. No task-specific object indices."""
import bisect


PHASES = ('grasp', 'transport', 'align', 'release')


def validate_segments(segments, length):
    cursor = 0
    if length < 1:
        raise ValueError('episode length must be positive')
    for segment in segments:
        if segment['phase'] not in PHASES:
            raise ValueError(f'unknown phase: {segment["phase"]}')
        if int(segment['start']) != cursor or int(segment['end']) <= cursor:
            raise ValueError('slices must be contiguous, ordered and nonempty')
        cursor = int(segment['end'])
        if cursor > length:
            raise ValueError('slice extends beyond episode')
    if cursor != length:
        raise ValueError('slices must cover the entire episode')


def phase_at(segments, index):
    if index < 0 or index >= segments[-1]['end']:
        raise IndexError('phase lookup outside episode')
    starts = [s['start'] for s in segments]
    return PHASES.index(segments[bisect.bisect_right(starts, index) - 1]['phase'])


def event_segments(gripper, close_threshold=0.0, close_is_greater=True,
                   align_before_release=12):
    """Weak offline labels, NOT a learned phase detector or success detector.

    Open->closed starts transport. A fixed window before a closed->open event
    is labeled align, and the subsequent open interval is labeled release.
    Initial approach/closure is grasp. Future release events are used only
    for OFFLINE labeling and must never be fed to a deployed router.
    """
    if len(gripper) == 0 or align_before_release < 0:
        raise ValueError('invalid gripper sequence or align window')
    closed = [(g >= close_threshold if close_is_greater else g <= close_threshold)
              for g in gripper]
    labels = ['grasp'] * len(closed)
    state = 'grasp'
    close_start = 0
    for i in range(len(closed)):
        if i > 0 and closed[i] and not closed[i - 1]:
            state, close_start = 'transport', i
        if i > 0 and not closed[i] and closed[i - 1]:
            for j in range(max(close_start + 1, i - align_before_release), i):
                labels[j] = 'align'
            state = 'release'
        labels[i] = state
    segments = []
    for i, label in enumerate(labels):
        if not segments or segments[-1]['phase'] != label:
            segments.append({'start': i, 'end': i + 1, 'phase': label})
        else:
            segments[-1]['end'] = i + 1
    validate_segments(segments, len(closed))
    return segments
