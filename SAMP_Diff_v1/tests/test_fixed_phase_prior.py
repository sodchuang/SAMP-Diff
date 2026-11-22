import pathlib
import sys
import unittest
import torch
import torch_dct

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.common.fixed_phase_prior import FixedPhasePrior
from diffusion_policy.common.action_slices import event_segments, validate_segments, phase_at


def metadata(streams=(1,), phases=((0,),), step=0, episode=0, task=0):
    batch = len(streams)
    return dict(stream_id=streams, phase_id=phases, step=[step]*batch,
                episode_id=[episode]*batch, task_id=[task]*batch)


class FixedPhaseTests(unittest.TestCase):
    def test_new_and_continuous(self):
        engine = FixedPhasePrior(10, 2)
        reference = torch.arange(60.).reshape(1, 6, 10)
        aligned, valid, phase, ctx = engine.prepare(metadata(), reference)
        self.assertFalse(valid.any())
        engine.commit(ctx, reference, phase)
        aligned, valid, _, _ = engine.prepare(metadata(step=2), reference)
        torch.testing.assert_close(aligned[0, 0], reference[0, 2])
        self.assertTrue(valid[0, :9].all())
        self.assertFalse(valid[0, 9])  # grasp gripper is fresh

    def test_episode_task_and_discontinuity(self):
        for change in (dict(episode=1), dict(task=1), dict(step=3)):
            engine = FixedPhasePrior(10, 2)
            ref = torch.ones(1, 6, 10)
            _, _, phase, ctx = engine.prepare(metadata(), ref)
            engine.commit(ctx, ref, phase)
            changed = dict(step=2)
            changed.update(change)
            self.assertFalse(engine.prepare(metadata(**changed), ref)[1].any())

    def test_two_arms_switch_independently(self):
        engine = FixedPhasePrior(20, 2)
        ref = torch.ones(1, 6, 20)
        _, _, phase, ctx = engine.prepare(metadata(phases=((1, 1),)), ref)
        engine.commit(ctx, ref, phase)
        _, valid, _, _ = engine.prepare(metadata(phases=((2, 1),), step=2), ref)
        self.assertFalse(valid[0, 3:9].any())
        self.assertTrue(valid[0, 13:19].all())
        self.assertTrue(valid[0, :3].all())

    def test_batch_reorder_uses_stream_identity(self):
        engine = FixedPhasePrior(10, 2)
        ref = torch.stack((torch.ones(6, 10), torch.ones(6, 10)*7))
        _, _, phase, ctx = engine.prepare(metadata((1, 2), ((1,), (1,))), ref)
        engine.commit(ctx, ref, phase)
        aligned, _, _, _ = engine.prepare(metadata((2, 1), ((1,), (1,)), step=2), ref)
        self.assertEqual(float(aligned[0, 0, 0]), 7.)
        self.assertEqual(float(aligned[1, 0, 0]), 1.)

    def test_reset_and_duplicate_stream(self):
        engine = FixedPhasePrior(10, 2)
        ref = torch.ones(1, 6, 10)
        _, _, phase, ctx = engine.prepare(metadata(), ref)
        engine.commit(ctx, ref, phase)
        engine.clear([1])
        self.assertFalse(engine.prepare(metadata(step=2), ref)[1].any())
        with self.assertRaises(ValueError):
            engine.prepare(metadata((1, 1), ((1,), (1,))), ref.expand(2, -1, -1))

    def test_source_fresh_and_fixed_profiles(self):
        engine = FixedPhasePrior(10, 2)
        ref = torch.ones(1, 6, 10)
        dct = lambda x: torch_dct.dct(x.transpose(1, 2), norm='ortho').transpose(1, 2)
        valid = torch.ones(1, 10, dtype=torch.bool)
        torch.manual_seed(4)
        result = engine.source(ref, valid, [[0]], dct)
        torch.manual_seed(4)
        noise = torch.randn_like(dct(ref))
        torch.testing.assert_close(result[:, :, 9], noise[:, :, 9])
        torch.testing.assert_close(result[:, :, 0], dct(ref)[:, :, 0] + .3*noise[:, :, 0])

    def test_slices_and_invalid_coverage(self):
        slices = event_segments([-1, -1, 1, 1, 1, 1, -1, -1], align_before_release=2)
        validate_segments(slices, 8)
        self.assertEqual([phase_at(slices, i) for i in (0, 2, 4, 6)], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            validate_segments([dict(start=1, end=8, phase='grasp')], 8)

    def test_layout_and_phase_shape_rejected(self):
        with self.assertRaises(ValueError):
            FixedPhasePrior(7, 2)
        with self.assertRaises(ValueError):
            FixedPhasePrior(20, 2).prepare(metadata(), torch.zeros(1, 6, 20))


if __name__ == '__main__':
    unittest.main()
