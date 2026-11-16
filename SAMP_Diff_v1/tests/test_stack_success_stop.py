"""Control-flow tests without requiring MuJoCo, torch, or gym."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


STEP = method(
    'diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py',
    'RobomimicLowdimWrapper', 'step',
    {'np': SimpleNamespace(concatenate=lambda values, axis: values)},
)


class StackSuccessStopTest(unittest.TestCase):
    def make_env(self, enabled, success=False, underlying_done=False):
        env = SimpleNamespace(
            env=SimpleNamespace(step=lambda a: ({'object': 1}, 0.0, underlying_done, {})),
            obs_keys=['object'], stack_stop_on_success=enabled,
            _update_tool_hang_stage_state=lambda: None,
            _update_stack_release_state=lambda: None,
            get_stack_release_flags=lambda: {
                'available': True, 'final_released_stacked': success,
                'strict_success': False},
        )
        return env

    def test_disabled_preserves_rollout(self):
        self.assertFalse(STEP(self.make_env(False, True), None)[2])

    def test_waits_for_confirmed_success(self):
        self.assertFalse(STEP(self.make_env(True, False), None)[2])

    def test_success_sets_done_without_changing_reward(self):
        _, reward, done, info = STEP(self.make_env(True, True), None)
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)
        self.assertTrue(info['stack_terminated_on_success'])

    def test_preserves_underlying_done(self):
        self.assertTrue(STEP(self.make_env(True, False, True), None)[2])

    def test_non_stack_does_not_terminate(self):
        env = self.make_env(True, True)
        env.get_stack_release_flags = lambda: {
            'available': False, 'final_released_stacked': True}
        self.assertFalse(STEP(env, None)[2])

    def test_hold_counter_requires_continuous_stack_and_retreat(self):
        update = method(
            'diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py',
            'RobomimicLowdimWrapper', '_update_stack_release_state', {},
        )
        flags = method(
            'diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py',
            'RobomimicLowdimWrapper', 'get_stack_release_flags', {},
        )
        task = SimpleNamespace(_check_success=lambda: True)
        env = SimpleNamespace(
            _stack_task=lambda: task, _stack_grasped=lambda t: False,
            _stack_eef_distance=lambda t: 0.1,
            _stack_gripper_open_width=lambda t: 0.05,
            stack_min_eef_distance=0.04, stack_min_gripper_open_width=0.04,
            stack_release_hold_steps=10, _stack_release_consecutive=0,
            _stack_release_max_consecutive=0, _stack_seen_released_stacked=False,
            _stack_seen_gripper_away=False,
        )
        for _ in range(9):
            update(env)
        self.assertFalse(flags(env)['strict_success'])
        update(env)
        self.assertTrue(flags(env)['strict_success'])
        task._check_success = lambda: False
        update(env)
        self.assertFalse(flags(env)['strict_success'])
        self.assertEqual(env._stack_release_consecutive, 0)

    def test_chunk_stops_before_remaining_actions(self):
        class Parent:
            def step(self, action):
                self.calls.append(action)
                return 0, 0, len(self.calls) == 2, {}

        tree = ast.parse((ROOT / 'diffusion_policy/gym_util/multistep_wrapper.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiStepWrapper')
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        stub = ast.ClassDef(name='Chunk', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                            keywords=[], body=[step], decorator_list=[])
        ns = {'Parent': Parent, 'aggregate': lambda values, *args: max(values),
              'dict_take_last_n': lambda *args: {}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])),
                     '<chunk-test>', 'exec'), ns)
        env = ns['Chunk']()
        env.calls, env.obs, env.reward, env.done = [], deque(), [], []
        env.info, env.n_obs_steps, env.max_episode_steps = {}, 2, 400
        env.reward_agg_method = 'max'
        env._get_obs, env._add_info = lambda n: 0, lambda info: None
        self.assertTrue(env.step(list(range(8)))[2])
        self.assertEqual(env.calls, [0, 1])


if __name__ == '__main__':
    unittest.main()
