"""CPU contract checks: video workers render, parent dummy stays headless."""
import ast
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/env_runner/robomimic_lowdim_runner.py'


class RenderContextTests(unittest.TestCase):
    def test_create_env_forwards_offscreen_flag(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_env')
        modules = {name: types.ModuleType(name) for name in (
            'robomimic', 'robomimic.utils', 'robomimic.utils.env_utils', 'robomimic.utils.obs_utils')}
        calls = []
        modules['robomimic.utils.env_utils'].create_env_from_metadata = lambda **kw: calls.append(kw)
        modules['robomimic.utils.obs_utils'].initialize_obs_modality_mapping_from_dict = lambda value: None
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        with patch.dict(sys.modules, modules):
            for flag in (False, True):
                namespace['create_env']({'env_name': 'ToolHang'}, ['object'], render_offscreen=flag)
                self.assertIs(calls[-1]['render_offscreen'], flag)

    def test_initial_and_recovered_vectors_use_headless_dummy(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'AsyncVectorEnv']
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn('dummy_env_fn', [k.arg for k in call.keywords])
        factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'env_fn')
        self.assertEqual(ast.unparse(factory.args.defaults[0]), 'bool(n_train_vis or n_test_vis)')
        dummy = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'partial')
        self.assertEqual(ast.literal_eval(dummy.keywords[0].value), False)


if __name__ == '__main__':
    unittest.main()
