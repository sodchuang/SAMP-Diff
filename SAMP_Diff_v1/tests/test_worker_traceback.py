"""Exercise the real error sender without requiring gym or MuJoCo."""
import ast
import contextlib
import io
import pathlib
import queue
import sys
import traceback
import unittest


class WorkerTracebackTests(unittest.TestCase):
    def test_empty_assertion_keeps_traceback_and_protocol(self):
        source = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/gym_util/async_vector_env.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_send_worker_error')
        namespace = dict(sys=sys, traceback=traceback)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        class Pipe:
            def send(self, value):
                self.value = value
        pipe, errors, output = Pipe(), queue.Queue(), io.StringIO()
        with contextlib.redirect_stderr(output):
            try:
                assert False
            except AssertionError:
                namespace['_send_worker_error'](pipe, 3, errors)
        index, kind, value = errors.get_nowait()
        self.assertEqual(index, 3)
        self.assertIs(kind, AssertionError)
        self.assertIsInstance(value, AssertionError)
        self.assertEqual(pipe.value, (None, False))
        self.assertIn('assert False', output.getvalue())
        self.assertIn('Traceback', output.getvalue())


if __name__ == '__main__':
    unittest.main()
