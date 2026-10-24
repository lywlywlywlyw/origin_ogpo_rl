"""Check JGCW path setup from outside the repository without starting training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class JGCWLauncherTests(unittest.TestCase):
    def launch(self, *, pythonpath=None, exit_code=0):
        source = Path(__file__).resolve().parents[1] / 'scripts/jgcw/square_shared_ogpo_bc.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repository with spaces'
            script = root / 'scripts/jgcw/square_shared_ogpo_bc.sh'
            script.parent.mkdir(parents=True)
            script.write_text(source.read_text())
            interpreter = root / '.venv/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text(
                f'#!{sys.executable}\nimport json, os, sys\n'
                'print(json.dumps(dict(cwd=os.getcwd(), interpreter=sys.argv[0], '
                'pythonpath=os.environ.get("PYTHONPATH"), args=sys.argv[1:])))\n'
                f'sys.exit({exit_code})\n')
            interpreter.chmod(0o755)
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            if pythonpath is not None:
                env['PYTHONPATH'] = pythonpath
            result = subprocess.run(
                ['bash', str(script), '--experiment.seed=3'], cwd=directory,
                env=env, text=True, capture_output=True)
            self.assertEqual(result.stderr, '')
            payload = json.loads(result.stdout)
            self.assertEqual(payload['cwd'], str(root))
            self.assertEqual(payload['interpreter'], str(interpreter))
            self.assertEqual(payload['pythonpath'], str(root) + (':' + pythonpath if pythonpath else ''))
            self.assertEqual(payload['args'], ['-m', 'ogpo.main',
                                             '--algo=jgcw_ogpo_square_shared', '--experiment.seed=3'])
            return result.returncode

    def test_unset_pythonpath_and_external_working_directory(self):
        self.assertEqual(self.launch(), 0)

    def test_existing_pythonpath_is_preserved(self):
        self.assertEqual(self.launch(pythonpath='/tmp/existing modules:/tmp/other'), 0)

    def test_python_exit_status_is_preserved(self):
        self.assertEqual(self.launch(exit_code=7), 7)
