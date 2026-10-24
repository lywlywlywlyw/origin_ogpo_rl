"""Regression checks for launcher precedence and async diagnostic plot steps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from ogpo.utils.log_utils import CsvLogger
from ogpo.utils.main_helper import LoggingHelper


class CoverageLauncherTests(unittest.TestCase):
    def launch_args(self, *overrides):
        script = Path(__file__).resolve().parents[1] / 'scripts/ogpo/square.sh'
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory)
            # Capture the actual launch arguments without starting training.
            (tools / 'uv').write_text(
                f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            (tools / 'bc').write_text('#!/bin/sh\ncat >/dev/null\necho 1\n')
            for name in ('uv', 'bc'):
                (tools / name).chmod(0o755)
            env = dict(os.environ, PATH=f'{tools}:{os.environ["PATH"]}')
            result = subprocess.run(['bash', str(script), *overrides], env=env,
                                    check=True, capture_output=True, text=True)
            return json.loads(result.stdout.splitlines()[-1])

    def test_online_only_checkpoint_options_are_forwarded(self):
        args = self.launch_args('--checkpoint_resume=false', '--bc_pi_steps=0',
                                '--restore_actor_path=/tmp/offline.pkl',
                                '--restore_critic_path=/tmp/offline.pkl', '--ep_resume=90000')
        for option in ('--checkpoint.resume=false', '--training.bc_pi_steps=0',
                       '--checkpoint.restore_actor_path=/tmp/offline.pkl',
                       '--checkpoint.restore_critic_path=/tmp/offline.pkl',
                       '--checkpoint.ep_resume=90000'):
            self.assertIn(option, args)

    def test_default_launcher_preserves_yaml_coverage_settings(self):
        args = self.launch_args()
        self.assertFalse(any(arg.startswith(('--agent.cal_action_coverage=',
                                             '--agent.coverage_')) for arg in args))

    def test_only_explicit_coverage_overrides_are_forwarded(self):
        options = ['cal_action_coverage=true', 'coverage_k=17', 'coverage_epsilon_a=0.25',
                   'coverage_sample_interval=3', 'coverage_rebuild_interval=250']
        args = self.launch_args(*(f'--{option}' for option in options))
        coverage = [arg for arg in args if arg.startswith(
            ('--agent.cal_action_coverage=', '--agent.coverage_'))]
        self.assertEqual(coverage, [f'--agent.{option}' for option in options])
        args = self.launch_args('--cal_action_coverage=false')
        coverage = [arg for arg in args if arg.startswith(
            ('--agent.cal_action_coverage=', '--agent.coverage_'))]
        self.assertEqual(coverage, ['--agent.cal_action_coverage=false'])


class AccuracyAxisTests(unittest.TestCase):
    def test_accuracy_uses_snapshot_axis_and_current_submission_step(self):
        wandb = Mock()
        with tempfile.TemporaryDirectory() as directory:
            logger = LoggingHelper({'q_accuracy': CsvLogger(str(Path(directory) / 'q.csv'))},
                                   wandb_logger=wandb)
            logger.log({'snapshot_step': 120, 'q_sz/mae': 2.5}, 'q_accuracy', step=9000)
            logger.close()
        self.assertEqual(wandb.define_metric.call_args_list, [
            call('q_accuracy/snapshot_step'),
            call('q_accuracy/*', step_metric='q_accuracy/snapshot_step', step_sync=False),
        ])
        payload = wandb.log.call_args.args[0]
        self.assertEqual(payload['q_accuracy/snapshot_step'], 120)
        self.assertEqual(payload['q_accuracy/q_sz/mae'], 2.5)
        self.assertEqual(wandb.log.call_args.kwargs['step'], 9000)

    def test_other_metrics_keep_their_default_axis(self):
        wandb = Mock()
        LoggingHelper({'online_agent': None}, wandb_logger=wandb)
        wandb.define_metric.assert_not_called()
