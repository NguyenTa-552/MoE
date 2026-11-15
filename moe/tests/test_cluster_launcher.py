"""Exercise submission and environment setup without contacting SLURM."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'moe/scripts/run_cluster.sh'


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(('CLUSTER_', 'MOE_', 'SLURM_')) and key != 'BASH_ENV'}


class ClusterLauncherTests(unittest.TestCase):
    def preview(self, *arguments, environment=None, script=LAUNCHER):
        env = clean_environment()
        if environment:
            env.update(environment)
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run(['bash', str(script), '--dry-run', *arguments],
                                  cwd=cwd, env=env, text=True, capture_output=True)

    def test_default_96gb_job_qos_and_quoted_paths(self):
        result = self.preview('source', 'config with spaces.json', 'runs/test run', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertEqual(args[0], 'sbatch')
        self.assertIn('--partition=student9696', args)
        self.assertIn('--gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_4g.96gb:1', args)
        self.assertIn(f'--chdir={ROOT}', args)
        self.assertEqual(args[-5:], ['--batch', 'source', 'config with spaces.json', 'runs/test run', '0'])
        self.assertIn('--qos=qos96', args)

    def test_48gb_override_and_explicit_empty_qos(self):
        result = self.preview('test', environment={'CLUSTER_PARTITION': 'student4848', 'CLUSTER_QOS': ''})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertIn('--gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb:1', args)
        self.assertFalse(any(arg.startswith('--qos=') for arg in args))

    def test_96gb_partition_and_explicit_qos(self):
        result = self.preview('test', environment={'CLUSTER_PARTITION': 'student9696',
                                                   'CLUSTER_QOS': 'account-approved-qos'})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertIn('--gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_4g.96gb:1', args)
        self.assertIn('--qos=account-approved-qos', args)

    def test_unknown_partition_requires_explicit_gres(self):
        result = self.preview('test', environment={'CLUSTER_PARTITION': 'researcher'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('CLUSTER_GPU_GRES', result.stderr)

    def test_legacy_submit_entry_uses_new_batch_environment(self):
        result = self.preview('student9696', '04:00:00', '32G', 'config.json', 'runs/study', '0', '1',
                              script=ROOT / 'moe/scripts/submit_source.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertIn('--time=04:00:00', args)
        self.assertIn('--mem=32G', args)
        self.assertIn(str(LAUNCHER), args)
        self.assertEqual(args[-6:], ['--batch', 'source', 'config.json', 'runs/study', '0', '1'])

    def test_batch_initializes_modules_and_preserves_assigned_mig(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'shell-init').write_text('module() { printf "%s\\n" "$*" >> "$CAPTURE_MODULES"; }\n')
            (root / 'activate').write_text('printf "%s\\n" "$*" >> "$CAPTURE_ACTIVATE"\n')
            fake_python = root / 'python'
            fake_python.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$CAPTURE_PYTHON"\nprintf "%s\\n" "${CUDA_VISIBLE_DEVICES-unset}" >> "$CAPTURE_VISIBLE"\n')
            fake_python.chmod(0o755)
            env = clean_environment()
            env.update(BASH_ENV=str(root / 'shell-init'), PATH=f'{root}:{env["PATH"]}',
                       SLURM_JOB_ID='123', MOE_CLUSTER_PROJECT_ROOT=str(ROOT),
                       CUDA_VISIBLE_DEVICES='MIG-assigned-device',
                       MOE_PYTHON=str(fake_python), CAPTURE_MODULES=str(root / 'modules'),
                       CAPTURE_ACTIVATE=str(root / 'activated'), CAPTURE_PYTHON=str(root / 'arguments'),
                       CAPTURE_VISIBLE=str(root / 'visible'))
            result = subprocess.run(['bash', str(LAUNCHER), '--batch', 'test'], env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'modules').read_text().splitlines(), ['load miniconda3', 'load cuda12.8'])
            self.assertEqual((root / 'activated').read_text().strip(), 'DAPOenv')
            self.assertEqual(set((root / 'visible').read_text().splitlines()), {'MIG-assigned-device'})
            self.assertIn(b'moe/tests', (root / 'arguments').read_bytes().split(b'\0'))

    def test_direct_batch_execution_without_allocation_is_rejected(self):
        result = subprocess.run(['bash', str(LAUNCHER), '--batch', 'test'], env=clean_environment(),
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SLURM allocation', result.stderr)

    def test_source_matrix_rejects_duplicate_seeds_before_running_tests(self):
        env = clean_environment()
        env['MOE_PYTHON'] = '/must-not-run-python'
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(['bash', str(ROOT / 'moe/scripts/run_experiments.sh'),
                'config.json', str(Path(tmp) / 'runs'), '0', '0'], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('distinct nonnegative integers', result.stderr)

    def test_source_matrix_forwards_retained_variant_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_python = root / 'python'
            fake_python.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$CAPTURE_PYTHON"\n')
            fake_python.chmod(0o755)
            env = clean_environment()
            env.update(MOE_PYTHON=str(fake_python), MOE_VARIANTS='B3 P1', MOE_DEVICE='cpu',
                       CAPTURE_PYTHON=str(root / 'args'))
            result = subprocess.run(['bash', str(ROOT / 'moe/scripts/run_experiments.sh'),
                'config.json', str(root / 'runs'), '0'], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = (root / 'args').read_bytes().split(b'\0')
            index = arguments.index(b'--variants')
            self.assertEqual(arguments[index + 1:index + 3], [b'B3', b'P1'])


if __name__ == '__main__':
    unittest.main()
