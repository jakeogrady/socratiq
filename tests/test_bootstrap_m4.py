from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPOSITORY_ROOT / "scripts/bootstrap_m4.sh"


class BootstrapScriptTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        completed = subprocess.run(  # noqa: S603 - repository-owned fixed command
            ["/bin/sh", "-n", str(BOOTSTRAP)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_check_only_validates_pinned_uv_without_writing(self) -> None:
        uv_binary = os.environ.get("SOCRATIQ_UV_BIN") or shutil.which("uv")
        if uv_binary is None:
            sibling_uv = Path(sys.executable).with_name("uv")
            uv_binary = str(sibling_uv) if sibling_uv.is_file() else None
        self.assertIsNotNone(uv_binary, "the bootstrap test requires pinned uv")
        environment = os.environ.copy()
        environment.update(
            {
                "SOCRATIQ_BOOTSTRAP_ALLOW_NON_M4": "1",
                "SOCRATIQ_BOOTSTRAP_ALLOW_DIRTY": "1",
                "SOCRATIQ_UV_BIN": str(uv_binary),
            }
        )
        completed = subprocess.run(  # noqa: S603 - repository-owned fixed command
            [str(BOOTSTRAP), "--check-only"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("uv: uv 0.9.18", completed.stdout)
        self.assertIn("no environment changes were made", completed.stdout)
