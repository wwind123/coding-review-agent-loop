"""Isolated Git child for the stdlib-only inspect entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# -I -S keeps the checkout and user site off sys.path. The verified caller
# passes its own package root because this source runs via Python -c.
sys.path.insert(0, sys.argv[1])
from coding_review_agent_loop.secure_git import local_command  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        return 125
    command, env, pass_fds = local_command(sys.argv[3:], environ=os.environ, checkout=Path.cwd())
    result = subprocess.run(command, env=env, pass_fds=pass_fds, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
