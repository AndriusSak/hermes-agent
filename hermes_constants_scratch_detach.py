"""The hourly scratch prune runs in a detached child process, not in the process that is due it.

A pass can take seconds to minutes (the idle scan of a large tree, the ``rmtree`` of a multi-GB
entry). Inline, it held every process start; on a daemon thread it died with every short-lived
``hermes`` command, the hourly stamp already spent. A detached child outlives the command that
launched it, so each due pass runs to the end, with the same rules as an inline one.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ``-I``: neither the launcher's cwd (a project ``utils/`` would shadow ours) nor PYTHON* env vars
# decide what the child imports; it gets the launcher's own ``sys.path`` (repo root, PM/venv
# site-packages), not ``hermes_bootstrap``, whose boot hook would schedule another prune.
_CHILD = ("import sys; sys.path[:] = sys.argv[4:]; "
          "import hermes_constants_scratch_detach as d; d.run(*sys.argv[1:4])")


def spawn_prune(scratch: Path) -> None:
    """Start one detached prune of *scratch*, the path the caller resolved (a child never
    re-resolves the profile home). A failed spawn is logged and dropped: the stamp is already
    spent, so the next pass is the next process past the hourly mark."""
    argv = [sys.executable, "-I"]
    if sys.dont_write_bytecode:
        argv.append("-B")
    if sys.pycache_prefix:  # a sealed payload must never get __pycache__ writes
        argv += ["-X", f"pycache_prefix={sys.pycache_prefix}"]
    argv += ["-c", _CHILD, os.fspath(scratch), str(os.getpid()), str(os.getppid()), *sys.path]
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway, windows_detach_popen_kwargs

    stdio = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    try:
        try:
            subprocess.Popen(argv, close_fds=True, **stdio, **windows_detach_popen_kwargs())
        except PermissionError:
            if sys.platform != "win32":
                raise
            # A job object without BREAKAWAY_OK rejects CREATE_BREAKAWAY_FROM_JOB.
            subprocess.Popen(argv, close_fds=True, **stdio, creationflags=windows_detach_flags_without_breakaway())
    except (OSError, RuntimeError) as exc:
        logger.debug("scratch prune: could not start the detached pass for %s: %s", scratch, exc)


def run(scratch: str, launcher_pid: str, launcher_parent_pid: str) -> None:
    """Child entry. The launcher's process tree stays the pass's own (never reaped; an entry it
    works inside is kept) after the launcher exits and this child is reparented away from it."""
    import hermes_constants_scratch
    from hermes_constants import prune_scratch_dir

    hermes_constants_scratch.adopt_launcher_lineage((int(launcher_pid), int(launcher_parent_pid)))
    prune_scratch_dir(Path(scratch))
