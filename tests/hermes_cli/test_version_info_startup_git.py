"""A Hermes start from a git checkout must not walk git for the release it never uses.

Each case runs in a fresh interpreter (the suite's conftest pre-seeds the version cache) with a
``git`` shim on PATH that logs every invocation before running the real git.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_logging_git(tmp_path: Path, code: str) -> tuple[dict, list[str]]:
    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("git not installed")
    home = tmp_path / "home"
    # Version resolution falls back to <HERMES_HOME>/hermes-agent when the executing tree has no
    # .git (sdist/archive test runs); give it a tagged checkout so the release walk always has one.
    repo = home / "hermes-agent"
    repo.mkdir(parents=True)
    git = [real_git, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "r"], check=True)
    subprocess.run([*git, "tag", "v1.2.3"], check=True)

    log = tmp_path / "git.log"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "git"
    shim.write_text(f'#!/bin/sh\necho "$*" >> "{log}"\nexec "{real_git}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(ROOT),
        "HERMES_HOME": str(home),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=120, check=False, stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    calls = log.read_text(encoding="utf-8-sig").splitlines() if log.exists() else []
    return json.loads(out.stdout.strip().splitlines()[-1]), calls


@pytest.mark.platforms("linux", "macos")
def test_provider_headers_run_no_git_until_read_and_cross_the_plugin_host(tmp_path):
    # A user override copying a bundled profile inherits its lazy headers; under
    # ``plugins.isolation: host`` they reach the parent only through the profile extraction.
    override = tmp_path / "gmi-eu"
    override.mkdir()
    (override / "__init__.py").write_text(
        "import dataclasses\n"
        "from providers import get_provider_profile, register_provider\n"
        "register_provider(dataclasses.replace(get_provider_profile('gmi'), name='gmi-eu', aliases=()))\n",
        encoding="utf-8",
    )
    code = (
        "import json, os, hermes_cli.config, providers\n"
        "providers.list_providers()\n"
        "log = os.environ['PATH'].split(os.pathsep)[0] + '/../git.log'\n"
        "before = open(log, encoding='utf-8-sig').read().splitlines() if os.path.exists(log) else []\n"
        "ua = dict(providers.get_provider_profile('fireworks').default_headers)['User-Agent']\n"
        "from hermes_cli.version_info import get_version_info\n"
        "from hermes_cli.plugin_host_profiles import load_hosted_profiles\n"
        f"hosted, = load_hosted_profiles({str(override)!r}, '_hermes_user_provider_gmi_eu')\n"
        "print(json.dumps({'before': before, 'ua': ua, 'base': get_version_info().base_version,\n"
        "                  'hosted': dict(hosted.default_headers)}))\n"
    )
    result, calls = _run_logging_git(tmp_path, code)

    assert result["before"] == [], "importing config + every provider plugin spawned git"
    assert result["ua"] == f"HermesAgent/{result['base']}"
    assert calls, "reading the header resolves the release on demand"
    assert result["hosted"] == {"User-Agent": f"HermesAgent/{result['base']}"}


@pytest.mark.platforms("linux", "macos")
def test_startup_checkout_sha_read_is_one_rev_parse(tmp_path):
    code = (
        "import json\n"
        "from hermes_cli.update_cmd_fleet import _current_checkout_sha\n"
        "sha = _current_checkout_sha()\n"
        "print(json.dumps({'sha': sha}))\n"
    )
    result, calls = _run_logging_git(tmp_path, code)

    assert calls == ["rev-parse HEAD"]
    assert result["sha"] and len(result["sha"]) == 40
