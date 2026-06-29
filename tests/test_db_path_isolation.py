"""Contract: ChunkHound must never write its database inside a code repo's
working tree, and must not auto-claim arbitrary git repos as projects.

These are local fork behaviors gated behind two env vars. The tests exist to
fail loudly if an upstream merge ever silently reverts them. Both behaviors
default to upstream when the env vars are unset, so the "unset" tests double as
documentation of the stock behavior.
"""

from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.utils.project_detection import find_project_root


# --- detection: don't auto-claim random git repos -------------------------


def test_require_config_ignores_bare_git_repo(tmp_path, monkeypatch):
    """With CHUNKHOUND_REQUIRE_PROJECT_CONFIG set, a git repo without a
    .chunkhound.json is NOT treated as a project (no surprise indexing)."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("CHUNKHOUND_REQUIRE_PROJECT_CONFIG", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path.parent)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        find_project_root()


def test_require_config_still_honors_explicit_marker(tmp_path, monkeypatch):
    """An explicit .chunkhound.json always opts a directory in, even under
    require-config mode."""
    (tmp_path / ".chunkhound.json").write_text("{}")
    monkeypatch.setenv("CHUNKHOUND_REQUIRE_PROJECT_CONFIG", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path.parent)
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == Path.cwd()


def test_git_repo_claimed_when_require_config_unset(tmp_path, monkeypatch):
    """Upstream behavior preserved: a git repo is claimed when the opt-in env
    var is absent."""
    (tmp_path / ".git").mkdir()
    monkeypatch.delenv("CHUNKHOUND_REQUIRE_PROJECT_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path.parent)
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == Path.cwd()


# --- default DB path: keep it out of the working tree ---------------------


def test_db_root_keeps_database_out_of_tree(tmp_path, monkeypatch):
    """With CHUNKHOUND_DB_ROOT set, a project with no explicit database.path
    resolves its DB under the central root, never inside the repo."""
    project = tmp_path / "myproj"
    project.mkdir()
    (project / ".chunkhound.json").write_text("{}")
    db_root = tmp_path / "central"

    monkeypatch.setenv("CHUNKHOUND_DB_ROOT", str(db_root))
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)
    # Isolate from any real ~/.config/chunkhound global config.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cfg = Config(target_dir=project)

    assert cfg.database.path == (db_root.resolve() / "myproj")
    assert project not in cfg.database.path.parents


def test_default_db_in_tree_when_db_root_unset(tmp_path, monkeypatch):
    """Upstream behavior preserved: without CHUNKHOUND_DB_ROOT the DB defaults
    to the in-tree .chunkhound/db location."""
    project = tmp_path / "myproj"
    project.mkdir()
    (project / ".chunkhound.json").write_text("{}")

    monkeypatch.delenv("CHUNKHOUND_DB_ROOT", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cfg = Config(target_dir=project)

    assert cfg.database.path == (project / ".chunkhound" / "db").resolve()


# --- MCP server must honor the opt-in gate too ----------------------------
#
# Regression guard: the `chunkhound mcp` positional path used to default to
# Path("."), and that non-None value bypassed the require-config walk-up (a
# non-None start_path is used verbatim by find_project_root). The MCP server
# therefore auto-claimed and indexed any session's cwd, opted in or not. The
# fix is to default the positional to None so the gated detection runs, and to
# resolve an absent path from the already-gated config.target_dir.


def test_mcp_command_without_path_honors_require_config(tmp_path, monkeypatch):
    """`chunkhound mcp` with no positional path in a bare git repo must NOT
    claim cwd when require-config is set - same gate the CLI walk-up uses."""
    from chunkhound.api.cli.main import create_parser

    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("CHUNKHOUND_REQUIRE_PROJECT_CONFIG", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path.parent)
    monkeypatch.chdir(tmp_path)

    args = create_parser().parse_args(["mcp"])
    # Mirrors Config's project-root resolution: the MCP command must feed
    # None (not a cwd default) so the gated walk-up runs and refuses.
    with pytest.raises(SystemExit):
        find_project_root(getattr(args, "path", None))


def test_mcp_resolves_to_gated_target_dir_when_no_path(tmp_path, monkeypatch):
    """With no positional path, the MCP command resolves the daemon's project
    dir from the config's already-gated target_dir, never a raw cwd default."""
    from types import SimpleNamespace

    from chunkhound.api.cli.commands.mcp import _resolve_project_dir

    project = tmp_path / "myproj"
    project.mkdir()
    (project / ".chunkhound.json").write_text("{}")
    monkeypatch.delenv("CHUNKHOUND_DB_ROOT", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cfg = Config(target_dir=project)
    args = SimpleNamespace(path=None)

    assert _resolve_project_dir(args, cfg) == project.resolve()


def test_mcp_explicit_path_overrides_config(tmp_path, monkeypatch):
    """An explicit positional path always wins (user intent), even under
    require-config - the bypass for deliberate paths stays intact."""
    from types import SimpleNamespace

    from chunkhound.api.cli.commands.mcp import _resolve_project_dir

    project = tmp_path / "myproj"
    project.mkdir()
    (project / ".chunkhound.json").write_text("{}")
    monkeypatch.delenv("CHUNKHOUND_DB_ROOT", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cfg = Config(target_dir=project)
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    args = SimpleNamespace(path=explicit)

    assert _resolve_project_dir(args, cfg) == explicit.resolve()
