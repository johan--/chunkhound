# ChunkHound LLM Context

## PROJECT_IDENTITY
ChunkHound: Semantic and regex search tool for codebases with MCP integration
Built: 100% by AI agents - NO human-written code
Purpose: Transform codebases into searchable knowledge bases for AI assistants

## MODIFICATION_RULES
**NEVER:**
- NEVER Use print() in MCP server (stdio.py, http_server.py, tools.py)
- NEVER Make single-row DB inserts in loops
- NEVER Use forward references (quotes) in type annotations unless needed

**ALWAYS:**
- ALWAYS Run smoke tests before committing: `uv run pytest tests/test_smoke.py -v -n auto`
- ALWAYS Run full test suite before pushing to a PR: `uv run pytest tests/ -v`
- ALWAYS Batch embeddings (min: 100, max: provider_limit)
- ALWAYS Use uv for all Python operations
- ALWAYS Update version via: `uv run scripts/update_version.py`

## KEY_COMMANDS
```bash
# Development
lint:      uv run ruff check chunkhound
typecheck: uv run mypy chunkhound
test:      uv run pytest
smoke:     uv run pytest tests/test_smoke.py -v -n auto  # MANDATORY before commits
full:      uv run pytest tests/ -v                     # MANDATORY before pushing to a PR
format:    uv run ruff format chunkhound

# Running
index:     uv run chunkhound index [directory]
mcp_stdio: uv run chunkhound mcp
mcp_http:  uv run chunkhound mcp http --port 5173
```

## VERSION_MANAGEMENT
Dynamic versioning via hatch-vcs - version derived from git tags.

```bash
# Create release
uv run scripts/update_version.py 4.1.0

# Create pre-release
uv run scripts/update_version.py 4.1.0b1
uv run scripts/update_version.py 4.1.0rc1

# Bump version
uv run scripts/update_version.py --bump minor      # v4.0.1 → v4.1.0
uv run scripts/update_version.py --bump minor b1   # v4.0.1 → v4.1.0b1
```

NEVER manually edit version strings - ALWAYS create git tags instead.

## PUBLISHING_PROCESS
Releases are now fully automated via GitHub Actions (OIDC Trusted Publishing).
See **RELEASING.md** for the authoritative step-by-step guide.

Quick summary:
1. Tag the version: `uv run scripts/update_version.py X.Y.Z`
2. Run smoke tests: `uv run pytest tests/test_smoke.py -v -n auto` (MANDATORY)
3. Create and publish a GitHub Release — `release.yml` handles the PyPI upload automatically.

Pre-releases (alpha/beta/RC) publish to **PyPI** (not TestPyPI) via `release-rc.yml` on tag push.
Do NOT use `uv publish` or `prepare_release.sh` manually — CI owns the publish step.

## TEST RELEASE (alpha to PyPI)

**If version not specified:** fetch latest version from PyPI, increment minor, append `a1`:
```bash
LATEST=$(pip index versions chunkhound 2>/dev/null | grep -oP '[\d.]+' | head -1)
# e.g. 4.0.3 → next minor = 4.1.0 → alpha = 4.1.0a1
```

**If version specified by user** (e.g. `4.2.0`): append `a1` → `4.2.0a1`

**Steps:**
```bash
# 1. Save current remote and switch to chunkhound org remote
ORIGINAL_REMOTE=$(git remote get-url origin)
git remote set-url origin https://github.com/chunkhound/chunkhound.git

# 2. Create the alpha tag
uv run scripts/update_version.py X.Y.Za1

# 3. Push the tag — triggers release-rc.yml → publishes to PyPI as pre-release
git push origin vX.Y.Za1

# 4. Revert remote back to original
git remote set-url origin "$ORIGINAL_REMOTE"
```

PyPI trusted publisher required for `release-rc.yml`:
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release-rc.yml`
- Environment: `pypi`

## DB_PATH_GOTCHAS
- **Preferred: pass project directory as positional arg** — `chunkhound search "query" /path/to/project` reads `.chunkhound.json` and resolves the DB correctly.
- **For MCP:** pass the project path positionally too (`chunkhound mcp /path/to/project`) so cwd doesn't matter. Add `--db <db-dir>` only when overriding the configured location.
- **`--db` with the wrong subpath silently returns 0 results** — no error, just empty. Always verify with a regex search first.
- **Two valid v5 DB layouts, driven by config:**
  - With `database.path` set in `.chunkhound.json` → chunkhound writes `<path>/chunks.db` (flat at that directory) + `<path>/chunks.db.root.json` sidecar.
  - Without `database.path` → defaults to `<project>/.chunkhound/db/chunks.db` (nested) + sidecar at the same place.
- **Indexed-root pinning:** the sidecar `chunks.db.root.json` records the project root at first write. Re-opening under a different root fails with `DuckDBIndexedRootMismatchError`. Pass the project root explicitly (positional arg) or run with cwd = project root.
- **`--db` always points at the DB *directory*, never the file** — passing `.../chunks.db` creates a nested `chunks.db/chunks.db`.
- **Config layering is `_deep_merge`:** `--config`/`CHUNKHOUND_CONFIG_FILE` is the base, project-local `.chunkhound.json` deep-merges on top per-field, env vars on top of that, CLI args win. A tiny per-project file (only `database.path` + project excludes) can inherit providers/models from a global defaults file — they merge, they don't replace.
- **Relative `path: ".chunkhound"` in `.chunkhound.json` resolves against CWD, not the project dir** — use absolute paths.
- **Pre-v4 flat-*file* `.chunkhound`** (where `.chunkhound` itself is a file, not a directory) blocks directory creation. Move aside before re-indexing.
- **Bloat baseline:** v5 indexes are ~250–300× source size (768-dim embeddings + HNSW + chunk-per-symbol granularity). 1 MB source → ~300 MB DB; plan disk accordingly.

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- Run `uv run mypy chunkhound` during reviews to catch Optional/type boundary issues
- All code patterns should be self-documenting

## TESTING_PHILOSOPHY
- Test external constraints, critical invariants, and user-facing contracts.
- Do NOT write tests for adapters, private helpers, mock behavior, or internal plumbing unless the test is the narrowest way to protect a real external contract.
- If a refactor could change the implementation without changing user-visible behavior, the test is probably too internal and should not exist.
- Prefer contract names like `test_cli_overrides_env` over implementation names like `test_extract_cli_overrides_calls_helper`.
- Use real business logic with fakes only at true external boundaries (network, filesystem, subprocess, third-party APIs).
- For provider integrations, test our contract with the provider: supported/unsupported feature gating, request validity constraints, explicit failures, and stable user-visible semantics. Do NOT test SDK mechanics or mirror every internal request-shaping helper.
- Before adding a test, ask: "Would a user, caller, CI contract, or external system notice if this broke?" If not, do not add the test.
- Prefer one higher-value contract test over many narrow implementation tests.
