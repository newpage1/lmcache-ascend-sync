# LMCache Sync Skill Design

## Overview

A native Claude Code skill (`lmcache-sync`) that monitors upstream LMCache releases and auto-generates adaptation PRs for LMCache-Ascend. Uses Claude Code's own capabilities for code understanding and adaptation, with Python helper scripts for mechanical operations (AST parsing, regex transforms, verification).

## Invocation

- **Slash command**: `/sync-lmcache`
- **Natural language**: triggered when user mentions "检查上游", "sync upstream", "新版本适配", "upstream release"
- **Scheduled**: daily via CronCreate (6:23 UTC)

## Configuration (embedded in SKILL.md)

| Key | Value |
|-----|-------|
| upstream_repo | `LMCache/LMCache` |
| downstream_repo | `LMCache/LMCache-Ascend` |
| fork_repo | `newpage1/LMCache-Ascend` |
| target_branch | `main` |
| min_version | `v0.4.3` |
| version_source | `lmcache_ascend/__init__.py` field `LMCACHE_UPSTREAM_TAG` |

## Pipeline: 6 Phases

### Phase 1: Version Detection

1. Read `lmcache_ascend/__init__.py` from downstream main branch via GitHub API (`gh api repos/{repo}/contents/lmcache_ascend/__init__.py?ref=main`)
2. Extract `LMCACHE_UPSTREAM_TAG` value via regex
3. Fetch latest upstream release: `gh api repos/LMCache/LMCache/releases/latest`
4. Compare versions. If already latest, log and exit.

### Phase 2: Change Analysis

1. Clone upstream to `/tmp/lmcache-sync/upstream` (if not exists, pull + fetch tags)
2. Run `git diff {from_version}..{to_version} -- {watched_files}` to get diffs
3. Call `python scripts/classify_changes.py --diff <diff_file> --patch-points config/patch_points.yaml`
4. Output: JSON with classified hunks, each tagged with ChangeType and corresponding Ascend files

**ChangeType categories**:
- `MECHANICAL_RENAME` / `IMPORT_CHANGE` / `CONFIG_ADDITION` → Phase 3
- `LOGIC_CHANGE` / `PARAM_CHANGE` / `NEW_METHOD` / `NEW_CLASS` → Phase 4
- `IRRELEVANT` → skip

### Phase 3: Deterministic Transforms

Call `python scripts/apply_transforms.py --source-dir /tmp/lmcache-sync/downstream/lmcache_ascend --rules config/transform_rules.yaml --classifications classifications.json`

What it does:
- String replacements (e.g., `is_tuple_format` → `is_separate_format`)
- Import unguarding (remove `if torch.npu.is_available()` wrappers)
- Config field renames and additions in `_CONFIG_DEFINITIONS`
- Word-boundary identifier renames

Rules are defined in `config/transform_rules.yaml` with 6 rule types: `replace`, `rename`, `import_unguard`, `config_rename`, `config_add`, `conditional_replace`.

### Phase 4: Claude Adaptation

For each `LOGIC_CHANGE` / `PARAM_CHANGE` / `NEW_METHOD` / `NEW_CLASS`:

1. Read upstream code at `{from_version}`: `git -C /tmp/lmcache-sync/upstream show {from_version}:{file_path}`
2. Read upstream code at `{to_version}`: `git -C /tmp/lmcache-sync/upstream show {to_version}:{file_path}`
3. Read current Ascend version of the file
4. Claude generates adapted code that:
   - Applies the equivalent upstream change
   - Preserves ALL Ascend-specific logic (torch.npu, lmc_ops, is_310p, NUMA, etc.)
   - Maintains exact indentation and style
5. Apply changes via Edit tool

**Prompt template** (internal to skill instructions):

```
Compare the upstream change:
- Before ({from_version}): {upstream_before}
- After ({to_version}): {upstream_after}

Current Ascend version:
{ascend_current}

Apply the equivalent change to the Ascend version while preserving all Ascend-specific logic.
```

### Phase 5: Verification (Tiered)

**Tier 1 — Local static analysis (always runs)**

Call `python scripts/verify_changes.py --original <original_dir> --modified <modified_dir>`

- **P0 Syntax**: `ast.parse` passes on every modified file
- **P0 Line count**: no file loses >30% lines (catches mass deletion)
- **P0 Class preservation**: all public classes still exist via AST walk
- **P0 Ascend keywords**: `torch.npu`, `lmc_ops`, `is_310p`, `NUMA` still present
- **P1 Method preservation**: all public methods of changed classes still exist
- **P1 Import structure**: no unexpected removal of critical imports (`lmcache_ascend`, `torch_npu`, `lmc_ops`)
- **P2 Docstring preservation**: class/module docstrings not accidentally deleted

**Tier 2 — Import & compilation check (requires Python + torch_npu env)**

Run in the Ascend dev environment:
```bash
cd /tmp/lmcache-sync/downstream
python -c "import ast; ast.parse(open('lmcache_ascend/__init__.py').read())"  # per-file
python -c "import lmcache_ascend"  # full package import
python -c "from lmcache_ascend.v1.npu_connector.npu_connectors import *"  # critical modules
```

**Tier 3 — Unit tests (requires full Ascend NPU environment)**

Run in the Ascend dev environment:
```bash
cd /tmp/lmcache-sync/downstream
pytest tests/v1/test_config.py -v            # config smoke test
pytest tests/v1/test_npu_connector.py -v      # connector test
pytest tests/v1/test_cache_engine.py -v       # cache engine test
pytest tests/test_version_integrity.py -v      # version integrity
```

**Tier 4 — Full build (optional, for kernel changes)**

Only if `csrc/` or kernel-related files changed:
```bash
pip install -e .  # triggers CMake build of AscendC kernels
```

**Flow**: Tier 1 always runs. Tier 2/3 run if Ascend environment is available (detect via `python -c "import torch_npu"`). If environment unavailable, skip Tier 2/3 but flag in PR description that verification is incomplete and needs manual check on Ascend environment.

### Phase 6: PR Creation

1. Clone fork: `git clone https://github.com/newpage1/LMCache-Ascend /tmp/lmcache-sync/fork`
2. Create branch: `git checkout -b sync/upstream-{version}`
3. Copy modified files into fork checkout
4. Update `LMCACHE_UPSTREAM_TAG` in `__init__.py` to new version
5. Commit: `git commit -m "feat: adapt to upstream LMCache {version}"`
6. Push: `git push origin sync/upstream-{version}`
7. Create PR: `gh pr create --repo LMCache/LMCache-Ascend --title "feat: adapt to upstream {version}" --body "{summary}"`

## File Structure

```
lmcache-ascend-sync/
├── skills/
│   └── lmcache-sync/
│       └── SKILL.md              # Main skill file
├── scripts/
│   ├── classify_changes.py       # Change classification (CLI entry point)
│   ├── apply_transforms.py       # Deterministic transforms (CLI entry point)
│   └── verify_changes.py         # Verification checks (CLI entry point)
├── config/
│   ├── patch_points.yaml         # 46 monkey-patch monitoring points
│   └── transform_rules.yaml      # Deterministic transform rules
└── (original src/ kept but no longer primary)
```

## Python Helper Scripts

### classify_changes.py

- **Input**: diff text file + patch_points.yaml path
- **Output**: JSON to stdout
- **No dependencies** beyond stdlib + pyyaml
- **Logic**: parse unified diff → classify each hunk by heuristic rules → map to Ascend files

### apply_transforms.py

- **Input**: source directory + rules YAML + classifications JSON
- **Output**: modified files in-place + summary JSON to stdout
- **No dependencies** beyond stdlib + pyyaml
- **Logic**: iterate rules → apply to matching files → output changes

### verify_changes.py

- **Input**: original directory + modified directory
- **Output**: JSON verification report to stdout
- **No dependencies** beyond stdlib
- **Logic**:
  - Tier 1 (always): AST syntax check, line count delta, class/method preservation, Ascend keyword check, import structure check
  - Returns structured report with per-file pass/fail and overall verdict

## Scheduled Check

After skill installation, set up daily monitoring:

```
CronCreate:
  cron: "23 6 * * *"
  prompt: "Check if LMCache upstream has a new release. If yes, run the full /sync-lmcache pipeline to generate an adaptation PR."
  recurring: true
  durable: true
```

## Error Handling

| Scenario | Action |
|----------|--------|
| No new release | Log and exit |
| Clone failure | Retry once, then fail with clear error |
| Classify failure | Fall back to treating all changes as LOGIC_CHANGE |
| Transform conflict | Log warning, skip conflicting rule |
| Verification failure | Output report, Claude reviews and fixes |
| PR creation failure | Save changes locally, report error to user |

## What Gets Removed

- `state.json` — version read from repo
- `method_patcher.py` — Claude replaces LLM API calls
- `pr_generator.py` — skill handles orchestration
- `sync_monitor.py` — skill handles monitoring
- LLM API key configuration — no external API needed
- `anthropic` / `openai` dependencies — no external LLM calls
