# lmcache-ascend-sync

Automated upstream LMCache release monitor and adaptation PR generator for [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend).

## What It Does

1. **Monitors** upstream [LMCache](https://github.com/LMCache/LMCache) for new releases (daily via GitHub Actions)
2. **Detects** breaking changes that affect LMCache-Ascend's monkey-patch points using:
   - Precise diff analysis on watched modules/classes/methods
   - Rebase simulation to catch integration conflicts
3. **Generates** adaptation code via Claude API
4. **Creates** pull requests on LMCache-Ascend automatically

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Release Check   │────>│  Conflict Detect  │────>│  PR Generation   │
│  (GitHub API)    │     │  (diff + rebase)  │     │  (Claude API)    │
└─────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
    state.json          patch_points.yaml          gh pr create
```

### Patch Points Monitored

LMCache-Ascend monkey-patches these upstream interfaces:

| Interface | Module | Type |
|-----------|--------|------|
| CUDA kernels | `lmcache/c_ops` | Module replacement |
| GPU Connectors | `gpu_connectors.py` | Class inheritance |
| Memory Allocator | `memory_management.py` | Class inheritance |
| vLLM Adapter | `vllm_v1_adapter.py` | Function replacement |
| CacheBlend | `compute/blend/` | Class inheritance |
| Token Hashing | `token_database.py` | Function replacement |
| NUMA Detection | `system_detection.py` | Function replacement |
| Cache Engine | `cache_engine.py` | Class inheritance |

See [config/patch_points.yaml](config/patch_points.yaml) for the full list.

## Setup

### GitHub Actions (Recommended)

1. Fork this repo or use it directly
2. Add repository secrets:
   - `ANTHROPIC_API_KEY` - Your Claude API key
3. The workflow runs daily at 6:00 UTC
4. You can also trigger manually via "Run workflow" with an optional target version

### Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Check for new releases
./run_local.sh --check-only

# Run full sync
export ANTHROPIC_API_KEY=sk-...
./run_local.sh

# Analyze a specific version
./run_local.sh --analyze v0.4.4
```

## Configuration

- `config/settings.yaml` - Main configuration (repos, schedule, Claude model, etc.)
- `config/patch_points.yaml` - Defines which upstream interfaces to monitor

## Requirements

- Python 3.11+
- `anthropic` >= 0.40.0
- `pyyaml` >= 6.0
- `requests` >= 2.28.0
- GitHub CLI (`gh`) for PR creation
- Git for repo operations
