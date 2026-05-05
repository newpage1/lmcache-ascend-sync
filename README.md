# lmcache-ascend-sync

Automated upstream LMCache release monitor and adaptation PR generator for [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend).

## What It Does

1. **Monitors** upstream [LMCache](https://github.com/LMCache/LMCache) for new releases (daily via GitHub Actions)
2. **Detects** breaking changes that affect LMCache-Ascend's monkey-patch points using:
   - Precise diff analysis on watched modules/classes/methods
   - Rebase simulation to catch integration conflicts
3. **Generates** adaptation code via LLM API (GLM-5.1 via Zhipu Anthropic-compatible API)
4. **Creates** pull requests on LMCache-Ascend via fork-based workflow

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Release Check   │────>│  Conflict Detect  │────>│  PR Generation   │
│  (GitHub API)    │     │  (diff + rebase)  │     │  (LLM API)       │
└─────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
    state.json          patch_points.yaml          gh pr create
```

### Monkey-Patch Points Monitored

LMCache-Ascend monkey-patches these upstream interfaces:

| Interface | Module | Type |
|-----------|--------|------|
| CUDA kernels | `lmcache/c_ops` | Module replacement |
| GPU Connectors | `gpu_connectors.py`, `gpu_connector/utils.py` | Class inheritance |
| NPU Connectors | `npu_connector/` | Subclass + factory replacement |
| Memory Allocator | `memory_management.py` | Class inheritance |
| vLLM Adapter | `vllm_v1_adapter.py` | Class replacement |
| Cache Engine | `cache_engine.py` | Class inheritance (AscendLMCacheEngine) |
| CacheBlend | `compute/blend/` | Class inheritance |
| Token Hashing | `token_database.py` | Function replacement |
| NUMA Detection | `system_detection.py` | Function replacement |
| RPC Utils | `rpc_utils.py` | Function replacement |
| Lookup Client | `lookup_client/` | Method monkey-patch |
| Storage Backend | `storage_backend/` | Type/signature updates |
| Proxy Memory Obj | `proxy_memory_obj.py` | Abstract method impl |
| Config | `config.py` | Config definition injection |
| vLLM Connector | vLLM `lmcache_connector.py` | Method delegation |

See [config/patch_points.yaml](config/patch_points.yaml) for the full list.

## Setup

### GitHub Actions (Recommended)

1. Fork this repo or use it directly
2. Add repository secrets:
   - `LLM_API_KEY` - Your LLM API key (Zhipu GLM)
   - `GITHUB_TOKEN` - GitHub token with repo/PR write access
3. The workflow runs daily at 6:00 UTC
4. State persists between runs via GitHub Actions artifacts
5. You can also trigger manually via "Run workflow" with an optional target version

### Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export LLM_API_KEY=your_zhipu_api_key
export GITHUB_TOKEN=your_github_token

# Check for new releases
python -m src.sync_monitor --check-only

# Run full sync (auto-detects current Ascend version)
python -m src.sync_monitor

# Analyze a specific version
python -m src.sync_monitor --analyze v0.4.4
```

## Configuration

- `config/settings.yaml` - Main configuration (repos, schedule, LLM model, etc.)
- `config/patch_points.yaml` - Defines which upstream interfaces to monitor

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `upstream.repo` | Upstream LMCache repo | `LMCache/LMCache` |
| `downstream.repo` | Target LMCache-Ascend repo | `LMCache/LMCache-Ascend` |
| `downstream.fork` | Fork for pushing branches | `newpage1/LMCache-Ascend` |
| `llm.model` | LLM model for code generation | `GLM-5.1` |
| `llm.base_url` | LLM API endpoint | Zhipu Anthropic-compatible API |

### Version Detection

The tool auto-detects the current upstream version tracked by LMCache-Ascend by reading `LMCACHE_UPSTREAM_TAG` from `lmcache_ascend/__init__.py`. It only generates diffs between the currently tracked version and the new release, ensuring incremental adaptation.

## Requirements

- Python 3.9+
- `anthropic` >= 0.40.0
- `openai` >= 1.0.0
- `pyyaml` >= 6.0
- `requests` >= 2.28.0
- GitHub CLI (`gh`) for PR creation
- Git for repo operations
