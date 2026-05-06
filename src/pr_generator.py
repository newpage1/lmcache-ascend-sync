"""
PR Generator - Uses LLM API to generate adaptation code and creates PRs.
Supports Anthropic-compatible APIs (Zhipu GLM) and OpenAI-compatible APIs.
"""

import json
import logging
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger("lmcache-sync.pr_generator")


class PRGenerator:
    """Generate adaptation PRs using LLM API (Anthropic-compatible or OpenAI-compatible)."""

    def __init__(self, config: dict, patch_points: dict):
        self.config = config
        self.patch_points = patch_points

        llm_config = config.get("llm", {})
        self.api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        )
        self.api_format = llm_config.get("api_format", "anthropic")
        self.base_url = llm_config.get("base_url", "https://open.bigmodel.cn/api/anthropic")
        self.model = llm_config.get("model", "GLM-5.1")
        self.max_tokens = llm_config.get("max_tokens", 8192)

    def generate(
        self,
        from_version: str,
        to_version: str,
        analysis: dict,
        rebase_result: dict,
    ) -> dict:
        """Generate adaptation code and create a PR.

        Returns dict with:
        - success: bool
        - pr_url: str (if success)
        - pr_number: int (if success)
        - error: str (if failed)
        """
        if not self.api_key:
            return {"success": False, "error": "LLM_API_KEY or ANTHROPIC_AUTH_TOKEN not set"}

        # Read downstream source files for context
        downstream_path = Path(self.config["sync"]["downstream_checkout"])
        ascend_sources = self._read_ascend_sources(downstream_path)

        # Read the upstream diff
        upstream_diff = analysis.get("diff_summary", "")

        # Build the prompt
        prompt = self._build_prompt(
            from_version, to_version, analysis, rebase_result,
            ascend_sources, upstream_diff,
        )

        # Call LLM API
        logger.info(f"Calling LLM API ({self.model} via {self.base_url})...")
        try:
            if self.api_format == "anthropic":
                client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                generated = response.content[0].text
            else:
                from openai import OpenAI
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                generated = response.choices[0].message.content
        except Exception as e:
            return {"success": False, "error": f"LLM API error: {e}"}

        # Parse the generated code
        logger.info(f"LLM response length: {len(generated)} chars")
        logger.debug(f"LLM response preview: {generated[:500]}")
        file_changes = self._parse_generated_code(generated)
        if not file_changes:
            logger.error(f"Failed to parse generated code. Raw response:\n{generated[:2000]}")
            return {
                "success": False,
                "error": "Failed to parse generated code from LLM response",
            }

        # Apply changes and create PR
        return self._create_pr(to_version, file_changes, analysis, rebase_result)

    def _read_ascend_sources(self, downstream_path: Path) -> dict[str, str]:
        """Read ALL LMCache-Ascend source files for comprehensive context."""
        sources = {}
        ascend_dir = downstream_path / "lmcache_ascend"
        if not ascend_dir.exists():
            return sources

        # Recursively read all Python files under lmcache_ascend/
        for py_file in ascend_dir.rglob("*.py"):
            rel_path = py_file.relative_to(ascend_dir)
            try:
                content = py_file.read_text()
                # Skip very large files (>50KB) but include their headers
                if len(content) > 50000:
                    lines = content.split("\n")
                    content = "\n".join(lines[:100]) + "\n# ... (truncated, %d lines total)\n" % len(lines)
                sources[str(rel_path)] = content
            except Exception:
                pass

        # Also read key project files
        extra_files = [
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
        ]
        for fname in extra_files:
            full_path = downstream_path / fname
            if full_path.exists():
                try:
                    sources[fname] = full_path.read_text()
                except Exception:
                    pass

        # Read test files
        tests_dir = downstream_path / "tests"
        if tests_dir.exists():
            for py_file in tests_dir.rglob("*.py"):
                rel_path = py_file.relative_to(downstream_path)
                try:
                    content = py_file.read_text()
                    if len(content) > 10000:
                        lines = content.split("\n")
                        content = "\n".join(lines[:50]) + "\n# ... (truncated)\n"
                    sources[str(rel_path)] = content
                except Exception:
                    pass

        return sources

    def _build_prompt(
        self,
        from_version: str,
        to_version: str,
        analysis: dict,
        rebase_result: dict,
        ascend_sources: dict[str, str],
        upstream_diff: str,
    ) -> str:
        """Build the LLM prompt for code generation."""
        conflicts_text = json.dumps(analysis.get("conflicts", []), indent=2)
        rebase_conflicts = json.dumps(
            rebase_result.get("conflict_files", []), indent=2
        )

        # Build per-file action list from conflicts
        file_action_list = self._build_file_action_list(
            analysis.get("conflicts", []), ascend_sources
        )

        sources_section = ""
        for path, content in ascend_sources.items():
            sources_section += f"\n### {path}\n```python\n{content}\n```\n"

        return textwrap.dedent(f"""\
            You are an expert Python developer maintaining LMCache-Ascend, a plugin that adapts
            the upstream LMCache project to run on Huawei Ascend NPUs.

            ## How LMCache-Ascend Works

            LMCache-Ascend does NOT fork the upstream repo. Instead, it works by:

            1. **sys.modules replacement**: Replacing `lmcache.c_ops` (CUDA kernels) with `lmcache_ascend.c_ops` (CANN kernels) at import time
            2. **Factory patching**: Patching `CreateGPUConnector` to return NPU connectors instead of GPU connectors
            3. **Class replacement**: Replacing `LMCacheConnectorV1Impl` with `LMCacheAscendConnectorV1Impl` and `LMCacheEngine` with `AscendLMCacheEngine`
            4. **Method monkey-patching**: Overriding specific methods like `wait_for_save`, `get_finished`, `handle_preemptions`
            5. **Config injection**: Adding Ascend-specific config definitions to `lmcache.v1.config._CONFIG_DEFINITIONS`

            All patches are applied in `lmcache_ascend/__init__.py` when the package is imported.

            Key Ascend-specific patterns:
            - `torch.npu` instead of `torch.cuda`
            - `torch.npu.Event` / `torch.npu.stream` instead of CUDA equivalents
            - CANN kernel calls via `lmcache_ascend.c_ops` (AscendC kernels)
            - NUMA detection for Ascend hardware topology
            - Short hash-based socket paths (Ascend path length limits)

            ## Task
            The upstream LMCache has released version {to_version} (Ascend currently tracks {from_version}).
            Analyze the upstream changes and generate updated LMCache-Ascend code.

            ## Detected Conflicts (from diff analysis)
            ```json
            {conflicts_text}
            ```

            ## Per-File Action Plan
            Based on the conflicts detected, here are the files that need changes:

            {file_action_list}

            ## Rebase Simulation Conflicts
            ```json
            {rebase_conflicts}
            ```

            ## Upstream Diff ({from_version} -> {to_version})
            ```diff
            {upstream_diff}
            ```

            ## Current LMCache-Ascend Source Files
            {sources_section}

            ## Instructions
            1. You MUST output EVERY file listed in the "Per-File Action Plan" above. Do NOT skip any.
            2. For each file:
               a. If an import path changed upstream, update the Ascend import to match
               b. If a method signature changed, update the Ascend override to match
               c. If a new abstract method was added upstream, add a compatible Ascend implementation
               d. If a class was renamed upstream, update the Ascend patch to reference the new name
            3. Preserve ALL Ascend-specific logic (NPU device, CANN kernels, NUMA, etc.)
            4. Preserve the `store_async` feature (AscendLMCacheEngine background thread)
            5. Update `LMCACHE_UPSTREAM_TAG` in `__init__.py` to `{to_version}`
            6. Do NOT remove any Ascend-only features or workarounds
            7. Output the COMPLETE file content for each file — do not use "..." or truncation

            ## Output Format
            For each file that needs changes, output EXACTLY:

            <<<FILE: lmcache_ascend/relative/path/to/file.py>>>
            # complete file content here
            <<<END>>>

            IMPORTANT:
            - File paths MUST start with `lmcache_ascend/` (not just the filename).
            - Output EVERY file from the action plan, not just one.
            - Output complete files — do not truncate or abbreviate.
        """)

    def _build_file_action_list(
        self, conflicts: list[dict], ascend_sources: dict[str, str]
    ) -> str:
        """Build a per-file action list from detected conflicts."""
        # Map conflict modules to Ascend files
        file_actions = {}
        for conflict in conflicts:
            module = conflict.get("module", "")
            ctype = conflict.get("type", "")
            desc = conflict.get("description", "")

            # Map upstream module to Ascend file(s)
            ascend_files = self._map_conflict_to_ascend_files(module, conflict)
            for af in ascend_files:
                if af not in file_actions:
                    file_actions[af] = []
                file_actions[af].append(f"[{ctype}] {desc}")

        if not file_actions:
            return "No specific file actions identified. Review the upstream diff and update any affected files."

        lines = []
        for filepath, actions in sorted(file_actions.items()):
            lines.append(f"\n### {filepath}")
            for action in actions:
                lines.append(f"  - {action}")

        return "\n".join(lines)

    def _map_conflict_to_ascend_files(
        self, module: str, conflict: dict
    ) -> list[str]:
        """Map an upstream module conflict to Ascend files that need updates."""
        # Direct mappings based on patch_points.yaml
        module_to_ascend = {
            "lmcache/v1/gpu_connector/gpu_connectors.py": [
                "lmcache_ascend/v1/npu_connector/npu_connectors.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/gpu_connector/utils.py": [
                "lmcache_ascend/v1/npu_connector/utils.py",
            ],
            "lmcache/v1/gpu_connector/__init__.py": [
                "lmcache_ascend/v1/npu_connector/__init__.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/cache_engine.py": [
                "lmcache_ascend/v1/cache_engine.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/config.py": [
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/integration/vllm/vllm_v1_adapter.py": [
                "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/memory_management.py": [
                "lmcache_ascend/v1/memory_management.py",
            ],
            "lmcache/v1/storage_backend/": [
                "lmcache_ascend/v1/storage_backend/",
            ],
            "lmcache/v1/rpc_utils.py": [
                "lmcache_ascend/v1/rpc_utils.py",
            ],
            "lmcache/v1/metadata.py": [
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/manager.py": [
                "lmcache_ascend/__init__.py",
            ],
        }

        ascend_files = module_to_ascend.get(module, [])
        if not ascend_files:
            # Default: __init__.py handles most patching
            ascend_files = ["lmcache_ascend/__init__.py"]

        return ascend_files

    def _parse_generated_code(self, generated: str) -> dict[str, str]:
        """Parse the FILE blocks from LLM response."""
        file_changes = {}
        parts = generated.split("<<<FILE:")
        for part in parts[1:]:  # Skip first (before first marker)
            if "<<<END>>>" not in part:
                continue
            header_and_content = part.split("<<<END>>>")[0]
            lines = header_and_content.split("\n", 1)
            if len(lines) < 2:
                continue
            file_path = lines[0].strip().rstrip(">").strip()
            content = lines[1]
            # Remove leading/trailing newlines from content
            content = content.strip("\n")

            # Normalize path: ensure it starts with lmcache_ascend/
            if not file_path.startswith("lmcache_ascend") and not file_path.startswith("tests/"):
                # Try common prefixes
                if file_path.startswith("lmcache/"):
                    # This is an upstream path, skip it - we only write to lmcache_ascend
                    logger.warning(f"Skipping upstream path in LLM output: {file_path}")
                    continue
                # Might be a bare filename, try prepending lmcache_ascend/
                if not file_path.startswith("/"):
                    file_path = f"lmcache_ascend/{file_path}"

            file_changes[file_path] = content
        return file_changes

    def _create_pr(
        self,
        to_version: str,
        file_changes: dict[str, str],
        analysis: dict,
        rebase_result: dict,
    ) -> dict:
        """Apply file changes and create a pull request."""
        downstream_path = Path(self.config["sync"]["downstream_checkout"])
        branch_name = f"{self.config['sync']['pr'].get('branch_prefix', 'sync/upstream-')}{to_version}"
        target_branch = self.config["downstream"]["target_branch"]

        # Create branch from target_branch (delete old branch if exists)
        subprocess.run(
            ["git", "-C", str(downstream_path), "checkout", target_branch],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(downstream_path), "branch", "-D", branch_name],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(downstream_path), "checkout", "-b", branch_name],
            capture_output=True,
            check=True,
        )

        # Apply changes
        for rel_path, content in file_changes.items():
            full_path = downstream_path / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Stage and commit
        subprocess.run(
            ["git", "-C", str(downstream_path), "add", "-A"],
            capture_output=True,
            check=True,
        )
        commit_msg = (
            f"chore: adapt to upstream LMCache {to_version}\n\n"
            f"Auto-generated adaptation for upstream release {to_version}.\n\n"
            f"Detected conflicts: {len(analysis.get('conflicts', []))}\n"
            f"Rebase conflicts: {len(rebase_result.get('conflict_files', []))}"
        )
        result = subprocess.run(
            ["git", "-C", str(downstream_path), "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"success": False, "error": f"Git commit failed: {result.stderr}"}

        # Push branch to fork
        fork_repo = self.config["downstream"].get("fork")
        upstream_repo = self.config["downstream"]["repo"]

        if fork_repo:
            # Add fork as remote and push there
            fork_url = f"https://github.com/{fork_repo}.git"
            subprocess.run(
                ["git", "-C", str(downstream_path), "remote", "add", "fork", fork_url],
                capture_output=True,
            )
            result = subprocess.run(
                ["git", "-C", str(downstream_path), "push", "fork", branch_name, "--force"],
                capture_output=True,
                text=True,
            )
        else:
            # Push directly to upstream (requires write access)
            remote_url = f"https://x-access-token:{os.environ.get('GITHUB_TOKEN', '')}@github.com/{upstream_repo}.git"
            subprocess.run(
                ["git", "-C", str(downstream_path), "remote", "set-url", "origin", remote_url],
                capture_output=True,
            )
            result = subprocess.run(
                ["git", "-C", str(downstream_path), "push", "origin", branch_name, "--force"],
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            logger.warning(f"Git push failed: {result.stderr}, trying gh CLI...")
            result = subprocess.run(
                ["git", "-C", str(downstream_path), "push", "fork" if fork_repo else "origin", branch_name],
                capture_output=True,
                text=True,
                env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
            )

        # Create PR via gh CLI: from fork to upstream
        pr_body = self._build_pr_body(to_version, analysis, rebase_result)
        title = self.config["sync"]["pr"]["title_template"].format(version=to_version)
        labels = self.config["sync"]["pr"].get("labels", [])
        draft = "--draft" if self.config["sync"]["pr"].get("draft", False) else ""

        cmd = [
            "gh", "pr", "create",
            "--repo", upstream_repo,
            "--title", title,
            "--body", pr_body,
            "--base", target_branch,
        ]
        if fork_repo:
            # head format: owner:branch for cross-repo PR
            fork_owner = fork_repo.split("/")[0]
            cmd.extend(["--head", f"{fork_owner}:{branch_name}"])
        else:
            cmd.extend(["--head", branch_name])
        if draft:
            cmd.append(draft)

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
        )

        if result.returncode != 0:
            # Check if PR already exists for this branch — that's OK, force-push already updated it
            if "already exists" in result.stderr:
                import re
                url_match = re.search(r'(https://github\.com/[^\s]+/pull/\d+)', result.stderr)
                existing_url = url_match.group(1) if url_match else None
                pr_number_match = re.search(r'/pull/(\d+)', result.stderr) if existing_url else None
                existing_number = int(pr_number_match.group(1)) if pr_number_match else None
                logger.info(f"PR already exists: {existing_url}. Branch was force-pushed, PR is updated.")
                return {
                    "success": True,
                    "pr_url": existing_url,
                    "pr_number": existing_number,
                }
            # Retry without labels if label error
            if "not found" in result.stderr and labels:
                logger.warning(f"Labels not found, retrying without labels")
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
                )

        if result.returncode != 0:
            # Save generated code locally even if PR creation fails
            output_dir = Path("generated") / to_version
            output_dir.mkdir(parents=True, exist_ok=True)
            for rel_path, content in file_changes.items():
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content)
            logger.info(f"Generated code saved to {output_dir}/")
            return {
                "success": False,
                "error": f"PR creation failed: {result.stderr}. Generated code saved to generated/{to_version}/",
            }

        pr_url = result.stdout.strip()
        pr_number = pr_url.split("/")[-1] if pr_url else None

        return {
            "success": True,
            "pr_url": pr_url,
            "pr_number": int(pr_number) if pr_number and pr_number.isdigit() else None,
        }

    def _build_pr_body(
        self, to_version: str, analysis: dict, rebase_result: dict
    ) -> str:
        """Generate PR body markdown."""
        conflicts = analysis.get("conflicts", [])
        rebase_files = rebase_result.get("conflict_files", [])

        conflict_lines = ""
        for c in conflicts:
            severity = c.get("severity", "unknown")
            desc = c.get("description", "")
            conflict_lines += f"- **[{severity}]** {desc}\n"

        rebase_lines = ""
        for f in rebase_files:
            rebase_lines += f"- `{f}`\n"

        return f"""## Upstream Sync: LMCache {to_version}

This PR is auto-generated by [lmcache-ascend-sync](https://github.com/newpage1/lmcache-ascend-sync).

### Detected Conflicts ({len(conflicts)})

{conflict_lines if conflict_lines else "None detected by diff analysis."}

### Rebase Simulation Conflicts ({len(rebase_files)})

{rebase_lines if rebase_lines else "No rebase conflicts."}

### Changed Files

Auto-adapted by Claude API based on upstream changes.

---
*This PR was generated automatically. Please review carefully before merging.*
"""
