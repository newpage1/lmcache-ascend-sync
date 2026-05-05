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
        """Read LMCache-Ascend source files that may need adaptation."""
        sources = {}
        ascend_dir = downstream_path / "lmcache_ascend"
        if not ascend_dir.exists():
            return sources

        key_files = [
            "__init__.py",
            "v1/npu_connector.py",
            "v1/cache_engine.py",
            "v1/memory_management.py",
            "v1/system_detection.py",
            "v1/tokens_hash.py",
            "integration/vllm/lmcache_ascend_connector_v1.py",
            "integration/vllm/vllm_v1_adapter.py",
        ]

        for rel_path in key_files:
            full_path = ascend_dir / rel_path
            if full_path.exists():
                try:
                    sources[rel_path] = full_path.read_text()
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
        """Build the Claude API prompt for code generation."""
        conflicts_text = json.dumps(analysis.get("conflicts", []), indent=2)
        rebase_conflicts = json.dumps(
            rebase_result.get("conflict_files", []), indent=2
        )

        sources_section = ""
        for path, content in ascend_sources.items():
            sources_section += f"\n### {path}\n```python\n{content}\n```\n"

        return textwrap.dedent(f"""\
            You are an expert Python developer maintaining LMCache-Ascend, a plugin that adapts
            the upstream LMCache project to run on Huawei Ascend NPUs.

            LMCache-Ascend works by monkey-patching specific upstream modules at import time,
            replacing CUDA kernels with CANN kernels, and subclassing GPU connectors as NPU connectors.

            ## Task
            The upstream LMCache has released version {to_version} (currently tracking {from_version}).
            Analyze the breaking changes and generate updated LMCache-Ascend code.

            ## Detected Conflicts
            ```json
            {conflicts_text}
            ```

            ## Rebase Simulation Conflicts
            ```json
            {rebase_conflicts}
            ```

            ## Upstream Diff (truncated)
            ```diff
            {upstream_diff}
            ```

            ## Current LMCache-Ascend Source Files
            {sources_section}

            ## Instructions
            1. For each conflict, determine what changes are needed in the LMCache-Ascend code.
            2. Generate the COMPLETE updated file content for each affected file.
            3. Maintain backward compatibility where possible.
            4. Keep all Ascend-specific logic (NPU device checks, CANN kernel calls, NUMA detection, etc.).
            5. Update method signatures only when the upstream interface actually changed.

            ## Output Format
            For each file that needs changes, output EXACTLY:

            <<<FILE: relative/path/to/file.py>>>
            # complete file content here
            <<<END>>>

            Only output files that need changes. Do not output unchanged files.
        """)

    def _parse_generated_code(self, generated: str) -> dict[str, str]:
        """Parse the FILE blocks from Claude's response."""
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

        # Create branch
        subprocess.run(
            ["git", "-C", str(downstream_path), "checkout", target_branch],
            capture_output=True,
            check=True,
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

        # Push branch
        remote_url = f"https://x-access-token:{os.environ.get('GITHUB_TOKEN', '')}@github.com/{self.config['downstream']['repo']}.git"
        subprocess.run(
            ["git", "-C", str(downstream_path), "remote", "set-url", "origin", remote_url],
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "-C", str(downstream_path), "push", "origin", branch_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Try with gh CLI as fallback
            result = subprocess.run(
                ["git", "-C", str(downstream_path), "push", "-u", "origin", branch_name],
                capture_output=True,
                text=True,
                env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
            )

        # Create PR via gh CLI
        pr_body = self._build_pr_body(to_version, analysis, rebase_result)
        title = self.config["sync"]["pr"]["title_template"].format(version=to_version)
        labels = self.config["sync"]["pr"].get("labels", [])
        draft = "--draft" if self.config["sync"]["pr"].get("draft", False) else ""

        cmd = [
            "gh", "pr", "create",
            "--repo", self.config["downstream"]["repo"],
            "--title", title,
            "--body", pr_body,
            "--head", branch_name,
            "--base", target_branch,
        ]
        if draft:
            cmd.append(draft)

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
        )

        if result.returncode != 0:
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
