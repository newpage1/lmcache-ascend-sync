"""
Method-level LLM Patcher - Uses LLM to generate surgical method-level changes.
Only sends the changed method (not entire files) to LLM, then uses AST
for precise method replacement.
"""

import ast
import logging
import textwrap
from dataclasses import dataclass
from typing import Optional

import anthropic

from .change_classifier import ChangeType, ClassifiedChange

logger = logging.getLogger("lmcache-sync.method_patcher")


@dataclass
class MethodPatch:
    filepath: str
    class_name: str
    method_name: str
    original_source: str
    patched_source: str


class MethodPatcher:
    """Use LLM to patch individual methods, not entire files."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/anthropic",
        model: str = "GLM-5.1",
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens

    def patch_methods(
        self,
        ascend_sources: dict[str, str],
        changes_needing_llm: list[ClassifiedChange],
        upstream_before_sources: dict[str, str],
        upstream_after_sources: dict[str, str],
    ) -> dict[str, str]:
        """Process all changes that need LLM intervention.

        Returns dict of filepath → modified source (only files that changed).
        """
        modified = {}

        for change in changes_needing_llm:
            for ascend_file in change.ascend_files:
                if ascend_file not in ascend_sources:
                    logger.debug(f"  Ascend file not found: {ascend_file}")
                    continue

                source = modified.get(ascend_file, ascend_sources[ascend_file])

                # Try to identify the specific method to patch
                method_info = self._identify_target_method(change, source)
                if not method_info:
                    logger.warning(
                        f"  Cannot identify target method for change in "
                        f"{change.hunk.file_path}, skipping"
                    )
                    continue

                class_name, method_name = method_info

                # Extract upstream method before/after
                upstream_before = self._extract_method_from_sources(
                    upstream_before_sources, change.hunk.file_path,
                    class_name, method_name,
                )
                upstream_after = self._extract_method_from_sources(
                    upstream_after_sources, change.hunk.file_path,
                    class_name, method_name,
                )

                # Extract current Ascend version
                ascend_method = self._extract_method(
                    source, class_name, method_name
                )
                if not ascend_method:
                    logger.warning(
                        f"  Method {class_name}.{method_name} not found in "
                        f"{ascend_file}"
                    )
                    continue

                # Generate patched method via LLM
                patched_method = self._call_llm(
                    method_name, class_name,
                    upstream_before, upstream_after,
                    ascend_method,
                )

                if not patched_method:
                    logger.warning(
                        f"  LLM failed to patch {class_name}.{method_name}"
                    )
                    continue

                # Replace method in source using AST
                new_source = self._replace_method(
                    source, class_name, method_name, patched_method
                )

                if new_source:
                    modified[ascend_file] = new_source
                    logger.info(
                        f"  Patched {class_name}.{method_name} in {ascend_file}"
                    )

        if modified:
            logger.info(
                f"Method patching: {len(modified)} files modified via LLM"
            )

        return modified

    def _identify_target_method(
        self, change: ClassifiedChange, source: str
    ) -> Optional[tuple[str, str]]:
        """Try to identify which class.method this change targets."""
        hunk = change.hunk

        # Look for method definitions in context and new lines
        all_lines = hunk.context_lines + hunk.old_lines + hunk.new_lines

        current_class = None
        current_method = None

        # Parse the source to find class/method near the line numbers
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Check if this method's line range overlaps with the hunk
                            if (item.lineno <= hunk.new_start <= item.end_lineno
                                    or item.lineno <= hunk.old_start <= item.end_lineno):
                                return (node.name, item.name)
        except SyntaxError:
            pass

        # Fallback: look for method names in the diff lines
        for line in all_lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                match = __import__("re").match(r'def\s+(\w+)', stripped)
                if match:
                    current_method = match.group(1)

        if current_method:
            # Find the class this method belongs to
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if item.name == current_method:
                                    return (node.name, current_method)
            except SyntaxError:
                pass

        return None

    @staticmethod
    def _extract_method(
        source: str, class_name: str, method_name: str
    ) -> Optional[str]:
        """Extract a method's source code using AST."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            return ast.get_source_segment(source, item)
        return None

    @staticmethod
    def _extract_method_from_sources(
        sources: dict[str, str],
        file_path: str,
        class_name: str,
        method_name: str,
    ) -> Optional[str]:
        """Extract method from a dict of file sources."""
        source = sources.get(file_path, "")
        if not source:
            return None
        return MethodPatcher._extract_method(source, class_name, method_name)

    @staticmethod
    def _replace_method(
        source: str,
        class_name: str,
        method_name: str,
        new_method_source: str,
    ) -> Optional[str]:
        """Replace a method in source using AST-based line tracking."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            # Get the line range of the original method
                            start_line = item.lineno - 1  # 0-indexed
                            end_line = item.end_lineno  # exclusive

                            # Build new source
                            new_lines = new_method_source.splitlines()

                            # Replace the method lines
                            result_lines = (
                                lines[:start_line]
                                + new_lines
                                + lines[end_line:]
                            )
                            return "\n".join(result_lines)

        return None

    def _call_llm(
        self,
        method_name: str,
        class_name: str,
        upstream_before: Optional[str],
        upstream_after: Optional[str],
        ascend_version: str,
    ) -> Optional[str]:
        """Call LLM with a focused method-level prompt."""
        prompt = self._build_prompt(
            method_name, class_name,
            upstream_before, upstream_after,
            ascend_version,
        )

        try:
            client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
            # Clean up: remove markdown code fences if present
            result = self._clean_llm_output(result)
            return result
        except Exception as e:
            logger.error(f"LLM API error in method patch: {e}")
            return None

    def _build_prompt(
        self,
        method_name: str,
        class_name: str,
        upstream_before: Optional[str],
        upstream_after: Optional[str],
        ascend_version: str,
    ) -> str:
        """Build a focused method-level prompt."""
        before_section = ""
        if upstream_before:
            before_section = f"""\
--- upstream before (old version)
```python
{upstream_before}
```"""

        after_section = ""
        if upstream_after:
            after_section = f"""\
+++ upstream after (new version)
```python
{upstream_after}
```"""

        diff_note = ""
        if upstream_before and upstream_after:
            diff_note = "\nCompare the two upstream versions above to understand what changed."
        elif not upstream_before:
            diff_note = "\nThis is a NEW method added in the upstream version."

        return textwrap.dedent(f"""\
            You are adapting LMCache code for Huawei Ascend NPUs.
            Task: Apply the upstream change to the Ascend version of `{class_name}.{method_name}`.

            {before_section}
            {after_section}
            {diff_note}

            Current Ascend version of `{class_name}.{method_name}`:
            ```python
            {ascend_version}
            ```

            Rules:
            1. Keep ALL Ascend-specific logic (torch.npu, lmc_ops, is_310p, NUMA, etc.)
            2. Only apply the equivalent of the upstream change
            3. Do NOT remove any existing Ascend workarounds
            4. Preserve the exact indentation level

            Output ONLY the modified method definition (starting with `def {method_name}(...)`).
            Do NOT output any explanation, markdown fences, or surrounding code.
        """)

    @staticmethod
    def _clean_llm_output(text: str) -> str:
        """Remove markdown code fences and leading/trailing whitespace."""
        text = text.strip()
        # Remove ```python ... ``` wrapper
        if text.startswith("```python"):
            text = text[len("```python"):]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
