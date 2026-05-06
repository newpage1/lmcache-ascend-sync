"""
Deterministic Transform Engine - Applies mechanical code transformations
using configurable rules (no LLM needed).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .change_classifier import ChangeType, ClassifiedChange

logger = logging.getLogger("lmcache-sync.transformer")


@dataclass
class TransformRule:
    name: str
    type: str  # replace, rename, import_unguard, config_add, config_rename, conditional_replace
    pattern: str = ""
    replacement: str = ""
    files: list[str] = None
    scope: str = "ascend_only"  # "ascend_only" or "all"
    description: str = ""
    # For config_rename
    old_field: str = ""
    new_field: str = ""
    target_file: str = ""
    # For config_add
    fields: list[dict] = None
    # For conditional_replace
    context_within: str = ""


class DeterministicTransformer:
    """Apply deterministic code transformations based on rules."""

    def __init__(self, rules_path: Optional[str] = None):
        self.rules = self._load_rules(rules_path)

    def _load_rules(self, path: Optional[str]) -> list[TransformRule]:
        if not path:
            # Default path
            default = Path(__file__).parent.parent / "config" / "transform_rules.yaml"
            if default.exists():
                path = str(default)
            else:
                return []

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            rules = []
            for r in data.get("rules", []):
                rules.append(TransformRule(
                    name=r["name"],
                    type=r["type"],
                    pattern=r.get("pattern", ""),
                    replacement=r.get("replacement", ""),
                    files=r.get("files"),
                    scope=r.get("scope", "ascend_only"),
                    description=r.get("description", ""),
                    old_field=r.get("old_field", ""),
                    new_field=r.get("new_field", ""),
                    target_file=r.get("target_file", ""),
                    fields=r.get("fields"),
                    context_within=r.get("context_within", ""),
                ))
            logger.info(f"Loaded {len(rules)} transform rules from {path}")
            return rules
        except Exception as e:
            logger.warning(f"Failed to load transform rules: {e}")
            return []

    def transform(
        self,
        ascend_sources: dict[str, str],
        classified_changes: list[ClassifiedChange],
    ) -> dict[str, str]:
        """Apply all deterministic transforms to Ascend sources.

        Returns modified sources (only files that were changed).
        """
        modified = {}

        # First, apply rules from transform_rules.yaml (chained: each rule works on previous output)
        working_sources = dict(ascend_sources)
        for rule in self.rules:
            changed_files = self._apply_rule(rule, working_sources)
            if changed_files:
                working_sources.update(changed_files)
                modified.update(changed_files)

        # Then, apply transforms inferred from classified changes
        for change in classified_changes:
            if change.change_type in (ChangeType.MECHANICAL_RENAME,
                                       ChangeType.IMPORT_CHANGE):
                changed = self._apply_classified_change(change, working_sources)
                if changed:
                    working_sources.update(changed)
                    modified.update(changed)

        if modified:
            logger.info(f"Deterministic transforms: {len(modified)} files modified")
            for path, content in modified.items():
                orig = ascend_sources.get(path, "")
                diff_lines = sum(
                    1 for a, b in zip(orig.splitlines(), content.splitlines())
                    if a != b
                )
                logger.info(f"  {path}: ~{diff_lines} lines changed")

        return modified

    def _apply_rule(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Apply a single rule to matching files."""
        results = {}

        if rule.type == "replace":
            results = self._apply_replace(rule, sources)
        elif rule.type == "rename":
            results = self._apply_rename(rule, sources)
        elif rule.type == "import_unguard":
            results = self._apply_import_unguard(rule, sources)
        elif rule.type == "conditional_replace":
            results = self._apply_conditional_replace(rule, sources)
        elif rule.type == "config_rename":
            results = self._apply_config_rename(rule, sources)
        elif rule.type == "config_add":
            results = self._apply_config_add(rule, sources)

        return results

    def _get_target_files(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> list[str]:
        """Get list of source files this rule should apply to."""
        if rule.files:
            return [f for f in rule.files if f in sources]
        if rule.target_file:
            return [rule.target_file] if rule.target_file in sources else []
        # No file filter — apply to all ascend files
        return list(sources.keys())

    def _apply_replace(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Simple string replacement."""
        results = {}
        for filepath in self._get_target_files(rule, sources):
            source = sources[filepath]
            if rule.pattern not in source:
                continue
            new_source = source.replace(rule.pattern, rule.replacement)
            if new_source != source:
                results[filepath] = new_source
                logger.debug(f"  [{rule.name}] Applied to {filepath}")
        return results

    def _apply_rename(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Identifier rename — uses word boundary matching."""
        results = {}
        pattern = re.compile(r'\b' + re.escape(rule.pattern) + r'\b')
        for filepath in self._get_target_files(rule, sources):
            source = sources[filepath]
            if not pattern.search(source):
                continue
            new_source = pattern.sub(rule.replacement, source)
            if new_source != source:
                results[filepath] = new_source
                logger.debug(f"  [{rule.name}] Applied to {filepath}")
        return results

    def _apply_import_unguard(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Remove conditional guard around import statement."""
        results = {}
        # Pattern may span multiple lines
        # Normalize whitespace in pattern for matching
        for filepath in self._get_target_files(rule, sources):
            source = sources[filepath]
            # Try to match the guarded import pattern
            # Handle variations: if torch.cuda.is_available():\n    import ...
            guard_pattern = re.compile(
                r'if\s+torch\.(cuda|npu)\.is_available\(\):\s*\n(\s+)'
                + re.escape(rule.replacement.replace("import ", "")),
                re.MULTILINE,
            )
            match = guard_pattern.search(source)
            if match:
                new_source = source[:match.start()] + rule.replacement + source[match.end():]
                results[filepath] = new_source
                logger.debug(f"  [{rule.name}] Applied to {filepath}")
                continue

            # Fallback: direct pattern matching
            if rule.pattern in source:
                new_source = source.replace(rule.pattern, rule.replacement)
                if new_source != source:
                    results[filepath] = new_source
                    logger.debug(f"  [{rule.name}] Applied to {filepath}")
        return results

    def _apply_conditional_replace(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Replace pattern only within specific context (e.g., inside methods)."""
        results = {}
        pattern = re.compile(re.escape(rule.pattern))

        for filepath in self._get_target_files(rule, sources):
            source = sources[filepath]
            if not pattern.search(source):
                continue

            lines = source.split("\n")
            new_lines = []
            in_context = False

            for line in lines:
                # Track context
                if rule.context_within and rule.context_within in line:
                    in_context = True
                elif rule.context_within == "def " and line.strip().startswith("def "):
                    in_context = True
                elif not line.strip() or (line[0] != ' ' and line[0] != '\t' and line.strip()):
                    # Dedent — exit context
                    if not line.strip().startswith(("if ", "for ", "while ", "with ", "else", "elif", "try", "except", "finally")):
                        in_context = False

                if in_context:
                    new_line = pattern.sub(rule.replacement, line)
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)

            new_source = "\n".join(new_lines)
            if new_source != source:
                results[filepath] = new_source
                logger.debug(f"  [{rule.name}] Applied to {filepath}")
        return results

    def _apply_config_rename(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Rename a config field in _CONFIG_DEFINITIONS."""
        results = {}
        filepath = rule.target_file
        if filepath not in sources:
            return results

        source = sources[filepath]
        # Rename in _CONFIG_DEFINITIONS dict key
        old_pattern = re.compile(
            r'["\']' + re.escape(rule.old_field) + r'["\']'
        )
        if old_pattern.search(source):
            new_source = old_pattern.sub(
                f'"{rule.new_field}"', source
            )
            # Also rename any references to the field
            ref_pattern = re.compile(r'\b' + re.escape(rule.old_field) + r'\b')
            new_source = ref_pattern.sub(rule.new_field, new_source)
            if new_source != source:
                results[filepath] = new_source
                logger.debug(f"  [{rule.name}] Applied to {filepath}")
        return results

    def _apply_config_add(
        self, rule: TransformRule, sources: dict[str, str]
    ) -> dict[str, str]:
        """Add new config fields to _CONFIG_DEFINITIONS."""
        results = {}
        filepath = rule.target_file
        if filepath not in sources or not rule.fields:
            return results

        source = sources[filepath]

        # Find _CONFIG_DEFINITIONS dict and add fields before the closing }
        # Look for the pattern of existing config entries to match style
        for field_def in rule.fields:
            field_name = field_def["name"]
            # Check if field already exists
            if f'"{field_name}"' in source or f"'{field_name}'" in source:
                logger.debug(f"  Config field '{field_name}' already exists, skipping")
                continue

            definition = field_def.get("definition", '{"type": str, "default": ""}')

            # Find insertion point: before the closing of _CONFIG_DEFINITIONS
            # Look for the last entry in the dict
            config_pattern = re.compile(
                r'(_CONFIG_DEFINITIONS\s*=\s*\{.*?)(\n\s*\})',
                re.DOTALL,
            )
            match = config_pattern.search(source)
            if match:
                # Insert before the closing brace
                indent = "    "
                new_entry = f'\n{indent}"{field_name}": {definition},'
                insert_pos = match.end() - len(match.group(2))
                source = source[:insert_pos] + new_entry + match.group(2)
                logger.debug(f"  Added config field '{field_name}'")

        if source != sources[filepath]:
            results[filepath] = source
        return results

    def _apply_classified_change(
        self,
        change: ClassifiedChange,
        sources: dict[str, str],
    ) -> dict[str, str]:
        """Apply a transform inferred from a classified change."""
        results = {}

        if change.change_type == ChangeType.MECHANICAL_RENAME:
            # Extract old → new from hunk
            for old_line, new_line in zip(change.hunk.old_lines, change.hunk.new_lines):
                old_s = old_line.strip()
                new_s = new_line.strip()
                if old_s != new_s:
                    # Find the differing part
                    for ascend_file in change.ascend_files:
                        if ascend_file not in sources:
                            continue
                        source = sources[ascend_file]
                        # Apply the same rename
                        pattern = re.compile(r'\b' + re.escape(old_s) + r'\b')
                        if pattern.search(source):
                            new_source = pattern.sub(new_s, source)
                            if new_source != source:
                                results[ascend_file] = new_source
                                logger.debug(
                                    f"  [classified rename] {old_s} → {new_s} "
                                    f"in {ascend_file}"
                                )
                    break  # Only process first difference

        elif change.change_type == ChangeType.IMPORT_CHANGE:
            for ascend_file in change.ascend_files:
                if ascend_file not in sources:
                    continue
                source = sources[ascend_file]
                # Apply import changes
                for old_line, new_line in zip(change.hunk.old_lines, change.hunk.new_lines):
                    old_import = old_line.strip()
                    new_import = new_line.strip()
                    if old_import in source:
                        new_source = source.replace(old_import, new_import)
                        if new_source != source:
                            results[ascend_file] = new_source
                            logger.debug(
                                f"  [classified import] {old_import} → {new_import} "
                                f"in {ascend_file}"
                            )

        return results
