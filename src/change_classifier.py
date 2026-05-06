"""
Change Classifier - Parses upstream diffs and classifies each hunk
into deterministic vs LLM-needed categories.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yaml

logger = logging.getLogger("lmcache-sync.classifier")


class ChangeType(Enum):
    MECHANICAL_RENAME = "mechanical_rename"
    IMPORT_CHANGE = "import_change"
    CONFIG_ADDITION = "config_addition"
    PARAM_CHANGE = "param_change"
    LOGIC_CHANGE = "logic_change"
    NEW_METHOD = "new_method"
    NEW_CLASS = "new_class"
    IRRELEVANT = "irrelevant"


@dataclass
class DiffHunk:
    file_path: str
    old_start: int
    new_start: int
    old_lines: list[str]  # Lines starting with '-'
    new_lines: list[str]  # Lines starting with '+'
    context_lines: list[str]  # Lines starting with ' '
    raw_header: str = ""


@dataclass
class ClassifiedChange:
    hunk: DiffHunk
    change_type: ChangeType
    confidence: float  # 0.0-1.0
    description: str
    # Mapping info: which Ascend file(s) this affects
    ascend_files: list[str] = field(default_factory=list)
    # For deterministic changes: the specific transform to apply
    transform_hint: Optional[str] = None


class ChangeClassifier:
    """Classify upstream diff hunks by change type."""

    def __init__(self, patch_points_path: Optional[str] = None):
        self.patch_points = self._load_patch_points(patch_points_path)
        self._relevant_modules = self._build_relevance_map()

    def _load_patch_points(self, path: Optional[str]) -> dict:
        if not path:
            return {}
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("patch_points", {})
        except Exception:
            return {}

    def _build_relevance_map(self) -> dict[str, list[str]]:
        """Build a mapping of upstream modules to Ascend files they affect."""
        # Hardcoded based on known patch point mappings
        return {
            "lmcache/v1/gpu_connector/gpu_connectors.py": [
                "lmcache_ascend/v1/npu_connector/npu_connectors.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/cache_engine.py": [
                "lmcache_ascend/v1/cache_engine.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/config.py": [
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/memory_management.py": [
                "lmcache_ascend/v1/memory_management.py",
            ],
            "lmcache/v1/storage_backend/": [
                "lmcache_ascend/v1/storage_backend/",
            ],
            "lmcache/v1/gpu_connector/__init__.py": [
                "lmcache_ascend/v1/npu_connector/__init__.py",
                "lmcache_ascend/__init__.py",
            ],
            "lmcache/v1/gpu_connector/utils.py": [
                "lmcache_ascend/v1/npu_connector/utils.py",
            ],
        }

    def classify_diff(self, diff_text: str) -> list[ClassifiedChange]:
        """Parse and classify all hunks in a unified diff."""
        hunks = self._parse_diff(diff_text)
        results = []

        for hunk in hunks:
            # Skip irrelevant files
            if not self._is_relevant_file(hunk.file_path):
                results.append(ClassifiedChange(
                    hunk=hunk,
                    change_type=ChangeType.IRRELEVANT,
                    confidence=1.0,
                    description=f"Irrelevant file: {hunk.file_path}",
                    ascend_files=[],
                ))
                continue

            change = self._classify_hunk(hunk)
            results.append(change)

        relevant = [c for c in results if c.change_type != ChangeType.IRRELEVANT]
        logger.info(
            f"Classified {len(results)} hunks: "
            f"{len(relevant)} relevant, "
            f"{len(results) - len(relevant)} irrelevant"
        )
        for c in relevant:
            logger.info(f"  [{c.change_type.value}] {c.description}")

        return results

    def _parse_diff(self, diff_text: str) -> list[DiffHunk]:
        """Parse unified diff format into DiffHunk objects."""
        hunks = []
        current_file = ""
        current_hunk_lines = []

        for line in diff_text.split("\n"):
            # File header
            file_match = re.match(r'^\+\+\+ b/(.+)$', line)
            if file_match:
                current_file = file_match.group(1)
                continue

            # Hunk header
            hunk_match = re.match(
                r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line
            )
            if hunk_match:
                # Save previous hunk if any
                if current_hunk_lines:
                    hunks.append(self._build_hunk(
                        current_file, current_hunk_lines
                    ))
                current_hunk_lines = [line]
                continue

            if current_hunk_lines:
                current_hunk_lines.append(line)

        # Don't forget the last hunk
        if current_hunk_lines:
            hunks.append(self._build_hunk(current_file, current_hunk_lines))

        return hunks

    def _build_hunk(self, file_path: str, lines: list[str]) -> DiffHunk:
        """Build a DiffHunk from raw hunk lines."""
        header = lines[0] if lines else ""
        old_lines = []
        new_lines = []
        context_lines = []

        for line in lines[1:]:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                context_lines.append(line[1:])

        hunk_match = re.match(
            r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', header
        )
        old_start = int(hunk_match.group(1)) if hunk_match else 0
        new_start = int(hunk_match.group(2)) if hunk_match else 0

        return DiffHunk(
            file_path=file_path,
            old_start=old_start,
            new_start=new_start,
            old_lines=old_lines,
            new_lines=new_lines,
            context_lines=context_lines,
            raw_header=header,
        )

    def _is_relevant_file(self, file_path: str) -> bool:
        """Check if this file is in our relevance map."""
        for module in self._relevant_modules:
            if file_path.startswith(module) or file_path == module:
                return True
        return False

    def _classify_hunk(self, hunk: DiffHunk) -> ClassifiedChange:
        """Classify a single diff hunk."""
        ascend_files = self._relevant_modules.get(
            hunk.file_path, ["lmcache_ascend/__init__.py"]
        )

        # Heuristic 1: Import-only changes
        if self._is_import_only(hunk):
            return ClassifiedChange(
                hunk=hunk,
                change_type=ChangeType.IMPORT_CHANGE,
                confidence=0.9,
                description=f"Import change in {hunk.file_path}",
                ascend_files=ascend_files,
                transform_hint="import_unguard",
            )

        # Heuristic 2: Pure rename (identifier changed, same structure)
        if self._is_mechanical_rename(hunk):
            return ClassifiedChange(
                hunk=hunk,
                change_type=ChangeType.MECHANICAL_RENAME,
                confidence=0.85,
                description=f"Rename in {hunk.file_path}: "
                            f"{self._describe_rename(hunk)}",
                ascend_files=ascend_files,
                transform_hint="rename",
            )

        # Heuristic 3: Config additions
        if self._is_config_addition(hunk):
            return ClassifiedChange(
                hunk=hunk,
                change_type=ChangeType.CONFIG_ADDITION,
                confidence=0.9,
                description=f"Config addition in {hunk.file_path}",
                ascend_files=ascend_files,
                transform_hint="config_add",
            )

        # Heuristic 4: New method/class
        if self._is_new_definition(hunk):
            new_name = self._get_new_definition_name(hunk)
            if new_name and new_name[0].isupper():
                change_type = ChangeType.NEW_CLASS
            else:
                change_type = ChangeType.NEW_METHOD
            return ClassifiedChange(
                hunk=hunk,
                change_type=change_type,
                confidence=0.8,
                description=f"New {change_type.value} in {hunk.file_path}: {new_name}",
                ascend_files=ascend_files,
            )

        # Heuristic 5: Parameter change (method signature modified)
        if self._is_param_change(hunk):
            return ClassifiedChange(
                hunk=hunk,
                change_type=ChangeType.PARAM_CHANGE,
                confidence=0.7,
                description=f"Parameter change in {hunk.file_path}",
                ascend_files=ascend_files,
            )

        # Default: logic change — needs LLM
        return ClassifiedChange(
            hunk=hunk,
            change_type=ChangeType.LOGIC_CHANGE,
            confidence=0.5,
            description=f"Logic change in {hunk.file_path} "
                        f"(lines {hunk.old_start}→{hunk.new_start})",
            ascend_files=ascend_files,
        )

    # --- Heuristic methods ---

    @staticmethod
    def _is_import_only(hunk: DiffHunk) -> bool:
        """Check if all changed lines are import statements."""
        all_lines = hunk.old_lines + hunk.new_lines
        if not all_lines:
            return False
        return all(
            line.strip().startswith(("import ", "from "))
            for line in all_lines
        )

    @staticmethod
    def _is_mechanical_rename(hunk: DiffHunk) -> bool:
        """Check if old and new lines are identical except for identifier names."""
        if len(hunk.old_lines) != len(hunk.new_lines):
            return False
        if not hunk.old_lines:
            return False

        for old, new in zip(hunk.old_lines, hunk.new_lines):
            old_stripped = old.strip()
            new_stripped = new.strip()
            if old_stripped == new_stripped:
                continue
            # Check if the only difference is a substring replacement
            # Find the differing parts
            if len(old_stripped) == len(new_stripped):
                diffs = sum(1 for a, b in zip(old_stripped, new_stripped) if a != b)
                if diffs / max(len(old_stripped), 1) < 0.3:
                    continue
            else:
                # Length differs — might be a rename with different length
                # Check if one is a substring of the other with a common prefix/suffix
                continue

        return True

    @staticmethod
    def _describe_rename(hunk: DiffHunk) -> str:
        """Extract what was renamed."""
        for old, new in zip(hunk.old_lines, hunk.new_lines):
            old_s = old.strip()
            new_s = new.strip()
            if old_s != new_s:
                return f"{old_s} → {new_s}"
        return ""

    @staticmethod
    def _is_config_addition(hunk: DiffHunk) -> bool:
        """Check if this is a config dict addition (new keys in a dict)."""
        # Config additions typically have:
        # - Only '+' lines (no '-' lines), adding key-value pairs
        # - Or renaming a key (one '-' line, one '+' line)
        if not hunk.new_lines:
            return False

        # Check if file is config.py or contains config patterns
        if "config" in hunk.file_path:
            new_text = "\n".join(hunk.new_lines)
            # Pattern: "key_name": { or "key_name": value
            config_pattern = re.compile(r'["\']\w+["\']\s*:')
            if config_pattern.search(new_text):
                return True

        return False

    @staticmethod
    def _is_new_definition(hunk: DiffHunk) -> bool:
        """Check if new lines contain a new def or class."""
        for line in hunk.new_lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                # Check it's truly new (not in old_lines)
                is_new = True
                for old_line in hunk.old_lines:
                    if stripped == old_line.strip():
                        is_new = False
                        break
                if is_new:
                    return True
        return False

    @staticmethod
    def _get_new_definition_name(hunk: DiffHunk) -> str:
        """Get the name of the new definition."""
        for line in hunk.new_lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                match = re.match(r'def\s+(\w+)', stripped)
                if match:
                    return match.group(1)
            elif stripped.startswith("class "):
                match = re.match(r'class\s+(\w+)', stripped)
                if match:
                    return match.group(1)
        return ""

    @staticmethod
    def _is_param_change(hunk: DiffHunk) -> bool:
        """Check if method signatures were modified (def line changed)."""
        old_defs = [l.strip() for l in hunk.old_lines if l.strip().startswith("def ")]
        new_defs = [l.strip() for l in hunk.new_lines if l.strip().startswith("def ")]

        if old_defs and new_defs:
            # Same method name but different signature
            for old_d, new_d in zip(old_defs, new_defs):
                old_name = re.match(r'def\s+(\w+)', old_d)
                new_name = re.match(r'def\s+(\w+)', new_d)
                if old_name and new_name and old_name.group(1) == new_name.group(1):
                    return True
        return False
