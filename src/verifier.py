"""
Change Verifier - Validates generated code quality before PR creation.

Checks: syntax, line count delta, class/method preservation, import preservation,
Ascend keyword preservation.
"""

import ast
import logging
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("lmcache-sync.verifier")

# Ascend-specific keywords that should be preserved
ASCEND_KEYWORDS = [
    "torch.npu",
    "lmc_ops",
    "is_310p",
    "Ascend",
    "npu.",
    "NPU",
    "CANN",
    "AscendLMCacheEngine",
    "NPUConnector",
    "npu_connector",
]


@dataclass
class CheckResult:
    name: str
    status: str  # "passed", "FAILED", "WARNING"
    message: str
    details: str = ""


@dataclass
class FileVerification:
    filepath: str
    checks: list[CheckResult] = field(default_factory=list)
    passed: bool = True

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "FAILED"]


@dataclass
class VerificationReport:
    results: list[FileVerification] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines = [f"验证报告: {len(self.results)} files checked"]
        lines.append("=" * 60)
        for fv in self.results:
            lines.append(f"\n文件: {fv.filepath}")
            for check in fv.checks:
                tag = f"[{check.status}]"
                lines.append(f"  {tag:10s} {check.name}: {check.message}")
            if not fv.passed:
                lines.append(f"  >>> 文件验证: {'PASSED' if fv.passed else 'FAILED'}")
        lines.append("=" * 60)
        lines.append(f"总结果: {'PASSED' if self.passed else 'FAILED — 拒绝合并'}")
        return "\n".join(lines)


class ChangeVerifier:
    """Verify generated code quality before PR creation."""

    def __init__(self, max_line_delta_pct: float = 30.0):
        self.max_line_delta_pct = max_line_delta_pct

    def verify(
        self,
        original_sources: dict[str, str],
        modified_sources: dict[str, str],
    ) -> VerificationReport:
        """Run all checks on modified files."""
        results = []
        all_passed = True

        for filepath, new_content in modified_sources.items():
            original = original_sources.get(filepath, "")
            fv = self._verify_file(filepath, original, new_content)
            results.append(fv)
            if not fv.passed:
                all_passed = False

        report = VerificationReport(results=results, passed=all_passed)
        logger.info(f"Verification result: {'PASSED' if all_passed else 'FAILED'}")
        for fv in results:
            if not fv.passed:
                for check in fv.failed_checks:
                    logger.warning(f"  {fv.filepath}: {check.name} - {check.message}")

        return report

    def _verify_file(
        self, filepath: str, original: str, modified: str
    ) -> FileVerification:
        """Run all checks on a single file."""
        checks = []

        # 1. Syntax check (always run first)
        checks.append(self._check_syntax(filepath, modified))

        # 2. Line count delta (only for existing files)
        if original:
            checks.append(self._check_line_count(filepath, original, modified))
            # 3. Class preservation
            checks.append(self._check_class_preservation(filepath, original, modified))
            # 4. Method preservation
            checks.append(self._check_method_preservation(filepath, original, modified))
            # 5. Import preservation
            checks.append(self._check_import_preservation(filepath, original, modified))
            # 6. Ascend keyword preservation
            checks.append(self._check_ascend_keywords(filepath, original, modified))
        else:
            checks.append(
                CheckResult("line_count", "passed", "New file, no delta check")
            )
            checks.append(
                CheckResult("class_preservation", "passed", "New file")
            )
            checks.append(
                CheckResult("method_preservation", "passed", "New file")
            )
            checks.append(
                CheckResult("import_preservation", "passed", "New file")
            )
            checks.append(
                CheckResult("ascend_keywords", "passed", "New file")
            )

        # A file fails if any check is FAILED
        passed = all(c.status != "FAILED" for c in checks)
        return FileVerification(filepath=filepath, checks=checks, passed=passed)

    def _check_syntax(self, filepath: str, code: str) -> CheckResult:
        """Check Python syntax with py_compile."""
        if not code.strip():
            return CheckResult("syntax", "FAILED", "Empty file")

        try:
            # Use ast.parse instead of py_compile to avoid file system writes
            ast.parse(code, filename=filepath)
            return CheckResult("syntax", "passed", "Valid Python syntax")
        except SyntaxError as e:
            return CheckResult(
                "syntax", "FAILED", f"Syntax error at line {e.lineno}: {e.msg}"
            )

    def _check_line_count(
        self, filepath: str, original: str, modified: str
    ) -> CheckResult:
        """Check line count delta is within threshold."""
        orig_lines = len(original.strip().split("\n"))
        new_lines = len(modified.strip().split("\n"))

        if orig_lines == 0:
            return CheckResult("line_count", "passed", "Original was empty")

        delta_pct = abs(new_lines - orig_lines) / orig_lines * 100

        if delta_pct > self.max_line_delta_pct:
            direction = "+" if new_lines > orig_lines else "-"
            return CheckResult(
                "line_count",
                "FAILED",
                f"{direction}{delta_pct:.0f}% lines ({orig_lines}→{new_lines}), "
                f"threshold {self.max_line_delta_pct}%",
            )
        return CheckResult(
            "line_count",
            "passed",
            f"{delta_pct:.1f}% delta ({orig_lines}→{new_lines})",
        )

    def _check_class_preservation(
        self, filepath: str, original: str, modified: str
    ) -> CheckResult:
        """Check all original classes still exist in modified file."""
        orig_classes = self._get_class_names(original)
        new_classes = self._get_class_names(modified)
        missing = orig_classes - new_classes

        if missing:
            return CheckResult(
                "class_preservation",
                "FAILED",
                f"Missing classes: {', '.join(sorted(missing))}",
            )
        return CheckResult(
            "class_preservation",
            "passed",
            f"All {len(orig_classes)} classes preserved",
        )

    def _check_method_preservation(
        self, filepath: str, original: str, modified: str
    ) -> CheckResult:
        """Check all original public methods still exist in modified file."""
        orig_methods = self._get_public_methods(original)
        new_methods = self._get_public_methods(modified)
        missing = orig_methods - new_methods

        if missing:
            return CheckResult(
                "method_preservation",
                "FAILED",
                f"Missing methods: {', '.join(sorted(missing))}",
            )
        return CheckResult(
            "method_preservation",
            "passed",
            f"All {len(orig_methods)} public methods preserved",
        )

    def _check_import_preservation(
        self, filepath: str, original: str, modified: str
    ) -> CheckResult:
        """Check all original imports are still present."""
        orig_imports = self._get_imports(original)
        new_imports = self._get_imports(modified)
        missing = orig_imports - new_imports

        if missing:
            # Import loss is a warning, not a hard failure
            return CheckResult(
                "import_preservation",
                "WARNING",
                f"Missing imports: {', '.join(sorted(missing))}",
            )
        return CheckResult(
            "import_preservation", "passed", "All imports preserved"
        )

    def _check_ascend_keywords(
        self, filepath: str, original: str, modified: str
    ) -> CheckResult:
        """Check Ascend-specific keywords are preserved."""
        lost_keywords = []
        for keyword in ASCEND_KEYWORDS:
            orig_count = original.count(keyword)
            new_count = modified.count(keyword)
            if orig_count > 0 and new_count < orig_count:
                lost_keywords.append(
                    f"{keyword} ({orig_count}→{new_count})"
                )

        if lost_keywords:
            return CheckResult(
                "ascend_keywords",
                "FAILED",
                f"Lost Ascend keywords: {', '.join(lost_keywords)}",
            )
        return CheckResult(
            "ascend_keywords", "passed", "Ascend keywords preserved"
        )

    # --- Helper methods ---

    @staticmethod
    def _get_class_names(source: str) -> set[str]:
        """Extract all class names from source using AST."""
        try:
            tree = ast.parse(source)
            return {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            }
        except SyntaxError:
            return set()

    @staticmethod
    def _get_public_methods(source: str) -> set[str]:
        """Extract all public method names (class methods, not starting with _) using AST."""
        methods = set()
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not item.name.startswith("_"):
                                methods.add(f"{node.name}.{item.name}")
        except SyntaxError:
            pass
        return methods

    @staticmethod
    def _get_imports(source: str) -> set[str]:
        """Extract all import statements as normalized strings."""
        imports = set()
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(a.name for a in node.names)
                    imports.add(f"from {module} import {names}")
        except SyntaxError:
            pass
        return imports
