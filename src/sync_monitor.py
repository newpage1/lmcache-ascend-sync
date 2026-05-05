"""
LMCache-Ascend Upstream Sync Monitor
=====================================

Monitors upstream LMCache releases, detects breaking changes for LMCache-Ascend,
and auto-generates adaptation PRs.

Usage:
    python -m src.sync_monitor                    # Full pipeline
    python -m src.sync_monitor --check-only       # Only check for new releases
    python -m src.sync_monitor --analyze <version> # Analyze specific version
    python -m src.sync_monitor --local             # Run locally (no GitHub Actions)
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lmcache-sync")


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_patch_points(points_path: Optional[str] = None) -> dict:
    """Load patch points configuration."""
    if points_path is None:
        points_path = Path(__file__).parent.parent / "config" / "patch_points.yaml"
    points_path = Path(points_path)
    if not points_path.exists():
        logger.error(f"Patch points file not found: {points_path}")
        sys.exit(1)
    with open(points_path) as f:
        return yaml.safe_load(f)


class ReleaseChecker:
    """Check upstream LMCache for new releases."""

    def __init__(self, config: dict):
        self.config = config
        self.upstream = config["upstream"]
        self.api_base = "https://api.github.com"

    def get_latest_release(self) -> Optional[dict]:
        """Get the latest release from upstream."""
        url = f"{self.api_base}/repos/{self.upstream['repo']}/releases/latest"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Failed to fetch latest release: {resp.status_code}")
            return None
        return resp.json()

    def get_all_releases(self) -> list[dict]:
        """Get all releases from upstream."""
        url = f"{self.api_base}/repos/{self.upstream['repo']}/releases"
        releases = []
        page = 1
        while True:
            resp = requests.get(url, params={"per_page": 100, "page": page}, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            releases.extend(data)
            page += 1
        return releases

    def get_new_releases(self, last_known_version: str) -> list[dict]:
        """Get releases newer than last_known_version."""
        all_releases = self.get_all_releases()
        import re

        new_releases = []
        found_known = False
        for release in all_releases:
            tag = release.get("tag_name", "")
            if tag == last_known_version:
                found_known = True
                break
            # Check tag pattern
            for pattern in self.upstream.get("tag_patterns", []):
                if re.match(pattern, tag):
                    new_releases.append(release)
                    break
            # Also stop if we hit the min_version
            if tag == self.upstream.get("min_version"):
                break
        return new_releases

    def get_downstream_current_version(self) -> Optional[str]:
        """Get the current LMCache version tracked by LMCache-Ascend.

        Reads LMCACHE_UPSTREAM_TAG from lmcache_ascend/__init__.py on the
        downstream main branch.  Falls back to reading from a local checkout
        or the config min_version.
        """
        # Try GitHub API first (no checkout needed)
        import base64
        import re

        repo = self.config["downstream"]["repo"]
        target_branch = self.config["downstream"]["target_branch"]
        for file_path in ["lmcache_ascend/__init__.py", "lmcache_ascend/_version.py"]:
            url = (
                f"{self.api_base}/repos/{repo}/contents/{file_path}"
                f"?ref={target_branch}"
            )
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    content = base64.b64decode(resp.json()["content"]).decode()
                    match = re.search(
                        r'LMCACHE_UPSTREAM_TAG\s*=\s*["\']?(v[\d.]+)["\']?',
                        content,
                    )
                    if match:
                        return match.group(1)
            except Exception:
                pass

        # Try local checkout
        downstream_path = Path(self.config["sync"]["downstream_checkout"])
        init_file = downstream_path / "lmcache_ascend" / "__init__.py"
        if init_file.exists():
            try:
                content = init_file.read_text()
                match = re.search(
                    r'LMCACHE_UPSTREAM_TAG\s*=\s*["\']?(v[\d.]+)["\']?',
                    content,
                )
                if match:
                    return match.group(1)
            except Exception:
                pass

        return self.upstream.get("min_version")

    def get_state_file_path(self) -> Path:
        """Path to the state file tracking last processed version."""
        return Path(__file__).parent.parent / "state.json"

    def get_last_processed_version(self) -> Optional[str]:
        """Read the last processed upstream version from state file."""
        state_path = self.get_state_file_path()
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                return state.get("last_processed_version")
        return None

    def save_state(self, version: str, status: str, details: dict = None):
        """Save processing state."""
        state = {
            "last_processed_version": version,
            "last_status": status,
            "last_run": __import__("datetime").datetime.utcnow().isoformat(),
        }
        if details:
            state["details"] = details
        state_path = self.get_state_file_path()
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)


class ConflictAnalyzer:
    """Analyze upstream changes for conflicts with LMCache-Ascend."""

    def __init__(self, config: dict, patch_points: dict):
        self.config = config
        self.patch_points = patch_points

    def clone_repos(self, upstream_ref: str) -> bool:
        """Clone upstream and downstream repos for analysis."""
        sync_paths = self.config["sync"]
        upstream_path = Path(sync_paths["upstream_checkout"])
        downstream_path = Path(sync_paths["downstream_checkout"])

        # Clone upstream
        if not self._clone_and_checkout(
            self.config["upstream"]["repo"],
            upstream_path,
            upstream_ref,
        ):
            return False

        # Clone downstream
        if not self._clone_and_checkout(
            self.config["downstream"]["repo"],
            downstream_path,
            self.config["downstream"]["target_branch"],
        ):
            return False

        return True

    def _clone_and_checkout(
        self, repo: str, path: Path, ref: str = None
    ) -> bool:
        """Clone a repo and checkout a specific ref."""
        url = f"https://github.com/{repo}.git"
        if path.exists():
            subprocess.run(["rm", "-rf", str(path)], check=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", url, str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to clone {repo}: {result.stderr}")
            return False
        if ref:
            checkout_cmd = ["git", "-C", str(path), "checkout", ref]
            result = subprocess.run(checkout_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to checkout {ref}: {result.stderr}")
        return True

    def get_upstream_diff(self, from_version: str, to_version: str) -> str:
        """Get diff between two upstream versions for watched files.

        Returns the full git diff for files that match patch points,
        plus a stat summary for all changed files.
        """
        upstream_path = Path(self.config["sync"]["upstream_checkout"])
        watched_files = self._get_watched_files()

        # Get full stat summary of all changes
        stat_cmd = [
            "git", "-C", str(upstream_path),
            "diff", "--stat", f"{from_version}..{to_version}",
        ]
        stat_result = subprocess.run(stat_cmd, capture_output=True, text=True)
        stat_summary = stat_result.stdout.strip()

        # Get detailed diffs for watched files
        diffs = []
        for file_path in watched_files:
            cmd = [
                "git", "-C", str(upstream_path),
                "diff", f"{from_version}..{to_version}", "--", file_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout.strip():
                diffs.append(f"=== {file_path} ===\n{result.stdout}")

        # Also check for new files added in to_version that match watched patterns
        diff_names_cmd = [
            "git", "-C", str(upstream_path),
            "diff", "--name-status", f"{from_version}..{to_version}",
        ]
        names_result = subprocess.run(diff_names_cmd, capture_output=True, text=True)
        new_files = []
        for line in names_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] == "A":
                new_files.append(parts[1])

        header = f"## Full Change Stat\n{stat_summary}\n"
        if new_files:
            header += f"\n## New Files Added\n" + "\n".join(f"- {f}" for f in new_files)

        if diffs:
            return header + "\n\n## Detailed Diffs (watched files)\n" + "\n\n".join(diffs)
        return header

    def _get_watched_files(self) -> list[str]:
        """Extract list of upstream files to watch from patch_points config."""
        files = set()
        for point in self.patch_points.get("patch_points", []):
            module = point["module"]
            # Convert module path to file path
            if module.endswith(".py"):
                files.add(module)
            elif module.endswith("/"):
                files.add(module)
            else:
                # module like lmcache.v1.cache_engine -> lmcache/v1/cache_engine.py
                file_path = module.replace(".", "/") + ".py"
                files.add(file_path)
        return sorted(files)

    def analyze_changes(self, from_version: str, to_version: str) -> dict:
        """Analyze upstream changes for breaking changes on patch points.

        Returns a dict with:
        - has_breaking_changes: bool
        - conflicts: list of conflict details
        - diff_summary: str
        """
        diff = self.get_upstream_diff(from_version, to_version)
        if not diff.strip():
            return {
                "has_breaking_changes": False,
                "conflicts": [],
                "diff_summary": "No changes detected in watched files.",
            }

        conflicts = self._detect_conflicts(diff, to_version)
        return {
            "has_breaking_changes": len(conflicts) > 0,
            "conflicts": conflicts,
            "diff_summary": diff[:50000],  # Allow larger diffs for better LLM context
        }

    def _detect_conflicts(self, diff: str, to_version: str) -> list[dict]:
        """Detect conflicts by matching diff hunks against patch points."""
        conflicts = []
        for point in self.patch_points.get("patch_points", []):
            module = point["module"]
            module_file = module if module.endswith(".py") else module.replace(".", "/") + ".py"
            # Check if diff contains changes to this module
            if module_file not in diff and not module.endswith("/"):
                continue

            # Check specific watched items
            if point["type"] == "class_inheritance":
                for cls in point.get("watch_classes", []):
                    class_name = cls["class"]
                    for method in cls.get("watch_methods", []):
                        # Look for method signature changes in diff
                        pattern = f"def {method}"
                        if pattern in diff:
                            # Check if the hunk is near the class
                            if class_name in diff:
                                conflicts.append({
                                    "type": "method_signature_change",
                                    "module": module,
                                    "class": class_name,
                                    "method": method,
                                    "severity": "high",
                                    "description": (
                                        f"Method {class_name}.{method} in {module} "
                                        f"may have changed in {to_version}"
                                    ),
                                })

            elif point["type"] == "function_replace":
                for func in point.get("watch_functions", []):
                    if f"def {func}" in diff:
                        conflicts.append({
                            "type": "function_change",
                            "module": module,
                            "function": func,
                            "severity": "high",
                            "description": (
                                f"Function {func} in {module} "
                                f"may have changed in {to_version}"
                            ),
                        })

            elif point["type"] == "sys.modules_replace":
                for func in point.get("key_functions", []):
                    if func in diff:
                        conflicts.append({
                            "type": "kernel_interface_change",
                            "module": module,
                            "function": func,
                            "severity": "critical",
                            "description": (
                                f"Kernel function {func} in {module} "
                                f"may have changed in {to_version}"
                            ),
                        })

        return conflicts

    def simulate_rebase(self, upstream_version: str) -> dict:
        """Simulate rebase of LMCache-Ascend onto new upstream version.

        Returns rebase result with conflict details.
        """
        downstream_path = Path(self.config["sync"]["downstream_checkout"])
        upstream_path = Path(self.config["sync"]["upstream_checkout"])

        # Get list of files that LMCache-Ascend modifies
        ascend_files = []
        for root, dirs, files in os.walk(downstream_path / "lmcache_ascend"):
            for f in files:
                if f.endswith(".py"):
                    rel = os.path.relpath(
                        os.path.join(root, f), downstream_path
                    )
                    ascend_files.append(rel)

        # Try git rebase --onto to detect conflicts
        # Create a test branch
        branch_name = f"sync-test-{upstream_version}"
        subprocess.run(
            ["git", "-C", str(downstream_path), "checkout", "-b", branch_name],
            capture_output=True,
        )

        result = subprocess.run(
            [
                "git", "-C", str(downstream_path),
                "rebase", "--no-commit", upstream_version,
            ],
            capture_output=True,
            text=True,
        )

        rebase_conflicts = []
        if result.returncode != 0:
            # Parse conflict files
            status_result = subprocess.run(
                ["git", "-C", str(downstream_path), "status", "--porcelain"],
                capture_output=True,
                text=True,
            )
            for line in status_result.stdout.splitlines():
                if line.startswith("UU") or line.startswith("AA"):
                    rebase_conflicts.append(line[3:].strip())

            # Abort rebase
            subprocess.run(
                ["git", "-C", str(downstream_path), "rebase", "--abort"],
                capture_output=True,
            )

        # Cleanup
        subprocess.run(
            ["git", "-C", str(downstream_path), "checkout", self.config["downstream"]["target_branch"]],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(downstream_path), "branch", "-D", branch_name],
            capture_output=True,
        )

        return {
            "has_conflicts": len(rebase_conflicts) > 0,
            "conflict_files": rebase_conflicts,
            "rebase_output": result.stdout + result.stderr,
        }


def main():
    parser = argparse.ArgumentParser(description="LMCache-Ascend Upstream Sync Monitor")
    parser.add_argument("--check-only", action="store_true", help="Only check for new releases")
    parser.add_argument("--analyze", type=str, help="Analyze a specific upstream version")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--patch-points", type=str, help="Path to patch points file")
    parser.add_argument("--local", action="store_true", help="Run locally (not in GitHub Actions)")
    args = parser.parse_args()

    config = load_config(args.config)
    patch_points = load_patch_points(args.patch_points)

    checker = ReleaseChecker(config)
    analyzer = ConflictAnalyzer(config, patch_points)

    if args.check_only:
        latest = checker.get_latest_release()
        if latest:
            logger.info(f"Latest upstream release: {latest['tag_name']}")
            last_processed = checker.get_last_processed_version()
            if last_processed:
                logger.info(f"Last processed: {last_processed}")
                if latest["tag_name"] == last_processed:
                    logger.info("Already up to date.")
                else:
                    logger.info(f"New release available: {latest['tag_name']}")
            else:
                logger.info(f"No previous state found. Latest: {latest['tag_name']}")
        return

    if args.analyze:
        target_version = args.analyze
        logger.info(f"Analyzing upstream version: {target_version}")
    else:
        # Check for new releases
        latest = checker.get_latest_release()
        if not latest:
            logger.error("Could not fetch latest release")
            sys.exit(1)

        last_processed = checker.get_last_processed_version()
        target_version = latest["tag_name"]

        if last_processed and target_version == last_processed:
            logger.info(f"Already processed {target_version}. Nothing to do.")
            return

    # Clone repos
    logger.info("Cloning repositories...")
    if not analyzer.clone_repos(target_version):
        logger.error("Failed to clone repositories")
        sys.exit(1)

    # Determine from version: prefer Ascend's current tracked version
    ascend_version = checker.get_downstream_current_version()
    last_processed = checker.get_last_processed_version() or ascend_version or config["upstream"]["min_version"]
    logger.info(f"Ascend current version: {ascend_version}")
    logger.info(f"Comparing {last_processed} -> {target_version}")

    # Phase 1: Diff analysis
    logger.info("Running conflict analysis...")
    analysis = analyzer.analyze_changes(last_processed, target_version)
    logger.info(f"Breaking changes detected: {analysis['has_breaking_changes']}")
    if analysis["conflicts"]:
        for c in analysis["conflicts"]:
            logger.warning(f"  [{c['severity']}] {c['description']}")

    # Phase 2: Rebase simulation
    logger.info("Running rebase simulation...")
    rebase_result = analyzer.simulate_rebase(target_version)
    logger.info(f"Rebase conflicts: {len(rebase_result.get('conflict_files', []))}")

    if not analysis["has_breaking_changes"] and not rebase_result["has_conflicts"]:
        logger.info("No breaking changes detected. Updating state.")
        checker.save_state(target_version, "no_changes")
        return

    # Phase 3: Generate adaptation PR (invoke PR generator)
    logger.info("Breaking changes detected. Invoking PR generator...")
    from src.pr_generator import PRGenerator

    pr_gen = PRGenerator(config, patch_points)
    result = pr_gen.generate(
        from_version=last_processed,
        to_version=target_version,
        analysis=analysis,
        rebase_result=rebase_result,
    )

    if result.get("success"):
        checker.save_state(
            target_version,
            "pr_created",
            {"pr_url": result.get("pr_url"), "pr_number": result.get("pr_number")},
        )
        logger.info(f"PR created: {result.get('pr_url')}")
    else:
        checker.save_state(target_version, "failed", {"error": result.get("error")})
        logger.error(f"Failed to create PR: {result.get('error')}")


if __name__ == "__main__":
    main()
