"""Tests for sync_monitor module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
import pytest

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sync_monitor import ReleaseChecker, ConflictAnalyzer, load_config, load_patch_points


@pytest.fixture
def config():
    """Load test config."""
    return {
        "upstream": {
            "repo": "LMCache/LMCache",
            "tag_patterns": ["^v0\\..*$"],
            "schedule": "0 6 * * *",
            "min_version": "v0.3.7",
        },
        "downstream": {
            "repo": "LMCache/LMCache-Ascend",
            "target_branch": "main",
            "branch_prefix": "sync/upstream-",
        },
        "sync": {
            "upstream_checkout": "/tmp/lmcache-sync-test/upstream",
            "downstream_checkout": "/tmp/lmcache-sync-test/downstream",
            "rebase": {"enabled": True, "max_conflicts": 10},
            "pr": {
                "labels": ["automated-sync"],
                "draft": False,
                "title_template": "chore: adapt to upstream LMCache {version}",
            },
        },
        "claude": {"model": "claude-sonnet-4-20250514", "max_tokens": 8192},
    }


@pytest.fixture
def patch_points():
    """Load test patch points."""
    return {
        "patch_points": [
            {
                "module": "lmcache/v1/gpu_connector/gpu_connectors.py",
                "type": "class_inheritance",
                "watch_classes": [
                    {
                        "class": "GPUConnectorInterface",
                        "watch_methods": ["to_gpu", "from_gpu"],
                    }
                ],
            },
            {
                "module": "lmcache/v1/memory_management.py",
                "type": "class_inheritance",
                "watch_classes": [
                    {
                        "class": "MixedMemoryAllocator",
                        "watch_methods": ["allocate", "__init__"],
                    }
                ],
            },
            {
                "module": "lmcache/c_ops",
                "type": "sys.modules_replace",
                "key_functions": ["multi_layer_kv_transfer", "host_register"],
            },
        ]
    }


class TestReleaseChecker:
    def test_init(self, config):
        checker = ReleaseChecker(config)
        assert checker.config == config
        assert checker.upstream["repo"] == "LMCache/LMCache"

    @patch("src.sync_monitor.requests.get")
    def test_get_latest_release(self, mock_get, config):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tag_name": "v0.4.4", "name": "v0.4.4"}
        mock_get.return_value = mock_resp

        checker = ReleaseChecker(config)
        release = checker.get_latest_release()
        assert release["tag_name"] == "v0.4.4"

    def test_state_file_roundtrip(self, config):
        checker = ReleaseChecker(config)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_path = Path(f.name)

        with patch.object(checker, "get_state_file_path", return_value=state_path):
            checker.save_state("v0.4.4", "pr_created", {"pr_url": "https://github.com/..."})
            version = checker.get_last_processed_version()
            assert version == "v0.4.4"

        os.unlink(state_path)


class TestConflictAnalyzer:
    def test_get_watched_files(self, config, patch_points):
        analyzer = ConflictAnalyzer(config, patch_points)
        files = analyzer._get_watched_files()
        assert "lmcache/v1/gpu_connector/gpu_connectors.py" in files
        assert "lmcache/v1/memory_management.py" in files
        assert "lmcache/c_ops.py" in files

    def test_detect_class_method_conflict(self, config, patch_points):
        analyzer = ConflictAnalyzer(config, patch_points)
        diff = """
diff --git a/lmcache/v1/gpu_connector/gpu_connectors.py b/lmcache/v1/gpu_connector/gpu_connectors.py
class GPUConnectorInterface:
-    def to_gpu(self, memory_obj, start, end, **kwargs):
+    def to_gpu(self, memory_obj, start, end, new_param=None, **kwargs):
"""
        conflicts = analyzer._detect_conflicts(diff, "v0.4.4")
        assert len(conflicts) > 0
        assert conflicts[0]["type"] == "method_signature_change"
        assert conflicts[0]["class"] == "GPUConnectorInterface"
        assert conflicts[0]["method"] == "to_gpu"

    def test_detect_kernel_conflict(self, config, patch_points):
        analyzer = ConflictAnalyzer(config, patch_points)
        diff = """
diff --git a/lmcache/csrc/ops.cpp b/lmcache/csrc/ops.cpp
 def multi_layer_kv_transfer(
"""
        conflicts = analyzer._detect_conflicts(diff, "v0.4.4")
        assert len(conflicts) > 0
        assert conflicts[0]["severity"] == "critical"

    def test_no_conflict_when_no_changes(self, config, patch_points):
        analyzer = ConflictAnalyzer(config, patch_points)
        analysis = analyzer.analyze_changes("v0.3.7", "v0.3.8")
        assert analysis["has_breaking_changes"] is False
        assert analysis["conflicts"] == []


class TestConfigLoading:
    def test_load_config(self):
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            assert "upstream" in config
            assert config["upstream"]["repo"] == "LMCache/LMCache"

    def test_load_patch_points(self):
        points_path = Path(__file__).parent.parent / "config" / "patch_points.yaml"
        if points_path.exists():
            points = load_patch_points(str(points_path))
            assert "patch_points" in points
            assert len(points["patch_points"]) > 0
