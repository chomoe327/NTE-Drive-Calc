# 盲筛标记规则、会话、评分与日志单元测试。
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.features.discard.log_store import MarkLogEntry, MarkLogSession, MarkLogStore
from src.features.discard.quality_rules import (
    QualityMarkRule,
    resolve_action,
    validate_rules,
)
from src.features.discard.scan_session import (
    InventoryChangedError,
    ScanSession,
    ScanSessionEntry,
    build_session_from_inventory,
    verify_signature,
)
from src.features.discard.scoring import build_mark_preview
from src.scanner.grid_navigation import generate_path_commands, moves_for_scan_index


def _drive(uid: str, scan_index: int, quality: str = "Gold") -> dict:
    return {
        "uid": uid,
        "item_type": "drive",
        "scan_index": scan_index,
        "quality": quality,
        "shape_id": "L1",
        "area": 1,
        "set_name": "测试套装",
        "main_stats": {"攻击力": 100.0, "暴击率": 10.0},
        "sub_stats": {"暴击率": 5.0},
    }


class MarkingQualityRulesTest(unittest.TestCase):
    def test_lock_must_exceed_discard(self):
        rule = QualityMarkRule("Gold", 18.0, 10.0, discard_below_enabled=True)
        self.assertIsNotNone(rule.validate())

    def test_middle_band_no_action(self):
        rules = {
            "Gold": QualityMarkRule("Gold", 10.0, 18.0, discard_below_enabled=True, lock_above_enabled=True),
        }
        self.assertEqual(resolve_action("Gold", 9.9, rules), "discard")
        self.assertIsNone(resolve_action("Gold", 14.0, rules))
        self.assertEqual(resolve_action("Gold", 18.0, rules), "lock")

    def test_blue_not_in_rules(self):
        rules = {"Gold": QualityMarkRule("Gold", 10.0, 18.0, discard_below_enabled=True)}
        self.assertIsNone(resolve_action("Blue", 0.0, rules))

    def test_independent_switches(self):
        discard_only = {"Purple": QualityMarkRule("Purple", 8.0, 15.0, discard_below_enabled=True)}
        lock_only = {"Purple": QualityMarkRule("Purple", 8.0, 15.0, lock_above_enabled=True)}
        self.assertEqual(resolve_action("Purple", 7.0, discard_only), "discard")
        self.assertIsNone(resolve_action("Purple", 7.0, lock_only))
        self.assertEqual(resolve_action("Purple", 16.0, lock_only), "lock")

    def test_validate_requires_enabled_rule(self):
        rules = {
            "Gold": QualityMarkRule("Gold", 10.0, 18.0),
            "Purple": QualityMarkRule("Purple", 8.0, 15.0),
        }
        self.assertIsNotNone(validate_rules(rules))


class MarkingScanSessionTest(unittest.TestCase):
    def test_build_session_from_inventory(self):
        inventory = [_drive("a", 1), _drive("b", 2)]
        session = build_session_from_inventory(inventory)
        self.assertEqual(session.total_drives, 2)
        self.assertEqual(len(session.entries), 2)

    def test_missing_scan_index_blocks(self):
        bad = _drive("x", 1)
        bad.pop("scan_index")
        with self.assertRaises(ValueError):
            build_session_from_inventory([bad])

    def test_signature_mismatch_raises(self):
        with self.assertRaises(InventoryChangedError):
            verify_signature("a", "b")


class MarkingScoringTest(unittest.TestCase):
    def test_preview_classifies_targets(self):
        inventory = [
            _drive("low", 1, "Gold"),
            _drive("high", 2, "Gold"),
            _drive("blue", 3, "Blue"),
        ]
        session = build_session_from_inventory(inventory)
        rules = {
            "Gold": QualityMarkRule("Gold", 10.0, 18.0, discard_below_enabled=True, lock_above_enabled=True),
            "Purple": QualityMarkRule("Purple", 8.0, 15.0, lock_above_enabled=True),
        }
        engine = SimpleNamespace(roles_db={"角色A": {"weights": {"攻击力": 1.0}}})

        def fake_score(drive, weights, max_weight):
            if drive.uid == "low":
                return 5.0
            if drive.uid == "high":
                return 20.0
            return 12.0

        engine.calculate_drive_score = fake_score
        engine._get_max_theoretical_weight = lambda weights: 1.0

        preview = build_mark_preview(session, ["角色A"], rules, engine, inventory)
        self.assertEqual(preview.discard_count, 1)
        self.assertEqual(preview.lock_count, 1)
        self.assertEqual(preview.blue_skipped, 1)
        self.assertEqual([t.scan_index for t in preview.targets], [1, 2])


class MarkingLogStoreTest(unittest.TestCase):
    def test_rollback_candidates_reverse_marked_order(self):
        session = MarkLogSession(
            id="s1",
            scan_session_id="snap",
            created_at="t0",
            rules={},
            roles=[],
            entries=[
                MarkLogEntry(1, "a", "discard", "marked", "2020-01-01T00:00:01Z"),
                MarkLogEntry(2, "b", "lock", "marked", "2020-01-01T00:00:02Z"),
                MarkLogEntry(3, "c", "discard", "skipped_already_marked"),
            ],
        )
        candidates = MarkLogStore(Path("unused")).rollback_candidates(session)
        self.assertEqual([e.scan_index for e in candidates], [2, 1])

    def test_persist_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "marking_log.json"
            store = MarkLogStore(path)
            session = MarkLogSession(
                id="s1",
                scan_session_id="snap",
                created_at="t0",
                rules={"Gold": {}},
                roles=["角色A"],
                entries=[MarkLogEntry(1, "a", "discard", "marked", "t1")],
            )
            store.append_session(session)
            loaded = store.latest_session()
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.entries[0].scan_index, 1)


class GridNavigationTest(unittest.TestCase):
    def test_moves_for_scan_index_matches_path(self):
        paths = generate_path_commands(10)
        self.assertEqual(moves_for_scan_index(1, 10), paths[0])
        self.assertEqual(moves_for_scan_index(10, 10), paths[9])


class MarkingExecutorTest(unittest.TestCase):
    @patch("src.features.discard.executor.BatchProcessor")
    @patch("src.features.discard.executor.mss.mss")
    @patch("src.features.discard.executor.GamepadScanner")
    @patch("src.features.discard.executor.vg", create=True)
    def test_skips_already_marked(self, _vg, scanner_cls, _mss, _batch):
        from src.features.discard.executor import MarkingExecutor
        from src.features.discard.scoring import MarkTarget

        session = ScanSession(
            session_id="s",
            created_at="t",
            total_drives=1,
            cols=7,
            entries=[
                ScanSessionEntry(1, "a", "Gold", "drive", "sig", None),
            ],
        )
        scanner = MagicMock()
        scanner_cls.return_value = scanner
        executor = MarkingExecutor(
            session,
            template_dir=Path("config/templates/marking"),
            config_dir=Path("config"),
        )
        executor.detector.detect_from_bgr = MagicMock(return_value={"discard": True, "lock": False})
        executor._verify_current_drive = MagicMock()
        executor._capture_bgr = MagicMock(return_value=MagicMock(shape=(100, 100, 3)))
        targets = [MarkTarget(1, "a", "Gold", 5.0, "discard")]
        results = executor.execute_targets(targets, switch_delay=0)
        self.assertEqual(results[0].status, "skipped_already_marked")
        scanner._apply_moves.assert_called_once()


if __name__ == "__main__":
    unittest.main()
