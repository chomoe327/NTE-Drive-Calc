# 盲筛标记规则、会话、评分与日志单元测试。
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.grade_scoring import (
    GRADE_LADDER,
    best_grade,
    evaluate_item_grades,
    grade_rank,
)
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
from src.features.discard.scoring import MarkTarget, build_mark_preview
from src.models.equipment import Drive, Tape
from src.scanner.grid_navigation import (
    _INVERT_MOVE,
    generate_path_commands,
    moves_between_scan_indices,
    moves_for_scan_index,
)


def _index_after_moves(from_index: int, moves: list[str], total: int) -> int:
    paths = generate_path_commands(total)
    index = from_index
    offset = 0
    while offset < len(moves):
        if index < total:
            forward = paths[index]
            if moves[offset : offset + len(forward)] == forward:
                offset += len(forward)
                index += 1
                continue
        if index > 1:
            backward = [_INVERT_MOVE[move] for move in reversed(paths[index - 1])]
            if moves[offset : offset + len(backward)] == backward:
                offset += len(backward)
                index -= 1
                continue
        raise AssertionError(f"无法解析移动序列，停在第 {index} 格，剩余 {moves[offset:]}")
    return index


def _drive(uid: str, scan_index: int, quality: str = "Gold", shape_id: str = "L1") -> dict:
    return {
        "uid": uid,
        "item_type": "drive",
        "scan_index": scan_index,
        "quality": quality,
        "shape_id": shape_id,
        "area": 1,
        "set_name": "测试套装",
        "main_stats": {"攻击力": 100.0, "暴击率": 10.0},
        "sub_stats": {"暴击率": 5.0},
    }


def _tape(uid: str, scan_index: int, quality: str = "Gold", set_name: str = "测试套装") -> dict:
    return {
        "uid": uid,
        "item_type": "tape",
        "scan_index": scan_index,
        "quality": quality,
        "shape_id": "TAPE_15",
        "area": 15,
        "set_name": set_name,
        "main_stats": "攻击力%",
        "sub_stats": {"攻击力": 10.0},
    }


class GradeScoringTest(unittest.TestCase):
    def test_grade_rank_order(self):
        self.assertLess(grade_rank("B"), grade_rank("SS"))

    def test_best_grade(self):
        self.assertEqual(best_grade(["B", "A", "SS"]), "SS")


class MarkingQualityRulesTest(unittest.TestCase):
    def test_lock_must_exceed_discard_grade(self):
        rule = QualityMarkRule("Gold", "SS", "B", discard_below_enabled=True)
        self.assertIsNotNone(rule.validate())

    def test_middle_band_no_action(self):
        rules = {
            "Gold": QualityMarkRule("Gold", "B", "SS", discard_below_enabled=True, lock_above_enabled=True),
        }
        self.assertEqual(resolve_action("Gold", "A", rules), None)
        self.assertEqual(resolve_action("Gold", "C", rules), "discard")
        self.assertEqual(resolve_action("Gold", "SS", rules), "lock")

    def test_blue_not_in_rules(self):
        rules = {"Gold": QualityMarkRule("Gold", "B", "SS", discard_below_enabled=True)}
        self.assertIsNone(resolve_action("Blue", "D", rules))

    def test_independent_switches(self):
        discard_only = {"Purple": QualityMarkRule("Purple", "B", "SS", discard_below_enabled=True)}
        lock_only = {"Purple": QualityMarkRule("Purple", "B", "SS", lock_above_enabled=True)}
        self.assertEqual(resolve_action("Purple", "C", discard_only), "discard")
        self.assertIsNone(resolve_action("Purple", "A", lock_only))
        self.assertEqual(resolve_action("Purple", "SS", lock_only), "lock")

    def test_validate_requires_enabled_rule(self):
        rules = {
            "Gold": QualityMarkRule("Gold"),
            "Purple": QualityMarkRule("Purple"),
        }
        self.assertIsNotNone(validate_rules(rules))


class MarkingScanSessionTest(unittest.TestCase):
    def test_build_session_from_inventory(self):
        inventory = [_drive("a", 1), _tape("t", 2)]
        session = build_session_from_inventory(inventory)
        self.assertEqual(session.total_drives, 2)
        self.assertEqual(len(session.entries), 2)
        types = {e.item_type for e in session.entries}
        self.assertEqual(types, {"drive", "tape"})

    def test_missing_scan_index_blocks(self):
        bad = _drive("x", 1)
        bad.pop("scan_index")
        with self.assertRaises(ValueError):
            build_session_from_inventory([bad])

    def test_signature_mismatch_raises(self):
        with self.assertRaises(InventoryChangedError):
            verify_signature("a", "b")


class MarkingScoringTest(unittest.TestCase):
    def _mock_orchestrator(self):
        orchestrator = SimpleNamespace(
            roles_db={"角色A": {"default_set": "测试套装", "weights": {"攻击力": 1.0}}},
            sets_db={"测试套装": {"shapes": ["L1"]}},
        )
        orchestrator._resolve_set_name = lambda name: name
        blueprints = {"角色A": [{"extra_pieces": []}]}
        return orchestrator, blueprints

    def test_preview_classifies_by_grade(self):
        inventory = [
            _drive("low", 1, "Gold"),
            _drive("high", 2, "Gold"),
            _drive("blue", 3, "Blue"),
        ]
        session = build_session_from_inventory(inventory)
        rules = {
            "Gold": QualityMarkRule("Gold", "B", "SS", discard_below_enabled=True, lock_above_enabled=True),
            "Purple": QualityMarkRule("Purple", "B", "SS", lock_above_enabled=True),
        }
        orchestrator, blueprints = self._mock_orchestrator()
        engine = MagicMock()
        engine._get_max_theoretical_weight = lambda weights: 1.0

        def fake_grade(item, selected_roles, orch, bps, eng):
            if item.uid == "low":
                return "C", "角色A", True
            if item.uid == "high":
                return "SS", "角色A", True
            return "D", None, False

        with patch("src.features.discard.scoring.evaluate_item_grades", side_effect=fake_grade):
            preview = build_mark_preview(
                session,
                ["角色A"],
                rules,
                engine,
                inventory,
                orchestrator,
                blueprints,
            )
        self.assertEqual(preview.discard_count, 1)
        self.assertEqual(preview.lock_count, 1)
        self.assertEqual(preview.blue_skipped, 1)
        self.assertEqual([t.scan_index for t in preview.targets], [1, 2])

    def test_no_usable_role_treated_as_discard(self):
        inventory = [_drive("orphan", 1, "Gold", shape_id="UNKNOWN")]
        session = build_session_from_inventory(inventory)
        rules = {"Gold": QualityMarkRule("Gold", "B", "SS", discard_below_enabled=True)}
        orchestrator, blueprints = self._mock_orchestrator()
        engine = MagicMock()
        preview = build_mark_preview(
            session,
            ["角色A"],
            rules,
            engine,
            inventory,
            orchestrator,
            blueprints,
        )
        self.assertEqual(preview.no_usable_role, 1)
        self.assertEqual(preview.discard_count, 1)
        self.assertEqual(preview.targets[0].max_grade, "D")


class RoleMultiSelectorTest(unittest.TestCase):
    def test_select_all_visible_roles(self):
        from src.ui.widgets import match_pinyin

        all_roles = {"A": {}, "B": {}, "C": {}}
        query = ""
        names = sorted(all_roles.keys())
        if query:
            names = [name for name in names if match_pinyin(name, query)]
        selected: list[str] = []
        for name in names:
            if name not in selected:
                selected.append(name)
        self.assertEqual(sorted(selected), ["A", "B", "C"])


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


class GridNavigationTest(unittest.TestCase):
    def test_moves_for_scan_index_matches_path(self):
        paths = generate_path_commands(10)
        self.assertEqual(moves_for_scan_index(1, 10), paths[0])
        self.assertEqual(moves_for_scan_index(10, 10), paths[9])

    def test_moves_between_forward_matches_scan_segments(self):
        total = 12
        paths = generate_path_commands(total)
        expected: list[str] = []
        for step in range(1, 7):
            expected.extend(paths[step])
        self.assertEqual(moves_between_scan_indices(1, 7, total), expected)

    def test_moves_between_forward_compound(self):
        total = 12
        paths = generate_path_commands(total)
        expected: list[str] = []
        for step in range(3, 8):
            expected.extend(paths[step])
        self.assertEqual(moves_between_scan_indices(3, 8, total), expected)

    def test_moves_between_backward_returns_to_start(self):
        total = 12
        for start, end in ((1, 7), (3, 8), (1, 12), (8, 12)):
            forward = moves_between_scan_indices(start, end, total)
            backward = moves_between_scan_indices(end, start, total)
            landed = _index_after_moves(start, forward + backward, total)
            self.assertEqual(landed, start, f"{start}->{end} 往返后应回到第 {start} 格")


class MarkingExecutorTest(unittest.TestCase):
    @patch("src.features.discard.executor.BatchProcessor")
    @patch("src.features.discard.executor.mss.mss")
    @patch("src.features.discard.executor.GamepadScanner")
    @patch("src.features.discard.executor.vg", create=True)
    def test_navigates_targets_in_scan_order(self, _vg, scanner_cls, _mss, _batch):
        from src.features.discard.executor import MarkingExecutor

        session = ScanSession(
            session_id="s",
            created_at="t",
            total_drives=12,
            cols=7,
            entries=[
                ScanSessionEntry(i, f"u{i}", "Gold", "drive", f"sig{i}", None)
                for i in range(1, 13)
            ],
        )
        scanner = MagicMock()
        scanner.wait_for_handoff = MagicMock()
        scanner.wake_inventory_selection = MagicMock()
        scanner_cls.return_value = scanner
        executor = MarkingExecutor(
            session,
            template_dir=Path("config/templates/marking"),
            config_dir=Path("config"),
        )
        executor.detector.detect_from_bgr = MagicMock(return_value={"discard": False, "lock": False})
        executor._verify_current_drive = MagicMock()
        executor._capture_bgr = MagicMock(return_value=MagicMock(shape=(100, 100, 3)))
        targets = [
            MarkTarget(3, "u3", "drive", "Gold", "C", None, "discard"),
            MarkTarget(8, "u8", "drive", "Gold", "C", None, "lock"),
        ]
        executor.execute_targets(targets)
        applied = [call.args[0] for call in scanner._apply_moves.call_args_list]
        self.assertEqual(applied[0], moves_between_scan_indices(1, 3, 12))
        self.assertEqual(applied[1], moves_between_scan_indices(3, 8, 12))

    @patch("src.features.discard.executor.BatchProcessor")
    @patch("src.features.discard.executor.mss.mss")
    @patch("src.features.discard.executor.GamepadScanner")
    @patch("src.features.discard.executor.vg", create=True)
    def test_skips_already_marked(self, _vg, scanner_cls, _mss, _batch):
        from src.features.discard.executor import MarkingExecutor

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
        scanner.wait_for_handoff = MagicMock()
        scanner_cls.return_value = scanner
        executor = MarkingExecutor(
            session,
            template_dir=Path("config/templates/marking"),
            config_dir=Path("config"),
        )
        executor.detector.detect_from_bgr = MagicMock(return_value={"discard": True, "lock": False})
        executor._verify_current_drive = MagicMock()
        executor._capture_bgr = MagicMock(return_value=MagicMock(shape=(100, 100, 3)))
        targets = [
            MarkTarget(1, "a", "drive", "Gold", "C", None, "discard"),
        ]
        results = executor.execute_targets(targets)
        self.assertEqual(results[0].status, "skipped_already_marked")


if __name__ == "__main__":
    unittest.main()
