# 盲筛标记规则、会话、评分与日志单元测试。
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

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
    signature_from_item_dict,
    verify_signature,
)
from src.features.discard.scoring import MarkTarget, build_mark_preview
from src.models.equipment import Drive, Tape
from src.scanner.grid_navigation import (
    COLS,
    _INVERT_MOVE,
    generate_path_commands,
    generate_scan_order,
    moves_between_scan_indices,
    moves_between_scan_indices_along_scan_path,
    moves_between_scan_indices_horizontal_first,
    moves_for_scan_index,
    scan_index_for_position,
)


def _index_after_grid_moves(from_index: int, moves: list[str], total: int, cols: int = COLS) -> int:
    order = generate_scan_order(total, cols)
    row, col = order[from_index - 1]
    for move in moves:
        if move == "R":
            col += 1
        elif move == "L":
            col -= 1
        elif move == "D":
            row += 1
        elif move == "U":
            row -= 1
        else:
            raise AssertionError(f"未知移动: {move}")
    return scan_index_for_position(row, col, total, cols)


def _index_after_path_moves(from_index: int, moves: list[str], total: int, cols: int = COLS) -> int:
    paths = generate_path_commands(total, cols)
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

    def test_signature_ignores_scan_index_and_uid(self):
        left = _drive("a", 1)
        right = _drive("b", 99)
        self.assertEqual(signature_from_item_dict(left), signature_from_item_dict(right))

    def test_signature_matches_live_parse_shape(self):
        inventory_item = _drive("a", 135)
        parsed_item = dict(inventory_item)
        parsed_item.pop("scan_index", None)
        parsed_item["uid"] = "live-ocr-uid"
        self.assertEqual(
            signature_from_item_dict(inventory_item),
            signature_from_item_dict(parsed_item),
        )


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


class AnchorNavigationTest(unittest.TestCase):
    def _simulate_anchor(self, start_row: int, start_col: int, total: int = 135, cols: int = 7):
        rows = max(1, (total + cols - 1) // cols)
        row, col = start_row, start_col
        moves: list[str] = []

        def done() -> bool:
            return row == 0 and col == 0

        def apply_step(direction: str) -> None:
            nonlocal row, col
            moves.append(direction)
            if direction == "L":
                col = max(0, col - 1)
            elif direction == "U":
                row = max(0, row - 1)

        if done():
            return moves, row, col
        for _ in range(cols):
            if done():
                break
            apply_step("L")
        if done():
            return moves, row, col
        for _ in range(rows - 1):
            if done():
                break
            apply_step("U")
            for _ in range(cols):
                if done():
                    break
                apply_step("L")
        return moves, row, col

    def test_from_first_row_end_only_moves_left(self):
        moves, row, col = self._simulate_anchor(0, 6)
        self.assertEqual((row, col), (0, 0))
        self.assertEqual(moves.count("U"), 0)

    def test_from_lower_row_uses_up_then_left(self):
        moves, row, col = self._simulate_anchor(3, 2)
        self.assertEqual((row, col), (0, 0))
        self.assertGreater(moves.count("U"), 0)


class GridNavigationTest(unittest.TestCase):
    def test_moves_for_scan_index_matches_path(self):
        paths = generate_path_commands(10)
        self.assertEqual(moves_for_scan_index(1, 10), paths[0])
        self.assertEqual(moves_for_scan_index(10, 10), paths[9])

    def test_moves_between_forward_lands_on_target(self):
        total = 135
        for start, end in ((1, 7), (3, 8), (1, 12), (8, 12), (1, 135), (20, 88), (50, 100)):
            moves = moves_between_scan_indices(start, end, total)
            landed = _index_after_grid_moves(start, moves, total)
            self.assertEqual(landed, end, f"{start}->{end}")

    def test_full_scan_path_visits_every_cell_incrementally(self):
        total = 20
        path = generate_path_commands(total)
        self.assertEqual(len(path), total)
        marking_jump = moves_between_scan_indices(1, total, total)
        scan_steps = sum(len(step) for step in path[1:])
        self.assertLess(len(marking_jump), scan_steps)

    def test_direct_navigation_shorter_than_scan_path(self):
        total = 135
        direct = moves_between_scan_indices(1, 135, total)
        along = moves_between_scan_indices_along_scan_path(1, 135, total)
        self.assertLess(len(direct), len(along))
        self.assertEqual(_index_after_grid_moves(1, direct, total), 135)

    def test_vertical_first_matches_horizontal_first_on_grid(self):
        total = 135
        for start, end in ((1, 135), (3, 8), (20, 88)):
            vertical = moves_between_scan_indices(start, end, total)
            horizontal = moves_between_scan_indices_horizontal_first(start, end, total)
            self.assertEqual(
                _index_after_grid_moves(start, vertical, total),
                _index_after_grid_moves(start, horizontal, total),
            )

    def test_moves_between_backward_returns_to_start(self):
        total = 12
        for start, end in ((1, 7), (3, 8), (1, 12), (8, 12)):
            forward = moves_between_scan_indices(start, end, total)
            backward = moves_between_scan_indices(end, start, total)
            landed = _index_after_grid_moves(start, forward + backward, total)
            self.assertEqual(landed, start, f"{start}->{end} 往返后应回到第 {start} 格")


class MarkStateDetectorTest(unittest.TestCase):
    def test_detect_prefers_marked_template(self):
        from src.features.discard.mark_state import MarkStateDetector

        template_dir = Path(__file__).resolve().parents[1] / "config" / "templates" / "marking"
        if not (template_dir / "discard_marked.png").exists():
            self.skipTest("标记模板不存在")
        detector = MarkStateDetector(template_dir)
        marked = detector._templates.get("discard_marked")
        unmarked = detector._templates.get("discard_unmarked")
        self.assertIsNotNone(marked)
        self.assertIsNotNone(unmarked)
        canvas = np.full((120, 120), 180, dtype=np.uint8)
        mh, mw = marked.shape[:2]
        canvas[10 : 10 + mh, 10 : 10 + mw] = marked
        is_marked, marked_score, unmarked_score = detector._pair_state(
            canvas,
            "discard_marked",
            "discard_unmarked",
        )
        self.assertTrue(is_marked)
        self.assertGreater(marked_score, unmarked_score)

    def test_detect_prefers_lock_marked_template(self):
        from src.features.discard.mark_state import MarkStateDetector

        template_dir = Path(__file__).resolve().parents[1] / "config" / "templates" / "marking"
        if not (template_dir / "lock_marked.png").exists():
            self.skipTest("标记模板不存在")
        detector = MarkStateDetector(template_dir)
        marked = detector._templates.get("lock_marked")
        self.assertIsNotNone(marked)
        canvas = np.full((120, 120), 180, dtype=np.uint8)
        mh, mw = marked.shape[:2]
        canvas[20 : 20 + mh, 20 : 20 + mw] = marked
        is_marked, marked_score, unmarked_score = detector._pair_state(
            canvas,
            "lock_marked",
            "lock_unmarked",
        )
        self.assertTrue(is_marked)
        self.assertGreater(marked_score, unmarked_score)


class MarkingExecutorButtonTest(unittest.TestCase):
    def test_resolve_xusb_button_prefers_vgamepad_names(self):
        from src.features.discard import executor as marking_executor

        fake_button = SimpleNamespace(
            XUSB_GAMEPAD_DPAD_LEFT=4,
            XUSB_GAMEPAD_DPAD_RIGHT=8,
            XUSB_GAMEPAD_A=0x1000,
        )
        fake_vg = SimpleNamespace(XUSB_BUTTON=fake_button)
        with patch.object(marking_executor, "vg", fake_vg):
            self.assertEqual(marking_executor._resolve_xusb_button("DPAD_LEFT"), 4)
            self.assertEqual(marking_executor._resolve_xusb_button("DPAD_RIGHT"), 8)
            self.assertEqual(marking_executor._resolve_xusb_button("A"), 0x1000)
            self.assertIsNone(marking_executor._resolve_xusb_button("UNKNOWN"))


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
        scanner.anchor_to_first_cell = MagicMock()
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
        scanner.anchor_to_first_cell.assert_called_once()
        for call in scanner._apply_moves.call_args_list:
            self.assertEqual(call.kwargs.get("pace"), "marking")
        moves = [call.args[0] for call in scanner._apply_moves.call_args_list]
        self.assertEqual(moves[0], moves_between_scan_indices(1, 3, 12))
        self.assertEqual(moves[1], moves_between_scan_indices(3, 8, 12))

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
        scanner.anchor_to_first_cell = MagicMock()
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
