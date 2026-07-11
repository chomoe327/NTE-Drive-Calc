import unittest
from unittest.mock import MagicMock, patch

from src.scanner.assembly_vision import (
    compute_position_error,
    detect_dragged_piece_anchor,
    estimate_position_error_from_origin,
    needs_position_correction,
)
from src.scanner.gamepad_assembly import GamepadAssemblyController, InventoryDriveNotFoundError, StickMove
from src.scanner.shape_recognizer import GoldShapeRecognizer


class FakeGamepad:
    def __init__(self):
        self.left_stick = (0.0, 0.0)
        self.pressed = set()
        self.updates = 0

    def left_joystick_float(self, x_value_float=0.0, y_value_float=0.0):
        self.left_stick = (float(x_value_float), float(y_value_float))

    def press_button(self, button):
        self.pressed.add(button)

    def release_button(self, button):
        self.pressed.discard(button)

    def update(self):
        self.updates += 1


class GamepadAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "hold_a_seconds": 0.01,
            "post_hold_a_seconds": 0.01,
            "settle_seconds": 0.01,
            "stick_update_interval_seconds": 0.01,
            "inventory_nav": {
                "tap_seconds": 0.01,
                "settle_seconds": 0.01,
                "right_stick_x": 1.0,
                "down_stick_y": -1.0,
            },
            "inventory_filter": {
                "enabled": True,
                "filter_button_2k": [132, 1338],
                "shape_click_select_2k": [2068, 538],
                "shape_modal_reset_2k": [990, 1215],
                "shape_modal_confirm_2k": [1560, 1215],
                "filter_panel_confirm_2k": [2287, 1327],
                "reset_before_select": True,
                "after_open_filter_seconds": 0.0,
                "after_open_shape_modal_seconds": 0.0,
                "after_reset_seconds": 0.0,
                "after_click_shape_seconds": 0.0,
                "after_confirm_modal_seconds": 0.0,
                "after_confirm_panel_seconds": 0.0,
                "shape_centers_2k": {
                    "H_2": [798, 498],
                    "V_2": [945, 498],
                },
            },
            "inventory_search": {
                "template_suffix": "_Gold.png",
                "purple_template_suffix": "_Purple.png",
                "match_qualities": ["Gold", "Purple"],
            },
            "position_correction": {
                "enabled": False,
            },
            "drag_moves": [
                {"label": "01_right", "stick_x": 0.95, "stick_y": 0.0, "duration_seconds": 0.02},
                {"label": "02_down", "stick_x": 0.0, "stick_y": -0.30, "duration_seconds": 0.01},
            ],
            "grid_origin_r": 1,
            "grid_origin_c": 0,
            "debug_capture": {
                "enabled": False,
            },
            "equip_transfer_dialog": {
                "enabled": True,
                "keywords": ["取消", "确认"],
                "confirm_nav": [],
                "detect_settle_seconds": 0.0,
                "confirm_a_hold_seconds": 0.01,
                "confirm_a_settle_seconds": 0.01,
                "post_confirm_seconds": 0.0,
            },
        }

    def test_moves_from_config_list(self):
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = self.calibration
        moves = controller._moves_from_config_list(1, 0)
        self.assertEqual(2, len(moves))
        self.assertEqual("01_right", moves[0].label)
        self.assertEqual(StickMove(0.95, 0.0, 0.02, "01_right"), moves[0])

    def test_moves_from_config_list_appends_target_nudges(self):
        calibration = dict(self.calibration)
        calibration["grid_origin_r"] = 1
        calibration["grid_origin_c"] = 0
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = calibration
        moves = controller._moves_from_config_list(2, 1)
        labels = [move.label for move in moves]
        self.assertIn("target_r_1", labels)
        self.assertIn("target_c_1", labels)

    def test_compute_position_error_returns_none_when_detection_missing(self):
        self.assertIsNone(compute_position_error(None, 2.0, 1, 3))
        self.assertFalse(
            needs_position_correction(0.0, 0.0, max_cell_error=0.45)
        )
        delta_r, delta_c = estimate_position_error_from_origin(1, 3, origin_r=1, origin_c=0)
        self.assertEqual((0.0, 3.0), (delta_r, delta_c))
        self.assertTrue(needs_position_correction(delta_r, delta_c, max_cell_error=0.45))

    def test_inventory_template_variants_default_to_gold_and_purple(self):
        calibration = dict(self.calibration)
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = calibration
        controller._shape_recognizer = GoldShapeRecognizer(
            template_dir="config/templates",
            template_suffix="_Gold.png",
            quality_template_suffixes={"Purple": "_Purple.png"},
        )

        variants = controller._inventory_template_variants_for_shape_match()
        self.assertIn("H_2", variants)
        self.assertEqual(2, len(variants["H_2"]))

    def test_inventory_template_variants_honor_explicit_quality(self):
        calibration = dict(self.calibration)
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = calibration
        controller._shape_recognizer = GoldShapeRecognizer(
            template_dir="config/templates",
            template_suffix="_Gold.png",
            quality_template_suffixes={"Purple": "_Purple.png"},
        )

        variants = controller._inventory_template_variants_for_shape_match("Purple")
        self.assertIn("H_2", variants)
        self.assertEqual(1, len(variants["H_2"]))

    def test_detect_dragged_piece_anchor_uses_inventory_template_variants(self):
        import cv2
        import numpy as np

        recognizer = GoldShapeRecognizer(
            template_dir="config/templates",
            template_suffix="_Gold.png",
            quality_template_suffixes={"Purple": "_Purple.png"},
        )
        template = recognizer.templates["H_2"]
        th, tw = template.shape[:2]
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        y0, x0 = 180, 320
        image[y0 : y0 + th, x0 : x0 + tw] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        board_region = (300, 150, 700, 500)
        result = detect_dragged_piece_anchor(
            image,
            "H_2",
            recognizer.templates_for_shape_match(["Gold", "Purple"]),
            board_region,
            min_confidence=0.30,
        )

        self.assertIsNotNone(result.get("anchor_r"))
        self.assertEqual("inventory_template", result["method"])
        self.assertGreaterEqual(float(result["confidence"]), 0.30)

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    def test_wake_gamepad_sends_stick_tap(self, _sleep):
        calibration = dict(self.calibration)
        calibration["debug_capture"] = {"enabled": False}
        fake_gamepad = FakeGamepad()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=calibration)
            controller._wake_gamepad()

        self.assertEqual((0.0, 0.0), fake_gamepad.left_stick)

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    def test_select_inventory_drive_applies_shape_filter_once(self, _sleep):
        fake_gamepad = FakeGamepad()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._wake_gamepad = MagicMock()
            controller._capture_debug = MagicMock()
            controller._click_point_2k = MagicMock()
            first = controller.select_inventory_drive_by_shape("H_2")
            second = controller.select_inventory_drive_by_shape("H_2")

        self.assertEqual("shape_filter", first["method"])
        self.assertEqual("H_2", second["shape_id"])
        # filter + shape select + reset + shape + modal confirm + panel confirm
        self.assertEqual(6, controller._click_point_2k.call_count)
        controller._wake_gamepad.assert_called()

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    def test_select_inventory_drive_refilters_when_shape_changes(self, _sleep):
        fake_gamepad = FakeGamepad()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._wake_gamepad = MagicMock()
            controller._capture_debug = MagicMock()
            controller._click_point_2k = MagicMock()
            controller.select_inventory_drive_by_shape("H_2")
            controller.select_inventory_drive_by_shape("V_2")

        self.assertEqual(12, controller._click_point_2k.call_count)

    def test_select_inventory_drive_raises_for_unknown_shape(self):
        fake_gamepad = FakeGamepad()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._wake_gamepad = MagicMock()
            controller._capture_debug = MagicMock()
            controller._click_point_2k = MagicMock()
            with self.assertRaises(InventoryDriveNotFoundError):
                controller.select_inventory_drive_by_shape("Trap_4_H")

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    def test_confirm_equip_transfer_presses_a_without_nav(self, _sleep):
        fake_gamepad = FakeGamepad()
        fake_button = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "vgamepad": MagicMock(
                    VX360Gamepad=lambda: fake_gamepad,
                    XUSB_BUTTON=MagicMock(XUSB_GAMEPAD_A=fake_button),
                )
            },
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._detect_equip_transfer_dialog = MagicMock(return_value=True)
            controller._capture_debug = MagicMock()
            confirmed = controller._confirm_equip_transfer_dialog()

        self.assertTrue(confirmed)
        self.assertEqual((0.0, 0.0), fake_gamepad.left_stick)
        self.assertGreaterEqual(fake_gamepad.updates, 2)

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    @patch("src.scanner.gamepad_assembly.time.perf_counter")
    def test_drag_to_grid_cell_keeps_a_held_until_finish(self, perf_counter, _sleep):
        clock = {"t": 0.0}

        def _tick():
            clock["t"] += 0.05
            return clock["t"]

        perf_counter.side_effect = _tick
        fake_gamepad = FakeGamepad()
        fake_button = MagicMock()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock(XUSB_GAMEPAD_A=fake_button))},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller.capture_screenshot = MagicMock(return_value="debug.png")
            controller.select_inventory_drive_by_shape = MagicMock(
                return_value={"shape_id": "H_2", "confidence": 1.0, "method": "shape_filter"}
            )
            controller._correct_drag_position = MagicMock(return_value=[])
            controller._handle_post_place_dialogs = MagicMock()
            moves = controller.drag_to_grid_cell(1, 0, piece_id="H_2")

        self.assertEqual(2, len(moves))
        self.assertNotIn(fake_button, fake_gamepad.pressed)
        self.assertEqual((0.0, 0.0), fake_gamepad.left_stick)
        controller.select_inventory_drive_by_shape.assert_called_once_with("H_2", quality=None)
        controller._correct_drag_position.assert_called_once_with("H_2", 1, 0, quality=None)
        controller._handle_post_place_dialogs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
