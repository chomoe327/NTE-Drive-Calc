import unittest
from unittest.mock import MagicMock, patch

from src.scanner.gamepad_assembly import GamepadAssemblyController, InventoryDriveNotFoundError, StickMove


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
            "inventory_search": {
                "max_steps": 4,
                "grid_columns": 4,
                "min_confidence": 0.58,
            },
            "drag_moves": [
                {"label": "01_right", "stick_x": 0.95, "stick_y": 0.0, "duration_seconds": 0.02},
                {"label": "02_down", "stick_x": 0.0, "stick_y": -0.30, "duration_seconds": 0.01},
            ],
            "debug_capture": {
                "enabled": False,
            },
        }

    def test_moves_from_config_list(self):
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = self.calibration
        moves = controller._moves_from_config_list(1, 0)
        self.assertEqual(2, len(moves))
        self.assertEqual("01_right", moves[0].label)
        self.assertEqual(StickMove(0.95, 0.0, 0.02, "01_right"), moves[0])

    def test_select_inventory_drive_wakes_before_search(self):
        calibration = dict(self.calibration)
        calibration["debug_capture"] = {"enabled": False}
        fake_gamepad = FakeGamepad()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=calibration)
            controller._wake_gamepad = MagicMock()
            controller._tap_inventory_right = MagicMock()
            controller._tap_inventory_down = MagicMock()
            controller._recognize_selected_drive_shape = MagicMock(
                return_value={"shape_id": "H_2", "confidence": 0.92}
            )
            controller._capture_debug = MagicMock()
            result = controller.select_inventory_drive_by_shape("H_2")

        controller._wake_gamepad.assert_called_once()
        controller._tap_inventory_right.assert_not_called()
        self.assertEqual("H_2", result["shape_id"])

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

    def test_select_inventory_drive_moves_until_shape_matches(self):
        fake_gamepad = FakeGamepad()
        recognitions = [
            {"shape_id": "V_2", "confidence": 0.90},
            {"shape_id": "L_3_BL", "confidence": 0.88},
            {"shape_id": "H_2", "confidence": 0.91},
        ]

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._wake_gamepad = MagicMock()
            controller._capture_debug = MagicMock()
            controller._recognize_selected_drive_shape = MagicMock(side_effect=recognitions)
            controller._tap_inventory_right = MagicMock()
            controller._tap_inventory_down = MagicMock()
            result = controller.select_inventory_drive_by_shape("H_2")

        self.assertEqual("H_2", result["shape_id"])
        self.assertEqual(2, controller._tap_inventory_right.call_count)

    def test_select_inventory_drive_raises_when_not_found(self):
        fake_gamepad = FakeGamepad()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock())},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._wake_gamepad = MagicMock()
            controller._capture_debug = MagicMock()
            controller._recognize_selected_drive_shape = MagicMock(
                return_value={"shape_id": "V_2", "confidence": 0.90}
            )
            controller._tap_inventory_right = MagicMock()
            controller._tap_inventory_down = MagicMock()
            with self.assertRaises(InventoryDriveNotFoundError):
                controller.select_inventory_drive_by_shape("H_2")

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    @patch("src.scanner.gamepad_assembly.time.perf_counter")
    def test_drag_to_grid_cell_keeps_a_held_until_finish(self, perf_counter, _sleep):
        perf_counter.side_effect = [0.0] * 40
        fake_gamepad = FakeGamepad()
        fake_button = MagicMock()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock(XUSB_GAMEPAD_A=fake_button))},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller.capture_screenshot = MagicMock(return_value="debug.png")
            controller.select_inventory_drive_by_shape = MagicMock(
                return_value={"shape_id": "H_2", "confidence": 0.95}
            )
            moves = controller.drag_to_grid_cell(1, 0, piece_id="H_2")

        self.assertEqual(2, len(moves))
        self.assertNotIn(fake_button, fake_gamepad.pressed)
        self.assertEqual((0.0, 0.0), fake_gamepad.left_stick)
        controller.select_inventory_drive_by_shape.assert_called_once_with("H_2")


if __name__ == "__main__":
    unittest.main()
