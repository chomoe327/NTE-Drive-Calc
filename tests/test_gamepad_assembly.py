import unittest
from unittest.mock import MagicMock, patch

from src.scanner.gamepad_assembly import GamepadAssemblyController, StickMove


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
            "drag_from_inventory_focus": {
                "stick_x": 0.72,
                "stick_y": -0.18,
                "duration_seconds": 0.02,
            },
            "grid_nudge": {
                "stick_x_per_col": 0.1,
                "stick_y_per_row": 0.1,
                "duration_per_step": 0.01,
            },
            "grid_origin_r": 1,
            "grid_origin_c": 0,
        }

    def test_build_drag_moves_for_origin_cell(self):
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = self.calibration
        moves = controller._build_drag_moves(1, 0)
        self.assertEqual(1, len(moves))
        self.assertEqual(StickMove(0.72, -0.18, 0.02), moves[0])

    def test_build_drag_moves_adds_row_and_col_nudges(self):
        controller = GamepadAssemblyController.__new__(GamepadAssemblyController)
        controller.calibration = self.calibration
        moves = controller._build_drag_moves(2, 1)
        self.assertEqual(3, len(moves))
        self.assertEqual(StickMove(0.0, 0.1, 0.01), moves[1])
        self.assertEqual(StickMove(0.1, 0.0, 0.01), moves[2])

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    @patch("src.scanner.gamepad_assembly.time.perf_counter")
    def test_drag_to_grid_cell_keeps_a_held_until_finish(self, perf_counter, _sleep):
        perf_counter.side_effect = [
            0.0, 0.0, 0.01, 0.01, 0.02, 0.02,
            0.03, 0.03, 0.04, 0.04, 0.05,
        ]
        fake_gamepad = FakeGamepad()
        fake_button = MagicMock()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock(XUSB_GAMEPAD_A=fake_button))},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller.drag_to_grid_cell(1, 0)

        self.assertNotIn(fake_button, fake_gamepad.pressed)
        self.assertEqual((0.0, 0.0), fake_gamepad.left_stick)
        self.assertGreaterEqual(fake_gamepad.updates, 4)

    @patch("src.scanner.gamepad_assembly.time.sleep", return_value=None)
    @patch("src.scanner.gamepad_assembly.time.perf_counter")
    def test_a_stays_pressed_during_stick_move(self, perf_counter, _sleep):
        perf_counter.side_effect = [0.0, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04]
        fake_gamepad = FakeGamepad()
        fake_button = MagicMock()

        with patch.dict(
            "sys.modules",
            {"vgamepad": MagicMock(VX360Gamepad=lambda: fake_gamepad, XUSB_BUTTON=MagicMock(XUSB_GAMEPAD_A=fake_button))},
        ):
            controller = GamepadAssemblyController(calibration=self.calibration)
            controller._begin_drag_hold()
            self.assertIn(fake_button, fake_gamepad.pressed)
            controller._move_stick_while_holding(0.72, -0.18, 0.02)
            self.assertIn(fake_button, fake_gamepad.pressed)
            controller._finish_drag_hold()

        self.assertNotIn(fake_button, fake_gamepad.pressed)


if __name__ == "__main__":
    unittest.main()
