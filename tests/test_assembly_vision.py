import unittest

import numpy as np

from src.scanner.assembly_vision import (
    correction_stick_for_error,
    expand_board_region,
    is_detection_outlier,
    position_error_magnitude,
)


class AssemblyVisionTests(unittest.TestCase):
    def test_expand_board_region_expands_more_to_the_left(self):
        region = expand_board_region((1000, 200, 1500, 700), 2560, 1440, margin_ratio=0.35)
        left_pad = 1000 - region[0]
        right_pad = region[2] - 1500
        self.assertGreater(left_pad, right_pad)

    def test_is_detection_outlier_flags_large_jump(self):
        self.assertTrue(is_detection_outlier(1.0, 3.5, 1.0, 1.0, max_cell_jump=1.2))
        self.assertFalse(is_detection_outlier(1.0, 1.2, 1.0, 1.0, max_cell_jump=1.2))

    def test_correction_stick_only_moves_axes_outside_tolerance(self):
        cfg = {
            "stick_x_per_col": 0.12,
            "stick_y_per_row": 0.14,
            "max_stick": 0.35,
            "max_cell_error": 0.45,
            "correction_stick_scale": 1.0,
            "nudge_seconds": 0.10,
        }
        stick_x, stick_y, duration = correction_stick_for_error(0.04, -1.02, cfg)
        self.assertAlmostEqual(0.0, stick_y)
        self.assertLess(stick_x, 0.0)
        self.assertEqual(0.10, duration)

    def test_position_error_magnitude_uses_max_axis(self):
        self.assertEqual(2.5, position_error_magnitude(0.2, 2.5))


if __name__ == "__main__":
    unittest.main()
