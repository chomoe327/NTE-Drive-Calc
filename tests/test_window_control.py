import ctypes
import sys
import unittest
from unittest.mock import patch

from src.scanner import window_control


class WindowControlTests(unittest.TestCase):
    def test_find_window_matches_first_candidate(self):
        windows = [
            window_control.WindowInfo(hwnd=100, title="NTE Drive Calc"),
            window_control.WindowInfo(hwnd=200, title="异环 - Project Mugen"),
        ]

        with patch.object(window_control, "list_visible_windows", return_value=windows):
            matched = window_control.find_window(["异环"])

        self.assertIsNotNone(matched)
        self.assertEqual(200, matched.hwnd)
        self.assertIn("异环", matched.title)

    def test_find_window_returns_none_when_not_found(self):
        windows = [window_control.WindowInfo(hwnd=100, title="Other App")]

        with patch.object(window_control, "list_visible_windows", return_value=windows):
            matched = window_control.find_window(["异环"])

        self.assertIsNone(matched)

    @unittest.skipUnless(sys.platform.startswith("win"), "Win32 activation test")
    def test_activate_window_success_when_foreground_matches(self):
        class FakeUser32:
            def IsWindow(self, _hwnd):
                return True

            def IsIconic(self, _hwnd):
                return False

            def ShowWindow(self, _hwnd, _cmd):
                return True

            def SetForegroundWindow(self, _hwnd):
                return True

            def GetForegroundWindow(self):
                return 321

        fake = FakeUser32()
        with patch.object(ctypes, "windll") as windll:
            windll.user32 = fake
            self.assertTrue(window_control.activate_window(321))

    def test_list_visible_windows_requires_windows(self):
        with patch.object(ctypes, "windll", None):
            with self.assertRaises(window_control.WindowControlError):
                window_control.list_visible_windows()


if __name__ == "__main__":
    unittest.main()
