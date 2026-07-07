import ctypes
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.scanner import window_control


class WindowControlTests(unittest.TestCase):
    def test_find_game_window_uses_exact_title(self):
        fake_user32 = MagicMock()
        fake_user32.FindWindowW.return_value = 1117414
        fake_user32.IsWindowVisible.return_value = True

        with patch.object(ctypes, "windll") as windll:
            windll.user32 = fake_user32
            matched = window_control.find_game_window("异环")

        fake_user32.FindWindowW.assert_called_once_with(None, "异环")
        self.assertIsNotNone(matched)
        self.assertEqual(1117414, matched.hwnd)
        self.assertEqual("异环", matched.title)

    def test_find_game_window_returns_none_when_missing(self):
        fake_user32 = MagicMock()
        fake_user32.FindWindowW.return_value = 0

        with patch.object(ctypes, "windll") as windll:
            windll.user32 = fake_user32
            matched = window_control.find_game_window("异环")

        self.assertIsNone(matched)

    def test_find_window_uses_first_candidate(self):
        fake_user32 = MagicMock()
        fake_user32.FindWindowW.side_effect = [0, 200]
        fake_user32.IsWindowVisible.return_value = True

        with patch.object(ctypes, "windll") as windll:
            windll.user32 = fake_user32
            matched = window_control.find_window(["Missing", "异环"])

        self.assertIsNotNone(matched)
        self.assertEqual(200, matched.hwnd)

    def test_activate_game_window_raises_when_missing(self):
        with patch.object(window_control, "find_game_window", return_value=None):
            with self.assertRaises(window_control.WindowControlError):
                window_control.activate_game_window("异环")

    def test_grant_foreground_permission_calls_allow_set_foreground_window(self):
        fake_user32 = MagicMock()
        fake_kernel32 = MagicMock()
        fake_kernel32.GetCurrentProcessId.return_value = 4321

        with patch.object(ctypes, "windll") as windll:
            windll.user32 = fake_user32
            windll.kernel32 = fake_kernel32
            window_control.grant_foreground_permission()

        fake_user32.AllowSetForegroundWindow.assert_any_call(4321)
        fake_user32.AllowSetForegroundWindow.assert_any_call(window_control.ASFW_ANY)

    @unittest.skipUnless(sys.platform.startswith("win"), "Win32 activation test")
    def test_activate_window_success_when_foreground_matches(self):
        class FakeUser32:
            def __init__(self):
                self._foreground = 0

            def IsWindow(self, _hwnd):
                return True

            def IsIconic(self, _hwnd):
                return False

            def ShowWindow(self, hwnd, _cmd):
                self._foreground = int(hwnd)
                return True

            def GetForegroundWindow(self):
                return self._foreground

            def GetWindowThreadProcessId(self, _hwnd, thread_id):
                thread_id.contents.value = 1
                return 1

            def AttachThreadInput(self, _from_thread, _to_thread, _attach):
                return True

            def keybd_event(self, *_args):
                return None

            def BringWindowToTop(self, hwnd):
                self._foreground = int(hwnd)
                return True

            def SetForegroundWindow(self, hwnd):
                self._foreground = int(hwnd)
                return True

            def SwitchToThisWindow(self, hwnd, _force):
                self._foreground = int(hwnd)

        fake = FakeUser32()
        with patch.object(window_control, "time") as time_mock, patch.object(ctypes, "windll") as windll:
            time_mock.sleep.return_value = None
            windll.user32 = fake
            windll.kernel32 = MagicMock(GetCurrentThreadId=lambda: 2)
            self.assertTrue(window_control.activate_window(321, retries=1, retry_delay_seconds=0))

    def test_list_visible_windows_requires_windows(self):
        with patch.object(ctypes, "windll", None):
            with self.assertRaises(window_control.WindowControlError):
                window_control.list_visible_windows()


if __name__ == "__main__":
    unittest.main()
