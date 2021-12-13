"""
Module for keeping an OS awake while performing a long-running task/operation.
"""
from contextlib import contextmanager
import platform
from typing import Optional


system_type: str = platform.system()
if system_type == "Windows":
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    # https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate

    def _activate_keep_awake(keep_screen_awake: bool = False) -> None:
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if keep_screen_awake:
            flags |= ES_DISPLAY_REQUIRED

        ctypes.windll.kernel32.SetThreadExecutionState(flags)

    def _deactivate_keep_awake() -> None:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

elif system_type == "Linux":
    import subprocess

    def _activate_keep_awake() -> None:
        subprocess.run(
            [u"systemctl", u"mask", u"sleep.target", u"suspend.target", u"hibernate.target", u"hybrid-sleep.target"])

    def _deactivate_keep_awake() -> None:
        subprocess.run(
            [u"systemctl", u"unmask", u"sleep.target", u"suspend.target", u"hibernate.target", u"hybrid-sleep.target"])


elif system_type == "Darwin":
    from subprocess import Popen, PIPE

    _darwin_process: Optional[Popen] = None

    def _activate_keep_awake(keep_screen_awake: bool = False) -> None:
        global _darwin_process

        if keep_screen_awake:
            args = ["-d", "-u", "-t 2592000"]
            _darwin_process = Popen([u"caffeinate"] + args, stdin=PIPE, stdout=PIPE)
        else:
            _darwin_process = Popen([u"caffeinate"], stdin=PIPE, stdout=PIPE)

    def _deactivate_keep_awake() -> None:
        global _darwin_process

        _darwin_process.terminate()
        _darwin_process.wait()


@contextmanager
def keep_awake(keep_screen_awake: bool = False):
    """
    Keeps the OS from sleeping until the 'with' block is completed.

    :param keep_screen_awake: Should the screen be kept awake/on as well. Linux will always keep the screen on anyway.
    :return: The generator object that can be used with 'with'.
    """
    _activate_keep_awake(keep_screen_awake)

    try:
        yield
    finally:
        _deactivate_keep_awake()
