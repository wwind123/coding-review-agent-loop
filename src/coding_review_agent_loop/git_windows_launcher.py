"""Windows Git launch boundary: assign a suspended process to a one-process Job.

This file runs in a separate isolated Python interpreter.  It has no checkout
imports and performs no Git probe before the Job has its limit and owns Git.
"""

from __future__ import annotations

import ctypes
import msvcrt
import subprocess
import sys
from ctypes import wintypes

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
HANDLE = wintypes.HANDLE
DWORD = wintypes.DWORD
SIZE_T = ctypes.c_size_t

CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
STARTF_USESTDHANDLES = 0x00000100
JobObjectExtendedLimitInformation = 9
INFINITE = 0xFFFFFFFF
HANDLE_FLAG_INHERIT = 0x00000001


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class BASIC_LIMITS(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", SIZE_T),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class EXTENDED_LIMITS(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", BASIC_LIMITS),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    ]


class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", DWORD), ("dwY", DWORD), ("dwXSize", DWORD),
        ("dwYSize", DWORD), ("dwXCountChars", DWORD), ("dwYCountChars", DWORD),
        ("dwFillAttribute", DWORD), ("dwFlags", DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", HANDLE), ("hStdOutput", HANDLE), ("hStdError", HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", HANDLE), ("hThread", HANDLE),
        ("dwProcessId", DWORD), ("dwThreadId", DWORD),
    ]


kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
kernel.CreateJobObjectW.restype = HANDLE
kernel.SetInformationJobObject.argtypes = (HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD)
kernel.SetInformationJobObject.restype = wintypes.BOOL
kernel.CreateProcessW.argtypes = (
    wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
    wintypes.BOOL, DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
    ctypes.POINTER(STARTUPINFO), ctypes.POINTER(PROCESS_INFORMATION),
)
kernel.CreateProcessW.restype = wintypes.BOOL
kernel.AssignProcessToJobObject.argtypes = (HANDLE, HANDLE)
kernel.AssignProcessToJobObject.restype = wintypes.BOOL
kernel.ResumeThread.argtypes = (HANDLE,)
kernel.ResumeThread.restype = DWORD
kernel.WaitForSingleObject.argtypes = (HANDLE, DWORD)
kernel.WaitForSingleObject.restype = DWORD
kernel.GetExitCodeProcess.argtypes = (HANDLE, ctypes.POINTER(DWORD))
kernel.GetExitCodeProcess.restype = wintypes.BOOL
kernel.TerminateProcess.argtypes = (HANDLE, DWORD)
kernel.TerminateProcess.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = (HANDLE,)
kernel.CloseHandle.restype = wintypes.BOOL
kernel.SetHandleInformation.argtypes = (HANDLE, DWORD, DWORD)
kernel.SetHandleInformation.restype = wintypes.BOOL


def _fail() -> int:
    print("Unsupported Git confinement: Windows Job Object launch failed.", file=sys.stderr)
    return 125


def main() -> int:
    if len(sys.argv) < 2:
        return _fail()
    git = sys.argv[1]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        return _fail()
    info = PROCESS_INFORMATION()
    started = False
    try:
        limits = EXTENDED_LIMITS()
        limits.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        if not kernel.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                               ctypes.byref(limits), ctypes.sizeof(limits)):
            return _fail()
        startup = STARTUPINFO()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = STARTF_USESTDHANDLES
        startup.hStdInput = HANDLE(msvcrt.get_osfhandle(0))
        startup.hStdOutput = HANDLE(msvcrt.get_osfhandle(1))
        startup.hStdError = HANDLE(msvcrt.get_osfhandle(2))
        for handle in (startup.hStdInput, startup.hStdOutput, startup.hStdError):
            if not kernel.SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT):
                return _fail()
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(sys.argv[1:]))
        if not kernel.CreateProcessW(git, command, None, None, True, CREATE_SUSPENDED,
                                     None, None, ctypes.byref(startup), ctypes.byref(info)):
            return _fail()
        started = True
        if not kernel.AssignProcessToJobObject(job, info.hProcess):
            return _fail()
        if kernel.ResumeThread(info.hThread) == INFINITE:
            return _fail()
        if kernel.WaitForSingleObject(info.hProcess, INFINITE) != 0:
            return _fail()
        exit_code = DWORD()
        if not kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code)):
            return _fail()
        return int(exit_code.value)
    finally:
        if started:
            kernel.TerminateProcess(info.hProcess, 125)
            kernel.CloseHandle(info.hThread)
            kernel.CloseHandle(info.hProcess)
        kernel.CloseHandle(job)


if __name__ == "__main__":
    raise SystemExit(main())
