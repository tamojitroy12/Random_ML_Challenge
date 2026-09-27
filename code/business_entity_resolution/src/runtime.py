"""
Runtime helpers shared by training and inference.

Console encoding, memory reporting, sleep prevention, timing, atomic file
writes and content hashing. Standard library + numpy only.
"""

import ctypes
import hashlib
import json
import os
import sys
import time

import numpy as np

_T0 = time.time()


def setup_console():
    """UTF-8 stdout/stderr (Windows pipes default to cp1252) and keep the machine awake."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    try:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: prevent sleep during long runs.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def mem_gb():
    """(current, peak) process memory in GB; (nan, nan) if unavailable."""
    try:
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1e9, pmc.PeakWorkingSetSize / 1e9
    except Exception:
        pass
    try:
        import resource  # POSIX fallback (peak only, KB on Linux)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        return float("nan"), peak
    except Exception:
        return float("nan"), float("nan")


def log(msg):
    cur, peak = mem_gb()
    print(f"[{time.time() - _T0:8.1f}s | mem {cur:5.2f} GB | peak {peak:5.2f} GB] {msg}", flush=True)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t
        log(f"{self.label}: {self.elapsed:.1f}s")
        return False


def safe_div(a, b, default=0.0):
    try:
        return a / b if b else default
    except Exception:
        return default


def atomic_savez(path, **arrays):
    """np.savez via a temp file + os.replace so a crash never leaves a truncated cache."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_npz(path):
    """Dict of arrays, or None when the file is missing/corrupt."""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as exc:
        print(f"  WARNING: could not load {path} ({exc}); it will be recomputed.", flush=True)
        return None


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()


def sha1_file(path):
    h = hashlib.sha1()
    if not os.path.isfile(path):
        return "missing"
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt_secs(s):
    s = float(s)
    if s < 120:
        return f"{s:.0f}s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"
