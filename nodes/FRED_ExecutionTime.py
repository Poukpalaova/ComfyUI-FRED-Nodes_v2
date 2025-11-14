from comfy.comfy_types.node_typing import ComfyNodeABC, IO
import time
import threading

HELP_MESSAGE = """
👑 FRED_ExecutionTime (run-safe reset)

Purpose:
- Measure the TOTAL prompt duration (seconds, 2 decimals) when executed at the very end of the workflow.
- Guarantee start_time reset on EVERY run, even when the graph itself does not change.

Usage:
- Connect the 'tail' input to the latest node of the workflow (the branch that truly finishes last).
- The timer starts when a new run key is detected in IS_CHANGED, and stops in finalize().

Output console print:
- [FRED_ExecTime] elapsed_sec: <float>
"""

class _ExecTimerSingleton:
    _inst = None
    _lock = threading.Lock()
    _monotone = 0  # fallback to distinguish runs if no reliable prompt info is present

    def __new__(cls):
        if cls._inst is None:
            with cls._lock:
                if cls._inst is None:
                    cls._inst = super().__new__(cls)
                    cls._inst._reset_internal()
        return cls._inst

    def _reset_internal(self):
        self.run_key = None
        self.start_time = None
        self.stopped_time = None

    @classmethod
    def next_monotone(cls):
        with cls._lock:
            cls._monotone += 1
            return cls._monotone

    def reset_for_key(self, key):
        self._reset_internal()
        self.run_key = key

    def ensure_started_for_key(self, key):
        # New run if key differs
        if self.run_key != key:
            self.reset_for_key(key)
        if self.start_time is None:
            self.start_time = time.perf_counter()

    def stop_and_get(self):
        if self.start_time is None:
            # Safety: if not started, start now to avoid None
            self.start_time = time.perf_counter()
        self.stopped_time = time.perf_counter()
        return self.stopped_time - self.start_time

_TIMER = _ExecTimerSingleton()

def _derive_run_key(prompt_obj):
    """
    Try to extract a per-run stable key from PROMPT.
    Some frontends include 'time', 'timestamp', or 'prompt_id'.
    As a fallback, return a monotone counter to force reset between runs.
    """
    try:
        if isinstance(prompt_obj, dict):
            # Option 1: direct field on the root object
            for k in ("time", "timestamp", "prompt_id", "run_id", "_time", "_ts"):
                if k in prompt_obj and prompt_obj[k] is not None:
                    return f"prompt:{prompt_obj[k]}"
            # Option 2: nested metadata
            meta = prompt_obj.get("_meta") or prompt_obj.get("_extra") or {}
            for k in ("time", "timestamp", "prompt_id", "run_id"):
                if k in meta and meta[k] is not None:
                    return f"prompt_meta:{meta[k]}"
        # Fallback: monotone counter
        return f"monotone:{_ExecTimerSingleton.next_monotone()}"
    except Exception:
        return f"monotone:{_ExecTimerSingleton.next_monotone()}"

class FRED_ExecutionTime(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Connect this input to the last node of your workflow (latest branch to finish)
                "tail": (IO.ANY,),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    # Now returns FLOAT (seconds) and STRING (formatted "X.XXs")
    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("execution_time_sec", "execution_time_str", "help")
    FUNCTION = "finalize"
    CATEGORY = "👑FRED/utility"
    DESCRIPTION = "Return total prompt duration in seconds (2 decimals). Connect 'tail' to the latest node of your workflow."

    @classmethod
    def IS_CHANGED(cls, *, extra_pnginfo=None, prompt=None, unique_id=None, **kwargs):
        run_key = _derive_run_key(prompt)
        _TIMER.ensure_started_for_key(run_key)
        # Force re-evaluation on every run
        return float("NaN")

    def finalize(self, tail=None, extra_pnginfo=None, prompt=None, unique_id=None):
        elapsed = _TIMER.stop_and_get()
        elapsed = round(float(elapsed), 2)
        print(f"[FRED_ExecTime] elapsed_sec: {elapsed}")
        # Provide a string version, e.g., "2.34s"
        elapsed_str = f"{elapsed:.2f}s"
        return (elapsed, elapsed_str, HELP_MESSAGE)

NODE_CLASS_MAPPINGS = {
    "FRED_ExecutionTime": FRED_ExecutionTime,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ExecutionTime": "👑 FRED Execution Time",
}
