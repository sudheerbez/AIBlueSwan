"""
Project AutoQuant — Sandboxed Code Executor
============================================
Safely executes LLM-generated factor code in a restricted environment.
"""

import io
import sys
import signal
import traceback
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schema — Execution Result
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    """Structured output from sandboxed code execution."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[Any] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None


# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    """Raised when code execution exceeds the allowed time limit."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Code execution timed out.")


# ---------------------------------------------------------------------------
# SafeCodeExecutor
# ---------------------------------------------------------------------------

# Modules allowed inside sandboxed code
ALLOWED_MODULES = frozenset({
    "pandas", "pd",
    "numpy", "np",
    "scipy", "scipy.stats", "scipy.optimize",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "json",
    "re",
    "ta",           # Technical analysis library
})


class SafeCodeExecutor:
    """
    Executes Python code strings in a sandboxed namespace.

    Security measures
    -----------------
    * Restricted built-in set (no ``open``, ``exec``, ``eval``, ``__import__``,
      ``compile``, ``globals``, ``locals``).
    * Only pre-approved modules are injected into the execution namespace.
    * Execution time is capped via ``signal.SIGALRM`` (Unix only).

    Parameters
    ----------
    timeout_seconds : int
        Maximum wall-clock seconds the code is allowed to run.
    """

    def __init__(self, timeout_seconds: int = 10) -> None:
        self.timeout_seconds = timeout_seconds

    # -- public API ----------------------------------------------------------

    def execute(
        self,
        code: str,
        *,
        extra_globals: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute *code* in a sandboxed namespace and return an
        ``ExecutionResult``.

        Parameters
        ----------
        code : str
            Python source code to execute.
        extra_globals : dict, optional
            Additional names to inject into the execution namespace (e.g. a
            price DataFrame for the factor function to consume).
        """
        sandbox_globals = self._build_sandbox(extra_globals)
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Save originals
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Set timeout (Unix only — silently skip on Windows or non-main threads)
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.timeout_seconds)
            except (AttributeError, OSError, ValueError):
                old_handler = None  # Windows / unsupported / not main thread

            exec(compile(code, "<autoquant-sandbox>", "exec"), sandbox_globals)  # noqa: S102

            # Cancel alarm
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, OSError, ValueError):
                pass

            # Attempt to extract a conventional return value
            return_value = sandbox_globals.get("result", sandbox_globals.get("signal_df", None))

            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_value=return_value,
            )

        except TimeoutError as exc:
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error_type="TimeoutError",
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )

        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Ensure alarm is cancelled
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _build_sandbox(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a restricted globals dict for ``exec()``."""
        import pandas as pd
        import numpy as np

        safe_builtins = {
            k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
            for k in (
                "abs", "all", "any", "bool", "dict", "enumerate", "filter",
                "float", "frozenset", "int", "isinstance", "len", "list",
                "map", "max", "min", "print", "range", "reversed", "round",
                "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
            )
        }

        namespace: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
        }

        # Optionally inject scipy / ta if available
        try:
            import scipy
            namespace["scipy"] = scipy
        except ImportError:
            pass

        try:
            import ta  # noqa: F811
            namespace["ta"] = ta
        except ImportError:
            pass

        if extra:
            namespace.update(extra)

        return namespace
