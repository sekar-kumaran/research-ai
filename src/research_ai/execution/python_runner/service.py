"""Sandboxed subprocess Python runner for small scientific calculations."""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

from research_ai.execution.sandbox import SandboxValidator


@dataclass
class PythonExecutionResult:
    ok: bool
    stdout: str
    error: str = ""
    latency_ms: float = 0.0
    validation_issues: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            **({"validation_issues": self.validation_issues} if self.validation_issues else {}),
        }


class PythonRunner:
    """Restricted subprocess Python runner for small scientific calculations.

    Two-layer safety:
    1. SandboxValidator performs AST-level static analysis before execution.
    2. The subprocess runs with ``-I`` (isolated mode) and only a tiny
       builtins surface exposed through exec().

    The runner is disabled unless ENABLE_PYTHON_EXECUTION=true. Even when
    enabled, this is NOT a container boundary — container-level isolation
    (Docker, gVisor) is the recommended production approach.
    """

    SAFE_BUILTINS = (
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
        "int", "len", "list", "max", "min", "pow", "print", "range",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
    )

    def __init__(
        self,
        enabled: bool,
        max_code_chars: int = 4000,
        timeout_seconds: int = 5,
    ) -> None:
        self.enabled = enabled
        self.max_code_chars = max_code_chars
        self.timeout_seconds = timeout_seconds
        self._validator = SandboxValidator()

    def run(self, code: str) -> PythonExecutionResult:
        started = time.perf_counter()

        if not self.enabled:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Python execution is disabled. Set ENABLE_PYTHON_EXECUTION=true to enable it.",
            )

        if len(code) > self.max_code_chars:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Code exceeds size limit ({self.max_code_chars} chars).",
            )

        # Layer 1: AST static analysis
        validation = self._validator.validate(code)
        if not validation.ok:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Code failed sandbox validation.",
                latency_ms=(time.perf_counter() - started) * 1000,
                validation_issues=validation.issues,
            )

        # Layer 2: subprocess execution with restricted builtins
        wrapper = (
            "import math, statistics\n"
            f"_safe = {self.SAFE_BUILTINS}\n"
            "_allowed = {name: getattr(__builtins__, name, None) for name in _safe}\n"
            "_allowed = {k: v for k, v in _allowed.items() if v is not None}\n"
            "_g = {'__builtins__': _allowed, 'math': math, 'statistics': statistics}\n"
            "exec(" + repr(code) + ", _g, {})\n"
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            latency = (time.perf_counter() - started) * 1000

            if completed.returncode != 0:
                return PythonExecutionResult(
                    ok=False,
                    stdout=completed.stdout,
                    error=completed.stderr.strip() or f"Process exited with {completed.returncode}",
                    latency_ms=latency,
                )

            sanitized = self._validator.sanitize_output(completed.stdout)
            return PythonExecutionResult(ok=True, stdout=sanitized, latency_ms=latency)

        except subprocess.TimeoutExpired:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Execution timed out after {self.timeout_seconds}s.",
            )
        except Exception as exc:
            return PythonExecutionResult(
                ok=False, stdout="", error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
