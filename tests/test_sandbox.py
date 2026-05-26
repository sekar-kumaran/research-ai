"""Tests for SandboxValidator — static analysis security layer.

Covers:
  - Forbidden AST node detection (Import, ImportFrom, Delete, Global, Nonlocal)
  - Forbidden name whole-word matching (BUG FIX: no false positives)
  - Dunder attribute detection
  - Dynamic execution detection (eval, exec, compile)
  - Line count limit
  - AST node count limit
  - Clean code passes validation
  - Sanitize output truncation and ANSI stripping
"""
from __future__ import annotations

import pytest

from research_ai.execution.sandbox.service import SandboxValidator


@pytest.fixture
def validator():
    return SandboxValidator()


# ---------------------------------------------------------------------------
# 1. Clean code passes
# ---------------------------------------------------------------------------

class TestCleanCodePasses:
    def test_simple_arithmetic(self, validator):
        result = validator.validate("x = 1 + 2\nprint(x)")
        assert result.ok, f"Expected pass, got: {result.issues}"

    def test_list_comprehension(self, validator):
        result = validator.validate("[x**2 for x in range(10)]")
        assert result.ok

    def test_statistics(self, validator):
        code = "data = [1, 2, 3, 4, 5]\nmean = sum(data) / len(data)\nprint(mean)"
        result = validator.validate(code)
        assert result.ok

    def test_string_operations(self, validator):
        result = validator.validate('text = "hello world"\nprint(text.upper())')
        assert result.ok

    def test_fibonacci(self, validator):
        code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
print(fib(10))
"""
        result = validator.validate(code)
        assert result.ok


# ---------------------------------------------------------------------------
# 2. Import blocking
# ---------------------------------------------------------------------------

class TestImportBlocking:
    def test_import_os_blocked(self, validator):
        result = validator.validate("import os")
        assert not result.ok
        assert any("Import" in issue or "os" in issue for issue in result.issues)

    def test_from_import_blocked(self, validator):
        result = validator.validate("from os import path")
        assert not result.ok

    def test_import_sys_blocked(self, validator):
        result = validator.validate("import sys")
        assert not result.ok

    def test_import_subprocess_blocked(self, validator):
        result = validator.validate("import subprocess")
        assert not result.ok


# ---------------------------------------------------------------------------
# 3. Forbidden names — whole-word matching (BUG FIX v3.1.1)
# ---------------------------------------------------------------------------

class TestForbiddenNameWholeWordMatching:
    """Verify false positives are eliminated by whole-word regex matching."""

    def test_open_standalone_blocked(self, validator):
        result = validator.validate("f = open('file.txt')")
        assert not result.ok, "Standalone 'open' must be blocked"

    def test_reopen_not_blocked(self, validator):
        """BUG FIX: 'reopen' contains 'open' as substring but is a legitimate name."""
        result = validator.validate("reopen_count = 5\nprint(reopen_count)")
        assert result.ok, f"'reopen' is a legitimate variable name, got issues: {result.issues}"

    def test_overlay_not_blocked_by_open(self, validator):
        """'overlap' does not contain the word 'open'."""
        result = validator.validate("overlap = 0.5\nprint(overlap)")
        assert result.ok, f"'overlap' is safe, got: {result.issues}"

    def test_socket_standalone_blocked(self, validator):
        result = validator.validate("socket = 1")
        assert not result.ok

    def test_eval_standalone_blocked(self, validator):
        result = validator.validate("result = eval('1+1')")
        assert not result.ok

    def test_os_standalone_blocked(self, validator):
        result = validator.validate("x = os.getcwd()")
        assert not result.ok

    def test_sys_standalone_blocked(self, validator):
        result = validator.validate("sys.exit(0)")
        assert not result.ok

    def test_inspect_standalone_blocked(self, validator):
        result = validator.validate("x = inspect.stack()")
        assert not result.ok


# ---------------------------------------------------------------------------
# 4. Dunder attribute access
# ---------------------------------------------------------------------------

class TestDunderAccess:
    def test_class_dunder_blocked(self, validator):
        result = validator.validate("x = ().__class__")
        assert not result.ok
        assert any("__class__" in issue or "Dunder" in issue for issue in result.issues)

    def test_builtins_dunder_blocked(self, validator):
        result = validator.validate("b = {}.__builtins__")
        assert not result.ok

    def test_dict_keys_allowed(self, validator):
        """Normal dict method access is allowed."""
        result = validator.validate("d = {}\nkeys = list(d.keys())")
        assert result.ok


# ---------------------------------------------------------------------------
# 5. Dynamic execution
# ---------------------------------------------------------------------------

class TestDynamicExecution:
    def test_eval_call_blocked(self, validator):
        result = validator.validate("eval('__import__(\"os\")')")
        assert not result.ok

    def test_exec_call_blocked(self, validator):
        result = validator.validate("exec('import os')")
        assert not result.ok

    def test_compile_call_blocked(self, validator):
        result = validator.validate("code = compile('x=1', '', 'exec')")
        assert not result.ok


# ---------------------------------------------------------------------------
# 6. Scope escape
# ---------------------------------------------------------------------------

class TestScopeEscape:
    def test_global_statement_blocked(self, validator):
        result = validator.validate("def f():\n    global x\n    x = 1")
        assert not result.ok

    def test_nonlocal_statement_blocked(self, validator):
        result = validator.validate("def outer():\n    x=1\n    def inner():\n        nonlocal x\n        x=2")
        assert not result.ok

    def test_del_statement_blocked(self, validator):
        result = validator.validate("x = [1,2,3]\ndel x[0]")
        assert not result.ok


# ---------------------------------------------------------------------------
# 7. Complexity limits
# ---------------------------------------------------------------------------

class TestComplexityLimits:
    def test_line_count_limit(self, validator):
        code = "\n".join(f"x_{i} = {i}" for i in range(200))
        result = validator.validate(code)
        assert not result.ok
        assert any("line limit" in issue.lower() for issue in result.issues)

    def test_ast_node_count_reported(self, validator):
        result = validator.validate("x = 1 + 2")
        assert result.ast_node_count > 0

    def test_syntax_error_reported(self, validator):
        result = validator.validate("def broken(:\n    pass")
        assert not result.ok
        assert any("Syntax error" in issue or "syntax" in issue.lower() for issue in result.issues)


# ---------------------------------------------------------------------------
# 8. Output sanitization
# ---------------------------------------------------------------------------

class TestSanitizeOutput:
    def test_ansi_codes_stripped(self, validator):
        ansi_text = "\x1b[31mred text\x1b[0m"
        cleaned = validator.sanitize_output(ansi_text)
        assert "\x1b" not in cleaned
        assert "red text" in cleaned

    def test_truncation_at_max_chars(self, validator):
        long_text = "x" * 10000
        cleaned = validator.sanitize_output(long_text, max_chars=100)
        assert len(cleaned) <= 100

    def test_short_output_unchanged(self, validator):
        result = validator.sanitize_output("hello world")
        assert result == "hello world"
