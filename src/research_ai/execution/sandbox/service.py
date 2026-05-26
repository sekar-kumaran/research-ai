"""Execution sandbox — static analysis and code validation layer.

DEFENCE-IN-DEPTH DESIGN
------------------------
This module is Layer 1 of a two-layer sandbox:
  Layer 1 (this file): AST-level static analysis — runs BEFORE execution.
  Layer 2 (python_runner.py): subprocess -I with restricted builtins.

Layer 1 rejects code without running it.  Layer 2 limits what the running
process can do even if Layer 1 misses something.

This is intentionally a software-layer defence, NOT a container boundary.
For production deployments processing untrusted user code, use:
  - Docker with seccomp/AppArmor profiles
  - gVisor (runsc) for kernel-level isolation
  - Firecracker MicroVMs for strongest isolation

WHY AST ANALYSIS?
-----------------
String matching on raw source code is insufficient (obfuscation bypasses it).
AST analysis works on the parsed syntax tree — the interpreter's actual view
of the code — making it much harder to bypass with encoding tricks.

We also do a pre-AST string scan as a fast first-pass for obvious forbidden
names, but the AST scan is the authoritative check.

FORBIDDEN NAME MATCHING (BUG FIX v3.1.1)
-----------------------------------------
Original code: `if name in lower`
  This is a plain substring search.  "open" would match in:
    - "reopened" → False positive (legitimate variable name)
    - "overlap"  → False positive ("open" not in "overlap"... wait, no:
                   "open" IS NOT in "overlap" — "over" + "lap" ≠ "open")
    Actually let me reconsider: "open" in "reopened" = True → False POSITIVE.
    "open" in "overlap" = False (o-v-e-r-l-a-p, no "open").

  Real false-positive examples:
    - variable named `reopen_file` → blocked because "open" is a substring
    - variable `socket_count` would be blocked by "socket"
    - import of `opensearch` library would be flagged by "open"

Fix: use whole-word regex matching (`\bname\b`) so "open" only matches the
standalone word "open", not "reopen", "opener", etc.  This reduces false
positives while maintaining security against the actual attack surface.

FORBIDDEN NODE TYPES (AST)
--------------------------
Import / ImportFrom:  Block all imports.  The subprocess already has no
  external packages accessible (run with -I), but imports at the AST level
  are still blocked to prevent code that tries to import standard library
  modules that might exist.

Delete:   `del x` can delete items from __builtins__ dict in some contexts.

Global / Nonlocal: scope escape mechanisms that could be used to reach
  outer interpreter namespaces.
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pre-compiled regex for whole-word forbidden name detection.
# Using word boundaries (\b) prevents "open" from matching "reopen",
# "socket" from matching "socket_count" within a larger identifier, etc.
# Built once at module load — reused for every validation call.
_FORBIDDEN_NAME_PATTERNS: dict[str, re.Pattern] = {}


def _build_name_pattern(name: str) -> re.Pattern:
    """Return a compiled regex that matches `name` as a whole word."""
    return re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]
    ast_node_count: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "issues": self.issues,
            "ast_node_count": self.ast_node_count,
        }


class SandboxValidator:
    """Static analysis validator that screens Python code before execution.

    Raises no exceptions — returns a ValidationResult with a flag and a
    list of human-readable issue descriptions so the caller can decide how
    to proceed.

    Usage:
        validator = SandboxValidator()
        result = validator.validate(user_code)
        if not result.ok:
            return error_response(result.issues)
    """

    # AST node types that signal dangerous operations.
    # These are Python AST class names (as returned by type(node).__name__).
    FORBIDDEN_NODE_TYPES = frozenset({
        "Import",        # `import os` — any import statement
        "ImportFrom",    # `from os import path` — from-import
        "Delete",        # `del x` — can delete builtins in some contexts
        "Global",        # `global x` — scope escape to module level
        "Nonlocal",      # `nonlocal x` — scope escape to enclosing function
    })

    # Identifiers that must not appear as standalone words in the code.
    # This is a defense-in-depth pre-AST scan.  The AST scan below is the
    # authoritative check; this catches obfuscated cases before parsing.
    #
    # BUG FIX v3.1.1: These are now matched as whole words (regex \b...\b)
    # rather than substrings to prevent false positives.  See module docstring.
    FORBIDDEN_NAMES = frozenset({
        "__import__", "__builtins__", "__class__", "__bases__",
        "open", "eval", "exec", "compile", "input",
        "os", "sys", "subprocess", "socket", "shutil",
        "pathlib", "pickle", "marshal", "ctypes", "cffi",
        "importlib", "pkgutil", "inspect",
    })

    # Code complexity limits — prevent DoS via deeply nested code generation
    MAX_LINES = 120
    MAX_NODES = 800

    def validate(self, code: str) -> ValidationResult:
        """Validate code and return a ValidationResult.

        Checks (in order):
          1. Line count limit
          2. Forbidden whole-word name scan (pre-AST, fast)
          3. AST parse + node type scan
          4. Dunder attribute access detection
          5. Dynamic execution call detection (eval/exec/compile)
          6. AST node count limit
        """
        issues: list[str] = []
        code = textwrap.dedent(code)

        # ------------------------------------------------------------------
        # 1. Line count guard
        # ------------------------------------------------------------------
        lines = code.splitlines()
        if len(lines) > self.MAX_LINES:
            issues.append(f"Code exceeds {self.MAX_LINES} line limit ({len(lines)} lines).")

        # ------------------------------------------------------------------
        # 2. Forbidden name scan — whole-word regex (BUG FIX v3.1.1)
        #
        # Pre-AST string scan catches obfuscated strings before they are
        # parsed.  Using whole-word matching prevents false positives:
        #   - "reopen" does NOT trigger "open"
        #   - "socket_count" does NOT trigger "socket"
        #   - "inspect" as a standalone word IS correctly caught
        # ------------------------------------------------------------------
        for name in self.FORBIDDEN_NAMES:
            pattern = _FORBIDDEN_NAME_PATTERNS.get(name)
            if pattern is None:
                pattern = _build_name_pattern(name)
                _FORBIDDEN_NAME_PATTERNS[name] = pattern
            if pattern.search(code):
                issues.append(f"Forbidden identifier detected: '{name}'.")

        # ------------------------------------------------------------------
        # 3–5. AST-level analysis
        # ------------------------------------------------------------------
        node_count = 0
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                node_count += 1
                node_type = type(node).__name__

                # 3. Structural node type check
                if node_type in self.FORBIDDEN_NODE_TYPES:
                    issues.append(f"Forbidden AST node: {node_type}.")

                # 4. Dunder attribute access: obj.__class__, obj.__dict__, etc.
                #    These provide backdoors to the interpreter's type system.
                if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                    issues.append(f"Dunder attribute access disallowed: '.{node.attr}'.")

                # 5. Dynamic execution: eval("os.system(...)") bypasses static checks.
                #    exec() and compile() have the same issue.
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in ("eval", "exec", "compile"):
                        issues.append(f"Dynamic execution call disallowed: {func.id}().")

        except SyntaxError as exc:
            issues.append(f"Syntax error: {exc}")

        # ------------------------------------------------------------------
        # 6. AST complexity limit
        # ------------------------------------------------------------------
        if node_count > self.MAX_NODES:
            issues.append(f"Code AST complexity exceeds limit ({node_count} nodes > {self.MAX_NODES}).")

        if issues:
            logger.warning("SandboxValidator rejected code: %s", issues[:3])

        return ValidationResult(ok=not issues, issues=issues, ast_node_count=node_count)

    @staticmethod
    def sanitize_output(text: str, max_chars: int = 8000) -> str:
        """Truncate and strip ANSI escape codes from execution output.

        max_chars prevents memory bloat from runaway print() loops.
        ANSI stripping prevents terminal injection via crafted output.
        """
        cleaned = text[:max_chars]
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", cleaned)
        return cleaned
