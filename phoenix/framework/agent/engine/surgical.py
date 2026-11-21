"""Specialized Deterministic Python System Tools: AST Error Diagnoser, Enclosing Scope Locator & Surgical Patcher.

Shifts error tracing, root-cause diagnosis, scope isolation, and AST-aligned
method replacement from the LLM into deterministic Python system tools so the
LLM only has to fix the broken code snippet.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from phoenix.framework.agent.tools._fs_utils import resolve_existing_path


def diagnose_python_error(
    error_msg: str,
    code_snippet: str = "",
    line_no: Optional[int] = None,
) -> List[str]:
    """Map Python AST/runtime tracebacks into concrete, actionable repair rules."""
    hints: List[str] = []
    err_low = (error_msg or "").lower()
    snip_low = (code_snippet or "").lower()

    if "'attribute' object has no attribute 'id'" in err_low or (
        "no attribute 'id'" in err_low and ("func" in snip_low or "attribute" in err_low)
    ):
        hints.append(
            "EXACT AST FIX (`node.func` can be `ast.Attribute` like `math.sin` or `ast.Name` like `sin`): "
            "Check `if isinstance(node.func, ast.Name): func_name = node.func.id` and "
            "`elif isinstance(node.func, ast.Attribute): func_name = node.func.attr` (and verify the base/module name like `math`). "
            "Do NOT access `.name` on `ast.Attribute` (it is `.attr`)."
        )

    if "unsupported node type" in err_low and "ast.expr" in err_low and "expression" not in err_low:
        hints.append(
            "EXACT AST FIX (`ast.Expr` wrapper): `ast.parse(text).body[0]` is an `ast.Expr` wrapping `.value`. "
            "Add `if isinstance(node, ast.Expr): return self.evaluate_node(node.value)` (or recursive call)."
        )

    if "unsupported node type" in err_low and "ast.attribute" in err_low:
        hints.append(
            "EXACT AST FIX (`ast.Attribute` like `math.pi`): When evaluating `ast.Attribute`, check "
            "`isinstance(node.value, ast.Name) and node.value.id == 'math'` and look up `node.attr` cleanly."
        )

    if "unsupported node type" in err_low and "ast.constant" in err_low:
        hints.append(
            "EXACT AST FIX (`ast.Constant`): Python 3.8+ represents numbers/strings as `ast.Constant(value=...)` (`return node.value`)."
        )

    if "is not defined" in err_low:
        m_undef = re.search(r"name ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"] is not defined", error_msg)
        if m_undef:
            sym = m_undef.group(1)
            if sym in ("sin", "cos", "tan", "tanh", "sqrt", "pi", "e"):
                hints.append(f"EXACT IMPORT FIX: `{sym}` belongs to `math`; ensure `import math` / `from math import {sym}` is present.")
            elif sym == "ast":
                hints.append("EXACT IMPORT FIX: Add `import ast` at top of file.")

    if "too many arguments" in err_low or "positional argument" in err_low:
        hints.append("SIGNATURE HINT: Check method `self` parameter or expected argument count at the failing call.")

    return hints


def find_enclosing_callable(code: str, target_line: int) -> Optional[Dict[str, Any]]:
    """Locate the innermost function/method (plus optional class) enclosing `target_line` (1-indexed)."""
    if not code or not target_line or target_line < 1:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    lines = code.splitlines()
    best: Optional[Dict[str, Any]] = None

    def _visit(nodes: List[ast.stmt], class_name: Optional[str] = None) -> None:
        nonlocal best
        for node in nodes:
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            if not (start <= target_line <= end):
                continue
            if isinstance(node, ast.ClassDef):
                _visit(node.body, class_name=node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raw_line = lines[start - 1] if 1 <= start <= len(lines) else ""
                indent = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
                qual = f"{class_name}.{node.name}" if class_name else node.name
                best = {
                    "class_name": class_name,
                    "func_name": node.name,
                    "qualname": qual,
                    "start_line": start,
                    "end_line": end,
                    "indent": indent,
                    "source": "\n".join(lines[start - 1 : end]),
                }
                _visit(node.body, class_name=class_name)

    _visit(tree.body)
    return best


def _find_matching_callable_targets(code: str, func_name: str, prefer_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find all top-level or class-method definitions named `func_name` in `code`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines()
    matches: List[Dict[str, Any]] = []

    def _scan(nodes: List[ast.stmt], cls_name: Optional[str] = None) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                _scan(node.body, cls_name=node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                raw_line = lines[start - 1] if 1 <= start <= len(lines) else ""
                indent = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
                matches.append({
                    "class_name": cls_name,
                    "func_name": node.name,
                    "qualname": f"{cls_name}.{node.name}" if cls_name else node.name,
                    "start_line": start,
                    "end_line": end,
                    "indent": indent,
                })

    _scan(tree.body)
    if prefer_class and len(matches) > 1:
        preferred = [m for m in matches if m["class_name"] == prefer_class]
        if preferred:
            return preferred
    return matches


class SurgicalCodePatcher:
    """Replaces a single function/method in-place via AST spans when the LLM returns only that snippet."""

    @staticmethod
    def apply_surgical_or_full(
        original_code: str,
        generated_code: str,
        focus_class: Optional[str] = None,
    ) -> Tuple[str, str, List[str]]:
        """
        If `original_code` is non-empty and `generated_code` consists strictly of one (or a few)
        standalone function definition(s) matching existing function/method names inside `original_code`
        (without classes/top-level script lines), surgically splice them into `original_code` at the
        exact AST span with preserved indentation. Otherwise return `generated_code` unchanged.
        """
        if not original_code.strip() or not generated_code.strip():
            return generated_code, "full_code", []

        try:
            gen_tree = ast.parse(generated_code)
            orig_tree = ast.parse(original_code)
        except SyntaxError:
            return generated_code, "full_code", []

        gen_top_funcs = [n for n in gen_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        gen_other = [
            n for n in gen_tree.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))
        ]
        # Only engage surgical splice when LLM emitted ONLY function(s) (no classes/top-level script)
        # while the original file contains additional top-level structure (classes, calls, constants, etc.)
        orig_has_extra_structure = any(
            not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))
            for n in orig_tree.body
        ) or len([n for n in orig_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]) > len(gen_top_funcs)

        if not gen_top_funcs or gen_other or not orig_has_extra_structure:
            return generated_code, "full_code", []

        working = original_code.replace("\r\n", "\n")
        gen_lines = generated_code.replace("\r\n", "\n").splitlines()
        applied_notes: List[str] = []

        for fn_node in gen_top_funcs:
            cands = _find_matching_callable_targets(working, fn_node.name, prefer_class=focus_class)
            if len(cands) != 1:
                return generated_code, "full_code", []
            target = cands[0]
            fn_src = textwrap.dedent("\n".join(gen_lines[fn_node.lineno - 1 : getattr(fn_node, "end_lineno", fn_node.lineno)]))
            indented_lines = [
                (target["indent"] + ln) if ln.strip() else ""
                for ln in fn_src.splitlines()
            ]
            work_lines = working.splitlines()
            new_lines = (
                work_lines[: target["start_line"] - 1]
                + indented_lines
                + work_lines[target["end_line"] :]
            )
            working = "\n".join(new_lines)
            if not working.endswith("\n"):
                working += "\n"
            applied_notes.append(f"Surgically spliced `{target['qualname']}` at L{target['start_line']}-L{target['end_line']}")

        # Preserve any new imports declared inside the snippet that aren't already in original
        gen_imports = [
            "\n".join(gen_lines[n.lineno - 1 : getattr(n, "end_lineno", n.lineno)])
            for n in gen_tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        for imp_stmt in reversed(gen_imports):
            if imp_stmt.strip() and imp_stmt.strip() not in working:
                working = imp_stmt.strip() + "\n" + working
                applied_notes.append(f"Preserved import `{imp_stmt.strip()}`")

        return working, "ast_surgical_method", applied_notes


class PythonSurgicalDiagnoseTool(BaseTool):
    """System tool that inspects Python file tracebacks, AST scopes, and exact root-cause fixes."""

    name = "python_diagnose"
    read_only = True
    description = (
        "Deterministically inspects a Python workspace file (AST symbols, security findings, enclosing function around a line, "
        "and concrete root-cause hints for an error message). Inputs: 'file_path' (str), optional 'error_line' (int), 'error_message' (str)."
    )

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__(read_only=True)
        self.runtime = runtime

    async def execute(
        self,
        file_path: str,
        error_line: Optional[int] = None,
        error_message: str = "",
        **kwargs,
    ) -> ToolResult:
        ok, resolved, err, _ = resolve_existing_path(file_path, self.runtime, allow_dir=False)
        if not ok:
            return ToolResult(success=False, output=None, error=err)
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except OSError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        scope = find_enclosing_callable(code, error_line) if error_line else None
        snippet = scope["source"] if scope else ""
        if not snippet and error_line:
            lines = code.splitlines()
            s = max(0, error_line - 6)
            e = min(len(lines), error_line + 5)
            snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in range(s, e))

        hints = diagnose_python_error(error_message, snippet, error_line)
        return ToolResult(
            success=True,
            output={
                "file_path": resolved,
                "enclosing_scope": scope,
                "error_hints": hints,
                "snippet": snippet,
            },
        )
