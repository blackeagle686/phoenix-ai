"""Plan 2: Direct Code / Search-Replace Patch Protocol & Symbolic Auto-Healer.

Separates code synthesis from fragile nested JSON escaping for local/small LLMs,
extracts SEARCH/REPLACE blocks or markdown Python fences, applies AST-aware healing
(e.g. injecting missing `math` imports), and applies bounded SyntaxGuard checks.
"""

from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from phoenix.framework.agent.engine.pre_analyzer import DiagnosticReport, FileDiagnostic, KNOWN_MATH_OPS
from phoenix.framework.agent.engine.surgical import SurgicalCodePatcher, normalize_python_snippet
from phoenix.framework.agent.tools._fs_utils import (
    SyntaxGuard,
    build_compact_diff,
    check_runtime_path,
    resolve_existing_path,
    resolve_write_path,
    smart_find_and_replace,
)
from phoenix.services.llm.base import _extract_json_dict

_SEARCH_REPLACE_RE = re.compile(
    r"<{4,}(?:\s*SEARCH)?[^\r\n]*\r?\n(.*?)\r?\n={4,}\s*\r?\n(.*?)\r?\n>{4,}(?:\s*REPLACE)?[^\r\n]*",
    re.DOTALL | re.IGNORECASE,
)
_FENCED_CODE_RE = re.compile(
    r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def _lang_spec_for_path(file_path: str) -> Dict[str, Any]:
    ext = os.path.splitext((file_path or "").lower())[1]
    specs = {
        ".py": {"lang": "Python", "fence": "python", "tags": ("python", "py"), "is_py": True},
        ".html": {"lang": "HTML5", "fence": "html", "tags": ("html", "htm"), "is_py": False},
        ".htm": {"lang": "HTML5", "fence": "html", "tags": ("html", "htm"), "is_py": False},
        ".css": {"lang": "CSS", "fence": "css", "tags": ("css", "scss", "sass"), "is_py": False},
        ".scss": {"lang": "SCSS", "fence": "scss", "tags": ("scss", "css"), "is_py": False},
        ".js": {"lang": "JavaScript", "fence": "javascript", "tags": ("javascript", "js", "jsx"), "is_py": False},
        ".jsx": {"lang": "JSX", "fence": "jsx", "tags": ("jsx", "javascript", "js"), "is_py": False},
        ".ts": {"lang": "TypeScript", "fence": "typescript", "tags": ("typescript", "ts", "tsx"), "is_py": False},
        ".tsx": {"lang": "TSX", "fence": "tsx", "tags": ("tsx", "typescript", "ts"), "is_py": False},
        ".json": {"lang": "JSON", "fence": "json", "tags": ("json",), "is_py": False},
        ".sql": {"lang": "SQL", "fence": "sql", "tags": ("sql",), "is_py": False},
        ".sh": {"lang": "Bash", "fence": "bash", "tags": ("bash", "sh", "shell"), "is_py": False},
        ".md": {"lang": "Markdown", "fence": "markdown", "tags": ("markdown", "md"), "is_py": False},
    }
    return specs.get(ext, {"lang": "source code", "fence": "", "tags": (), "is_py": not ext})

_MATH_CONSTANTS: Set[str] = {"pi", "e", "tau", "inf", "nan"}
_HEALABLE_MATH_NAMES: Set[str] = set(KNOWN_MATH_OPS) | _MATH_CONSTANTS


@dataclass
class PatchApplyResult:
    """Structured outcome from applying a model's code or patch output to disk."""

    success: bool
    file_path: str
    resolved_path: str
    mode: str = "full_code"  # search_replace | full_code | json_tool | none
    healed_actions: List[str] = field(default_factory=list)
    diff: str = ""
    error: Optional[str] = None
    syntax_valid: bool = True


class SymbolicHealer:
    """Deterministic AST-based repair for common small-LLM coding omissions."""

    @staticmethod
    def heal_python_code(code: str, target_path: str = "") -> Tuple[str, List[str]]:
        """Auto-heal missing `math` imports and strip stray markdown noise cleanly."""
        actions: List[str] = []
        text = (code or "").replace("\r\n", "\n").strip("\r\n")
        if target_path and not target_path.lower().endswith(".py"):
            return text + ("\n" if text else ""), actions

        # Remove accidentally captured markdown fence edges and normalize leading indentation
        lines = text.splitlines()
        while lines and lines[0].strip().startswith("```"):
            lines.pop(0)
            actions.append("Stripped leading markdown fence")
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
            actions.append("Stripped trailing markdown fence")
        text = normalize_python_snippet("\n".join(lines))

        # Deterministic symbolic repairs for common small-LLM AST calculator / deprecation bugs
        if "isinstance(node, ast.Num)" in text or "isinstance(node, ast.Str)" in text:
            new_text = re.sub(
                r"if\s+isinstance\(\s*node\s*,\s*ast\.Num\s*\)\s*:\s*\n(\s*)return\s+node\.n\s*\n\s*el"
                r"if\s+isinstance\(\s*node\s*,\s*ast\.Str\s*\)\s*:\s*\n\s*return\s+node\.s",
                r"if isinstance(node, ast.Constant):\n\1return node.value",
                text,
            )
            new_text = re.sub(
                r"elif\s+isinstance\(\s*node\s*,\s*ast\.Num\s*\)\s*:\s*\n(\s*)return\s+node\.n\s*\n\s*el"
                r"if\s+isinstance\(\s*node\s*,\s*ast\.Str\s*\)\s*:\s*\n\s*return\s+node\.s",
                r"elif isinstance(node, ast.Constant):\n\1return node.value",
                new_text,
            )
            new_text = re.sub(
                r"isinstance\(\s*node\s*,\s*ast\.Num\s*\)",
                "isinstance(node, ast.Constant)",
                new_text,
            )
            new_text = re.sub(
                r"isinstance\(\s*node\s*,\s*ast\.Str\s*\)",
                "isinstance(node, ast.Constant)",
                new_text,
            )
            new_text = re.sub(r"\breturn\s+node\.[ns]\b", "return node.value", new_text)
            if new_text != text:
                text = new_text
                actions.append("Healed deprecated ast.Num/ast.Str to ast.Constant(node.value)")

        if "return func_name(*args)" in text and "func_name" in text:
            replacement_call = (
                "return (MATH_FUNCTIONS[func_name] if func_name in MATH_FUNCTIONS else getattr(math, func_name))(*args)"
                if "MATH_FUNCTIONS" in text
                else "return getattr(math, func_name)(*args)"
            )
            text = text.replace("return func_name(*args)", replacement_call)
            actions.append("Healed string `func_name(*args)` call to resolve callable via MATH_FUNCTIONS/math")

        if "MATH_FUNCTIONS['pi']" in text or 'MATH_FUNCTIONS["pi"]' in text:
            text = text.replace("MATH_FUNCTIONS['pi']", "math.pi").replace('MATH_FUNCTIONS["pi"]', "math.pi")
            actions.append("Healed `MATH_FUNCTIONS['pi']` constant lookup to `math.pi`")

        if "MATH_FUNCTIONS[node.attr]" in text:
            text = text.replace(
                "MATH_FUNCTIONS[node.attr]",
                "(MATH_FUNCTIONS[node.attr] if node.attr in MATH_FUNCTIONS else getattr(math, node.attr))",
            )
            actions.append("Healed `MATH_FUNCTIONS[node.attr]` attribute resolution fallback to `getattr(math, node.attr)`")

        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text + ("\n" if text and not text.endswith("\n") else ""), actions

        # Inspect AST for broken evaluate_binary_op / evaluate_node patterns
        src_lines = text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_binary_op":
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", start_l)
                fn_src = "\n".join(src_lines[start_l - 1 : end_l])
                if "MATH_FUNCTIONS" in fn_src or "Div" not in fn_src or "ast.Attribute" in fn_src:
                    def_line = src_lines[start_l - 1]
                    indent = def_line[: len(def_line) - len(def_line.lstrip())]
                    body_indent = indent + "    "
                    fixed_fn = [
                        f"{indent}def evaluate_binary_op(self, node):",
                        f"{body_indent}left = self.evaluate_node(node.left)",
                        f"{body_indent}right = self.evaluate_node(node.right)",
                        f"{body_indent}if isinstance(node.op, ast.Add):",
                        f"{body_indent}    return left + right",
                        f"{body_indent}elif isinstance(node.op, ast.Sub):",
                        f"{body_indent}    return left - right",
                        f"{body_indent}elif isinstance(node.op, ast.Mult):",
                        f"{body_indent}    return left * right",
                        f"{body_indent}elif isinstance(node.op, ast.Div):",
                        f"{body_indent}    return left / right",
                        f"{body_indent}elif isinstance(node.op, ast.FloorDiv):",
                        f"{body_indent}    return left // right",
                        f"{body_indent}elif isinstance(node.op, ast.Mod):",
                        f"{body_indent}    return left % right",
                        f"{body_indent}elif isinstance(node.op, ast.Pow):",
                        f"{body_indent}    return left ** right",
                        f"{body_indent}raise ValueError(f'Unsupported binary operation: {{type(node.op).__name__}}')",
                    ]
                    src_lines = src_lines[: start_l - 1] + fixed_fn + src_lines[end_l:]
                    text = "\n".join(src_lines)
                    actions.append("Healed `evaluate_binary_op` to handle Add/Sub/Mult/Div/FloorDiv/Mod/Pow")
                    tree = ast.parse(text)
                    break

        src_lines = text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_node":
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", start_l)
                fn_src = "\n".join(src_lines[start_l - 1 : end_l])
                if "ast.Call" in fn_src and "ast.Attribute" not in fn_src:
                    def_line = src_lines[start_l - 1]
                    indent = def_line[: len(def_line) - len(def_line.lstrip())]
                    body_indent = indent + "    "
                    attr_branch = [
                        f"{body_indent}elif isinstance(node, ast.Attribute):",
                        f"{body_indent}    if isinstance(node.value, ast.Name) and node.value.id == 'math':",
                        f"{body_indent}        return getattr(math, node.attr)",
                        f"{body_indent}    return getattr(self.evaluate_node(node.value), node.attr)",
                    ]
                    # Insert right before the final `else:` inside evaluate_node if present
                    insert_idx = end_l
                    for idx in range(end_l - 1, start_l - 1, -1):
                        if src_lines[idx].strip().startswith("else:"):
                            insert_idx = idx
                            break
                    src_lines = src_lines[:insert_idx] + attr_branch + src_lines[insert_idx:]
                    text = "\n".join(src_lines)
                    actions.append("Healed `evaluate_node` to support `ast.Attribute` (e.g. `math.pi`)")
                    tree = ast.parse(text)
                    break

        defined: Set[str] = set()
        imported_math_module = False
        imported_math_names: Set[str] = set()
        used_bare_math: Set[str] = set()
        uses_math_prefix = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    defined.add(name)
                    if alias.name == "math" and (alias.asname is None or alias.asname == "math"):
                        imported_math_module = True
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    defined.add(name)
                    if node.module == "math":
                        imported_math_names.add(alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args + node.args.kwonlyargs:
                        defined.add(arg.arg)
                    if node.args.vararg:
                        defined.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        defined.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.comprehension):
                for tgt in ast.walk(node.target):
                    if isinstance(tgt, ast.Name):
                        defined.add(tgt.id)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
                uses_math_prefix = True
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in _HEALABLE_MATH_NAMES and node.id not in defined:
                    used_bare_math.add(node.id)

        imports_to_inject: List[str] = []
        if uses_math_prefix and not imported_math_module:
            imports_to_inject.append("import math")
            actions.append("Auto-injected missing `import math`")
        missing_bare = sorted(n for n in used_bare_math if n not in imported_math_names)
        if missing_bare:
            imports_to_inject.append(f"from math import {', '.join(missing_bare)}")
            actions.append(f"Auto-injected `from math import {', '.join(missing_bare)}`")

        if imports_to_inject:
            code_lines = text.splitlines()
            insert_at = 0
            # Place after module docstring / __future__ imports if present
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
                if isinstance(tree.body[0].value.value, str):
                    insert_at = getattr(tree.body[0], "end_lineno", 1)
            while insert_at < len(code_lines) and code_lines[insert_at].strip().startswith("from __future__"):
                insert_at += 1
            for idx, imp_line in enumerate(imports_to_inject):
                code_lines.insert(insert_at + idx, imp_line)
            text = "\n".join(code_lines)

        if text and not text.endswith("\n"):
            text += "\n"
        return text, actions


class PatchProtocol:
    """Generates focused prompts and parses SEARCH/REPLACE, fenced code, or JSON outputs reliably."""

    @staticmethod
    def build_prompt(
        report: DiagnosticReport,
        previous_feedback: str = "",
        best_score: int = 0,
        focus_scope: Optional[Dict[str, Any]] = None,
        diagnosis_hints: Optional[List[str]] = None,
        force_search_replace: bool = False,
        target_file: Optional[str] = None,
        sibling_context: Optional[Dict[str, str]] = None,
        project_files: Optional[List[str]] = None,
    ) -> str:
        c = report.contract
        primary_path = target_file or (c.target_files[0] if c.target_files else "target.py")
        lang_spec = _lang_spec_for_path(primary_path)
        is_py = bool(lang_spec["is_py"])
        fence_tag = str(lang_spec["fence"] or "")
        lang_label = str(lang_spec["lang"])

        fd: Optional[FileDiagnostic] = next(
            (f for f in report.files if f.exists and (f.requested_path == primary_path or os.path.basename(f.requested_path) == os.path.basename(primary_path))),
            None,
        )
        if fd is None and target_file is None:
            fd = next((f for f in report.files if f.exists), None)
        surgical_mode = bool(is_py and focus_scope and focus_scope.get("source") and fd is not None and fd.ast_valid and not c.requires_security)

        err_core = (
            c.traceback_error.split(":", 1)[-1].strip().lower()
            if c.traceback_error and ":" in c.traceback_error
            else (c.traceback_error or "").strip().lower()
        )
        stale_traceback = bool(
            c.has_traceback
            and err_core
            and previous_feedback
            and err_core not in previous_feedback.lower()
        )
        task_text = (
            f"Fix the active runtime error inside `{focus_scope['qualname']}` in `{primary_path}`."
            if stale_traceback and surgical_mode and focus_scope is not None
            else report.prompt
        )

        lines: List[str] = [
            f"TASK: {task_text}",
            f"TARGET FILE TO GENERATE NOW: `{primary_path}` ({lang_label})",
        ]
        all_files = project_files or list(c.target_files or [])
        if len(all_files) > 1:
            lines.append("FULL PROJECT STRUCTURE: " + ", ".join(all_files))
            lines.append(
                f"CRITICAL SCOPE RULE: Write ONLY the complete contents of `{primary_path}` ({lang_label}) in this response. "
                "Do NOT write code for other files in this response."
            )
            ext = os.path.splitext(primary_path.lower())[1]
            same_dir = os.path.dirname(primary_path.replace("\\", "/"))
            siblings_same_dir = [
                os.path.basename(p) for p in all_files
                if p != primary_path and os.path.dirname(p.replace("\\", "/")) == same_dir
            ]
            if ext == ".py" and "fastapi" in report.prompt.lower():
                lines.append(
                    "FASTAPI ARCHITECTURE: Use FastAPI + CORSMiddleware (`allow_origins=['*']`, `allow_methods=['*']`, `allow_headers=['*']`), "
                    "a clean Pydantic request model, safe arithmetic evaluation (no raw `eval`), clear JSON error handling (`HTTPException`), "
                    "and `uvicorn.run(app, host='0.0.0.0', port=8000)` under `if __name__ == '__main__':`."
                )
            elif ext in (".html", ".htm"):
                css_sibs = [s for s in siblings_same_dir if s.endswith(".css")]
                js_sibs = [s for s in siblings_same_dir if s.endswith(".js")]
                link_hints = []
                if css_sibs:
                    link_hints.append(f'<link rel="stylesheet" href="{css_sibs[0]}">')
                if js_sibs:
                    link_hints.append(f'<script src="{js_sibs[0]}" defer></script>')
                if link_hints:
                    lines.append("HTML LINKS: Include " + " and ".join(link_hints) + " with clean semantic IDs.")
            elif ext in (".js", ".ts"):
                lines.append(
                    "JS INTEGRATION: Wire exact DOM element IDs from the HTML file and call the exact backend FastAPI endpoint URL "
                    "(default `http://127.0.0.1:8000`) using `fetch(...)` with `Content-Type: application/json` and clean error display."
                )

        if sibling_context:
            sib_lines = ["ALREADY CREATED PROJECT FILES (match these exact routes, JSON keys, IDs & classes):"]
            for s_path, s_summary in sibling_context.items():
                if s_path != primary_path and s_summary:
                    sib_lines.append(f"- `{s_path}`:\n{s_summary}")
            if len(sib_lines) > 1:
                lines.append("\n".join(sib_lines))

        if is_py and c.requires_security:
            lines.append(
                "SECURITY MANDATE: NO eval(), NO exec(), NO compile(), NO __import__(), NO os/subprocess/open()."
            )
        if is_py and c.has_traceback and c.traceback_error and not stale_traceback:
            line_hint = f" at line(s) {c.traceback_lines}" if c.traceback_lines else ""
            lines.append(f"REPORTED RUNTIME ERROR{line_hint}: {c.traceback_error}")
            if any("ast.Expr" in h for h in c.ast_node_hints):
                lines.append(
                    "AST BUGFIX RULE: `ast.parse(expr)` returns an `ast.Module` whose `.body[0]` is `ast.Expr(value=...)`. "
                    "Handle `isinstance(node, ast.Expr)` by returning `evaluate_node(node.value)`, AND handle `ast.Expression` (`node.body`) and `ast.Constant`!"
                )
        if is_py and c.required_ops:
            lines.append(f"REQUIRED OPERATIONS: {', '.join(c.required_ops)}")
        if is_py and c.is_calculator_domain and c.requires_security:
            lines.append(
                "SAFE PATTERN: Use `ast.parse(expr, mode='eval')` + recursive AST node evaluator "
                "(BinOp/UnaryOp/Constant/Name/Call) with an explicit dict mapping math ops like "
                + (", ".join(f"'{op}': math.{op}" for op in (c.required_ops or ["sin", "cos", "tanh"])))
                + ", raising ValueError on any unsupported node or expression."
            )

        if diagnosis_hints:
            lines.append("SYSTEM ERROR DIAGNOSIS (APPLY THIS FIX):\n" + "\n".join(f"- {h}" for h in diagnosis_hints))

        if fd is not None and (not surgical_mode or fd.line_count <= 220):
            if is_py and fd.security_findings and (
                c.requires_security
                or any(tok in f for f in fd.security_findings for tok in ("eval", "exec", "compile", "__import__"))
            ):
                lines.append("DETECTED RISKS IN CURRENT FILE: " + "; ".join(fd.security_findings[:6]))
            if is_py and fd.missing_ops:
                lines.append("CURRENTLY MISSING OPS: " + ", ".join(fd.missing_ops))
            lines.append(f"CURRENT CONTENT OF `{fd.requested_path}`:\n```{fence_tag}\n{fd.raw_content}\n```")

        if surgical_mode and focus_scope is not None:
            lines.append(
                f"SURGICAL FOCUS (`{focus_scope['qualname']}`, L{focus_scope['start_line']}-L{focus_scope['end_line']}):\n"
                f"```python\n{focus_scope['source']}\n```"
            )

        if previous_feedback:
            lines.append(
                f"VERIFICATION REPORT FROM PREVIOUS ATTEMPT (Best Score So Far: {best_score}/100):\n"
                f"{previous_feedback}\n"
                "Fix ALL listed issues directly. Do NOT repeat the exact same failing code."
            )

        if force_search_replace:
            lines.append(
                "OUTPUT FORMAT (SURGICAL DIFF ONLY): Do NOT copy the whole function unchanged. "
                "Emit ONLY a targeted SEARCH/REPLACE block changing the failing lines:\n"
                "<<<<<<< SEARCH\n<exact existing lines to change>\n=======\n<fixed lines>\n>>>>>>> REPLACE"
            )
        elif surgical_mode and focus_scope is not None:
            lines.append(
                f"OUTPUT FORMAT: Return either the complete fixed file OR ONLY the fixed `def {focus_scope['func_name']}(...):` function(s) "
                f"inside a single ```python ... ``` block (the system will splice standalone functions safely back into `{primary_path}`). "
                "Do NOT echo the original code first and do NOT wrap in JSON."
            )
        else:
            fence_example = f"```{fence_tag} ... ```" if fence_tag else "``` ... ```"
            lines.append(
                f"OUTPUT FORMAT: Emit ONLY a single {fence_example} code block containing the complete {lang_label} code for `{primary_path}`. "
                "Do NOT repeat the original unchanged code before your fix, do NOT write long prose, and do NOT wrap code inside JSON."
            )
        return "\n\n".join(lines)

    @staticmethod
    def _validate_non_python_snippet(code: str, ext: str) -> bool:
        s = (code or "").strip()
        if not s:
            return False
        if ext in (".html", ".htm"):
            return "<" in s and ">" in s and not s.lstrip().startswith(("def ", "import ", "from "))
        if ext in (".css", ".scss", ".sass", ".less"):
            return "{" in s and "}" in s and ":" in s and "<html" not in s.lower()
        if ext in (".js", ".jsx", ".ts", ".tsx"):
            return not re.search(r"^\s*(?:from\s+\w+\s+import|def\s+\w+\s*\()", s, flags=re.MULTILINE)
        if ext == ".json":
            try:
                json.loads(s)
                return True
            except Exception:
                return False
        return True

    @staticmethod
    def parse_model_output(raw_output: str, old_content: str = "", target_path: str = "") -> Tuple[str, Any]:
        """Parse model output into ('search_replace', pairs) | ('full_code', code_str) | ('json_tool', payload) | ('none', raw)."""
        text = (raw_output or "").strip()
        if "<think>" in text:
            stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if stripped:
                text = stripped
        if not text:
            return "none", ""

        lang_spec = _lang_spec_for_path(target_path)
        is_py = bool(lang_spec["is_py"])
        ext = os.path.splitext((target_path or "").lower())[1]
        preferred_tags = set(lang_spec["tags"])

        # 1. Explicit SEARCH/REPLACE blocks
        sr_matches = _SEARCH_REPLACE_RE.findall(text)
        if sr_matches:
            pairs = [(m[0].strip("\r\n"), m[1].strip("\r\n")) for m in sr_matches if m[0].strip()]
            if pairs:
                return "search_replace", pairs

        # 2. Markdown code fences (language-aware + prefer changed blocks over verbatim echoes of old_content)
        fences: List[Tuple[str, str]] = [
            ((m.group(1) or "").strip().lower(), m.group(2))
            for m in _FENCED_CODE_RE.finditer(text)
            if m.group(2).strip()
        ]
        if not fences:
            # Handle truncated/unclosed trailing ```<lang> fence from local models
            unclosed = re.search(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.+)$", text, flags=re.DOTALL)
            if unclosed and unclosed.group(2).strip():
                fences = [((unclosed.group(1) or "").strip().lower(), unclosed.group(2))]

        if fences:
            old_norm = (
                normalize_python_snippet(old_content).strip()
                if (is_py and old_content)
                else textwrap.dedent(old_content).strip()
            )
            valid_fences: List[Tuple[bool, bool, int, int, str]] = []
            for idx, (tag, blk) in enumerate(fences):
                tag_matches = (not preferred_tags) or (tag in preferred_tags) or (tag == "")
                if is_py:
                    norm_blk = normalize_python_snippet(blk)
                    try:
                        ast.parse(norm_blk)
                        is_changed = (not old_norm) or (norm_blk.strip() != old_norm)
                        valid_fences.append((tag_matches, is_changed, len(norm_blk), idx, norm_blk))
                    except SyntaxError:
                        pass
                else:
                    norm_blk = textwrap.dedent(blk.replace("\r\n", "\n")).strip("\r\n")
                    if PatchProtocol._validate_non_python_snippet(norm_blk, ext):
                        is_changed = (not old_norm) or (norm_blk.strip() != old_norm)
                        valid_fences.append((tag_matches, is_changed, len(norm_blk), idx, norm_blk))
            if valid_fences:
                # Prefer blocks matching the target file's language tag AND differing from old_content
                lang_matched = [item for item in valid_fences if item[0]]
                pool = lang_matched if lang_matched else valid_fences
                changed_only = [item for item in pool if item[1]]
                pool = changed_only if changed_only else pool
                max_len = max(item[2] for item in pool)
                substantial = [item for item in pool if item[2] >= int(max_len * 0.65)]
                best_item = max(substantial, key=lambda x: (x[3], x[2])) if substantial else max(pool, key=lambda x: (x[2], x[3]))
                return "full_code", best_item[4]
            # Fallback to tag-matched or longest fence
            tag_fences = [blk for tag, blk in fences if (not preferred_tags or tag in preferred_tags)]
            chosen = max(tag_fences or [blk for _, blk in fences], key=len)
            return "full_code", (normalize_python_snippet(chosen) if is_py else textwrap.dedent(chosen.replace("\r\n", "\n")).strip("\r\n"))

        # 3. Fallback if model emitted structured JSON tool output anyway
        if text.startswith("{") and ext != ".json":
            parsed_json = _extract_json_dict(text)
            if isinstance(parsed_json, dict):
                tools = parsed_json.get("tools_to_call")
                if isinstance(tools, list) and tools:
                    for tc in tools:
                        if not isinstance(tc, dict):
                            continue
                        args = tc.get("arguments") or {}
                        if isinstance(args, dict):
                            if isinstance(args.get("content"), str) and args["content"].strip():
                                return "full_code", args["content"]
                            edits = args.get("edits")
                            if isinstance(edits, list) and edits:
                                sr = [
                                    (str(e.get("target") or e.get("search") or ""), str(e.get("replacement") or e.get("replace") or ""))
                                    for e in edits
                                    if isinstance(e, dict) and (e.get("target") or e.get("search"))
                                ]
                                if sr:
                                    return "search_replace", sr

        # 4. Raw unfenced source code fallback (language-aware)
        if not is_py:
            if PatchProtocol._validate_non_python_snippet(text, ext):
                return "full_code", textwrap.dedent(text.replace("\r\n", "\n")).strip("\r\n")
            return "none", text

        if any(tok in text for tok in ("def ", "import ", "from ", "class ")):
            norm_raw = normalize_python_snippet(text)
            try:
                ast.parse(norm_raw)
                return "full_code", norm_raw
            except SyntaxError:
                raw_lines = text.splitlines()
                start_candidates = [
                    i for i, ln in enumerate(raw_lines)
                    if ln.lstrip().startswith(("import ", "from ", "def ", "async def ", "class ", "@"))
                ]
                for st in start_candidates:
                    for end in range(len(raw_lines), st, -1):
                        cand = normalize_python_snippet("\n".join(raw_lines[st:end]))
                        if not cand:
                            continue
                        try:
                            ast.parse(cand)
                            return "full_code", cand
                        except SyntaxError:
                            continue

        return "none", text

    @classmethod
    def apply_output(
        cls,
        raw_output: str,
        target_path: str,
        runtime: Any,
        syntax_guard: Optional[SyntaxGuard] = None,
        focus_class: Optional[str] = None,
    ) -> PatchApplyResult:
        """Apply parsed SEARCH/REPLACE or full code safely onto the workspace file."""
        guard = syntax_guard or SyntaxGuard(max_rollbacks=3)
        norm_target = (target_path or "").replace("\\", "/")
        old_content = ""
        # Explicit subdirectory paths (e.g. `calc_proj/front/c.js`) resolve directly so a stray root `c.js` never hijacks them
        if "/" in norm_target and not os.path.isabs(target_path):
            ok_w, direct_abs, w_err = resolve_write_path(target_path, runtime)
            if not ok_w:
                return PatchApplyResult(
                    success=False,
                    file_path=target_path,
                    resolved_path=direct_abs,
                    mode="none",
                    error=w_err or f"Invalid target path: {target_path}",
                    syntax_valid=False,
                )
            resolved = direct_abs
            if os.path.isfile(resolved):
                try:
                    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                        old_content = f.read()
                except OSError:
                    old_content = ""
        else:
            ok_exist, existing_abs, _, _ = resolve_existing_path(target_path, runtime, allow_dir=False)
            if ok_exist and os.path.isfile(existing_abs):
                resolved = existing_abs
                try:
                    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                        old_content = f.read()
                except OSError:
                    old_content = ""
            else:
                ok_w, resolved, w_err = resolve_write_path(target_path, runtime)
                if not ok_w:
                    return PatchApplyResult(
                        success=False,
                        file_path=target_path,
                        resolved_path=resolved,
                        mode="none",
                        error=w_err or f"Invalid target path: {target_path}",
                        syntax_valid=False,
                    )

        mode, payload = cls.parse_model_output(raw_output, old_content=old_content, target_path=resolved)
        if mode == "none":
            lang_spec = _lang_spec_for_path(resolved)
            fence_tag = lang_spec["fence"] or "code"
            return PatchApplyResult(
                success=False,
                file_path=target_path,
                resolved_path=resolved,
                mode="none",
                error=f"Model output did not contain a valid ```{fence_tag}``` block or SEARCH/REPLACE patch for '{target_path}'.",
                syntax_valid=False,
            )

        is_py = resolved.lower().endswith(".py")
        healed_actions: List[str] = []
        if mode == "search_replace":
            working = old_content
            for search_blk, replace_blk in payload:
                updated, err, _ = smart_find_and_replace(working, search_blk, replace_blk)
                if err:
                    return PatchApplyResult(
                        success=False,
                        file_path=target_path,
                        resolved_path=resolved,
                        mode=mode,
                        error=f"Patch apply error: {err}",
                        syntax_valid=True,
                    )
                working = updated or working
            new_content, healed_actions = SymbolicHealer.heal_python_code(working, resolved)
        elif is_py:
            candidate_code, resolved_mode, surgical_notes = SurgicalCodePatcher.apply_surgical_or_full(
                old_content,
                str(payload),
                focus_class=focus_class,
            )
            if resolved_mode:
                mode = resolved_mode
            new_content, healed_actions = SymbolicHealer.heal_python_code(candidate_code, resolved)
            if surgical_notes:
                healed_actions = list(surgical_notes) + list(healed_actions)
        else:
            new_content, healed_actions = SymbolicHealer.heal_python_code(str(payload), resolved)

        # Prevent destructive shrinkage of existing non-trivial files into tiny stubs
        if old_content and len(old_content.splitlines()) >= 8 and len(new_content.splitlines()) < 3:
            return PatchApplyResult(
                success=False,
                file_path=target_path,
                resolved_path=resolved,
                mode=mode,
                healed_actions=healed_actions,
                error="Refused destructive rewrite: replacement collapsed existing file into < 3 lines.",
                syntax_valid=False,
            )

        if is_py:
            allow_write, rollback_err, _ = guard.check_python(resolved, new_content, enabled=True)[:3]
            if not allow_write:
                return PatchApplyResult(
                    success=False,
                    file_path=target_path,
                    resolved_path=resolved,
                    mode=mode,
                    healed_actions=healed_actions,
                    error=rollback_err or "SyntaxError in generated code.",
                    syntax_valid=False,
                )

        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)

        diff = build_compact_diff(old_content, new_content, resolved) if old_content else "(new file created)"
        return PatchApplyResult(
            success=True,
            file_path=target_path,
            resolved_path=resolved,
            mode=mode,
            healed_actions=healed_actions,
            diff=diff,
            syntax_valid=True,
        )
