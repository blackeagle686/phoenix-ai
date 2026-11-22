"""Plan 1: Deterministic Pre-Cognition & AST/Security Diagnostic Engine.

Performs zero-LLM workspace discovery, file hydration, AST symbol extraction,
security risk detection, and intent contract extraction BEFORE the first LLM call.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from phoenix.framework.agent.cognition.schema import PlanSchema, TaskSchema, TaskType
from phoenix.framework.agent.tools._fs_utils import (
    get_workspace_root,
    is_ignored_dir,
    resolve_existing_path,
)

KNOWN_MATH_OPS: Tuple[str, ...] = (
    "sin", "cos", "tan", "tanh", "sinh", "cosh",
    "asin", "acos", "atan", "asinh", "acosh", "atanh",
    "sqrt", "log", "log10", "log2", "exp", "pow",
    "factorial", "radians", "degrees", "fabs", "floor", "ceil",
)

DANGEROUS_CALL_NAMES: Set[str] = {
    "eval", "exec", "compile", "__import__", "breakpoint",
}

DANGEROUS_IO_CALLS: Set[str] = {
    "open", "input",
}

DANGEROUS_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pty", "socket",
    "ctypes", "pickle", "marshal", "importlib", "builtins",
}

DANGEROUS_DUNDERS: Set[str] = {
    "__subclasses__", "__bases__", "__mro__", "__globals__",
    "__builtins__", "__import__", "__code__", "__closure__",
}

_SECURITY_KEYWORDS = (
    "secure", "security", "unsafe", "danger", "dangerous", "harden",
    "sanitize", "sandbox", "safe", "vulnerability", "injection",
    "no eval", "without eval", "remove eval", "calculation only",
    "calculation operation", "just for calculation", "only math",
    "آمن", "أمان", "امان", "حماية", "خطر", "تأمين",
)

_CALC_ONLY_KEYWORDS = (
    "calculation", "calculator", "calc", "math", "arithmetic", "equation",
    "equations", "geomitric", "geometric", "trigonometric", "trig",
)

_FILENAME_REGEX = re.compile(
    r"(?<![A-Za-z0-9_./\\-])([A-Za-z0-9_./\\-]+\.(?:py|js|jsx|ts|tsx|json|ya?ml|toml|md|txt|sql|sh|csv))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_BACKTICK_SYMBOL_REGEX = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?`")


@dataclass
class IntentContract:
    """Deterministic contract inferred from the user request."""

    archetype: str = "general"  # code_modify | code_create | code_qa | general
    target_files: List[str] = field(default_factory=list)
    requires_security: bool = False
    is_calculator_domain: bool = False
    has_traceback: bool = False
    traceback_error: Optional[str] = None
    traceback_lines: List[int] = field(default_factory=list)
    ast_node_hints: List[str] = field(default_factory=list)
    forbidden_constructs: List[str] = field(default_factory=list)
    required_ops: List[str] = field(default_factory=list)
    required_symbols: List[str] = field(default_factory=list)


@dataclass
class FileDiagnostic:
    """Static AST + security scan result for a single workspace file."""

    requested_path: str
    resolved_path: str
    exists: bool = False
    line_count: int = 0
    raw_content: str = ""
    numbered_content: str = ""
    ast_valid: bool = True
    ast_error: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    defined_names: Set[str] = field(default_factory=set)
    security_findings: List[str] = field(default_factory=list)
    missing_ops: List[str] = field(default_factory=list)
    has_top_level_input: bool = False
    has_interactive_loop: bool = False


@dataclass
class DiagnosticReport:
    """Unified pre-cognition report passed to the Playbook, LLM, and Verifier."""

    prompt: str
    contract: IntentContract
    files: List[FileDiagnostic] = field(default_factory=list)
    pre_read_paths: List[str] = field(default_factory=list)
    summary_brief: str = ""


def _number_lines(content: str, max_lines: int = 160) -> str:
    lines = content.splitlines()
    if not lines:
        return "(empty file)"
    if len(lines) <= max_lines:
        return "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
    head_cnt = max_lines // 2
    tail_cnt = max_lines - head_cnt
    head = [f"{i + 1}: {lines[i]}" for i in range(head_cnt)]
    omitted = len(lines) - max_lines
    tail_start = len(lines) - tail_cnt
    tail = [f"{i + 1}: {lines[i]}" for i in range(tail_start, len(lines))]
    return "\n".join(head + [f"... ({omitted} lines omitted) ..."] + tail)


def scan_python_ast(
    code: str,
    requires_security: bool = False,
    required_ops: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically inspect Python code via AST without executing it."""
    result: Dict[str, Any] = {
        "ast_valid": True,
        "ast_error": None,
        "symbols": [],
        "defined_names": set(),
        "referenced_ops": set(),
        "security_findings": [],
        "has_top_level_input": False,
        "has_interactive_loop": False,
        "missing_ops": [],
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["ast_valid"] = False
        result["ast_error"] = f"SyntaxError line {exc.lineno}: {exc.msg}"
        return result

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            end_ln = getattr(node, "end_lineno", node.lineno)
            result["symbols"].append(f"def {node.name}({', '.join(args)}) [L{node.lineno}-L{end_ln}]")
            result["defined_names"].add(node.name)
        elif isinstance(node, ast.ClassDef):
            end_ln = getattr(node, "end_lineno", node.lineno)
            result["symbols"].append(f"class {node.name} [L{node.lineno}-L{end_ln}]")
            result["defined_names"].add(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result["defined_names"].add(item.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    result["defined_names"].add(tgt.id)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "input":
                    result["has_top_level_input"] = True
        elif isinstance(node, (ast.Expr, ast.While, ast.If)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "input":
                    result["has_top_level_input"] = True
                    if isinstance(node, ast.While) or any(isinstance(w, ast.While) for w in ast.walk(node)):
                        result["has_interactive_loop"] = True

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)
        if isinstance(node, ast.While):
            if any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "input" for sub in ast.walk(node)):
                result["has_interactive_loop"] = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_mod = alias.name.split(".")[0]
                if base_mod in DANGEROUS_MODULES:
                    result["security_findings"].append(f"L{lineno}: imports dangerous module '{base_mod}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_mod = node.module.split(".")[0]
                if base_mod in DANGEROUS_MODULES:
                    result["security_findings"].append(f"L{lineno}: imports from dangerous module '{base_mod}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
                if fn_name in DANGEROUS_CALL_NAMES:
                    result["security_findings"].append(f"L{lineno}: unsafe call to '{fn_name}()'")
                elif requires_security and fn_name in DANGEROUS_IO_CALLS and fn_name == "open":
                    result["security_findings"].append(f"L{lineno}: file I/O call 'open()' in restricted calculation context")
                if fn_name in KNOWN_MATH_OPS:
                    result["referenced_ops"].add(fn_name)
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in KNOWN_MATH_OPS:
                    result["referenced_ops"].add(attr)
                if isinstance(node.func.value, ast.Name) and node.func.value.id in DANGEROUS_MODULES:
                    result["security_findings"].append(
                        f"L{lineno}: dangerous module call '{node.func.value.id}.{attr}()'"
                    )
        elif isinstance(node, ast.Attribute):
            if node.attr in DANGEROUS_DUNDERS:
                result["security_findings"].append(f"L{lineno}: restricted dunder attribute '{node.attr}'")
            if node.attr in KNOWN_MATH_OPS:
                result["referenced_ops"].add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if val in KNOWN_MATH_OPS:
                result["referenced_ops"].add(val)

    for op in (required_ops or []):
        if (
            op not in result["defined_names"]
            and op not in result["referenced_ops"]
            and not re.search(rf"\b{re.escape(op)}\b", code)
        ):
            result["missing_ops"].append(op)

    return result


class PreAnalyzer:
    """Zero-LLM pre-cognition analyzer for files, contracts, AST diagnostics, and playbooks."""

    @staticmethod
    def _extract_contract(prompt: str, runtime: Any = None) -> IntentContract:
        text = str(prompt or "")
        low = text.lower()

        requires_security = any(kw in low for kw in _SECURITY_KEYWORDS)
        is_calc = any(kw in low for kw in _CALC_ONLY_KEYWORDS)

        has_traceback = bool(
            re.search(
                r"traceback\s*\(most recent call last\)|file\s+[\"'][^\"']+[\"'],\s+line\s+\d+|^\s*[A-Za-z_]*Error:",
                text,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        traceback_lines: List[int] = []
        for m in re.finditer(r"\bline\s+(\d+)\b", text):
            ln_no = int(m.group(1))
            if ln_no not in traceback_lines:
                traceback_lines.append(ln_no)
        traceback_error: Optional[str] = None
        if has_traceback:
            for ln in reversed([x.strip() for x in text.splitlines() if x.strip()]):
                if re.match(r"^[A-Za-z_]*(?:Error|Exception):", ln) or "Unsupported node type" in ln:
                    traceback_error = ln[:220]
                    break
        ast_node_hints: List[str] = []
        for m in re.finditer(r"ast\.([A-Za-z_][A-Za-z0-9_]*)", text):
            hint = f"ast.{m.group(1)}"
            if hint not in ast_node_hints and m.group(1) not in ("parse", "walk", "AST", "dump"):
                ast_node_hints.append(hint)

        # Extract requested math operations only from the user's descriptive text (never quoted traceback code)
        ops_source = (
            re.split(r"traceback\s*\(most recent call last\)|file\s+[\"']", text, maxsplit=1, flags=re.IGNORECASE)[0]
            if has_traceback
            else text
        ).lower()
        required_ops: List[str] = []
        for op in KNOWN_MATH_OPS:
            if re.search(rf"\b{re.escape(op)}\b", ops_source):
                required_ops.append(op)

        required_symbols: List[str] = []
        for match in _BACKTICK_SYMBOL_REGEX.finditer(text):
            sym = match.group(1)
            if sym not in KNOWN_MATH_OPS and "." not in sym and sym not in required_symbols:
                required_symbols.append(sym)

        # File extraction from prompt (explicit extensions, workspace stems, or single .py workspace fallback)
        candidates: List[str] = []
        for m in _FILENAME_REGEX.finditer(text):
            cand = m.group(1).strip().strip("'\",.;:")
            if cand and cand not in candidates:
                candidates.append(cand)

        ws_root = get_workspace_root(runtime) if runtime is not None else None
        if not candidates and runtime is not None and getattr(runtime, "workspace_root", None) and os.path.isdir(ws_root):
            py_files: List[str] = []
            for root, dirs, files in os.walk(ws_root):
                dirs[:] = [d for d in sorted(dirs) if not is_ignored_dir(d)]
                for fname in sorted(files):
                    if fname.startswith("."):
                        continue
                    rel_f = os.path.relpath(os.path.join(root, fname), ws_root)
                    stem = os.path.splitext(fname)[0].lower()
                    if len(stem) >= 3 and re.search(rf"\b{re.escape(stem)}\b", low):
                        if rel_f not in candidates:
                            candidates.append(rel_f)
                    if fname.lower().endswith(".py"):
                        py_files.append(rel_f)
                if len(py_files) >= 15:
                    break
            if not candidates and len(py_files) == 1 and (
                requires_security or is_calc or any(k in low for k in ("code", "file", "fix", "update", "script", "refactor", "bug"))
            ):
                candidates.append(py_files[0])

        resolved_targets: List[str] = []
        for cand in candidates:
            ok, resolved, _, _ = resolve_existing_path(cand, runtime, allow_dir=False)
            if ok:
                rel = os.path.relpath(resolved, ws_root) if ws_root else resolved
                if rel not in resolved_targets:
                    resolved_targets.append(rel)
            else:
                if cand not in resolved_targets:
                    resolved_targets.append(cand)

        # Detect archetype cleanly
        create_verbs = ("create", "new file", "write a new", "generate file", "أنشئ", "انشئ")
        qa_verbs = ("explain", "summarize", "what does", "read ", "inspect ", "analyze ", "اشرح", "اقرأ")
        modify_verbs = (
            "update", "fix", "edit", "modify", "make it", "secure", "harden",
            "refactor", "support", "souport", "add ", "improve", "upgrade",
            "traceback", "exception", "valueerror", "attributeerror", "nameerror", "typeerror",
            "عدل", "صلح", "حسن", "دعم",
        )

        existing_targets = []
        for tf in resolved_targets:
            ok, abs_p, _, _ = resolve_existing_path(tf, runtime, allow_dir=False)
            if ok and os.path.isfile(abs_p):
                existing_targets.append(tf)

        if existing_targets and any(v in low for v in qa_verbs) and not (any(v in low for v in modify_verbs) or has_traceback):
            archetype = "code_qa"
        elif existing_targets and (any(v in low for v in modify_verbs) or requires_security or required_ops or has_traceback):
            archetype = "code_modify"
        elif resolved_targets and (any(v in low for v in create_verbs) or not existing_targets):
            archetype = "code_create"
        else:
            archetype = "general"

        forbidden = sorted(DANGEROUS_CALL_NAMES | DANGEROUS_MODULES) if requires_security else []
        return IntentContract(
            archetype=archetype,
            target_files=resolved_targets,
            requires_security=requires_security,
            is_calculator_domain=is_calc,
            has_traceback=has_traceback,
            traceback_error=traceback_error,
            traceback_lines=traceback_lines,
            ast_node_hints=ast_node_hints,
            forbidden_constructs=forbidden,
            required_ops=required_ops,
            required_symbols=required_symbols,
        )

    @classmethod
    def analyze(cls, prompt: str, runtime: Any = None) -> DiagnosticReport:
        """Analyze prompt + workspace files deterministically before invoking any LLM."""
        contract = cls._extract_contract(prompt, runtime=runtime)
        diagnostics: List[FileDiagnostic] = []
        pre_read_paths: List[str] = []

        for tf in contract.target_files[:5]:
            ok, abs_path, _, _ = resolve_existing_path(tf, runtime, allow_dir=False)
            if ok and os.path.isfile(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except OSError:
                    content = ""
                pre_read_paths.append(os.path.realpath(abs_path))
                lines = content.splitlines()
                ast_info: Dict[str, Any] = {}
                if abs_path.lower().endswith(".py"):
                    ast_info = scan_python_ast(
                        content,
                        requires_security=contract.requires_security,
                        required_ops=contract.required_ops,
                    )
                diagnostics.append(
                    FileDiagnostic(
                        requested_path=tf,
                        resolved_path=abs_path,
                        exists=True,
                        line_count=len(lines),
                        raw_content=content,
                        numbered_content=_number_lines(content),
                        ast_valid=ast_info.get("ast_valid", True),
                        ast_error=ast_info.get("ast_error"),
                        symbols=ast_info.get("symbols", []),
                        defined_names=ast_info.get("defined_names", set()),
                        security_findings=ast_info.get("security_findings", []),
                        missing_ops=ast_info.get("missing_ops", []),
                        has_top_level_input=ast_info.get("has_top_level_input", False),
                        has_interactive_loop=ast_info.get("has_interactive_loop", False),
                    )
                )
            else:
                diagnostics.append(
                    FileDiagnostic(
                        requested_path=tf,
                        resolved_path=abs_path or tf,
                        exists=False,
                    )
                )

        # Build compact diagnostic brief
        brief_parts: List[str] = []
        if diagnostics:
            brief_parts.append("PRE-ANALYSIS DIAGNOSTIC BRIEF (auto-inspected before LLM execution):")
            for fd in diagnostics:
                if not fd.exists:
                    brief_parts.append(f"- {fd.requested_path}: NEW FILE (does not exist yet)")
                    continue
                brief_parts.append(f"- File `{fd.requested_path}` ({fd.line_count} lines, ast_valid={fd.ast_valid}):")
                if fd.ast_error:
                    brief_parts.append(f"  * Syntax Error: {fd.ast_error}")
                if fd.symbols:
                    brief_parts.append(f"  * Symbols: {', '.join(fd.symbols[:12])}")
                if fd.security_findings:
                    brief_parts.append(f"  * SECURITY RISKS DETECTED: {'; '.join(fd.security_findings[:8])}")
                if fd.missing_ops:
                    brief_parts.append(f"  * MISSING REQUESTED OPS: {', '.join(fd.missing_ops)}")
                brief_parts.append(f"  * Current Content:\n```python\n{fd.numbered_content}\n```")

        if contract.requires_security and contract.is_calculator_domain:
            brief_parts.append(
                "SAFE ARCHITECTURE RULE: Do NOT use eval(), exec(), compile(), __import__(), or os/subprocess. "
                "Instead use `ast.parse(expr, mode='eval')` with an explicit recursive node evaluator supporting "
                "ast.Constant, ast.BinOp (+, -, *, /, //, %, **), ast.UnaryOp (+, -), ast.Name (pi, e), and "
                "whitelisted ast.Call functions via `math`."
            )

        return DiagnosticReport(
            prompt=prompt,
            contract=contract,
            files=diagnostics,
            pre_read_paths=pre_read_paths,
            summary_brief="\n".join(brief_parts),
        )

    @classmethod
    def build_playbook_plan(cls, prompt: str, report: DiagnosticReport) -> Optional[PlanSchema]:
        """Deterministically construct a minimal cohesive PlanSchema for well-defined code tasks."""
        c = report.contract
        if c.archetype not in ("code_modify", "code_create") or not c.target_files:
            return None

        target_str = ", ".join(c.target_files)
        expectations: List[str] = [f"{target_str} has valid Python AST syntax and preserves required functionality"]
        if c.requires_security:
            expectations.append("Zero unsafe calls (no eval/exec/compile/__import__/os/subprocess)")
        if c.required_ops:
            expectations.append(f"Supports operations: {', '.join(c.required_ops)}")

        task_type = TaskType.UPDATE if c.archetype == "code_modify" else TaskType.WRITE
        return PlanSchema(
            objective=prompt[:240],
            tasks=[
                TaskSchema(
                    task_id="t1",
                    task_type=task_type,
                    description=prompt,
                    target_files=list(c.target_files),
                    expected_result="; ".join(expectations),
                )
            ],
        )
