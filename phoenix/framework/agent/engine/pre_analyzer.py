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
    check_runtime_path,
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
    r"(?<![A-Za-z0-9_./\\-])([A-Za-z0-9_./\\-]+\.(?:py|js|jsx|ts|tsx|html|htm|css|scss|sass|less|json|ya?ml|toml|md|txt|sql|sh|csv|xml|svg|vue|svelte))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_DIR_SLASH_REGEX = re.compile(
    r"(?<![A-Za-z0-9_.:/\\-])([A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)*/)(?![A-Za-z0-9_])"
)
_NAMED_FOLDER_REGEX = re.compile(
    r"\b(?:folder|directory|dir)\s+(?:called|named)\s+[`'\"]?([A-Za-z0-9_][A-Za-z0-9_.-]*)/?[`'\"]?",
    re.IGNORECASE,
)
_COMMON_SUBDIR_NAMES: Set[str] = {
    "front", "frontend", "client", "web", "ui", "public", "static", "templates",
    "assets", "styles", "css", "js", "scripts", "back", "backend", "server",
    "api", "app", "src", "lib", "core", "routes", "models", "schemas",
    "services", "utils", "helpers", "tests", "test", "components", "pages",
    "views", "controllers", "db", "database", "config", "docs",
}
_BACKTICK_SYMBOL_REGEX = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?`")


def _damerau_levenshtein(a: str, b: str) -> int:
    """Edit distance supporting single adjacent character transposition (e.g. `cacl_proj` <-> `calc_proj`)."""
    la, lb = len(a), len(b)
    if a == b:
        return 0
    if abs(la - lb) > 2:
        return abs(la - lb)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _extract_structured_paths(text: str, *, has_traceback: bool = False) -> List[str]:
    """Extract target file paths while preserving natural-language folder hierarchies (e.g. `calc_proj/front/c.html`)."""
    raw_files: List[Tuple[int, str]] = []
    for m in _FILENAME_REGEX.finditer(text):
        cand = m.group(1).strip().strip("'\",.;:")
        if not cand or cand.lower().startswith(("http://", "https://")):
            continue
        # Skip URL hostnames like `example.com/a.js`
        prefix = text[max(0, m.start() - 10) : m.start()].lower()
        if "://" in prefix:
            continue
        raw_files.append((m.start(), cand.replace("\\", "/")))

    if not raw_files:
        return []
    if has_traceback:
        seen_tb: List[str] = []
        for _, f in raw_files:
            if f not in seen_tb:
                seen_tb.append(f)
        return seen_tb

    # Collect directory cues (explicit `foo/` tokens + `folder called foo`)
    raw_dirs: List[Tuple[int, str, str]] = []
    covered_spans: List[Tuple[int, int]] = []
    for m in _DIR_SLASH_REGEX.finditer(text):
        d_raw = m.group(1).strip("/")
        if not d_raw or "." in d_raw.split("/")[-1]:
            continue
        # Skip if immediately preceded by `:` or `://` or `/` (URL or absolute system path like `/kaggle/working/...`)
        prev_ch = text[m.start() - 1] if m.start() > 0 else ""
        if prev_ch in ("/", ":"):
            continue
        ctx_before = text[max(0, m.start() - 45) : m.start()].lower()
        raw_dirs.append((m.start(), d_raw, ctx_before))
        covered_spans.append((m.start(), m.end()))

    for m in _NAMED_FOLDER_REGEX.finditer(text):
        if any(s <= m.start(1) <= e for s, e in covered_spans):
            continue
        d_raw = m.group(1).strip("/")
        if d_raw and "." not in d_raw:
            ctx_before = text[max(0, m.start() - 30) : m.start()].lower()
            raw_dirs.append((m.start(1), d_raw, ctx_before + " called"))

    if not raw_dirs:
        seen_flat: List[str] = []
        for _, f in raw_files:
            if f not in seen_flat:
                seen_flat.append(f)
        return seen_flat

    # Canonicalize single-typo duplicate root directory names (e.g. `cacl_proj` + `calc_proj` -> `calc_proj`)
    low_prompt = text.lower()
    single_dirs = [d for _, d, _ in raw_dirs if "/" not in d]
    unique_single = list(dict.fromkeys(single_dirs))
    alias_map: Dict[str, str] = {}
    for i in range(len(unique_single)):
        for j in range(i + 1, len(unique_single)):
            a, b = unique_single[i], unique_single[j]
            al, bl = a.lower(), b.lower()
            if len(al) >= 5 and len(bl) >= 5 and al not in _COMMON_SUBDIR_NAMES and bl not in _COMMON_SUBDIR_NAMES:
                if _damerau_levenshtein(al, bl) <= 1:
                    # Prefer spelling whose root stem appears elsewhere in prompt (e.g. `calc` in `calculator`), else later spelling
                    a_stem = al.split("_")[0].split("-")[0]
                    b_stem = bl.split("_")[0].split("-")[0]
                    a_hits = len(re.findall(re.escape(a_stem), low_prompt)) if len(a_stem) >= 3 else 0
                    b_hits = len(re.findall(re.escape(b_stem), low_prompt)) if len(b_stem) >= 3 else 0
                    canon = a if a_hits > b_hits else b
                    alias_map[a] = canon
                    alias_map[b] = canon

    def _canon_dir(d: str) -> str:
        parts = [alias_map.get(seg, seg) for seg in d.split("/") if seg]
        return "/".join(parts)

    events: List[Tuple[int, str, str, str]] = []
    for pos, d_raw, ctx in raw_dirs:
        events.append((pos, "dir", _canon_dir(d_raw), ctx))
    for pos, f_raw in raw_files:
        events.append((pos, "file", f_raw, ""))
    events.sort(key=lambda e: e[0])

    root_dir: Optional[str] = None
    active_dir: Optional[str] = None
    resolved: List[str] = []
    sub_cues = ("contain", "inside", "under", "subfolder", "called", "named", "with", "into")

    for _, kind, val, ctx in events:
        if kind == "dir":
            if "/" in val:
                active_dir = val
                if root_dir is None:
                    root_dir = val.split("/")[0]
            else:
                vlow = val.lower()
                if root_dir is not None and val == root_dir:
                    active_dir = root_dir
                elif root_dir is not None and (
                    vlow in _COMMON_SUBDIR_NAMES
                    or active_dir == root_dir
                    or any(cue in ctx for cue in sub_cues)
                ):
                    active_dir = f"{root_dir}/{val}"
                elif root_dir is None and vlow not in _COMMON_SUBDIR_NAMES:
                    root_dir = val
                    active_dir = val
                else:
                    active_dir = val
        else:
            if "/" in val:
                segs = val.split("/")
                segs[0] = alias_map.get(segs[0], segs[0])
                final_path = "/".join(segs)
            elif active_dir:
                final_path = f"{active_dir}/{val}"
            else:
                final_path = val
            if final_path not in resolved:
                resolved.append(final_path)

    return resolved


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

        # File extraction from prompt (structured folder hierarchies, explicit extensions, workspace stems, or single .py fallback)
        candidates: List[str] = _extract_structured_paths(text, has_traceback=has_traceback)

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
        existing_targets: List[str] = []
        for cand in candidates:
            # Explicit relative subdirectory path (e.g. `calc_proj/front/c.js`): check exact location so a stray root `c.js` never hijacks it!
            if "/" in cand and not os.path.isabs(cand) and not has_traceback:
                allowed, abs_direct, _ = check_runtime_path(runtime, cand)
                if allowed and os.path.isfile(abs_direct):
                    rel = os.path.relpath(abs_direct, ws_root).replace("\\", "/") if ws_root else cand
                    if rel not in resolved_targets:
                        resolved_targets.append(rel)
                    if rel not in existing_targets:
                        existing_targets.append(rel)
                else:
                    if cand not in resolved_targets:
                        resolved_targets.append(cand)
                continue

            ok, resolved, _, _ = resolve_existing_path(cand, runtime, allow_dir=False)
            if ok and os.path.isfile(resolved):
                rel = os.path.relpath(resolved, ws_root).replace("\\", "/") if ws_root else resolved
                if rel not in resolved_targets:
                    resolved_targets.append(rel)
                if rel not in existing_targets:
                    existing_targets.append(rel)
            else:
                if cand not in resolved_targets:
                    resolved_targets.append(cand)

        # Detect archetype cleanly
        create_verbs = ("create", "new file", "write a new", "generate file", "scaffold", "build a ", "build simple", "أنشئ", "انشئ")
        qa_verbs = ("explain", "summarize", "what does", "read ", "inspect ", "analyze ", "اشرح", "اقرأ")
        modify_verbs = (
            "update", "fix", "edit", "modify", "make it", "secure", "harden",
            "refactor", "support", "souport", "add ", "improve", "upgrade",
            "traceback", "exception", "valueerror", "attributeerror", "nameerror", "typeerror",
            "عدل", "صلح", "حسن", "دعم",
        )

        has_missing_targets = len(existing_targets) < len(resolved_targets)
        if existing_targets and not has_missing_targets and any(v in low for v in qa_verbs) and not (any(v in low for v in modify_verbs) or has_traceback):
            archetype = "code_qa"
        elif resolved_targets and has_missing_targets and (any(v in low for v in create_verbs) or not existing_targets):
            archetype = "code_create"
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
