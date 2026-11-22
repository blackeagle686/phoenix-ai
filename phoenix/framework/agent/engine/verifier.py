"""Plan 3: Execution-Driven Verifier & Adversarial Sandbox Probe Engine.

Replaces subjective self-grading by weak LLMs with deterministic AST syntax/security
gates, symbol & math-operation verification, live execution smoke tests, and
adversarial injection probes inside RestrictedPythonRuntime.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from phoenix.framework.agent.engine.pre_analyzer import (
    DiagnosticReport,
    scan_python_ast,
)
from phoenix.framework.agent.engine.surgical import (
    diagnose_python_error,
    find_enclosing_callable,
)
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.tools._fs_utils import resolve_existing_path

ADVERSARIAL_PROBES: List[str] = [
    "__import__('os').system('echo PWNED')",
    "open('/etc/passwd').read()",
    "eval('2 + 2')",
    "exec('x = 1')",
    "(1).__class__.__mro__[1].__subclasses__()",
]

MATH_FUNC_EXPECTATIONS: Dict[str, tuple] = {
    "sin": (0, 0.0),
    "cos": (0, 1.0),
    "tan": (0, 0.0),
    "tanh": (0, 0.0),
    "sinh": (0, 0.0),
    "cosh": (0, 1.0),
    "sqrt": (4, 2.0),
    "exp": (0, 1.0),
    "log": (1, 0.0),
    "log10": (1, 0.0),
    "atan": (0, 0.0),
    "asin": (0, 0.0),
    "acos": (1, 0.0),
}

_PREFERRED_EVAL_NAMES: Tuple[str, ...] = (
    "calculate", "calc", "evaluate", "eval_expr", "evaluate_expression",
    "solve", "compute", "run_calc", "safe_calc", "safe_eval", "math_eval",
)

_EXPR_ARG_NAMES = frozenset({
    "expr", "expression", "formula", "equation", "math_expr", "expr_str",
    "raw_expr", "query", "statement", "text", "s", "inp", "input_str",
})


def _accepts_single_arg(fn_node: ast.AST, is_method: bool = False) -> bool:
    """Return True if *fn_node* can be invoked with a single positional argument."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    pos_args = fn_node.args.args[1:] if is_method else fn_node.args.args
    required_pos = max(0, len(pos_args) - len(fn_node.args.defaults))
    if required_pos > 1:
        return False
    if len(pos_args) == 0 and fn_node.args.vararg is None:
        return False
    if any(default is None for default in fn_node.args.kw_defaults):
        return False
    return True


def _looks_like_expr_evaluator(fn_node: ast.AST, is_method: bool = False) -> bool:
    """Return True only when *fn_node* plausibly evaluates a string math expression."""
    if not _accepts_single_arg(fn_node, is_method=is_method):
        return False
    fn_low = fn_node.name.lower()
    if fn_low in _PREFERRED_EVAL_NAMES:
        return True
    if "eval" in fn_low or "expr" in fn_low:
        return True
    pos_args = fn_node.args.args[1:] if is_method else fn_node.args.args
    first_arg = pos_args[0].arg.lower() if pos_args else ""
    if any(tok in fn_low for tok in ("calc", "solve", "compute")) and first_arg in _EXPR_ARG_NAMES:
        return True
    return False


@dataclass
class VerificationReport:
    """Deterministic multi-gate score and actionable failure feedback."""

    passed: bool
    score: int  # 0-100
    rating: int  # 1-10
    checks_passed: List[str] = field(default_factory=list)
    discrepancies: List[str] = field(default_factory=list)
    corrective_action: Optional[str] = None
    functional_outputs: List[str] = field(default_factory=list)
    focus_line: Optional[int] = None
    focus_scope: Optional[Dict[str, Any]] = None
    diagnosis_hints: List[str] = field(default_factory=list)


def _is_ast_node_visitor(fn_node: ast.FunctionDef) -> bool:
    """Detect methods/functions that take an ast.AST node rather than an expression string."""
    arg_names = [a.arg.lower() for a in fn_node.args.args if a.arg != "self"]
    if not arg_names:
        return False
    calls_parse = any(
        isinstance(sub, ast.Attribute) and sub.attr == "parse"
        for sub in ast.walk(fn_node)
    )
    if calls_parse:
        return False
    first_arg = arg_names[0]
    fn_low = fn_node.name.lower()
    if "node" in first_arg or "ast" in first_arg or "node" in fn_low or fn_low.startswith("visit"):
        return True
    # Also inspect AST attribute checks on the parameter (e.g. node.left / node.body / node.value)
    node_attrs = {"left", "right", "operand", "value", "body", "func", "args", "op"}
    for sub in ast.walk(fn_node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == fn_node.args.args[-len(arg_names)].arg
            and sub.attr in node_attrs
        ):
            return True
    return False


def _node_contains_input_or_interactive_loop(node: ast.AST, interactive_funcs: Optional[set] = None) -> bool:
    """Return True if an AST subtree directly invokes `input()`, reads `stdin`, or calls an interactive function."""
    inter = interactive_funcs or set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                if sub.func.id == "input" or sub.func.id in inter:
                    return True
            elif isinstance(sub.func, ast.Attribute):
                if sub.func.attr in inter:
                    return True
                if (
                    sub.func.attr in ("read", "readline", "readlines")
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == "stdin"
                ):
                    return True
        elif isinstance(sub, ast.While):
            for inner in ast.walk(sub):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id == "input" or inner.func.id in inter:
                        return True
    return False


def _find_interactive_functions(tree: ast.Module) -> set:
    """Build transitive call graph of functions/methods containing interactive input() or CLI while loops."""
    fn_nodes: List[Tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_nodes.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn_nodes.append((item.name, item))

    interactive: set = set()
    for fname, fnode in fn_nodes:
        if _node_contains_input_or_interactive_loop(fnode):
            interactive.add(fname)

    # Transitive closure: if main() calls run_cli() which calls input(), mark main() interactive too
    changed = True
    while changed:
        changed = False
        for fname, fnode in fn_nodes:
            if fname in interactive:
                continue
            if _node_contains_input_or_interactive_loop(fnode, interactive):
                interactive.add(fname)
                changed = True
    return interactive


def _neutralize_lines_preserving_count(lines: List[str], node: ast.AST) -> None:
    """Replace an AST statement's span with `pass` + blank lines so 1-indexed line numbers stay exact."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1) - 1
    if start < 0 or start >= len(lines):
        return
    raw_first = lines[start]
    indent = raw_first[: len(raw_first) - len(raw_first.lstrip(" \t"))]
    lines[start] = f"{indent}pass"
    for idx in range(start + 1, min(end + 1, len(lines))):
        lines[idx] = ""


def _prepare_module_for_sandbox(
    code: str,
    required_ops: List[str],
) -> Tuple[str, List[str], List[str], Dict[str, List[str]], List[str], Dict[str, List[str]]]:
    """Neutralize blocking top-level/CLI interactive loops while keeping line numbers exact and preserving non-interactive tests."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, [], [], {}, [], {}

    interactive_funcs = _find_interactive_functions(tree)
    lines = code.splitlines()
    any_reusable_funcs: List[str] = []
    discovered_str_funcs: List[str] = []
    discovered_node_funcs: List[str] = []
    discovered_classes: List[str] = []
    class_str_methods: Dict[str, List[str]] = {}
    class_node_methods: Dict[str, List[str]] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.name not in ("main", "run_console", "cli")
                and node.name not in required_ops
                and node.name not in interactive_funcs
            ):
                any_reusable_funcs.append(node.name)
                if len(node.args.args) >= 1:
                    if _is_ast_node_visitor(node):
                        discovered_node_funcs.append(node.name)
                    elif _looks_like_expr_evaluator(node, is_method=False):
                        discovered_str_funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            discovered_classes.append(node.name)
            s_methods: List[str] = []
            n_methods: List[str] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    if item.name in interactive_funcs:
                        continue
                    if len(item.args.args) >= 2:  # self + at least 1 argument
                        if _is_ast_node_visitor(item):
                            n_methods.append(item.name)
                        elif _looks_like_expr_evaluator(item, is_method=True):
                            s_methods.append(item.name)
            if s_methods:
                class_str_methods[node.name] = s_methods
            if n_methods:
                class_node_methods[node.name] = n_methods

    has_reusable_api = bool(
        any_reusable_funcs or discovered_str_funcs or discovered_node_funcs or class_str_methods or class_node_methods or discovered_classes
    )

    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                start = node.lineno - 1
                if start < len(lines):
                    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
                    lines[start] = f"{indent}if True:"
                # Keep non-interactive self-test statements active; neutralize interactive CLI or daemon `while True` loops
                for sub_stmt in node.body:
                    is_unbounded_while_true = (
                        has_reusable_api
                        and isinstance(sub_stmt, ast.While)
                        and isinstance(sub_stmt.test, ast.Constant)
                        and bool(sub_stmt.test.value) is True
                    )
                    if is_unbounded_while_true or _node_contains_input_or_interactive_loop(sub_stmt, interactive_funcs):
                        _neutralize_lines_preserving_count(lines, sub_stmt)
        elif has_reusable_api and _node_contains_input_or_interactive_loop(node, interactive_funcs):
            # Also neutralize bare top-level CLI loops (`while True: input(...)`) or bare `main()` calls
            # when the module defines functions/classes to verify, preserving 1:1 line offsets.
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                _neutralize_lines_preserving_count(lines, node)

    cleaned = "\n".join(lines)

    ordered_str_funcs: List[str] = []
    for name in list(_PREFERRED_EVAL_NAMES) + discovered_str_funcs:
        if name not in discovered_node_funcs and name not in interactive_funcs and name not in ordered_str_funcs:
            ordered_str_funcs.append(name)

    return cleaned, ordered_str_funcs, discovered_classes, class_str_methods, discovered_node_funcs, class_node_methods


def _build_sandbox_harness(
    module_code: str,
    required_ops: List[str],
    requires_security: bool,
    is_calculator: bool,
    ast_node_hints: Optional[List[str]] = None,
) -> str:
    """Build an isolated Python test harness compatible with RestrictedPythonRuntime AST constraints."""
    (
        safe_module_code,
        candidate_funcs,
        candidate_classes,
        class_str_methods,
        node_funcs,
        class_node_methods,
    ) = _prepare_module_for_sandbox(module_code, required_ops)

    # Generate NameError-safe namespace binding without restricted `globals()`/`vars()`/`dir()`/`hasattr()`
    capture_lines: List[str] = ["    _ns = {}"]
    names_to_capture = list(dict.fromkeys(candidate_funcs + node_funcs + list(required_ops) + candidate_classes))
    for sym in names_to_capture:
        if sym.isidentifier():
            capture_lines.append(f"    try:\n        _ns[{sym!r}] = {sym}\n    except Exception:\n        pass")
    capture_block = "\n".join(capture_lines)

    def _build_class_probes(target_var: str, methods_map: Dict[str, List[str]]) -> str:
        lines_out: List[str] = []
        for cname in candidate_classes:
            methods = methods_map.get(cname, [])
            preferred = [m for m in _PREFERRED_EVAL_NAMES if m in methods] + [m for m in methods if m not in _PREFERRED_EVAL_NAMES]
            for mname in preferred:
                if cname.isidentifier() and mname.isidentifier():
                    lines_out.append(
                        f"        if {target_var} is None:\n"
                        f"            try:\n"
                        f"                _inst = {cname}()\n"
                        f"                _m = _inst.{mname}\n"
                        f"                if callable(_m):\n"
                        f"                    {target_var} = _m\n"
                        f"            except Exception:\n"
                        f"                pass"
                    )
        return "\n".join(lines_out) if lines_out else "        pass"

    class_str_probe_block = _build_class_probes("_expr_fn", class_str_methods)
    class_node_probe_block = _build_class_probes("_node_fn", class_node_methods)

    harness_cfg = json.dumps({
        "required_ops": required_ops,
        "requires_security": requires_security,
        "is_calculator": is_calculator,
        "candidate_funcs": candidate_funcs,
        "node_funcs": node_funcs,
        "ast_node_hints": ast_node_hints or [],
        "probes": ADVERSARIAL_PROBES,
        "expectations": {k: list(v) for k, v in MATH_FUNC_EXPECTATIONS.items()},
    })

    return f"""
import ast as _v_ast
import json as _v_json
import math as _v_math

# --- BEGIN TARGET MODULE ---
{safe_module_code}
# --- END TARGET MODULE ---

def _v_run_checks():
    _cfg = _v_json.loads({harness_cfg!r})
{capture_block}
    _out = {{"passed": [], "failed": [], "outputs": [], "security_breach": False}}

    _expr_fn = None
    for _name in _cfg["candidate_funcs"]:
        _cand = _ns.get(_name)
        if callable(_cand):
            _expr_fn = _cand
            break

    if _expr_fn is None:
{class_str_probe_block}

    _node_fn = None
    for _nname in _cfg["node_funcs"]:
        _ncand = _ns.get(_nname)
        if callable(_ncand):
            _node_fn = _ncand
            break

    if _node_fn is None:
{class_node_probe_block}

    def _looks_like_safe_rejection(val):
        if val is None or isinstance(val, bool):
            return True
        if isinstance(val, str):
            low = val.lower()
            if any(w in low for w in ("error", "invalid", "unsafe", "forbidden", "blocked", "not allowed", "unsupported", "denied", "disallowed")):
                return True
            if "pwned" in low or "root:" in low or "subclasses" in low:
                return False
        return False

    if _expr_fn is not None:
        # 1. Basic arithmetic check
        try:
            _r_arith = _expr_fn("2 + 3 * 4")
            _num = float(_r_arith) if not isinstance(_r_arith, (int, float)) else _r_arith
            if abs(_num - 14.0) < 1e-6:
                _out["passed"].append("arith: 2+3*4 == 14")
                _out["outputs"].append("calc('2 + 3 * 4') -> " + repr(_r_arith))
            else:
                _out["failed"].append("Arithmetic check failed: evaluator returned " + repr(_r_arith) + " for '2 + 3 * 4' instead of 14")
        except Exception as _e:
            _out["failed"].append("Arithmetic check raised " + type(_e).__name__ + " on '2 + 3 * 4': " + str(_e))

        # 2. Required trig/math ops via evaluator OR standalone function
        for _op in _cfg["required_ops"]:
            _exp_pair = _cfg["expectations"].get(_op)
            if not _exp_pair:
                continue
            _arg, _expected = _exp_pair[0], _exp_pair[1]
            _expr_str = _op + "(" + str(_arg) + ")"
            _fn_op = _ns.get(_op)
            _ok_op = False
            _err_op = None
            try:
                _res_op = _expr_fn(_expr_str)
                _val_op = float(_res_op)
                if abs(_val_op - _expected) < 1e-5:
                    _ok_op = True
                    _out["passed"].append("op:" + _expr_str + "==" + str(_res_op))
                    _out["outputs"].append(_expr_str + " -> " + repr(_res_op))
                else:
                    _err_op = _expr_str + " returned " + repr(_res_op) + " (expected " + str(_expected) + ")"
            except Exception as _e:
                _err_op = _expr_str + " raised " + type(_e).__name__ + ": " + str(_e)

            if not _ok_op and callable(_fn_op):
                try:
                    _res_direct = _fn_op(_arg)
                    _val_direct = float(_res_direct)
                    if abs(_val_direct - _expected) < 1e-5:
                        _ok_op = True
                        _out["passed"].append("func:" + _op + "(" + str(_arg) + ")==" + str(_res_direct))
                        _out["outputs"].append(_op + "(" + str(_arg) + ") -> " + repr(_res_direct))
                except Exception:
                    pass

            if not _ok_op and _err_op:
                _out["failed"].append(_err_op)

        # 3. Adversarial injection probes
        if _cfg["requires_security"]:
            for _probe in _cfg["probes"]:
                try:
                    _p_res = _expr_fn(_probe)
                    if not _looks_like_safe_rejection(_p_res):
                        _out["failed"].append("SECURITY BREACH: malicious input " + repr(_probe) + " executed and returned " + repr(_p_res))
                        _out["security_breach"] = True
                    else:
                        _out["passed"].append("blocked_probe:" + _probe[:24])
                except Exception:
                    _out["passed"].append("blocked_probe:" + _probe[:24])
    elif _node_fn is not None:
        # AST node visitor check (test inner expression node .body first, or full ast.Expression)
        try:
            _parsed_eval = _v_ast.parse("2 + 3 * 4", mode="eval")
            try:
                _r_node = _node_fn(_parsed_eval.body)
            except Exception:
                _r_node = _node_fn(_parsed_eval)
            if abs(float(_r_node) - 14.0) < 1e-6:
                _out["passed"].append("ast_eval: 2+3*4 == 14")
                _out["outputs"].append("evaluate_node(ast.parse('2 + 3 * 4')) -> " + repr(_r_node))
            else:
                _out["failed"].append("AST node check failed: returned " + repr(_r_node) + " for 2+3*4 instead of 14")
        except Exception as _e:
            _out["failed"].append("AST node check raised " + type(_e).__name__ + ": " + str(_e))

        if any("Expr" in _h for _h in _cfg["ast_node_hints"]):
            try:
                _expr_stmt = _v_ast.parse("2 + 3 * 4").body[0]
                _r_expr = _node_fn(_expr_stmt)
                if abs(float(_r_expr) - 14.0) < 1e-6:
                    _out["passed"].append("ast_expr_node_supported")
                    _out["outputs"].append("evaluate_node(ast.parse('2 + 3 * 4').body[0]) -> " + repr(_r_expr))
                else:
                    _out["failed"].append("ast.Expr node returned " + repr(_r_expr) + " instead of 14")
            except Exception as _e:
                _out["failed"].append("ast.Expr node check (ast.parse(...).body[0]) raised " + type(_e).__name__ + ": " + str(_e))
    else:
        for _op in _cfg["required_ops"]:
            _fn_op = _ns.get(_op)
            _exp_pair = _cfg["expectations"].get(_op)
            if callable(_fn_op) and _exp_pair:
                _arg, _expected = _exp_pair[0], _exp_pair[1]
                try:
                    _res_d = _fn_op(_arg)
                    if abs(float(_res_d) - _expected) < 1e-5:
                        _out["passed"].append("func:" + _op + "(" + str(_arg) + ")==" + str(_res_d))
                        _out["outputs"].append(_op + "(" + str(_arg) + ") -> " + repr(_res_d))
                    else:
                        _out["failed"].append("Function " + _op + "(" + str(_arg) + ") returned " + repr(_res_d) + ", expected " + str(_expected))
                except Exception as _e:
                    _out["failed"].append("Function " + _op + " raised " + type(_e).__name__ + ": " + str(_e))

    print("__VERIFY_JSON__:" + _v_json.dumps(_out))

_v_run_checks()
"""


class ExecutionVerifier:
    """Deterministic multi-gate verifier executing AST, security, and live sandbox checks."""

    @staticmethod
    async def verify(
        report: DiagnosticReport,
        runtime: Any,
        candidate_files: Optional[List[str]] = None,
    ) -> VerificationReport:
        c = report.contract
        targets: List[str] = []
        for fp in (candidate_files or []) + list(c.target_files or []):
            if fp and fp not in targets:
                targets.append(fp)

        if not targets:
            return VerificationReport(
                passed=True,
                score=85,
                rating=8,
                checks_passed=["No file targets required verification"],
            )

        checks_passed: List[str] = []
        discrepancies: List[str] = []
        functional_outputs: List[str] = []
        focus_line: Optional[int] = None
        focus_scope: Optional[Dict[str, Any]] = None
        diagnosis_hints: List[str] = []
        score = 0

        py_files_checked = 0
        seen_real_paths = set()
        for tf in targets[:6]:
            ok, resolved, err, _ = resolve_existing_path(tf, runtime, allow_dir=False)
            if not ok or not os.path.isfile(resolved):
                if not any(tf in d for d in discrepancies):
                    discrepancies.append(f"Expected file '{tf}' is missing on disk ({err or 'not found'}).")
                continue

            real_resolved = os.path.realpath(resolved)
            if real_resolved in seen_real_paths:
                continue
            seen_real_paths.add(real_resolved)

            try:
                with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
            except OSError as exc:
                discrepancies.append(f"Cannot read '{tf}': {exc}")
                continue

            if not code.strip():
                discrepancies.append(f"File '{os.path.basename(resolved)}' is empty.")
                continue

            if not resolved.lower().endswith(".py"):
                checks_passed.append(f"exists:{os.path.basename(resolved)}")
                score = max(score, 80)
                continue

            py_files_checked += 1
            ast_info = scan_python_ast(
                code,
                requires_security=c.requires_security,
                required_ops=c.required_ops,
            )

            # Gate 1: Syntax (+25)
            if not ast_info["ast_valid"]:
                discrepancies.append(f"{os.path.basename(resolved)}: {ast_info['ast_error']}")
                continue
            score += 25
            checks_passed.append(f"ast_valid:{os.path.basename(resolved)}")

            # Gate 2: Security AST Gate (+25)
            sec_findings = ast_info["security_findings"]
            if c.requires_security and sec_findings:
                discrepancies.append(
                    f"Security policy violation in {os.path.basename(resolved)}: {'; '.join(sec_findings[:4])}"
                )
            else:
                score += 25
                checks_passed.append("security_ast_clean")

            # Gate 3: Required Ops / Symbols (+20)
            missing_ops = ast_info["missing_ops"]
            missing_syms = [
                sym for sym in c.required_symbols
                if sym not in ast_info["defined_names"]
            ]
            if missing_ops:
                discrepancies.append(
                    f"Missing requested operations in {os.path.basename(resolved)}: {', '.join(missing_ops)}"
                )
            elif missing_syms:
                discrepancies.append(
                    f"Missing required symbols in {os.path.basename(resolved)}: {', '.join(missing_syms)}"
                )
            else:
                score += 20
                checks_passed.append("required_symbols_present")

            # Gate 4: Live Functional + Adversarial Sandbox Execution (+30)
            if not sec_findings:
                sb_rt = (
                    runtime
                    if isinstance(runtime, RestrictedPythonRuntime)
                    else RestrictedPythonRuntime(workspace_root=getattr(runtime, "workspace_root", None))
                )
                harness = _build_sandbox_harness(
                    code,
                    required_ops=c.required_ops,
                    requires_security=c.requires_security,
                    is_calculator=c.is_calculator_domain,
                    ast_node_hints=c.ast_node_hints,
                )
                if isinstance(sb_rt, RestrictedPythonRuntime):
                    probe_timeout = min(float(getattr(sb_rt, "timeout", 30.0) or 30.0), 5.0)
                    exec_res = await sb_rt.execute_code(
                        harness,
                        language="python",
                        timeout=probe_timeout,
                        max_loop_iterations=10000,
                    )
                else:
                    exec_res = await sb_rt.execute_code(harness, language="python")
                parsed_out: Optional[Dict[str, Any]] = None
                if exec_res.success and exec_res.output:
                    for ln in exec_res.output.splitlines():
                        if ln.startswith("__VERIFY_JSON__:"):
                            try:
                                parsed_out = json.loads(ln[len("__VERIFY_JSON__:"):])
                            except Exception:
                                parsed_out = None
                            break

                if not exec_res.success:
                    err_msg = exec_res.error or "Sandbox execution failed"
                    line_match = re.search(r"at line (\d+)", err_msg)
                    h_lines = harness.splitlines()
                    mod_begin_idx = next(
                        (i for i, l in enumerate(h_lines) if "# --- BEGIN TARGET MODULE ---" in l),
                        -1,
                    )
                    src_line_txt = ""
                    src_snippet = ""
                    if line_match:
                        ln_no = int(line_match.group(1))
                        if 1 <= ln_no <= len(h_lines):
                            src_line_txt = h_lines[ln_no - 1].strip()
                            src_snippet = f" -> `{src_line_txt}`"
                        if mod_begin_idx >= 0 and ln_no > mod_begin_idx + 1:
                            cand_user_ln = ln_no - (mod_begin_idx + 1)
                            if 1 <= cand_user_ln <= len(code.splitlines()):
                                focus_line = cand_user_ln
                    if focus_line is None and src_line_txt:
                        for c_idx, c_line in enumerate(code.splitlines(), start=1):
                            if c_line.strip() == src_line_txt:
                                focus_line = c_idx
                                break
                    if mod_begin_idx >= 0:
                        # Map any sandbox frame markers `@L<n>` in trace back to user file line numbers
                        def _map_trace_ln(m: re.Match) -> str:
                            raw_ln = int(m.group(1))
                            u_ln = raw_ln - (mod_begin_idx + 1)
                            return f"@L{u_ln}" if 1 <= u_ln <= len(code.splitlines()) else f"@sandboxL{raw_ln}"

                        err_msg = re.sub(r"@L(\d+)", _map_trace_ln, err_msg)
                        if line_match:
                            raw_err_ln = int(line_match.group(1))
                            u_err_ln = raw_err_ln - (mod_begin_idx + 1)
                            if 1 <= u_err_ln <= len(code.splitlines()):
                                err_msg = re.sub(r"at line \d+", f"at line {u_err_ln}", err_msg, count=1)

                    if focus_line is not None and focus_scope is None:
                        focus_scope = find_enclosing_callable(code, focus_line)
                    for hint in diagnose_python_error(err_msg, src_line_txt, focus_line, full_code=code):
                        if hint not in diagnosis_hints:
                            diagnosis_hints.append(hint)
                    loc_label = f"line {focus_line}" if focus_line is not None else "sandbox"
                    scope_label = f" in `{focus_scope['qualname']}`" if focus_scope else ""
                    discrepancies.append(
                        f"Sandbox runtime error in {os.path.basename(resolved)} at {loc_label}{scope_label}: {err_msg}{src_snippet}"
                    )
                elif parsed_out is not None:
                    checks_passed.extend(parsed_out.get("passed", []))
                    functional_outputs.extend(parsed_out.get("outputs", []))
                    sb_failures = parsed_out.get("failed", [])
                    if sb_failures:
                        discrepancies.extend(sb_failures[:4])
                        for hint in diagnose_python_error("\n".join(sb_failures), code, focus_line, full_code=code):
                            if hint not in diagnosis_hints:
                                diagnosis_hints.append(hint)
                    else:
                        score += 30
                else:
                    score += 30
                    checks_passed.append("sandbox_import_ok")

            if discrepancies and focus_scope is None and c.traceback_lines:
                for tb_ln in reversed(c.traceback_lines):
                    scope_cand = find_enclosing_callable(code, tb_ln)
                    if scope_cand is not None:
                        focus_line = tb_ln
                        focus_scope = scope_cand
                        break

        if py_files_checked > 1:
            score = min(100, score // py_files_checked)

        score = max(0, min(100, score))
        passed = (len(discrepancies) == 0) and (score >= 90 or (score >= 70 and not c.requires_security))
        rating = max(1, min(10, round(score / 10))) if not passed else max(8, min(10, round(score / 10)))
        corrective = "\n".join(f"- {d}" for d in discrepancies[:4]) if discrepancies else None

        return VerificationReport(
            passed=passed,
            score=score,
            rating=rating,
            checks_passed=checks_passed,
            discrepancies=discrepancies,
            corrective_action=corrective,
            functional_outputs=functional_outputs,
            focus_line=focus_line,
            focus_scope=focus_scope,
            diagnosis_hints=diagnosis_hints,
        )
