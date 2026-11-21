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


def _prepare_module_for_sandbox(code: str, required_ops: List[str]) -> Tuple[str, List[str], List[str], Dict[str, List[str]]]:
    """Strip `if __name__ == '__main__':` + replace blocked `input(...)` stubs, and collect callable targets."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, [], [], {}

    lines = code.splitlines()
    ranges_to_clear: List[Tuple[int, int]] = []
    discovered_funcs: List[str] = []
    discovered_classes: List[str] = []
    class_methods: Dict[str, List[str]] = {}

    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno) - 1
                ranges_to_clear.append((start, end))
        elif isinstance(node, ast.FunctionDef):
            if node.name not in ("main", "run_console", "cli") and node.name not in required_ops:
                if len(node.args.args) >= 1:
                    discovered_funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            discovered_classes.append(node.name)
            methods: List[str] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    if len(item.args.args) >= 2:  # self + at least 1 argument
                        methods.append(item.name)
            if methods:
                class_methods[node.name] = methods

    for start, end in ranges_to_clear:
        for i in range(start, min(end + 1, len(lines))):
            lines[i] = "pass" if i == start else ""

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\binput\s*\([^()\r\n]*\)", "'0'", cleaned)

    ordered_funcs: List[str] = []
    for name in list(_PREFERRED_EVAL_NAMES) + discovered_funcs:
        if name not in ordered_funcs:
            ordered_funcs.append(name)
    return cleaned, ordered_funcs, discovered_classes, class_methods


def _build_sandbox_harness(
    module_code: str,
    required_ops: List[str],
    requires_security: bool,
    is_calculator: bool,
) -> str:
    """Build an isolated Python test harness compatible with RestrictedPythonRuntime AST constraints."""
    safe_module_code, candidate_funcs, candidate_classes, class_methods = _prepare_module_for_sandbox(
        module_code, required_ops
    )

    # Generate NameError-safe namespace binding without restricted `globals()`/`vars()`/`dir()`/`hasattr()`
    capture_lines: List[str] = ["    _ns = {}"]
    names_to_capture = list(dict.fromkeys(candidate_funcs + list(required_ops) + candidate_classes))
    for sym in names_to_capture:
        if sym.isidentifier():
            capture_lines.append(f"    try:\n        _ns[{sym!r}] = {sym}\n    except Exception:\n        pass")
    capture_block = "\n".join(capture_lines)

    # Discover callable instance methods directly from AST ClassDef without calling hasattr/getattr
    class_probe_lines: List[str] = []
    for cname in candidate_classes:
        methods = class_methods.get(cname, [])
        preferred = [m for m in _PREFERRED_EVAL_NAMES if m in methods] + [m for m in methods if m not in _PREFERRED_EVAL_NAMES]
        for mname in preferred:
            if cname.isidentifier() and mname.isidentifier():
                class_probe_lines.append(
                    f"        if _expr_fn is None:\n"
                    f"            try:\n"
                    f"                _inst = {cname}()\n"
                    f"                _m = _inst.{mname}\n"
                    f"                if callable(_m):\n"
                    f"                    _expr_fn = _m\n"
                    f"            except Exception:\n"
                    f"                pass"
                )
    class_probe_block = "\n".join(class_probe_lines) if class_probe_lines else "        pass"

    harness_cfg = json.dumps({
        "required_ops": required_ops,
        "requires_security": requires_security,
        "is_calculator": is_calculator,
        "candidate_funcs": candidate_funcs,
        "probes": ADVERSARIAL_PROBES,
        "expectations": {k: list(v) for k, v in MATH_FUNC_EXPECTATIONS.items()},
    })

    return f"""
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
{class_probe_block}

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
        # 1. Basic arithmetic check (only require strict match when calculator/math ops requested)
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
                )
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
                    src_snippet = ""
                    if line_match:
                        ln_no = int(line_match.group(1))
                        h_lines = harness.splitlines()
                        if 1 <= ln_no <= len(h_lines):
                            src_snippet = f" -> `{h_lines[ln_no - 1].strip()}`"
                    discrepancies.append(f"Sandbox runtime error in {os.path.basename(resolved)}: {err_msg}{src_snippet}")
                elif parsed_out is not None:
                    checks_passed.extend(parsed_out.get("passed", []))
                    functional_outputs.extend(parsed_out.get("outputs", []))
                    sb_failures = parsed_out.get("failed", [])
                    if sb_failures:
                        discrepancies.extend(sb_failures[:4])
                    else:
                        score += 30
                else:
                    score += 30
                    checks_passed.append("sandbox_import_ok")

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
        )
