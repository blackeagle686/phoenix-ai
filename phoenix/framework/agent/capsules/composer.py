"""PHX v2 AST-Level Composer, Scoped LLM Glue Engine, Sandbox Verifier & Mega-Capsule Miner.

Implements FR-15 (AST Composition & Symbol/Route Collision Guard), FR-16 (Scoped LLM Glue),
FR-17 (Embedded Sandbox Verification + Quarantine), and FR-18 (RF Blueprint Router & HDBSCAN Miner).
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from phoenix.framework.agent.capsules.registry import CapsuleRegistry
from phoenix.framework.agent.capsules.schema import (
    CapsuleASTWiring,
    CapsuleContract,
    CapsuleSemanticProfile,
    CapsuleStack,
    CapsuleVerification,
    CodeCapsule,
)
from phoenix.framework.agent.capsules.solver import CapsuleConstraintSolver, SolveResult
from phoenix.framework.agent.engine.pre_analyzer import scan_python_ast
from phoenix.framework.agent.engine.surgical import normalize_python_snippet
from phoenix.framework.agent.kernel.features import FeatureExtractor
from phoenix.framework.agent.kernel.models import CalibratedRFRouter
from phoenix.framework.agent.kernel.schema import ExecutionTrace, OutcomeStatus
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime


@dataclass
class CompositionResult:
    success: bool
    assembled_code: str = ""
    capsule_ids: List[str] = field(default_factory=list)
    auto_added_dependencies: List[str] = field(default_factory=list)
    middleware_order: List[str] = field(default_factory=list)
    interface_signatures: List[str] = field(default_factory=list)
    filled_glue_slots: List[str] = field(default_factory=list)
    verifier_passed: bool = False
    tests_run: int = 0
    errors: List[str] = field(default_factory=list)
    security_findings: List[str] = field(default_factory=list)


def _validate_and_render_params(capsule: CodeCapsule, overrides: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """Validate parameter types & numeric/enum constraints before rendering capsule template."""
    vals: Dict[str, Any] = {}
    errors: List[str] = []
    ov = overrides or {}
    for p in capsule.parameters:
        raw_val = ov.get(p.name, p.default)
        if p.type == "int":
            try:
                val = int(raw_val)
            except (TypeError, ValueError):
                errors.append(f"Parameter '{p.name}' of '{capsule.capsule_id}' must be int, got {raw_val!r}.")
                continue
            if p.constraints and ".." in p.constraints:
                lo_s, hi_s = p.constraints.split("..", 1)
                if not (int(lo_s) <= val <= int(hi_s)):
                    errors.append(f"Parameter '{p.name}'={val} violates constraint '{p.constraints}'.")
                    continue
            vals[p.name] = val
        elif p.type == "identifier":
            s = str(raw_val or "")
            if not s.isidentifier():
                errors.append(f"Parameter '{p.name}'={s!r} is not a valid Python identifier.")
                continue
            vals[p.name] = s
        else:
            vals[p.name] = raw_val

    if errors:
        return "", errors

    src = capsule.ast_wiring.template_source
    if vals:
        try:
            src = src.format(**vals)
        except Exception as exc:
            return "", [f"Template formatting failed for '{capsule.capsule_id}': {exc}"]
    return src, []


def _extract_interface_signatures(tree: ast.Module) -> List[str]:
    """Section 5.6: Extract compiled interface signatures passed to Scoped LLM Glue synthesis."""
    sigs: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                f"{m.name}({', '.join(a.arg for a in m.args.args)})"
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            sigs.append(f"class {node.name}: [{', '.join(methods)}]")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            sigs.append(f"def {node.name}({args})")
    return sigs


class ASTCapsuleComposer:
    """FR-15/16/17: Deterministic AST-level composer with collision guard, scoped LLM glue, and sandbox verifier."""

    @classmethod
    def compose_ast(
        cls,
        ordered_capsules: Sequence[CodeCapsule],
        *,
        param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        custom_slot_bodies: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Parse every capsule into AST, check Global Symbol & Route Collision Table,
        hoist/deduplicate imports cleanly, splice optional custom_logic_slots, and unparse.
        Returns `(assembled_code, interface_signatures, filled_slots, errors)`.
        """
        imports_seen: Set[str] = set()
        import_nodes: List[ast.stmt] = []
        body_nodes: List[ast.stmt] = []
        symbol_table: Dict[str, str] = {}   # symbol_name -> capsule_id
        route_table: Dict[str, str] = {}    # route_spec -> capsule_id
        errors: List[str] = []
        filled_slots: List[str] = []
        overrides_map = param_overrides or {}
        slots_map = custom_slot_bodies or {}

        for cap in ordered_capsules:
            # Check route collisions across contract.provides
            for prov in cap.contract.provides:
                if prov.startswith("route:"):
                    if prov in route_table:
                        errors.append(
                            f"Route collision: '{prov}' declared by both '{route_table[prov]}' and '{cap.capsule_id}'."
                        )
                    else:
                        route_table[prov] = cap.capsule_id

            rendered, param_errs = _validate_and_render_params(cap, overrides_map.get(cap.capsule_id))
            if param_errs:
                errors.extend(param_errs)
                continue

            try:
                cap_tree = ast.parse(rendered)
            except SyntaxError as exc:
                errors.append(f"AST syntax error in capsule '{cap.capsule_id}' at line {exc.lineno}: {exc.msg}")
                continue

            for stmt in cap_tree.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    imp_str = ast.unparse(stmt)
                    if imp_str not in imports_seen:
                        imports_seen.add(imp_str)
                        import_nodes.append(stmt)
                    continue

                # Global Symbol Collision check (Section 5.6)
                if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    sym = stmt.name
                    if sym in symbol_table:
                        errors.append(
                            f"Symbol collision: '{sym}' defined in both '{symbol_table[sym]}' and '{cap.capsule_id}'."
                        )
                        continue
                    symbol_table[sym] = cap.capsule_id

                    # FR-16: Splice scoped custom logic body strictly inside allowed slot function signature
                    if sym in cap.ast_wiring.custom_logic_slots and sym in slots_map:
                        slot_code = normalize_python_snippet(slots_map[sym])
                        try:
                            slot_tree = ast.parse(slot_code)
                            if (
                                len(slot_tree.body) == 1
                                and isinstance(slot_tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                                and slot_tree.body[0].name == sym
                            ):
                                # Preserve original typed signature while updating body strictly
                                stmt.body = slot_tree.body[0].body
                            else:
                                stmt.body = slot_tree.body
                            filled_slots.append(sym)
                        except SyntaxError as se:
                            errors.append(f"Invalid syntax in custom_logic_slot '{sym}': {se.msg}")
                            continue

                body_nodes.append(stmt)

        if errors:
            return "", [], filled_slots, errors

        # Attach deterministic wiring manifest metadata as module-level constant
        mw_names = [
            c.capsule_id
            for c in sorted(ordered_capsules, key=lambda c: (c.contract.middleware_order_priority, c.capsule_id))
            if "app.middlewares" in c.ast_wiring.injection_points
        ]
        manifest_assign = ast.parse(
            f"__PHX_CAPSULES__ = {[c.capsule_id for c in ordered_capsules]!r}\n"
            f"__PHX_MIDDLEWARE_ORDER__ = {mw_names!r}\n"
        ).body

        combined_tree = ast.Module(body=import_nodes + manifest_assign + body_nodes, type_ignores=[])
        ast.fix_missing_locations(combined_tree)
        assembled = ast.unparse(combined_tree) + "\n"
        sigs = _extract_interface_signatures(combined_tree)
        return assembled, sigs, filled_slots, []

    @classmethod
    async def synthesize_scoped_glue(
        cls,
        llm: Any,
        slot_name: str,
        requirement: str,
        interface_signatures: Sequence[str],
    ) -> Optional[str]:
        """FR-16: Invoke stateless LLM Battery strictly scoped to a single custom_logic_slot."""
        if llm is None:
            return None
        sig_block = "\n".join(f"  - {s}" for s in interface_signatures)
        prompt = (
            f"Write ONLY the Python function `def {slot_name}(...):` implementing this custom hook requirement.\n"
            f"Requirement: {requirement}\n"
            f"Available Compiled Interface Signatures (use ONLY these symbols):\n{sig_block}\n"
            "Return ONLY valid Python code inside a ```python ... ``` block. Do NOT modify other functions or use eval/exec/os."
        )
        raw = await llm.generate(prompt, session_id=None)
        m = re.search(r"```(?:python)?\s*(.*?)```", str(raw or ""), re.DOTALL | re.IGNORECASE)
        code = m.group(1) if m else str(raw or "")
        return normalize_python_snippet(code)

    @classmethod
    async def compose_and_verify(
        cls,
        registry: CapsuleRegistry,
        requested_capsule_ids: Sequence[str],
        *,
        runtime: Optional[Any] = None,
        param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        custom_slot_bodies: Optional[Dict[str, str]] = None,
        custom_slot_requirements: Optional[Dict[str, str]] = None,
        llm: Optional[Any] = None,
        language: str = "python",
        framework: Optional[str] = None,
        orm: Optional[str] = None,
        quarantine_on_failure: bool = True,
    ) -> CompositionResult:
        """Execute full Composition Pipeline: Solve DAG -> AST Compose -> Scoped Glue -> Embedded Sandbox Verify."""
        solve_res: SolveResult = CapsuleConstraintSolver.solve(
            registry,
            requested_capsule_ids,
            language=language,
            framework=framework,
            orm=orm,
        )
        if not solve_res.success:
            return CompositionResult(
                success=False,
                errors=solve_res.conflicts + solve_res.missing_requirements,
            )

        slot_bodies = dict(custom_slot_bodies or {})
        # First pass to extract compiled interface signatures if LLM glue synthesis is requested
        assembled, sigs, filled_slots, comp_errs = cls.compose_ast(
            solve_res.ordered_capsules,
            param_overrides=param_overrides,
            custom_slot_bodies=slot_bodies,
        )
        if comp_errs:
            return CompositionResult(success=False, errors=comp_errs)

        if custom_slot_requirements and llm is not None:
            for slot_name, req_desc in custom_slot_requirements.items():
                if slot_name not in slot_bodies:
                    synthesized = await cls.synthesize_scoped_glue(llm, slot_name, req_desc, sigs)
                    if synthesized:
                        slot_bodies[slot_name] = synthesized
            assembled, sigs, filled_slots, comp_errs = cls.compose_ast(
                solve_res.ordered_capsules,
                param_overrides=param_overrides,
                custom_slot_bodies=slot_bodies,
            )
            if comp_errs:
                return CompositionResult(success=False, errors=comp_errs)

        # FR-17: Embedded Deterministic Verification (AST syntax + Security Scan + Unit/Integration Tests in Sandbox)
        ast_scan = scan_python_ast(assembled, requires_security=True)
        if not ast_scan["ast_valid"]:
            return CompositionResult(
                success=False,
                assembled_code=assembled,
                capsule_ids=[c.capsule_id for c in solve_res.ordered_capsules],
                errors=[f"Assembled AST invalid: {ast_scan['ast_error']}"],
            )
        if ast_scan["security_findings"]:
            return CompositionResult(
                success=False,
                assembled_code=assembled,
                capsule_ids=[c.capsule_id for c in solve_res.ordered_capsules],
                security_findings=ast_scan["security_findings"],
                errors=["Security policy violation in assembled code: " + "; ".join(ast_scan["security_findings"])],
            )

        sb_rt = (
            runtime
            if isinstance(runtime, RestrictedPythonRuntime)
            else RestrictedPythonRuntime(workspace_root=getattr(runtime, "workspace_root", None), timeout=5.0)
        )

        # Run every capsule's attached unit/integration test inside isolated RestrictedPythonRuntime
        tests_run = 0
        test_errors: List[str] = []
        for cap in solve_res.ordered_capsules:
            tsrc = (cap.verification.test_source or "").strip()
            if not tsrc:
                continue
            tests_run += 1
            test_program = f"{assembled}\n# --- CAPSULE TEST: {cap.capsule_id} ---\n{tsrc}\nprint('CAPSULE_TEST_OK')\n"
            res = await sb_rt.execute_code(test_program, language="python", timeout=5.0, max_loop_iterations=10000)
            if not res.success or "CAPSULE_TEST_OK" not in (res.output or ""):
                err_msg = f"Capsule test failed for '{cap.capsule_id}': {res.error or res.output}"
                test_errors.append(err_msg)
                if quarantine_on_failure and not filled_slots:
                    registry.quarantine(cap.capsule_id, err_msg)

        passed = len(test_errors) == 0
        return CompositionResult(
            success=passed,
            assembled_code=assembled,
            capsule_ids=[c.capsule_id for c in solve_res.ordered_capsules],
            auto_added_dependencies=solve_res.auto_added_dependencies,
            middleware_order=solve_res.middleware_order,
            interface_signatures=sigs,
            filled_glue_slots=filled_slots,
            verifier_passed=passed,
            tests_run=tests_run,
            errors=test_errors,
        )


class BlueprintMLRouter:
    """FR-18: Random Forest Blueprint Router mapping user prompt vectors directly to verified capsule_ids."""

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = float(confidence_threshold)
        self._fe = FeatureExtractor()
        self._router = CalibratedRFRouter()

    def fit_from_traces(self, traces: Sequence[ExecutionTrace]) -> bool:
        verified = [
            t for t in traces
            if t.outcome.verifier_passed
            and t.outcome.status == OutcomeStatus.SUCCESS
            and t.capsule_ids
            and t.state_vector
        ]
        if len(verified) < 6:
            return False
        X = np.vstack([np.asarray(t.state_vector, dtype=np.float32) for t in verified])
        y = ["|".join(sorted(t.capsule_ids)) for t in verified]
        self._router.fit(X, y)
        return self._router.fitted

    def predict_blueprint(self, prompt: str) -> Tuple[List[str], float, bool]:
        """Return `(capsule_ids, calibrated_prob, high_confidence_hit)` without calling any LLM."""
        if not self._router.fitted:
            return [], 0.0, False
        xv = self._fe.extract(prompt)
        pred_key, prob, _, _ = self._router.predict(xv)
        if not pred_key:
            return [], 0.0, False
        cids = [c for c in pred_key.split("|") if c]
        return cids, prob, bool(prob >= self.confidence_threshold)


class MegaCapsuleMiner:
    """FR-18: Cold-Path HDBSCAN / frequency miner freezing recurring verified capsule bundles into Mega-Capsules."""

    def __init__(self, min_occurrences: int = 20, min_success_rate: float = 0.95):
        self.min_occurrences = max(2, int(min_occurrences))
        self.min_success_rate = float(min_success_rate)

    def mine_and_register(
        self,
        registry: CapsuleRegistry,
        traces: Sequence[ExecutionTrace],
    ) -> List[CodeCapsule]:
        bundle_stats: Dict[Tuple[str, ...], List[bool]] = {}
        for tr in traces:
            if len(tr.capsule_ids) < 2:
                continue
            key = tuple(sorted(set(tr.capsule_ids)))
            ok = bool(tr.outcome.verifier_passed and tr.outcome.status == OutcomeStatus.SUCCESS)
            bundle_stats.setdefault(key, []).append(ok)

        created: List[CodeCapsule] = []
        for bundle_key, outcomes in bundle_stats.items():
            succ = sum(1 for o in outcomes if o)
            if succ < self.min_occurrences:
                continue
            sr = succ / float(len(outcomes))
            if sr < self.min_success_rate:
                continue
            caps = [registry.get(cid) for cid in bundle_key]
            if any(c is None for c in caps):
                continue
            valid_caps = [c for c in caps if c is not None]
            assembled, _, _, errs = ASTCapsuleComposer.compose_ast(valid_caps)
            if errs or not assembled:
                continue

            digest = hashlib.blake2b("|".join(bundle_key).encode("utf-8"), digest_size=4).hexdigest()
            mega_id = f"cap.mega.bundle_{digest}.v1"
            if registry.get(mega_id) is not None:
                continue

            all_provides: List[str] = []
            all_tags: List[str] = ["mega_capsule"]
            all_tests: List[str] = []
            for c in valid_caps:
                for p in c.contract.provides:
                    if p not in all_provides:
                        all_provides.append(p)
                for tg in c.semantic_profile.tags:
                    if tg not in all_tags:
                        all_tags.append(tg)
                if c.verification.test_source.strip():
                    all_tests.append(c.verification.test_source.strip())

            mega = CodeCapsule(
                capsule_id=mega_id,
                version="1.0.0",
                stack=valid_caps[0].stack,
                semantic_profile=CapsuleSemanticProfile(
                    summary=f"Auto-mined Mega-Capsule combining {', '.join(bundle_key)} ({succ} verified runs, {sr:.0%} success)",
                    tags=all_tags,
                ),
                contract=CapsuleContract(
                    provides=all_provides,
                    requires=[],
                    conflicts_with=[],
                    middleware_order_priority=min(c.contract.middleware_order_priority for c in valid_caps),
                ),
                ast_wiring=CapsuleASTWiring(
                    template_source=assembled,
                    injection_points=["app.routers", "app.middlewares", "app.di_container"],
                ),
                verification=CapsuleVerification(
                    security_audited=all(c.verification.security_audited for c in valid_caps),
                    test_source="\n".join(all_tests),
                ),
                is_mega_capsule=True,
                constituent_ids=list(bundle_key),
            )
            registry.register(mega)
            created.append(mega)
        return created
