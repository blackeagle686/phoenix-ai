"""Neuro-Symbolic Engine Runner: Deterministic Playbook + PatchProtocol + Score-Guided Repair.

Executes structured code modification/creation workflows with:
  1. Zero-LLM PreAnalyzer workspace/AST/security diagnostics
  2. Code/Diff synthesis via PatchProtocol + SymbolicHealer (no fragile nested JSON)
  3. Live RestrictedPythonRuntime execution + adversarial verification
  4. Monotonic Best-Snapshot Retention (never lets a bad retry overwrite a better solution)
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional

from phoenix.framework.agent.engine.pre_analyzer import DiagnosticReport, PreAnalyzer
from phoenix.framework.agent.engine.protocol import PatchProtocol
from phoenix.framework.agent.engine.verifier import ExecutionVerifier, VerificationReport
from phoenix.framework.agent.tools._fs_utils import SyntaxGuard, resolve_existing_path
from phoenix.services.llm.moe import moe_route_context


class NeuroSymbolicRunner:
    """System-orchestrated cognitive loop for deterministic reliability on any LLM scale."""

    @staticmethod
    def _snapshot_files(file_paths: List[str], runtime: Any) -> Dict[str, Optional[str]]:
        snap: Dict[str, Optional[str]] = {}
        for fp in file_paths:
            ok, resolved, _, _ = resolve_existing_path(fp, runtime, allow_dir=False)
            target = resolved if ok else fp
            if target and os.path.isfile(target):
                try:
                    with open(target, "r", encoding="utf-8", errors="replace") as f:
                        snap[target] = f.read()
                except OSError:
                    snap[target] = None
            elif target:
                snap[target] = None
        return snap

    @staticmethod
    def _restore_snapshot(snapshot: Dict[str, Optional[str]]) -> None:
        for path, content in snapshot.items():
            if content is None:
                continue
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    @classmethod
    async def run_code_workflow(
        cls,
        prompt: str,
        llm: Any,
        runtime: Any,
        report: Optional[DiagnosticReport] = None,
        max_attempts: int = 4,
        emit_phase: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Run a deterministic Diagnose -> Patch -> Heal -> Verify -> Monotonic Repair loop."""

        async def _emit(phase: str, info: Dict[str, Any]) -> None:
            if emit_phase is not None:
                res = emit_phase(phase, info)
                if hasattr(res, "__await__"):
                    await res

        diag = report or PreAnalyzer.analyze(prompt, runtime=runtime)
        targets = list(diag.contract.target_files or [])
        if not targets:
            return {"handled": False, "reason": "No code file targets identified"}

        primary_target = targets[0]
        syntax_guard = SyntaxGuard(max_rollbacks=max(3, max_attempts))

        initial_snapshot = cls._snapshot_files(targets, runtime)
        best_score = -1
        best_snapshot: Dict[str, Optional[str]] = dict(initial_snapshot)
        best_report: Optional[VerificationReport] = None
        best_diff = ""
        healed_notes: List[str] = []
        previous_feedback = ""
        last_ver_feedback = ""
        last_written_code: Optional[str] = None
        repeated_same_code = 0
        focus_scope: Optional[Dict[str, Any]] = None
        diagnosis_hints: List[str] = []
        attempts_run = 0

        # Pre-flight deterministic execution trace before attempt 1 when modifying an existing file
        if diag.contract.archetype == "code_modify":
            initial_ver = await ExecutionVerifier.verify(diag, runtime=runtime)
            best_score = initial_ver.score
            best_report = initial_ver
            focus_scope = initial_ver.focus_scope
            diagnosis_hints = list(initial_ver.diagnosis_hints or [])
            if initial_ver.discrepancies:
                previous_feedback = (
                    f"- Current File Score: {initial_ver.score}/100\n"
                    "- Detected failing checks on disk:\n"
                    + "\n".join(f"  * {d}" for d in initial_ver.discrepancies)
                )
                last_ver_feedback = previous_feedback

        for attempt in range(1, max(1, max_attempts) + 1):
            attempts_run = attempt
            if attempt > 1:
                diag = PreAnalyzer.analyze(prompt, runtime=runtime)

            prompt_text = PatchProtocol.build_prompt(
                diag,
                previous_feedback=previous_feedback,
                best_score=max(0, best_score),
                focus_scope=focus_scope,
                diagnosis_hints=diagnosis_hints,
                force_search_replace=False,
            )
            await _emit(
                "acting",
                {"task_id": "t1", "attempt": attempt, "message": f"Synthesizing patch for {primary_target} (attempt {attempt})..."},
            )

            gen_kwargs: Dict[str, Any] = {"session_id": None}
            if repeated_same_code >= 1 or attempt >= 2:
                try:
                    sig = inspect.signature(llm.generate)
                    if "temperature" in sig.parameters:
                        gen_kwargs["temperature"] = 0.45 if repeated_same_code >= 1 else 0.25
                except (TypeError, ValueError):
                    pass

            with moe_route_context(role="coder", attempt=attempt, escalate=(attempt >= 2)):
                raw_reply = await llm.generate(prompt_text, **gen_kwargs)

            apply_res = PatchProtocol.apply_output(
                str(raw_reply or ""),
                target_path=primary_target,
                runtime=runtime,
                syntax_guard=syntax_guard,
                focus_class=(focus_scope or {}).get("class_name"),
            )

            if not apply_res.success:
                apply_err = apply_res.error or "Failed to parse or apply code output."
                # Reset repeated_same_code so next attempt falls back to standard ```python``` block format
                # while preserving the underlying verification error context!
                repeated_same_code = 0
                base_fb = last_ver_feedback or previous_feedback
                previous_feedback = (
                    f"{base_fb}\n- FORMAT ERROR ON LAST ATTEMPT: {apply_err} "
                    "Return valid Python code inside a ```python ... ``` code block."
                ).strip()
                await _emit(
                    "reacting",
                    {"task_id": "t1", "message": f"Patch apply failed on attempt {attempt}: {apply_err[:300]}"},
                )
                continue

            current_code = ""
            if os.path.isfile(apply_res.resolved_path):
                with open(apply_res.resolved_path, "r", encoding="utf-8", errors="replace") as f:
                    current_code = f.read()

            is_duplicate_attempt = last_written_code is not None and current_code == last_written_code
            last_written_code = current_code

            if apply_res.healed_actions:
                for act in apply_res.healed_actions:
                    if act not in healed_notes:
                        healed_notes.append(act)

            await _emit(
                "watching",
                {"task_id": "t1", "message": f"Running deterministic AST, security & sandbox verification on {primary_target}..."},
            )
            ver = await ExecutionVerifier.verify(
                diag,
                runtime=runtime,
                candidate_files=[apply_res.resolved_path],
            )
            focus_scope = ver.focus_scope or focus_scope
            if ver.diagnosis_hints:
                diagnosis_hints = list(ver.diagnosis_hints)

            has_real_diff = bool(apply_res.diff) and apply_res.diff != "(no changes)"
            if ver.score > best_score:
                best_score = ver.score
                best_snapshot = cls._snapshot_files([apply_res.resolved_path], runtime)
                best_report = ver
                best_diff = apply_res.diff
            elif ver.score == best_score:
                # Never overwrite a real non-empty diff with "(no changes)" on equal score
                if has_real_diff or not best_diff or best_diff == "(no changes)":
                    best_snapshot = cls._snapshot_files([apply_res.resolved_path], runtime)
                    best_report = ver
                    if has_real_diff:
                        best_diff = apply_res.diff
            else:
                # Monotonic protection: never allow a worse retry to corrupt a higher-scoring file on disk
                cls._restore_snapshot(best_snapshot)

            if ver.passed:
                return {
                    "handled": True,
                    "success": True,
                    "attempts": attempts_run,
                    "target_files": targets,
                    "resolved_file": apply_res.resolved_path,
                    "verification": ver,
                    "diff": best_diff,
                    "healed_actions": healed_notes,
                }

            passed_str = ", ".join(ver.checks_passed) if ver.checks_passed else "none"
            failed_str = "\n".join(f"  * {d}" for d in ver.discrepancies) if ver.discrepancies else "  * Unknown check failure"
            previous_feedback = (
                f"- Score: {ver.score}/100\n"
                f"- Passed checks: {passed_str}\n"
                f"- Failed checks:\n{failed_str}"
            )
            last_ver_feedback = previous_feedback
            if is_duplicate_attempt:
                repeated_same_code += 1
                previous_feedback += "\n- WARNING: Output matched previous failing code identically. Apply the exact AST/runtime fix above."
            else:
                repeated_same_code = 0

            await _emit(
                "reacting",
                {
                    "task_id": "t1",
                    "score": ver.score,
                    "passed_checks": ver.checks_passed,
                    "discrepancies": ver.discrepancies,
                    "message": (
                        f"Score {ver.score}/100 ({len(ver.discrepancies)} issue(s)); "
                        f"failed: {'; '.join(ver.discrepancies)[:320]}"
                    ),
                },
            )
            if repeated_same_code >= 2:
                break

        # Ensure highest-scoring snapshot stays on disk even if max_attempts exhausted
        cls._restore_snapshot(best_snapshot)
        return {
            "handled": True,
            "success": bool(best_report and best_report.passed),
            "attempts": attempts_run,
            "target_files": targets,
            "resolved_file": primary_target,
            "verification": best_report,
            "diff": best_diff,
            "healed_actions": healed_notes,
            "error": best_report.corrective_action if best_report else previous_feedback,
        }
