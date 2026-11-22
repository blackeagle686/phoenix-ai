"""Neuro-Symbolic Engine Runner: Deterministic Playbook + PatchProtocol + Score-Guided Repair.

Executes structured code modification/creation workflows with:
  1. Zero-LLM PreAnalyzer workspace/AST/security diagnostics
  2. Code/Diff synthesis via PatchProtocol + SymbolicHealer (no fragile nested JSON)
  3. Live RestrictedPythonRuntime execution + adversarial verification
  4. Monotonic Best-Snapshot Retention (never lets a bad retry overwrite a better solution)
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from typing import Any, Callable, Dict, List, Optional

from phoenix.framework.agent.engine.pre_analyzer import DiagnosticReport, PreAnalyzer
from phoenix.framework.agent.engine.protocol import PatchProtocol
from phoenix.framework.agent.engine.verifier import ExecutionVerifier, VerificationReport
from phoenix.framework.agent.tools._fs_utils import SyntaxGuard, check_runtime_path, resolve_existing_path
from phoenix.services.llm.moe import moe_route_context


class NeuroSymbolicRunner:
    """System-orchestrated cognitive loop for deterministic reliability on any LLM scale."""

    @staticmethod
    def _order_project_targets(targets: List[str]) -> List[str]:
        """Order multi-file targets so backend APIs & contracts generate before HTML -> CSS -> JS."""
        priority = {
            ".py": 10, ".sql": 12, ".json": 15, ".toml": 15, ".yaml": 15, ".yml": 15,
            ".html": 20, ".htm": 20,
            ".css": 30, ".scss": 30, ".sass": 30,
            ".js": 40, ".jsx": 40, ".ts": 40, ".tsx": 40,
            ".md": 50, ".txt": 50,
        }
        return [
            t for _, _, t in sorted(
                ((priority.get(os.path.splitext(t.lower())[1], 45), idx, t) for idx, t in enumerate(targets))
            )
        ]

    @staticmethod
    def _extract_sibling_summary(file_path: str, content: str) -> str:
        """Build a compact cross-file contract summary (API routes, DOM IDs, CSS selectors) for small LLMs."""
        ext = os.path.splitext(file_path.lower())[1]
        lines = [ln for ln in (content or "").splitlines() if ln.strip()]
        if ext == ".py":
            highlights: List[str] = []
            for idx, ln in enumerate(lines):
                s = ln.strip()
                if s.startswith(("@app.", "@router.", "class ", "def ", "async def ")):
                    highlights.append("  " + s[:120])
                elif ": " in s and idx > 0 and any("BaseModel" in h for h in highlights[-4:]):
                    highlights.append("    " + s[:100])
            if highlights:
                return "\n".join(highlights[:18])
        elif ext in (".html", ".htm"):
            ids = list(dict.fromkeys(re.findall(r'\bid\s*=\s*["\']([^"\']+)["\']', content)))
            classes = list(dict.fromkeys(re.findall(r'\bclass\s*=\s*["\']([^"\']+)["\']', content)))
            parts = []
            if ids:
                parts.append("  DOM IDs: " + ", ".join(ids[:20]))
            if classes:
                parts.append("  CSS Classes: " + ", ".join(classes[:20]))
            if parts:
                return "\n".join(parts)
        elif ext in (".css", ".scss"):
            sels = list(dict.fromkeys(re.findall(r"^([.#A-Za-z0-9_,\s:\-\[\]=]+)\s*\{", content, flags=re.MULTILINE)))
            if sels:
                return "  Selectors: " + ", ".join(s.strip()[:40] for s in sels[:16])
        return "  " + "\n  ".join(lines[:10])

    @staticmethod
    def _snapshot_files(file_paths: List[str], runtime: Any) -> Dict[str, Optional[str]]:
        snap: Dict[str, Optional[str]] = {}
        for fp in file_paths:
            norm_fp = (fp or "").replace("\\", "/")
            if "/" in norm_fp and not os.path.isabs(fp):
                allowed, target, _ = check_runtime_path(runtime, fp)
                if not allowed:
                    continue
            else:
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

    @classmethod
    async def _run_multi_file_workflow(
        cls,
        *,
        prompt: str,
        llm: Any,
        runtime: Any,
        diag: DiagnosticReport,
        targets: List[str],
        max_attempts: int,
        emit_phase: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Dependency-ordered per-file synthesis & verification for multi-file/full-stack projects."""

        async def _emit(phase: str, info: Dict[str, Any]) -> None:
            if emit_phase is not None:
                res = emit_phase(phase, info)
                if hasattr(res, "__await__"):
                    await res

        ordered = cls._order_project_targets(targets)
        per_file_budget = min(3, max(2, max_attempts))
        sibling_context: Dict[str, str] = {}
        resolved_files: List[str] = []
        resolved_by_target: Dict[str, str] = {}
        file_diffs: Dict[str, str] = {}
        healed_notes: List[str] = []
        total_attempts = 0
        failed_files: List[str] = []

        for file_idx, tf in enumerate(ordered, start=1):
            syntax_guard = SyntaxGuard(max_rollbacks=max(3, per_file_budget))
            init_snap = cls._snapshot_files([tf], runtime)
            best_f_score = -1
            best_f_snap = dict(init_snap)
            best_f_path = tf
            best_f_diff = ""
            file_passed = False
            prev_fb = ""

            for attempt in range(1, per_file_budget + 1):
                total_attempts += 1
                prompt_text = PatchProtocol.build_prompt(
                    diag,
                    previous_feedback=prev_fb,
                    best_score=max(0, best_f_score),
                    target_file=tf,
                    sibling_context=sibling_context,
                    project_files=ordered,
                )
                await _emit(
                    "acting",
                    {
                        "task_id": f"f{file_idx}",
                        "attempt": attempt,
                        "message": f"Synthesizing {tf} ({file_idx}/{len(ordered)}, attempt {attempt})...",
                    },
                )

                gen_kwargs: Dict[str, Any] = {"session_id": None}
                try:
                    sig = inspect.signature(llm.generate)
                    if attempt >= 2 and "temperature" in sig.parameters:
                        gen_kwargs["temperature"] = 0.35
                except (TypeError, ValueError):
                    pass

                with moe_route_context(role="coder", attempt=attempt, escalate=(attempt >= 2)):
                    raw_reply = await llm.generate(prompt_text, **gen_kwargs)

                apply_res = PatchProtocol.apply_output(
                    str(raw_reply or ""),
                    target_path=tf,
                    runtime=runtime,
                    syntax_guard=syntax_guard,
                )
                if not apply_res.success:
                    err_txt = apply_res.error or f"Failed to parse or write {tf}."
                    prev_fb = f"- FORMAT ERROR ON LAST ATTEMPT: {err_txt}"
                    await _emit("reacting", {"task_id": f"f{file_idx}", "message": f"{tf} apply failed: {err_txt[:240]}"})
                    continue

                if apply_res.healed_actions:
                    for act in apply_res.healed_actions:
                        if act not in healed_notes:
                            healed_notes.append(act)

                await _emit(
                    "watching",
                    {"task_id": f"f{file_idx}", "message": f"Verifying {tf} ({file_idx}/{len(ordered)})..."},
                )
                f_ver = await ExecutionVerifier.verify(
                    diag,
                    runtime=runtime,
                    candidate_files=[apply_res.resolved_path],
                    only_candidates=True,
                )
                if f_ver.score >= best_f_score:
                    best_f_score = f_ver.score
                    best_f_snap = cls._snapshot_files([apply_res.resolved_path], runtime)
                    best_f_path = apply_res.resolved_path
                    best_f_diff = apply_res.diff
                else:
                    cls._restore_snapshot(best_f_snap)

                if f_ver.passed:
                    file_passed = True
                    break

                prev_fb = "\n".join(f"  * {d}" for d in (f_ver.discrepancies or ["Check failed"]))
                await _emit(
                    "reacting",
                    {
                        "task_id": f"f{file_idx}",
                        "score": f_ver.score,
                        "message": f"{tf} score {f_ver.score}/100: {'; '.join(f_ver.discrepancies)[:260]}",
                    },
                )

            if not file_passed:
                cls._restore_snapshot(best_f_snap)
                failed_files.append(tf)

            if os.path.isfile(best_f_path):
                resolved_files.append(best_f_path)
                resolved_by_target[tf] = best_f_path
                file_diffs[tf] = best_f_diff or "(created)"
                try:
                    with open(best_f_path, "r", encoding="utf-8", errors="replace") as f:
                        sibling_context[tf] = cls._extract_sibling_summary(tf, f.read())
                except OSError:
                    pass

        await _emit(
            "watching",
            {"task_id": "project", "message": f"Running final cross-file verification across {len(ordered)} files..."},
        )
        final_ver = await ExecutionVerifier.verify(
            diag,
            runtime=runtime,
            candidate_files=resolved_files or ordered,
            only_candidates=False,
        )
        all_ok = bool(final_ver.passed and not failed_files)
        summary_lines: List[str] = []
        for tf in ordered:
            p = resolved_by_target.get(tf)
            if p and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        n_lines = len(f.read().splitlines())
                    summary_lines.append(f"- {tf}: {n_lines} lines")
                except OSError:
                    summary_lines.append(f"- {tf}: created")
            else:
                summary_lines.append(f"- {tf}: MISSING")
        summary_diff = "\n".join(summary_lines)
        return {
            "handled": True,
            "success": all_ok,
            "attempts": total_attempts,
            "target_files": ordered,
            "resolved_file": resolved_files[0] if resolved_files else ordered[0],
            "resolved_files": resolved_files,
            "verification": final_ver,
            "diff": summary_diff,
            "file_diffs": file_diffs,
            "healed_actions": healed_notes,
            "error": None if all_ok else (final_ver.corrective_action or f"Failed files: {', '.join(failed_files)}"),
        }

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

        if len(targets) > 1:
            return await cls._run_multi_file_workflow(
                prompt=prompt,
                llm=llm,
                runtime=runtime,
                diag=diag,
                targets=targets,
                max_attempts=max_attempts,
                emit_phase=emit_phase,
            )

        primary_target = targets[0]
        syntax_guard = SyntaxGuard(max_rollbacks=max(3, max_attempts))

        initial_snapshot = cls._snapshot_files(targets, runtime)
        best_score = -1
        best_snapshot: Dict[str, Optional[str]] = dict(initial_snapshot)
        best_report: Optional[VerificationReport] = None
        best_diff = ""
        any_effective_change = False
        best_resolved_path = primary_target
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

            orig_content = initial_snapshot.get(apply_res.resolved_path)
            changed_from_initial = (
                orig_content is None
                or current_code.replace("\r\n", "\n").strip() != orig_content.replace("\r\n", "\n").strip()
            )
            has_real_diff = bool(apply_res.diff) and apply_res.diff != "(no changes)" and changed_from_initial

            # Guard: a `code_modify` request MUST actually change the target file — never declare VERIFIED on a no-op echo!
            if diag.contract.archetype == "code_modify" and not changed_from_initial:
                repeated_same_code += 1
                base_fb = last_ver_feedback or previous_feedback
                previous_feedback = (
                    f"{base_fb}\n"
                    f"- NO-OP FAILURE ON ATTEMPT {attempt}: Your output produced ZERO changes to `{primary_target}` "
                    "(the file is identical to the original). You MUST actually modify the code to fix the user's "
                    "reported bug and apply all requested improvements. Do NOT echo the original code unchanged."
                ).strip()
                await _emit(
                    "reacting",
                    {
                        "task_id": "t1",
                        "message": f"Attempt {attempt} made no changes to {primary_target}; retrying with explicit fix instruction...",
                    },
                )
                if repeated_same_code >= 2:
                    break
                continue

            if changed_from_initial:
                any_effective_change = True
                best_resolved_path = apply_res.resolved_path

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

            if ver.score > best_score:
                best_score = ver.score
                best_snapshot = cls._snapshot_files([apply_res.resolved_path], runtime)
                best_report = ver
                best_diff = apply_res.diff
                best_resolved_path = apply_res.resolved_path
            elif ver.score == best_score:
                # Never overwrite a real non-empty diff with "(no changes)" on equal score
                if has_real_diff or not best_diff or best_diff == "(no changes)":
                    best_snapshot = cls._snapshot_files([apply_res.resolved_path], runtime)
                    best_report = ver
                    best_resolved_path = apply_res.resolved_path
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
        modify_satisfied = diag.contract.archetype != "code_modify" or any_effective_change
        final_success = bool(best_report and best_report.passed and modify_satisfied)
        final_error = (
            f"No changes were applied to {primary_target} after {attempts_run} attempt(s)."
            if not modify_satisfied
            else (best_report.corrective_action if best_report else previous_feedback)
        )
        return {
            "handled": True,
            "success": final_success,
            "attempts": attempts_run,
            "target_files": targets,
            "resolved_file": best_resolved_path,
            "verification": best_report,
            "diff": best_diff or "(no changes)",
            "healed_actions": healed_notes,
            "error": None if final_success else final_error,
        }
