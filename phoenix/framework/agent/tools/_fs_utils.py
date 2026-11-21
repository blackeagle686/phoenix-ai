"""Shared deterministic filesystem, search, patch matching, and bounded syntax guard utilities.

Designed to shift path discipline, whitespace recovery, search noise filtering, and
syntax safety from the LLM into deterministic system logic while keeping outputs compact.
"""

from __future__ import annotations

import ast
import difflib
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import model_validator
except ImportError:
    model_validator = None


class EditItemSchema(BaseModel):
    """Canonical search/replace item supporting legacy 'search'/'replace' dict keys via pre-validation."""

    target: str = Field(..., description="Exact target substring or block to locate and replace")
    replacement: str = Field(..., description="Replacement text or code block")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _normalize_legacy_keys(cls, data: Any) -> Any:
            if isinstance(data, dict):
                d = dict(data)
                if "target" not in d and "search" in d:
                    d["target"] = d.pop("search")
                if "replacement" not in d and "replace" in d:
                    d["replacement"] = d.pop("replace")
                return d
            return data

IGNORE_DIRS = frozenset({
    ".git", "venv", "env", ".venv", "agent_venv", "git_venv",
    "__pycache__", ".idea", ".vscode", "node_modules", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
})


def is_ignored_dir(dirname: str) -> bool:
    return dirname in IGNORE_DIRS or dirname.endswith("_venv") or dirname.startswith(".")


def get_workspace_root(runtime: Any = None) -> str:
    ws = getattr(runtime, "workspace_root", None) if runtime is not None else None
    return os.path.realpath(ws) if ws else os.getcwd()


def check_runtime_path(runtime: Any, file_path: str) -> Tuple[bool, str, Optional[str]]:
    """Validate `file_path` against `runtime.workspace_root` (symlink-safe via realpath)."""
    if runtime is not None and hasattr(runtime, "_resolve_and_check_path"):
        return runtime._resolve_and_check_path(file_path)

    base = get_workspace_root(runtime)
    candidate = file_path if os.path.isabs(file_path) else os.path.join(base, file_path)
    abs_path = os.path.abspath(candidate)
    ws = getattr(runtime, "workspace_root", None) if runtime is not None else None
    if ws:
        real_root = os.path.realpath(ws)
        real_target = os.path.realpath(abs_path)
        try:
            if os.path.commonpath([real_root, real_target]) != real_root:
                return (
                    False,
                    abs_path,
                    f"Security Violation: Path '{file_path}' resolves outside workspace root '{real_root}'.",
                )
        except ValueError:
            return False, abs_path, f"Security Violation: Invalid path '{file_path}'."
    return True, abs_path, None


def _scan_workspace_candidates(
    base_dir: str,
    target_name: str,
    allow_dir: bool = False,
    max_entries: int = 2500,
) -> Tuple[List[str], List[str]]:
    """Find exact basename matches (never arbitrary substrings) + close fuzzy paths for hints."""
    exact: List[str] = []
    rel_pool: List[str] = []
    visited = 0
    lower_target = target_name.lower()

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in sorted(dirs) if not is_ignored_dir(d)]
        if allow_dir:
            for d in dirs:
                full_d = os.path.join(root, d)
                if d == target_name:
                    exact.append(full_d)
                rel_pool.append(os.path.relpath(full_d, base_dir))
                visited += 1
        for f in sorted(files):
            if f.startswith("."):
                continue
            full_f = os.path.join(root, f)
            if f == target_name or f.lower() == lower_target:
                exact.append(full_f)
            rel_pool.append(os.path.relpath(full_f, base_dir))
            visited += 1
            if visited >= max_entries:
                break
        if visited >= max_entries:
            break

    return exact, rel_pool


def resolve_existing_path(
    file_path: str,
    runtime: Any = None,
    allow_dir: bool = False,
) -> Tuple[bool, str, Optional[str], Dict[str, Any]]:
    """Deterministically resolve an existing file/directory inside workspace_root.

    Rules:
    - Enforces runtime sandbox jailing BEFORE touching or leaking outside paths.
    - Direct relative/absolute match first (zero tree walk).
    - If not found directly, searches ONLY for exact filename matches inside workspace:
        * 1 match  -> resolves unambiguously and records `resolved_from`.
        * 2+ match -> fails loudly with `candidates` list (never silently edits wrong file).
        * 0 match  -> fails with closest 4 `suggestions` as hints to save LLM turns.
    """
    if not file_path or not str(file_path).strip():
        return False, "", "Empty 'file_path' provided.", {}

    allowed, abs_path, sec_err = check_runtime_path(runtime, file_path)
    if not allowed:
        return False, abs_path, sec_err, {}

    if os.path.exists(abs_path):
        if os.path.isdir(abs_path) and not allow_dir:
            return False, abs_path, f"Path '{file_path}' is a directory, expected a file.", {}
        return True, abs_path, None, {}

    # Also check cwd if workspace_root was None or path passed relative to repo root
    ws_root = get_workspace_root(runtime)
    base_name = os.path.basename(file_path.rstrip("/\\"))
    if not base_name:
        return False, abs_path, f"Path not found: '{file_path}'.", {}

    exact, rel_pool = _scan_workspace_candidates(ws_root, base_name, allow_dir=allow_dir)
    if len(exact) == 1:
        resolved = exact[0]
        ok2, resolved_checked, sec_err2 = check_runtime_path(runtime, resolved)
        if not ok2:
            return False, resolved_checked, sec_err2, {}
        return True, resolved_checked, None, {"resolved_from": file_path}

    if len(exact) > 1:
        rels = [os.path.relpath(p, ws_root) for p in exact[:6]]
        return (
            False,
            abs_path,
            f"Ambiguous path '{file_path}': found {len(exact)} matches: {rels}. Specify exact relative path.",
            {"candidates": rels},
        )

    close = difflib.get_close_matches(file_path.lstrip("./"), rel_pool, n=4, cutoff=0.45)
    hint = f" Did you mean: {close}?" if close else ""
    return False, abs_path, f"Path not found: '{file_path}'.{hint}", {"suggestions": close}


def resolve_write_path(
    file_path: str,
    runtime: Any = None,
) -> Tuple[bool, str, Optional[str]]:
    """Resolve destination path for creating/writing without fuzzy-matching existing files."""
    if not file_path or not str(file_path).strip():
        return False, "", "Empty 'file_path' provided."
    return check_runtime_path(runtime, file_path)


def _line_number_at(text: str, char_idx: int) -> int:
    return text.count("\n", 0, char_idx) + 1


def _find_closest_snippet(content_lf: str, search_lf: str, window_pad: int = 2) -> str:
    """Return the closest matching multi-line snippet with 1-based line numbers."""
    file_lines = content_lf.splitlines()
    if not file_lines:
        return ""
    search_lines = [ln.strip() for ln in search_lf.splitlines() if ln.strip()]
    probe = search_lines[0] if search_lines else search_lf.strip()
    if not probe:
        return ""

    best_idx = 0
    best_ratio = 0.0
    for idx, line in enumerate(file_lines):
        ratio = difflib.SequenceMatcher(None, probe, line.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    if best_ratio < 0.25:
        return ""

    block_len = max(1, min(8, len(search_lf.splitlines())))
    start = max(0, best_idx - window_pad)
    end = min(len(file_lines), best_idx + block_len + window_pad)
    return "\n".join(f"L{i + 1}: {file_lines[i]}" for i in range(start, end))


def smart_find_and_replace(
    content: str,
    search: str,
    replace: str,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Deterministic multi-tier text replacement that eliminates fragile LLM whitespace misses.

    Tier 1: Exact unique substring match (CRLF-aware).
    Tier 2: Unique line-trimmed / common-indent-tolerant match (only when unambiguous!).
    Failure: Returns exact line numbers if ambiguous (>1 match), or closest 5-line snippet
             if 0 matches so the LLM can fix its edit immediately without re-reading the file.
    """
    if not search:
        return content, None, {"match_mode": "noop"}

    use_crlf = "\r\n" in content
    content_lf = content.replace("\r\n", "\n")
    search_lf = search.replace("\r\n", "\n")
    replace_lf = replace.replace("\r\n", "\n")

    # Tier 1: Exact match
    exact_cnt = content_lf.count(search_lf)
    if exact_cnt == 1:
        updated_lf = content_lf.replace(search_lf, replace_lf, 1)
        final = updated_lf.replace("\n", "\r\n") if use_crlf else updated_lf
        return final, None, {"match_mode": "exact"}

    if exact_cnt > 1:
        occ_lines = []
        pos = 0
        while len(occ_lines) < 8:
            found = content_lf.find(search_lf, pos)
            if found == -1:
                break
            occ_lines.append(_line_number_at(content_lf, found))
            pos = found + len(search_lf)
        return (
            None,
            f"Target text is ambiguous ({exact_cnt} occurrences at lines {occ_lines}). Include 1-2 surrounding lines.",
            {"occurrences_at_lines": occ_lines},
        )

    # Tier 2: Line-stripped / whitespace-tolerant unique block match
    file_lines = content_lf.splitlines(keepends=True)
    # Remove trailing empty element distinction cleanly
    file_stripped = [ln.rstrip("\r\n").strip() for ln in file_lines]
    search_lines = search_lf.splitlines()
    # Trim leading/trailing pure blank lines in search only if search had multiple lines
    search_norm = [ln.strip() for ln in search_lines]
    # Avoid trivially matching empty-only patterns
    if any(search_norm):
        win = len(search_norm)
        norm_hits: List[int] = []
        for i in range(0, len(file_stripped) - win + 1):
            if file_stripped[i : i + win] == search_norm:
                norm_hits.append(i)

        if len(norm_hits) == 1:
            hit_idx = norm_hits[0]
            # Preserve base indentation from first non-empty matched line vs search line
            orig_first = file_lines[hit_idx].rstrip("\r\n")
            search_first = search_lines[0] if search_lines else ""
            orig_indent = orig_first[: len(orig_first) - len(orig_first.lstrip(" \t"))]
            search_indent = search_first[: len(search_first) - len(search_first.lstrip(" \t"))]

            rep_lines = replace_lf.splitlines()
            if orig_indent != search_indent:
                adjusted_rep: List[str] = []
                for rln in rep_lines:
                    if not rln.strip():
                        adjusted_rep.append("")
                    elif search_indent and rln.startswith(search_indent):
                        adjusted_rep.append(orig_indent + rln[len(search_indent) :])
                    elif not rln[0].isspace():
                        adjusted_rep.append(orig_indent + rln)
                    else:
                        adjusted_rep.append(rln)
                rep_block = "\n".join(adjusted_rep)
            else:
                rep_block = replace_lf

            had_trailing_nl = "".join(file_lines[hit_idx : hit_idx + win]).endswith("\n")
            if had_trailing_nl and rep_block and not rep_block.endswith("\n"):
                rep_block += "\n"

            updated_lf = (
                "".join(file_lines[:hit_idx])
                + rep_block
                + "".join(file_lines[hit_idx + win :])
            )
            final = updated_lf.replace("\n", "\r\n") if use_crlf else updated_lf
            return final, None, {"match_mode": "whitespace_normalized", "matched_line": hit_idx + 1}

        if len(norm_hits) > 1:
            lines_1b = [h + 1 for h in norm_hits[:8]]
            return (
                None,
                f"Target text matched {len(norm_hits)} whitespace-normalized locations (lines {lines_1b}). Include more context.",
                {"occurrences_at_lines": lines_1b},
            )

    # Tier 3: Token-saving diagnostic hint
    closest = _find_closest_snippet(content_lf, search_lf)
    hint_msg = f"\nClosest match in file:\n{closest}" if closest else ""
    return (
        None,
        f"Target content not found in file: {search_lf[:120]!r}.{hint_msg}",
        {"closest_snippet": closest},
    )


def build_compact_diff(old_text: str, new_text: str, file_path: str, max_lines: int = 32) -> str:
    """Build a short unified diff snippet showing changed regions only."""
    old_lines = old_text.replace("\r\n", "\n").splitlines()
    new_lines = new_text.replace("\r\n", "\n").splitlines()
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{os.path.basename(file_path)}",
            tofile=f"b/{os.path.basename(file_path)}",
            n=2,
            lineterm="",
        )
    )
    if not diff:
        return "(no changes)"
    if len(diff) > max_lines:
        return "\n".join(diff[:max_lines] + [f"... ({len(diff) - max_lines} more diff lines)"])
    return "\n".join(diff)


class SyntaxGuard:
    """Bounded AST syntax verifier for Python files.

    Rolls back invalid `.py` writes/edits up to `max_rollbacks` consecutive times
    per file path so broken code is never left on disk accidentally. Once
    `max_rollbacks` consecutive failures are exceeded, permits the write with an
    explicit warning and resets the counter to break infinite LLM retry loops.
    """

    def __init__(self, max_rollbacks: int = 3) -> None:
        self.max_rollbacks = max(0, int(max_rollbacks))
        self._fail_counts: Dict[str, int] = {}

    def reset(self, file_path: str) -> None:
        self._fail_counts.pop(os.path.realpath(file_path), None)

    def check_python(
        self,
        file_path: str,
        code: str,
        enabled: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[str], int]:
        """Validate Python syntax if `file_path` ends with `.py` and `enabled=True`.

        Returns:
            (allow_write, rollback_error, syntax_warning, attempt_count)
        """
        if not enabled or not file_path.lower().endswith(".py"):
            return True, None, None, 0

        key = os.path.realpath(file_path)
        try:
            ast.parse(code, filename=file_path)
            self._fail_counts.pop(key, None)
            return True, None, None, 0
        except SyntaxError as exc:
            err_detail = f"SyntaxError at line {exc.lineno}: {exc.msg}"
            if exc.text:
                err_detail += f" -> {exc.text.strip()}"

            fails = self._fail_counts.get(key, 0) + 1
            if fails <= self.max_rollbacks:
                self._fail_counts[key] = fails
                return (
                    False,
                    f"Rolled back write (syntax attempt {fails}/{self.max_rollbacks}) to protect file: {err_detail}",
                    None,
                    fails,
                )

            # Exceeded rollback budget -> break infinite loop cleanly, save with explicit warning
            self._fail_counts.pop(key, None)
            warning = (
                f"Saved despite syntax error after {self.max_rollbacks} rollbacks "
                f"to avoid infinite retry loop: {err_detail}"
            )
            return True, None, warning, fails


def get_workspace_snapshot(runtime: Any = None, max_files: int = 25) -> str:
    """Return a concise single-string summary of workspace files so the LLM knows what already exists."""
    ws = getattr(runtime, "workspace_root", None) if runtime is not None else None
    if not ws:
        return ""
    real_ws = os.path.realpath(ws)
    if not os.path.isdir(real_ws):
        return ""

    entries: List[str] = []
    truncated = False
    for root, dirs, files in os.walk(real_ws):
        dirs[:] = [d for d in sorted(dirs) if not is_ignored_dir(d)]
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, real_ws)
            try:
                size = os.path.getsize(full_path)
                if size <= 200_000:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        line_cnt = sum(1 for _ in f)
                    entries.append(f"{rel_path} ({line_cnt}L)")
                else:
                    entries.append(f"{rel_path} ({size // 1024}KB)")
            except OSError:
                entries.append(rel_path)

            if len(entries) >= max_files:
                truncated = True
                break
        if truncated:
            break

    if not entries:
        return "(empty workspace)"
    suffix = " ..." if truncated else ""
    return ", ".join(entries) + suffix


def inspect_workspace_file(runtime: Any, file_path: str, preview_lines: int = 35) -> Dict[str, Any]:
    """Deterministically inspect an on-disk workspace file (existence, line count, Python AST, preview)."""
    ok, resolved_path, err, _ = resolve_existing_path(file_path, runtime, allow_dir=False)
    if not ok or not os.path.isfile(resolved_path):
        return {
            "file_path": resolved_path or file_path,
            "exists": False,
            "error": err or f"File not found: {file_path}",
        }

    try:
        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        return {"file_path": resolved_path, "exists": True, "error": str(exc)}

    lines = content.splitlines()
    info: Dict[str, Any] = {
        "file_path": resolved_path,
        "exists": True,
        "line_count": len(lines),
        "char_count": len(content),
        "ast_valid": True,
        "ast_error": None,
        "symbols": [],
    }
    if resolved_path.lower().endswith(".py"):
        try:
            tree = ast.parse(content, filename=resolved_path)
            symbols = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(f"def {node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"class {node.name}")
            info["symbols"] = symbols
        except SyntaxError as exc:
            info["ast_valid"] = False
            info["ast_error"] = f"SyntaxError line {exc.lineno}: {exc.msg}"

    if len(lines) <= preview_lines:
        info["preview"] = content
    else:
        head = "\n".join(lines[: preview_lines // 2])
        tail = "\n".join(lines[-(preview_lines // 2) :])
        info["preview"] = f"{head}\n... ({len(lines) - preview_lines} lines omitted) ...\n{tail}"
    return info
