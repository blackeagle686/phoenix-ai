"""Deterministic, token-efficient filesystem & search tools with workspace jailing and bounded syntax rollback."""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from phoenix.framework.agent.cognition.schema import (
    FileContent,
    FileReadResult,
    FileSearchMatch,
    MultiBlockUpdateEdit,
    ReplacementChunk,
)
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.tools._fs_utils import (
    EditItemSchema,
    SyntaxGuard,
    build_compact_diff,
    check_runtime_path,
    is_ignored_dir,
    resolve_existing_path,
    resolve_write_path,
    smart_find_and_replace,
)
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

_ALLOWED_INDEX_EXTS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss",
    ".json", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini",
    ".sh", ".bash", ".c", ".cpp", ".h", ".java", ".go", ".rs",
})


def _dump_model(model: Any) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def get_blocks_from_lines(
    lines: List[str],
    file_path: str,
    block_size: int = 200,
    start_line_offset: int = 1,
) -> Tuple[List[FileContent], int]:
    total_lines = len(lines)
    if total_lines == 0:
        return [], 0

    step = max(1, int(block_size))
    blocks: List[FileContent] = []
    for i in range(0, total_lines, step):
        chunk_end = min(i + step, total_lines)
        blocks.append(
            FileContent(
                file_path=file_path,
                content_block="".join(lines[i:chunk_end]),
                from_line=start_line_offset + i,
                to_line=start_line_offset + chunk_end - 1,
                block_summary=None,
            )
        )
    return blocks, total_lines


class _FileWriteArgs(BaseModel):
    file_path: str = Field(..., description="Target file path inside workspace")
    content: str = Field(..., description="Text or code content to write")
    from_line: int = Field(default=1, ge=1, description="Optional 1-indexed start line for partial replacement")
    to_line: Optional[int] = Field(default=None, ge=1, description="Optional 1-indexed inclusive end line for partial replacement")
    expected_target: Optional[str] = Field(default=None, description="Optional expected text inside [from_line:to_line] to prevent off-by-one overwrites")
    verify_syntax: bool = Field(default=True, description="Validate Python AST syntax before saving .py files (max 3 rollbacks)")


class _FileEditArgs(BaseModel):
    file_path: str = Field(..., description="Path to the file to edit")
    edits: Optional[List[EditItemSchema]] = Field(default=None, description="Search/replace blocks: [{'target': '...', 'replacement': '...'}]")
    chunks: Optional[List[ReplacementChunk]] = Field(default=None, description="Line-range replacements: [{'from_line': 10, 'to_line': 15, 'replacement_content': '...', 'target_content': '...'}]")
    upsert: bool = Field(default=False, description="When True, append replacements if search text is missing or create file if missing")
    verify_syntax: bool = Field(default=True, description="Validate Python AST syntax before saving .py files (max 3 rollbacks)")


class _FileSearchArgs(BaseModel):
    query: str = Field(..., description="Substring or regex pattern to search for")
    path: str = Field(default=".", description="File or directory path inside workspace (defaults to '.')")
    file_glob: Optional[str] = Field(default=None, description="Optional filename glob filter, e.g. '*.py'")
    is_regex: bool = Field(default=False, description="Treat query as a regular expression")
    case_sensitive: bool = Field(default=False, description="Perform case-sensitive matching")
    max_matches: int = Field(default=50, ge=1, le=500, description="Maximum matching lines returned")


def _extract_outline(file_path: str, content: str) -> Dict[str, Any]:
    file_data: Dict[str, Any] = {"language": "", "imports": [], "functions": [], "classes": []}
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        file_data["language"] = "python"
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    file_data["imports"].extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    file_data["imports"].extend(f"{mod}.{a.name}" for a in node.names)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    file_data["functions"].append(
                        f"{node.name} (L{node.lineno}-L{getattr(node, 'end_lineno', node.lineno)})"
                    )
                elif isinstance(node, ast.ClassDef):
                    file_data["classes"].append(
                        f"{node.name} (L{node.lineno}-L{getattr(node, 'end_lineno', node.lineno)})"
                    )
        except SyntaxError:
            pass
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        file_data["language"] = "javascript/typescript"
        file_data["imports"].extend(re.findall(r'import\s+.*?\s+from\s+[\'"](.*?)[\'"]', content))
        file_data["classes"].extend(m.group(1) for m in re.finditer(r"class\s+([a-zA-Z0-9_]+)", content))
    else:
        file_data["language"] = ext.lstrip(".") or "text"

    file_data["imports"] = sorted(set(file_data["imports"]))
    file_data["functions"] = sorted(set(file_data["functions"]))
    file_data["classes"] = sorted(set(file_data["classes"]))
    return file_data


class FileReadTool(BaseTool):
    name = "file_read"
    read_only = True
    param_aliases = {"path": "file_path"}
    description = (
        "Reads a file (or lists a workspace directory) within optional 1-indexed line bounds and returns content blocks + outline. "
        "Inputs: 'file_path' (str), optional 'from_line' (int=1), 'to_line' (int), 'block_size' (int=200)."
    )

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__(read_only=True)
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(
        self,
        file_path: str = ".",
        from_line: int = 1,
        to_line: Optional[int] = None,
        block_size: int = 200,
        include_outline: bool = True,
        max_files: int = 80,
        **kwargs,
    ) -> ToolResult:
        try:
            target = kwargs.get("path") or file_path or "."
            ok, resolved_path, err, meta = await asyncio.to_thread(
                resolve_existing_path, target, self.runtime, True
            )
            if not ok:
                return ToolResult(success=False, output=meta or None, error=err)

            if os.path.isdir(resolved_path):
                def _scan_dir():
                    files_outline: Dict[str, Any] = {}
                    summary_lines: List[str] = []
                    truncated = False
                    count = 0
                    for root, dirs, files in os.walk(resolved_path):
                        dirs[:] = [d for d in sorted(dirs) if not is_ignored_dir(d)]
                        for fname in sorted(files):
                            if fname.startswith("."):
                                continue
                            ext = os.path.splitext(fname)[1].lower()
                            if ext not in _ALLOWED_INDEX_EXTS:
                                continue
                            if count >= max_files:
                                truncated = True
                                break
                            fp = os.path.join(root, fname)
                            try:
                                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                                    raw = f.read()
                            except Exception:
                                continue
                            line_cnt = len(raw.splitlines())
                            summary_lines.append(f"{os.path.relpath(fp, resolved_path)} ({line_cnt} lines)\n")
                            if include_outline and count < 25:
                                ol = _extract_outline(fp, raw)
                                ol["total_lines"] = line_cnt
                                files_outline[fp] = ol
                            else:
                                files_outline[fp] = {"total_lines": line_cnt}
                            count += 1
                        if truncated:
                            break
                    return summary_lines, files_outline, truncated

                summary_lines, files_outline, truncated = await asyncio.to_thread(_scan_dir)
                blocks, total_lines = get_blocks_from_lines(summary_lines, resolved_path, block_size=block_size)
                res_model = FileReadResult(file_path=resolved_path, content=blocks, total_lines=total_lines)
                out_dict = _dump_model(res_model)
                out_dict["is_directory"] = True
                out_dict["truncated"] = truncated
                out_dict["files"] = files_outline
                out_dict.update(meta)
                return ToolResult(success=True, output=out_dict)

            read_result = await self.runtime.execute_io("read", resolved_path)
            if not read_result.success:
                return ToolResult(success=False, output=None, error=read_result.error)

            raw_content = read_result.output or ""
            all_lines = raw_content.splitlines(keepends=True)
            total_file_lines = len(all_lines)

            start_idx = max(0, (from_line or 1) - 1)
            end_idx = to_line if to_line is not None else total_file_lines
            selected_lines = all_lines[start_idx:end_idx]

            blocks, _ = get_blocks_from_lines(
                selected_lines,
                resolved_path,
                block_size=max(1, block_size),
                start_line_offset=start_idx + 1,
            )
            res_model = FileReadResult(file_path=resolved_path, content=blocks, total_lines=total_file_lines)
            out_dict = _dump_model(res_model)
            if include_outline:
                out_dict["outline"] = await asyncio.to_thread(_extract_outline, resolved_path, raw_content)
            out_dict.update(meta)
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileWriteTool(BaseTool):
    name = "file_write"
    args_schema = _FileWriteArgs
    param_aliases = {"write_content": "content", "path": "file_path"}
    description = (
        "Creates or overwrites a file (or replaces a line range when from_line/to_line are provided). "
        "Inputs: 'file_path' (str), 'content' (str). Optional: 'from_line', 'to_line', 'expected_target', 'verify_syntax'."
    )

    def __init__(self, runtime: Optional[Any] = None, max_syntax_rollbacks: int = 3):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()
        self._syntax_guard = SyntaxGuard(max_rollbacks=max_syntax_rollbacks)

    async def execute(
        self,
        file_path: str,
        content: Optional[str] = None,
        from_line: int = 1,
        to_line: Optional[int] = None,
        expected_target: Optional[str] = None,
        verify_syntax: bool = True,
        write_content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            actual_content = content if content is not None else write_content
            if actual_content is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Missing required 'content' argument for file_write (refusing to overwrite file with empty string).",
                )

            is_partial = (from_line or 1) > 1 or to_line is not None
            meta: Dict[str, Any] = {}
            if is_partial:
                ok, resolved_path, err, meta = await asyncio.to_thread(
                    resolve_existing_path, file_path, self.runtime, False
                )
                if not ok:
                    return ToolResult(success=False, output=meta or None, error=err)
            else:
                ok, resolved_path, err = await asyncio.to_thread(resolve_write_path, file_path, self.runtime)
                if not ok:
                    return ToolResult(success=False, output=None, error=err)

            old_content = ""
            if os.path.exists(resolved_path):
                read_old = await self.runtime.execute_io("read", resolved_path)
                if read_old.success and read_old.output is not None:
                    old_content = read_old.output

            if not is_partial:
                final_content = actual_content
            else:
                lines = old_content.splitlines(keepends=True)
                start_idx = max(0, (from_line or 1) - 1)
                end_line = to_line if to_line is not None else max(1, len(lines))

                if expected_target is not None:
                    slice_text = "".join(lines[start_idx:end_line])
                    if expected_target.replace("\r\n", "\n") not in slice_text.replace("\r\n", "\n"):
                        return ToolResult(
                            success=False,
                            output={"actual_lines": slice_text},
                            error=(
                                f"Line guard mismatch at L{start_idx + 1}-L{end_line}: "
                                "'expected_target' did not match lines on disk. Read lines again before writing."
                            ),
                        )

                if start_idx > len(lines):
                    lines.extend(["\n"] * (start_idx - len(lines)))

                new_lines = actual_content.splitlines(keepends=True)
                if actual_content and not actual_content.endswith("\n"):
                    if new_lines:
                        new_lines[-1] += "\n"
                    else:
                        new_lines.append("\n")

                lines[start_idx:end_line] = new_lines
                final_content = "".join(lines)

            allow, rb_err, syn_warn, _ = self._syntax_guard.check_python(
                resolved_path, final_content, enabled=verify_syntax
            )
            if not allow:
                return ToolResult(success=False, output={"rolled_back": True}, error=rb_err)

            write_result = await self.runtime.execute_io("edit", resolved_path, final_content)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)

            total_lines = len(final_content.splitlines())
            written_lines = len(actual_content.splitlines())
            out: Dict[str, Any] = {
                "file_path": resolved_path,
                "success": True,
                "total_lines": total_lines,
                "lines_written": written_lines,
                "diff": build_compact_diff(old_content, final_content, resolved_path)
                if old_content
                else f"Created {os.path.basename(resolved_path)} ({total_lines} lines).",
            }
            if syn_warn:
                out["syntax_warning"] = syn_warn
            out.update(meta)
            return ToolResult(success=True, output=out)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileAppendTool(BaseTool):
    name = "file_append"
    param_aliases = {"write_content": "content", "path": "file_path"}
    description = "Appends text to the end of a file inside the workspace. Inputs: 'file_path' (str), 'content' (str)."

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(
        self,
        file_path: str,
        content: Optional[str] = None,
        write_content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            actual_content = content if content is not None else write_content
            if actual_content is None:
                return ToolResult(success=False, output=None, error="Missing 'content' argument for file_append.")

            ok, resolved_path, err = await asyncio.to_thread(resolve_write_path, file_path, self.runtime)
            if not ok:
                return ToolResult(success=False, output=None, error=err)

            append_result = await self.runtime.execute_io("append", resolved_path, actual_content)
            if not append_result.success:
                return ToolResult(success=False, output=None, error=append_result.error)

            read_result = await self.runtime.execute_io("read", resolved_path)
            lines = read_result.output.splitlines() if (read_result.success and read_result.output) else []

            return ToolResult(
                success=True,
                output={
                    "file_path": resolved_path,
                    "success": True,
                    "appended_lines": len(actual_content.splitlines()),
                    "total_lines": len(lines),
                    "tail_preview": "\n".join(lines[-5:]),
                },
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileEditTool(BaseTool):
    name = "file_edit"
    args_schema = _FileEditArgs
    param_aliases = {"path": "file_path"}
    description = (
        "Deterministically edits a file using exact/whitespace-safe search+replace 'edits' or line-range 'chunks'. "
        "Inputs: 'file_path' (str), and either 'edits' ([{'target': '...', 'replacement': '...'}]) "
        "or 'chunks' ([{'from_line': int, 'to_line': int, 'replacement_content': str, 'target_content'?: str}])."
    )

    def __init__(self, runtime: Optional[Any] = None, max_syntax_rollbacks: int = 3):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()
        self._syntax_guard = SyntaxGuard(max_rollbacks=max_syntax_rollbacks)

    async def execute(
        self,
        file_path: str,
        chunks: Optional[List[Union[ReplacementChunk, Dict[str, Any]]]] = None,
        edits: Optional[List[Union[MultiBlockUpdateEdit, Dict[str, Any]]]] = None,
        upsert: bool = False,
        verify_syntax: bool = True,
        **kwargs,
    ) -> ToolResult:
        try:
            if not chunks and not edits:
                return ToolResult(
                    success=False,
                    output=None,
                    error="No edits provided. Supply non-empty 'edits' or 'chunks'.",
                )

            meta: Dict[str, Any] = {}
            ok, resolved_path, err, meta = await asyncio.to_thread(
                resolve_existing_path, file_path, self.runtime, False
            )
            if not ok:
                if upsert:
                    ok_w, resolved_path, err_w = await asyncio.to_thread(resolve_write_path, file_path, self.runtime)
                    if not ok_w:
                        return ToolResult(success=False, output=None, error=err_w)
                else:
                    return ToolResult(success=False, output=meta or None, error=err)

            old_content = ""
            if os.path.exists(resolved_path):
                read_result = await self.runtime.execute_io("read", resolved_path)
                if read_result.success and read_result.output is not None:
                    old_content = read_result.output

            applied_count = 0
            match_modes: List[str] = []

            if chunks:
                lines = old_content.splitlines(keepends=True)
                processed: List[ReplacementChunk] = [
                    ReplacementChunk(**c) if isinstance(c, dict) else c for c in chunks
                ]
                processed.sort(key=lambda c: c.from_line, reverse=True)

                for chunk in processed:
                    start_idx = max(0, chunk.from_line - 1)
                    end_idx = chunk.to_line

                    if chunk.target_content:
                        slice_text = "".join(lines[start_idx:end_idx])
                        if chunk.target_content.replace("\r\n", "\n").strip() not in slice_text.replace("\r\n", "\n"):
                            return ToolResult(
                                success=False,
                                output={"actual_lines": slice_text, "rolled_back": True},
                                error=(
                                    f"Chunk target_content mismatch at L{chunk.from_line}-L{chunk.to_line}. "
                                    "Aborted without modifying file."
                                ),
                            )

                    if start_idx > len(lines):
                        lines.extend(["\n"] * (start_idx - len(lines)))

                    new_lines = chunk.replacement_content.splitlines(keepends=True)
                    if chunk.replacement_content and not chunk.replacement_content.endswith("\n"):
                        if new_lines:
                            new_lines[-1] += "\n"
                        else:
                            new_lines.append("\n")

                    lines[start_idx:end_idx] = new_lines
                    applied_count += 1
                    match_modes.append("line_chunk")

                final_content = "".join(lines)
            else:
                updated = old_content
                for edit in edits or []:
                    if isinstance(edit, BaseModel):
                        search = getattr(edit, "target", "")
                        replace = getattr(edit, "replacement", "")
                    else:
                        search = edit.get("target", edit.get("search", ""))
                        replace = edit.get("replacement", edit.get("replace", ""))

                    if not search:
                        continue

                    new_text, find_err, find_meta = smart_find_and_replace(updated, search, replace)
                    if find_err is None and new_text is not None:
                        updated = new_text
                        applied_count += 1
                        match_modes.append(str(find_meta.get("match_mode", "exact")))
                    elif upsert and not find_meta.get("occurrences_at_lines"):
                        if updated and not updated.endswith("\n"):
                            updated += "\n"
                        updated += replace
                        if replace and not replace.endswith("\n"):
                            updated += "\n"
                        applied_count += 1
                        match_modes.append("upsert_append")
                    else:
                        return ToolResult(
                            success=False,
                            output={"applied_before_abort": 0, **find_meta},
                            error=find_err or f"Search text not found: {search[:100]}",
                        )

                final_content = updated

            allow, rb_err, syn_warn, _ = self._syntax_guard.check_python(
                resolved_path, final_content, enabled=verify_syntax
            )
            if not allow:
                return ToolResult(success=False, output={"rolled_back": True}, error=rb_err)

            write_result = await self.runtime.execute_io("edit", resolved_path, final_content)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)

            out: Dict[str, Any] = {
                "file_path": resolved_path,
                "success": True,
                "applied_count": applied_count,
                "match_modes": match_modes,
                "total_lines": len(final_content.splitlines()),
                "diff": build_compact_diff(old_content, final_content, resolved_path),
            }
            if syn_warn:
                out["syntax_warning"] = syn_warn
            out.update(meta)
            return ToolResult(success=True, output=out)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileSearchTool(BaseTool):
    name = "file_search"
    read_only = True
    args_schema = _FileSearchArgs
    param_aliases = {"search_query": "query", "pattern": "query", "file_path": "path"}
    description = (
        "Fast pure-Python text/regex search inside workspace files or directories (auto-skips .git, venvs, node_modules). "
        "Inputs: 'query' (str), optional 'path' (str='.'), 'file_glob' (str), 'is_regex' (bool), 'case_sensitive' (bool), 'max_matches' (int=50)."
    )

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__(read_only=True)
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(
        self,
        query: Optional[str] = None,
        path: str = ".",
        file_glob: Optional[str] = None,
        is_regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 50,
        search_query: Optional[str] = None,
        pattern: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            final_query = query if query is not None else (search_query if search_query is not None else pattern)
            target_path = file_path or path or "."

            if not final_query:
                return ToolResult(success=False, output=None, error="Missing required 'query' argument to search.")

            ok, resolved_path, err, meta = await asyncio.to_thread(
                resolve_existing_path, target_path, self.runtime, True
            )
            if not ok:
                return ToolResult(success=False, output=meta or None, error=err)

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex_obj = re.compile(final_query if is_regex else re.escape(final_query), flags)
                ci_regex = re.compile(final_query if is_regex else re.escape(final_query), re.IGNORECASE) if case_sensitive else None
            except re.error as r_err:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid regular expression {final_query!r}: {r_err}",
                )

            cap = max(1, min(int(max_matches or 50), 500))

            def _search_sync() -> Dict[str, Any]:
                matches: List[Dict[str, Any]] = []
                ci_hint_count = 0
                files_searched = 0
                truncated = False

                def _scan_file(fp: str) -> None:
                    nonlocal ci_hint_count, files_searched, truncated
                    if truncated:
                        return
                    try:
                        if os.path.getsize(fp) > 2 * 1024 * 1024:
                            return
                        with open(fp, "rb") as fb:
                            sample = fb.read(1024)
                            if b"\x00" in sample:
                                return
                        with open(fp, "r", encoding="utf-8", errors="replace") as f:
                            files_searched += 1
                            for line_num, line in enumerate(f, start=1):
                                if regex_obj.search(line):
                                    match_obj = FileSearchMatch(
                                        line_number=line_num,
                                        line_content=line.strip()[:240],
                                        block_index=(line_num - 1) // 100,
                                    )
                                    rec = _dump_model(match_obj)
                                    rec["file_path"] = fp
                                    matches.append(rec)
                                    if len(matches) >= cap:
                                        truncated = True
                                        return
                                elif ci_regex is not None and len(matches) == 0 and ci_hint_count < 20:
                                    if ci_regex.search(line):
                                        ci_hint_count += 1
                    except Exception:
                        return

                if os.path.isfile(resolved_path):
                    _scan_file(resolved_path)
                else:
                    for root, dirs, files in os.walk(resolved_path):
                        dirs[:] = [d for d in sorted(dirs) if not is_ignored_dir(d)]
                        for fname in sorted(files):
                            if fname.startswith("."):
                                continue
                            if file_glob and not fnmatch.fnmatch(fname, file_glob):
                                continue
                            _scan_file(os.path.join(root, fname))
                            if truncated:
                                break
                        if truncated:
                            break

                return {
                    "matches": matches,
                    "ci_hint_count": ci_hint_count,
                    "files_searched": files_searched,
                    "truncated": truncated,
                }

            res = await asyncio.to_thread(_search_sync)
            out: Dict[str, Any] = {
                "file_path": resolved_path,
                "success": True,
                "matches": res["matches"],
                "total_matches": len(res["matches"]),
                "files_searched": res["files_searched"],
                "truncated": res["truncated"],
            }
            if len(res["matches"]) == 0 and res["ci_hint_count"] > 0:
                out["hint"] = (
                    f"Found {res['ci_hint_count']}+ matches ignoring case. "
                    "Retry with case_sensitive=False."
                )
            out.update(meta)
            return ToolResult(success=True, output=out)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileDeleteTool(BaseTool):
    name = "file_delete"
    param_aliases = {"path": "file_path"}
    description = "Safely deletes an exact file path inside the workspace. Input: 'file_path' (str)."

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            ok, checked_path, sec_err = await asyncio.to_thread(
                check_runtime_path, self.runtime, file_path
            )
            if not ok:
                return ToolResult(success=False, output=None, error=sec_err)

            if not os.path.exists(checked_path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Safety Check Failed: Exact file not found at {file_path}. Auto-correction is disabled for deletion.",
                )

            exe_result = await self.runtime.execute_io("delete", checked_path)
            if not exe_result.success:
                return ToolResult(success=False, output=None, error=exe_result.error)
            return ToolResult(success=True, output=f"Successfully deleted {checked_path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
