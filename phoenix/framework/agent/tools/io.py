import os
import re
import ast
import json
import shlex
import asyncio
import difflib
from typing import List, Dict, Union, Optional, Any

from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from phoenix.framework.agent.cognition.schema import (
    FileContent,
    FileReadResult,
    FileWriteResult,
    FileUpdateResult,
    ReplacementChunk,
    FileSearchMatch,
    FileSearchResult,
)


def _auto_correct_path(
    file_path: str,
    is_edit_mode: bool = True,
    tool_name: str = "Tool",
    workspace_root: Optional[str] = None,
) -> str:
    base_dir = workspace_root or os.getcwd()

    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path

    # Direct check relative to base_dir or cwd
    candidate_abs = file_path if os.path.isabs(file_path) else os.path.abspath(os.path.join(base_dir, file_path))
    if os.path.exists(candidate_abs):
        return candidate_abs

    parent_dir = os.path.dirname(file_path) or "."
    base_name = os.path.basename(file_path)
    if os.path.exists(parent_dir) and os.path.exists(os.path.join(parent_dir, base_name)):
        return os.path.abspath(os.path.join(parent_dir, base_name))

    corrected_path = None

    if is_edit_mode and base_name:
        search_root = base_dir
        curr = search_root
        while curr and curr != "/":
            if os.path.isdir(os.path.join(curr, ".git")):
                search_root = curr
                break
            curr = os.path.dirname(curr)

        IGNORE_DIRS = {
            ".git", "venv", "env", "agent_venv", "git_venv",
            "__pycache__", ".idea", ".vscode", "node_modules", "dist", "build",
        }

        exact_matches = []
        partial_matches = []

        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
            for f in files:
                if f == base_name:
                    exact_matches.append(os.path.join(root, f))
                elif base_name.lower() in f.lower():
                    partial_matches.append(os.path.join(root, f))

        candidates = exact_matches if exact_matches else partial_matches

        if len(candidates) == 1:
            corrected_path = candidates[0]
        elif len(candidates) > 1:
            rel_file_path = file_path if not os.path.isabs(file_path) else os.path.relpath(file_path, search_root)
            rel_candidates = [os.path.relpath(c, search_root) for c in candidates]
            matches = difflib.get_close_matches(rel_file_path, rel_candidates, n=1, cutoff=0.1)
            if matches:
                corrected_path = os.path.join(search_root, matches[0])
            else:
                corrected_path = candidates[0]

    if corrected_path:
        return corrected_path

    # For creation or non-existing target, normalize to absolute path and ensure parent directory exists
    if not is_edit_mode:
        parent = os.path.dirname(candidate_abs)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    return candidate_abs


def get_blocks_from_lines(lines: List[str], file_path: str, block_size: int = 100, start_line_offset: int = 1) -> tuple[List[FileContent], int]:
    total_lines = len(lines)
    if total_lines == 0:
        return [], 0

    blocks = []
    for i in range(0, total_lines, block_size):
        chunk_end = min(i + block_size, total_lines)
        content_block = "".join(lines[i:chunk_end])
        blocks.append(
            FileContent(
                file_path=file_path,
                content_block=content_block,
                from_line=start_line_offset + i,
                to_line=start_line_offset + chunk_end - 1,
                block_summary=None,
            )
        )
    return blocks, total_lines


class FileReadTool(BaseTool):
    name = "file_read"
    description = (
        "Reads a file or inspects a directory and returns its content blocks, line counts, and code structure outline. "
        "Input: 'file_path' (str), optional 'from_line' (int, 1-indexed), 'to_line' (int), 'block_size' (int)."
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    def _indexing_file(self, file_path: str, content: Optional[str] = None) -> tuple[Dict[str, Any], str]:
        file_data: Dict[str, Any] = {
            "language": "",
            "imports": [],
            "functions": [],
            "classes": [],
        }

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if content is None:
            if not os.path.exists(file_path):
                return file_data, ""
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                return file_data, ""

        if ext == ".py":
            file_data["language"] = "python"
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_data["imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            file_data["imports"].append(f"{module}.{alias.name}")
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
        elif ext in [".js", ".jsx", ".ts", ".tsx"]:
            file_data["language"] = "javascript/typescript"
            import_matches = re.findall(r'import\s+.*?\s+from\s+[\'"](.*?)[\'"]', content)
            file_data["imports"].extend(import_matches)
            for m in re.finditer(r"class\s+([a-zA-Z0-9_]+)", content):
                file_data["classes"].append(m.group(1))
        else:
            file_data["language"] = ext.lstrip(".") or "text"

        file_data["imports"] = sorted(set(file_data["imports"]))
        file_data["functions"] = sorted(set(file_data["functions"]))
        file_data["classes"] = sorted(set(file_data["classes"]))
        return file_data, content

    async def execute(
        self,
        file_path: str,
        from_line: int = 1,
        to_line: Optional[int] = None,
        block_size: int = 100,
        **kwargs,
    ) -> ToolResult:
        try:
            workspace_root = getattr(self.runtime, "workspace_root", None)
            resolved_path = _auto_correct_path(
                file_path,
                is_edit_mode=True,
                tool_name="FileReadTool",
                workspace_root=workspace_root,
            )

            if not os.path.exists(resolved_path):
                return ToolResult(success=False, output=None, error=f"File or directory not found: {file_path}")

            if os.path.isdir(resolved_path):
                IGNORE_DIRS = {
                    ".git", "venv", "env", "agent_venv", "git_venv",
                    "__pycache__", ".idea", ".vscode", "node_modules", "dist", "build",
                }
                ALLOWED_EXTENSIONS = {
                    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss",
                    ".json", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini",
                    ".sh", ".bash", ".c", ".cpp", ".h", ".java", ".go", ".rs",
                }
                files_outline: Dict[str, Any] = {}
                summary_lines: List[str] = []

                for root, dirs, files in os.walk(resolved_path):
                    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
                    for file in sorted(files):
                        if file.startswith("."):
                            continue
                        _, ext = os.path.splitext(file)
                        if ext.lower() in ALLOWED_EXTENSIONS:
                            fp = os.path.join(root, file)
                            outline, raw_text = await asyncio.to_thread(self._indexing_file, fp)
                            line_cnt = len(raw_text.splitlines())
                            outline["total_lines"] = line_cnt
                            files_outline[fp] = outline
                            summary_lines.append(f"{fp} ({line_cnt} lines)\n")

                blocks, total_lines = get_blocks_from_lines(summary_lines, resolved_path, block_size=block_size)
                res_model = FileReadResult(
                    file_path=resolved_path,
                    content=blocks,
                    total_lines=total_lines,
                )
                out_dict = res_model.model_dump() if hasattr(res_model, "model_dump") else res_model.dict()
                out_dict["is_directory"] = True
                out_dict["files"] = files_outline
                return ToolResult(success=True, output=out_dict)

            # Single file read via runtime
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
            outline, _ = self._indexing_file(resolved_path, content=raw_content)

            result = FileReadResult(
                file_path=resolved_path,
                content=blocks,
                total_lines=total_file_lines,
            )
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            out_dict["outline"] = outline
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileWriteTool(BaseTool):
    name = "file_write"
    description = (
        "Writes or overwrites content in a file. "
        "Payload must include: 'file_path' (str), and 'write_content' or 'content' (str, the actual code/text). "
        "Optional: 'from_line' (int, 1-indexed), 'to_line' (int)."
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(
        self,
        file_path: str,
        write_content: Optional[str] = None,
        from_line: int = 1,
        to_line: Optional[int] = None,
        content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            actual_content = write_content if write_content is not None else content
            if actual_content is None:
                actual_content = ""

            workspace_root = getattr(self.runtime, "workspace_root", None)
            is_partial = from_line > 1 or to_line is not None
            resolved_path = _auto_correct_path(
                file_path,
                is_edit_mode=is_partial,
                tool_name="FileWriteTool",
                workspace_root=workspace_root,
            )

            if not is_partial:
                write_result = await self.runtime.execute_io("edit", resolved_path, actual_content)
                if not write_result.success:
                    return ToolResult(success=False, output=None, error=write_result.error)
                lines = actual_content.splitlines(keepends=True)
            else:
                read_result = await self.runtime.execute_io("read", resolved_path)
                lines = read_result.output.splitlines(keepends=True) if (read_result.success and read_result.output) else []

                start_idx = max(0, from_line - 1)
                end_line = to_line if to_line is not None else max(1, len(lines))

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
                write_result = await self.runtime.execute_io("edit", resolved_path, final_content)
                if not write_result.success:
                    return ToolResult(success=False, output=None, error=write_result.error)

            blocks, total_lines = get_blocks_from_lines(lines, resolved_path)
            result = FileWriteResult(
                file_path=resolved_path,
                content=blocks,
                total_lines=total_lines,
            )
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileAppendTool(BaseTool):
    name = "file_append"
    description = "Appends content to the end of a file and returns FileWriteResult. Input: 'file_path' (str), 'content' (str)."

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, content: str = "", write_content: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            actual_content = content if content else (write_content or "")
            workspace_root = getattr(self.runtime, "workspace_root", None)
            resolved_path = _auto_correct_path(
                file_path,
                is_edit_mode=os.path.exists(file_path),
                tool_name="FileAppendTool",
                workspace_root=workspace_root,
            )

            append_result = await self.runtime.execute_io("append", resolved_path, actual_content)
            if not append_result.success:
                return ToolResult(success=False, output=None, error=append_result.error)

            read_result = await self.runtime.execute_io("read", resolved_path)
            lines = read_result.output.splitlines(keepends=True) if (read_result.success and read_result.output) else []

            blocks, total_lines = get_blocks_from_lines(lines, resolved_path)
            result = FileWriteResult(
                file_path=resolved_path,
                content=blocks,
                total_lines=total_lines,
            )
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileEditTool(BaseTool):
    name = "file_edit"
    description = (
        "Updates/Edits a file using line-based chunks or search/replace edits. "
        "Payload must include: 'file_path' (str), and either 'chunks' (list of dicts with 'from_line', 'to_line', 'replacement_content') "
        "or 'edits' (list of dicts with 'search' and 'replace')."
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(
        self,
        file_path: str,
        chunks: Optional[List[Union[ReplacementChunk, Dict[str, Any]]]] = None,
        edits: Optional[list] = None,
        upsert: bool = True,
        **kwargs,
    ) -> ToolResult:
        try:
            workspace_root = getattr(self.runtime, "workspace_root", None)
            resolved_path = _auto_correct_path(
                file_path,
                is_edit_mode=True,
                tool_name="FileEditTool",
                workspace_root=workspace_root,
            )

            read_result = await self.runtime.execute_io("read", resolved_path)
            lines = read_result.output.splitlines(keepends=True) if (read_result.success and read_result.output) else []

            if chunks:
                processed_chunks = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        processed_chunks.append(ReplacementChunk(**chunk))
                    else:
                        processed_chunks.append(chunk)

                processed_chunks.sort(key=lambda c: c.from_line, reverse=True)

                for chunk in processed_chunks:
                    start_idx = max(0, chunk.from_line - 1)
                    end_idx = chunk.to_line

                    if start_idx > len(lines):
                        lines.extend(["\n"] * (start_idx - len(lines)))

                    new_lines = chunk.replacement_content.splitlines(keepends=True)
                    if chunk.replacement_content and not chunk.replacement_content.endswith("\n"):
                        if new_lines:
                            new_lines[-1] += "\n"
                        else:
                            new_lines.append("\n")

                    lines[start_idx:end_idx] = new_lines

            elif edits:
                content = "".join(lines)
                updated = content

                for edit in edits:
                    search = edit.get("search", edit.get("target", ""))
                    replace = edit.get("replace", edit.get("replacement", ""))
                    if not search:
                        continue

                    if search in updated:
                        updated = updated.replace(search, replace)
                    elif upsert:
                        if updated and not updated.endswith("\n"):
                            updated += "\n"
                        updated += replace
                        if replace and not replace.endswith("\n"):
                            updated += "\n"
                    else:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Search text not found: {search[:80]}",
                        )
                lines = updated.splitlines(keepends=True)

            final_content = "".join(lines)
            write_result = await self.runtime.execute_io("edit", resolved_path, final_content)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)

            blocks, total_lines = get_blocks_from_lines(lines, resolved_path)
            result = FileUpdateResult(
                file_path=resolved_path,
                success=True,
                content=blocks,
                total_lines=total_lines,
            )
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileSearchTool(BaseTool):
    name = "file_search"
    description = (
        "Searches for a query string or regex pattern in a file or directory. "
        "Input: 'file_path' (str), 'search_query' (str), optional 'is_regex' (bool), 'case_sensitive' (bool)."
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(
        self,
        file_path: Optional[str] = None,
        search_query: Optional[str] = None,
        is_regex: bool = False,
        case_sensitive: bool = False,
        path: Optional[str] = None,
        pattern: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            target_path = file_path or path
            final_query = search_query or pattern or query

            if not target_path:
                return ToolResult(success=False, output=None, error="No file path provided to search.")
            if not final_query:
                return ToolResult(success=False, output=None, error="No search query/pattern provided.")

            grep_flags = "-n"
            if not case_sensitive:
                grep_flags += " -i"
            if is_regex:
                grep_flags += " -E"
            else:
                grep_flags += " -F"

            is_dir = os.path.isdir(target_path)
            if is_dir:
                grep_flags += " -r"

            cmd = f"grep {grep_flags} {shlex.quote(final_query)} {shlex.quote(target_path)}"
            search_result = await self.runtime.execute_command(cmd)

            matches = []
            if search_result.success and search_result.output:
                for line in search_result.output.splitlines():
                    if not line.strip():
                        continue
                    parts = line.split(":", 2 if is_dir else 1)
                    if len(parts) >= 2:
                        try:
                            if is_dir and len(parts) >= 3:
                                line_num = int(parts[1])
                                line_content = parts[2]
                            else:
                                line_num = int(parts[0])
                                line_content = parts[1]

                            block_idx = max(0, line_num - 1) // 100
                            matches.append(
                                FileSearchMatch(
                                    line_number=line_num,
                                    line_content=line_content.strip(),
                                    block_index=block_idx,
                                )
                            )
                        except ValueError:
                            pass

            result = FileSearchResult(
                file_path=target_path,
                matches=matches,
                total_matches=len(matches),
            )
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileDeleteTool(BaseTool):
    name = "file_delete"
    description = "Deletes a file. Input: 'file_path' (str)."

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Safety Check Failed: Exact file not found at {file_path}. Auto-correction is disabled for deletion.",
                )

            exe_result = await self.runtime.execute_io("delete", file_path)
            if not exe_result.success:
                return ToolResult(success=False, output=None, error=exe_result.error)
            return ToolResult(success=True, output=f"Successfully deleted {file_path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
