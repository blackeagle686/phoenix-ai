from phoenix.framework.agent.tools.base import BaseTool, ToolResult
import re
from typing import List, Dict, Union, Optional, Any
from phoenix.framework.agent.cognition.schema import FileContent
from phoenix.framework.agent.cognition.schema import *
from phoenix.services.llm.openai import OpenAILLM
from phoenix.core.config import config

import os
import asyncio
from itertools import islice
import ast
import json
import difflib


MODEL_NAME = getattr(config, "LLM_MODEL", "LongCat-2.0-Preview")
MODEL_API_KEY = getattr(config, "LLM_API_KEY", "ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G")
MODEL_BASE_URL = getattr(config, "LLM_BASE_URL", "https://api.longcat.chat/openai")


    

def _auto_correct_path(file_path: str, is_edit_mode: bool = True, tool_name: str = "Tool") -> str:
    import os
    import difflib

    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
        
    if not os.path.isabs(file_path) and not is_edit_mode:
        raise ValueError(
            f"[{tool_name}] SAFETY ERROR: Refusing to create file using relative or ambiguous path '{file_path}'. "
            f"You MUST provide an ABSOLUTE full path to ensure the file is placed correctly. "
            f"The current working directory is '{os.getcwd()}'."
        )

    parent_dir = os.path.dirname(file_path) or '.'
    base_name = os.path.basename(file_path)
    
    # Direct check
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    if os.path.exists(parent_dir) and os.path.exists(os.path.join(parent_dir, base_name)):
        return os.path.abspath(os.path.join(parent_dir, base_name))
        
    corrected_path = None

    if is_edit_mode:
        search_root = os.getcwd()
        # Find project root by looking for .git
        curr = search_root
        while curr and curr != '/':
            if os.path.isdir(os.path.join(curr, '.git')):
                search_root = curr
                break
            curr = os.path.dirname(curr)
            
        IGNORE_DIRS = {'.git', 'venv', 'env', '__pycache__', '.idea', '.vscode', 'node_modules', 'dist', 'build'}
        
        exact_matches = []
        partial_matches = []
        
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
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
        print(f"[{tool_name}] Path '{file_path}' corrected to: '{corrected_path}'")
        return corrected_path
    else:
        # If absolute path but parent doesn't exist, auto create it safely
        if os.path.isabs(file_path) and not is_edit_mode:
            parent = os.path.dirname(file_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
        return file_path

def get_blocks_from_lines(lines: List[str], file_path: str, block_size: int = 100) -> tuple[List[FileContent], int]:
    total_lines = len(lines)
    if total_lines == 0:
        return [], 0
        
    blocks = []
    for i in range(0, total_lines, block_size):
        chunk_end = min(i + block_size, total_lines)
        content_block = "".join(lines[i:chunk_end])
        blocks.append(FileContent(
            file_path=file_path,
            content_block=content_block,
            from_line=i + 1,
            to_line=chunk_end,
            block_summary=None
        ))
    return blocks, total_lines

class FileReadTool(BaseTool):
    name = "file_read"
    description = (
        "Use this tool to read, analysis and understanding the files content.\n"
        "Use this for python files and for directorys files \n"
        "\n Analyzes a file or directory and provides a comprehensive overall summary. Input: 'file_path' (str)."
        "\n this tool provide summary of a file or directory with information about"
        "\n language, imports, functions, classes and file summary"
        "\n and for the directory it provide summary of all the files in the directory \n"
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    def _indexing_file(self, file_path: str):

        file_data = {
            "language": "", 
            "imports": [], 
            "functions": [], 
            "classes": [],
            "file_summary": "" 
        }

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if not os.path.exists(file_path):
            return file_data, ""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return file_data, ""

        if ext == '.py':
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
                    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        file_data["functions"].append(f"{node.name} (L{node.lineno}-L{getattr(node, 'end_lineno', node.lineno)})")
                    elif isinstance(node, ast.ClassDef):
                        file_data["classes"].append(f"{node.name} (L{node.lineno}-L{getattr(node, 'end_lineno', node.lineno)})")
            except SyntaxError:
                pass

        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            file_data["language"] = "javascript/typescript"
            import_matches = re.findall(r'import\s+.*?\s+from\s+[\'"](.*?)[\'"]', content)
            file_data["imports"].extend(import_matches)
            
            line_starts = [0] + [m.start() + 1 for m in re.finditer(r'\n', content)]
            def get_line(offset):
                import bisect
                return bisect.bisect_right(line_starts, offset)

            for m in re.finditer(r'(?:function\s+([a-zA-Z0-9_]+))|(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>', content):
                func_name = m.group(1) or m.group(2)
                if func_name:
                    lineno = get_line(m.start())
                    file_data["functions"].append(f"{func_name} (L{lineno})")
                    
            for m in re.finditer(r'class\s+([a-zA-Z0-9_]+)', content):
                class_name = m.group(1)
                lineno = get_line(m.start())
                file_data["classes"].append(f"{class_name} (L{lineno})")

        elif ext in ['.html']:
            file_data["language"] = "html"
        elif ext in ['.css', '.scss']:
            file_data["language"] = "css"
        elif ext in ['.sh', '.bash']:
            file_data["language"] = "shell"
        elif ext in ['.json']:
            file_data["language"] = "json"
        elif ext in ['.md', '.txt']:
            file_data["language"] = "text"
        else:
            file_data["language"] = ext.replace('.', '')

        file_data["imports"] = list(set(file_data["imports"]))
        file_data["functions"] = list(set(file_data["functions"]))
        file_data["classes"] = list(set(file_data["classes"]))
        
        return file_data, content

    async def execute(self, file_path: str, block_size: int = 500, **kwargs) -> ToolResult:

        try:
            llm = OpenAILLM(
                model=MODEL_NAME,
                api_key=MODEL_API_KEY,
                base_url=MODEL_BASE_URL
            )
            await llm.init()

            IGNORE_DIRS = {'.git', 'venv', 'env', '__pycache__', '.idea', '.vscode', 'node_modules', 'dist', 'build'}
            ALLOWED_EXTENSIONS = {
                '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', 
                '.json', '.md', '.txt', '.yml', '.yaml', '.toml', '.ini', 
                '.sh', '.bash', '.c', '.cpp', '.h', '.java', '.go', '.rs'
            }
            
            target_files = []
            
            file_path = _auto_correct_path(file_path, is_edit_mode=True, tool_name="FileReadTool")

            if not os.path.exists(file_path):
                return ToolResult(success=False, output=None, error=f"File or directory not found: {file_path}")

            if os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path):
                    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
                    for file in files:
                        if file.startswith('.'):
                            continue
                        _, ext = os.path.splitext(file)
                        if ext.lower() in ALLOWED_EXTENSIONS:
                            target_files.append(os.path.join(root, file))
            elif os.path.isfile(file_path):
                target_files.append(file_path)

            if not target_files:
                return ToolResult(success=True, output="No valid text or code files found to analyze.")

            async def _understand_file(target_file: str):
                file_data, content = self._indexing_file(target_file)
                if not content:
                    return target_file, file_data

                outline = f"Language: {file_data['language']}\nImports: {', '.join(file_data['imports'][:20])}\nClasses: {', '.join(file_data['classes'])}\nFunctions: {', '.join(file_data['functions'])}"
                sample_content = content[:2000]
                
                try:
                    summary = await llm.generate(
                        f"Provide a brief summary (max 100 words) of this file's purpose based on its outline and sample content.\nOutline:\n{outline}\n\nSample Content:\n{sample_content}",
                        max_tokens=150
                    )
                    file_data["file_summary"] = summary
                except Exception:
                    file_data["file_summary"] = "Error generating summary."
                    
                return target_file, file_data

            file_results = await asyncio.gather(*[_understand_file(tf) for tf in target_files])
            
            final_output = {}
            global_context = ""
            for fp, data in file_results:
                final_output[fp] = data
                global_context += f"File: {fp}\nSummary: {data['file_summary']}\n\n"
            
            global_summary = ""
            if global_context.strip():
                try:
                    global_summary = await llm.generate(
                        f"Provide a comprehensive overall summary for these files based on their individual summaries:\n{global_context}",
                        max_tokens=500
                    )
                except Exception as e:
                    global_summary = f"Failed to generate global summary: {e}"

            result_dict = {
                "global_summary": global_summary,
                "files": final_output
            }
            
            return ToolResult(success=True, output=json.dumps(result_dict, indent=2))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class FileWriteTool(BaseTool):
    name = "file_write"
    description = (
        "Writes or overwrites content in a file. "
        "Payload must include: 'file_path' (str), 'write_content' (str, the actual code/text). "
        "Optional: 'from_line' (int, 1-indexed), 'to_line' (int)."
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, write_content: str = None, from_line: int = 1, to_line: Optional[int] = None, content: str = None, **kwargs) -> ToolResult:
        try:
            actual_content = write_content if write_content is not None else content
            if actual_content is None:
                actual_content = ""
                
            file_path = _auto_correct_path(file_path, is_edit_mode=(from_line > 1 or to_line is not None), tool_name="FileWriteTool")

            read_result = await self.runtime.execute_io("read", file_path)
            lines = read_result.output.splitlines(keepends=True) if (read_result.success and read_result.output) else []
                
            start_idx = max(0, from_line - 1)
            end_line = to_line if to_line is not None else max(1, len(lines))
            
            if start_idx > len(lines):
                lines.extend(["\n"] * (start_idx - len(lines)))
                
            new_lines = actual_content.splitlines(keepends=True)
            if actual_content and not actual_content.endswith('\n'):
                if new_lines:
                    new_lines[-1] += '\n'
                else:
                    new_lines.append('\n')
                    
            lines[start_idx:end_line] = new_lines
            
            final_content = "".join(lines)
            write_result = await self.runtime.execute_io("edit", file_path, final_content)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)
                
            blocks, total_lines = get_blocks_from_lines(lines, file_path)
            result = FileWriteResult(
                file_path=file_path,
                content=blocks,
                total_lines=total_lines
            )
            return ToolResult(success=True, output=result.dict())
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class FileAppendTool(BaseTool):
    name = "file_append"
    description = "Appends content to the end of a file and returns FileWriteResult."

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            file_path = _auto_correct_path(file_path, is_edit_mode=True, tool_name="FileAppendTool")
            
            append_result = await self.runtime.execute_io("append", file_path, content)
            if not append_result.success:
                return ToolResult(success=False, output=None, error=append_result.error)
                
            read_result = await self.runtime.execute_io("read", file_path)
            lines = read_result.output.splitlines(keepends=True) if (read_result.success and read_result.output) else []
            
            blocks, total_lines = get_blocks_from_lines(lines, file_path)
            result = FileWriteResult(
                file_path=file_path,
                content=blocks,
                total_lines=total_lines
            )
            return ToolResult(success=True, output=result.dict())
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class FileEditTool(BaseTool):
    name = "file_edit"
    description = (
        "Updates/Edits a file using line-based chunks. "
        "Payload must include: 'file_path' (str), 'chunks' (list of dicts). "
        "Each chunk dict must have: 'from_line' (int, 1-indexed start), "
        "'to_line' (int, 1-indexed end, inclusive), and 'replacement_content' (str, the new code/text to insert). "
        "Optional chunk field: 'target_content' (str). "
        "Example chunk: {\"from_line\": 1, \"to_line\": 2, \"replacement_content\": \"def sum(a, b):\\n    return a + b\\n\"}"
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, chunks: List[Union[ReplacementChunk, Dict[str, Any]]] = None, edits: list = None, upsert: bool = True, **kwargs) -> ToolResult:
        try:
            file_path = _auto_correct_path(file_path, is_edit_mode=True, tool_name="FileEditTool")
            
            read_result = await self.runtime.execute_io("read", file_path)
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
                    start_idx = chunk.from_line - 1
                    end_idx = chunk.to_line
                    
                    if start_idx > len(lines):
                        lines.extend(["\n"] * (start_idx - len(lines)))
                        
                    new_lines = chunk.replacement_content.splitlines(keepends=True)
                    if chunk.replacement_content and not chunk.replacement_content.endswith('\n'):
                        if new_lines:
                            new_lines[-1] += '\n'
                        else:
                            new_lines.append('\n')
                            
                    lines[start_idx:end_idx] = new_lines
                    
            elif edits:
                content = "".join(lines)
                updated = content
                applied = 0
                upserted = 0

                for edit in edits:
                    search = edit.get("search", "")
                    replace = edit.get("replace", "")
                    if not search:
                        continue

                    if search in updated:
                        updated = updated.replace(search, replace)
                        applied += 1
                    elif upsert:
                        if updated and not updated.endswith("\n"):
                            updated += "\n"
                        updated += replace
                        if replace and not replace.endswith("\n"):
                            updated += "\n"
                        upserted += 1
                    else:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Search text not found: {search[:80]}"
                        )
                lines = updated.splitlines(keepends=True)

            final_content = "".join(lines)
            write_result = await self.runtime.execute_io("edit", file_path, final_content)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)

            blocks, total_lines = get_blocks_from_lines(lines, file_path)
            result = FileUpdateResult(
                file_path=file_path,
                success=True,
                content=blocks,
                total_lines=total_lines
            )
            return ToolResult(success=True, output=result.dict())
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class FileSearchTool(BaseTool):
    name = "file_search"
    description = "Searches for a query string or regex pattern in a file or directory using bash grep."

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str = None, search_query: str = None, is_regex: bool = False, case_sensitive: bool = False, path: str = None, pattern: str = None, query: str = None, **kwargs) -> ToolResult:
        try:
            target_path = file_path or path
            final_query = search_query or pattern or query
            
            if not target_path:
                return ToolResult(success=False, output=None, error="No file path provided to search.")
            if not final_query:
                return ToolResult(success=False, output=None, error="No search query/pattern provided.")
                
            # Optimize search using execute_command
            import shlex
            
            grep_flags = "-n"
            if not case_sensitive:
                grep_flags += " -i"
            if is_regex:
                grep_flags += " -E"
            else:
                grep_flags += " -F"
                
            # If target_path is a directory, add recursive flag
            # Wait, since we are doing this strictly via runtime, we might not have `os.path.isdir`.
            # Let's check with a command first if it's a directory
            check_cmd = f"test -d {shlex.quote(target_path)} && echo 'dir' || echo 'file'"
            check_result = await self.runtime.execute_command(check_cmd)
            is_dir = "dir" in check_result.output
            
            if is_dir:
                grep_flags += " -r"
                
            cmd = f"grep {grep_flags} {shlex.quote(final_query)} {shlex.quote(target_path)}"
            search_result = await self.runtime.execute_command(cmd)
            
            matches = []
            if search_result.success and search_result.output:
                for line in search_result.output.splitlines():
                    if not line.strip(): continue
                    parts = line.split(":", 2 if is_dir else 1)
                    if len(parts) >= 2:
                        try:
                            # format: file:line:content or line:content
                            if is_dir and len(parts) >= 3:
                                line_num = int(parts[1])
                                line_content = parts[2]
                            else:
                                line_num = int(parts[0])
                                line_content = parts[1]
                                
                            block_idx = max(0, line_num - 1) // 100
                            matches.append(FileSearchMatch(
                                line_number=line_num,
                                line_content=line_content.strip(),
                                block_index=block_idx
                            ))
                        except ValueError:
                            pass
                            
            result = FileSearchResult(
                file_path=target_path,
                matches=matches,
                total_matches=len(matches)
            )
            return ToolResult(success=True, output=result.dict())
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
        import os
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, output=None, error=f"Safety Check Failed: Exact file not found at {file_path}. Auto-correction is disabled for deletion.")
                
            exe_result = await self.runtime.execute_io("delete", file_path)
            if not exe_result.success:
                return ToolResult(success=False, output=None, error=exe_result.error)
            return ToolResult(success=True, output=f"Successfully deleted {file_path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


if __name__ == "__main__":
    import asyncio
    analysis = FileReadTool()
    result = asyncio.run(analysis.execute("io.py"))
    print(result.output)