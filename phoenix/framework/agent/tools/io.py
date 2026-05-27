from phoenix.framework.agent.tools.base import BaseTool, ToolResult
import os
import re
from phoenix.framework.agent.cognition.
class FileReadTool(BaseTool):
    name = "file_read"
    description = "Reads the content of a file. Input: 'file_path' (str). Optional: 'chunk_size' (int) to return a list of chunks."

    async def execute(self, file_path: str, chunk_size: int = None, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, output="", error=f"File not found: {file_path}")
            # If chunk_size is provided and > 0, read the file in successive chunks
            if chunk_size and isinstance(chunk_size, int) and chunk_size > 0:
                chunks = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    while True:
                        part = f.read(chunk_size)
                        if not part:
                            break
                        chunks.append(part)
                return ToolResult(success=True, output=chunks)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Writes content to a file. Input: 'file_path' (str), 'content' (str)."

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(success=True, output=f"Successfully wrote to {file_path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class FileAppendTool(BaseTool):
    name = "file_append"
    description = "Appends content to a file. Input: 'file_path' (str), 'content' (str)."

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(success=True, output=f"Successfully appended to {file_path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class FileEditTool(BaseTool):
    name = "file_edit"
    description = (
        "Search-and-replace patching tool with upsert behavior. "
        "Input: 'file_path' (str), 'edits' (list of {search, replace}), "
        "'upsert' (bool, optional, default True)."
    )

    async def execute(self, file_path: str, edits: list, upsert: bool = True, **kwargs) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

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
                        output=f"Applied {applied}, upserted {upserted}.",
                        error=f"Search text not found: {search[:80]}"
                    )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated)

            return ToolResult(
                success=True,
                output=f"Successfully edited {file_path} (applied={applied}, upserted={upserted})"
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class FileSearchTool(BaseTool):
    name = "file_search"
    description = "Searches for a specific string or regex pattern in a file or directory. Input: 'path' (str), 'pattern' (str), 'is_regex' (bool, optional)."

    async def execute(self, path: str, pattern: str, is_regex: bool = False, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(path):
                return ToolResult(success=False, output="", error=f"Path not found: {path}")

            results = []
            
            if os.path.isfile(path):
                files_to_search = [path]
            else:
                files_to_search = []
                for root, _, files in os.walk(path):
                    for file in files:
                        files_to_search.append(os.path.join(root, file))

            for file_path in files_to_search:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            match = False
                            if is_regex:
                                if re.search(pattern, line):
                                    match = True
                            else:
                                if pattern in line:
                                    match = True
                                    
                            if match:
                                results.append(f"{file_path}:{i+1}: {line.strip()}")
                except UnicodeDecodeError:
                    # Skip binary files
                    continue

            if not results:
                return ToolResult(success=True, output="No matches found.")
                
            return ToolResult(success=True, output="\n".join(results))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
