from phoenix.framework.agent.tools.base import BaseTool, ToolResult
import os
import re
from typing import List, Dict, Union, Optional, Any
from phoenix.framework.agent.cognition.planner.schema import FileContent
from phoenix.framework.agent.cognition.actor.schema import *

def get_file_content_blocks(file_path: str, block_size: int = 100) -> tuple[List[FileContent], int]:
    if not os.path.exists(file_path):
        return [], 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        # Fallback for binary or unreadable files
        return [], 0
        
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
    description = "Reads a file and returns its structured block content matching FileReadResult schema."

    async def execute(self, file_path: str, read_percentage: int = 100, block_size: int = 100, from_line: int = 1, to_line: Optional[int] = None, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, output=None, error=f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            total_lines = len(lines)
            
            # Slicing the file content based on from_line and to_line
            start_idx = max(0, from_line - 1)
            end_idx = total_lines if to_line is None else min(total_lines, to_line)
            
            # Apply read_percentage constraint if specified
            if read_percentage < 100:
                subset_len = int((end_idx - start_idx) * (read_percentage / 100))
                end_idx = start_idx + max(1, subset_len)
                
            sliced_lines = lines[start_idx:end_idx]
            
            # Build blocks for the sliced content
            blocks = []
            for i in range(0, len(sliced_lines), block_size):
                chunk_end = min(i + block_size, len(sliced_lines))
                content_block = "".join(sliced_lines[i:chunk_end])
                blocks.append(FileContent(
                    file_path=file_path,
                    content_block=content_block,
                    from_line=start_idx + i + 1,
                    to_line=start_idx + chunk_end,
                    block_summary=None
                ))
                
            result = FileReadResult(
                file_path=file_path,
                content=blocks,
                total_lines=total_lines
            )
            return ToolResult(success=True, output=result.dict())
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Writes/Appends content to a file at a specific line range and returns FileWriteResult."

    async def execute(self, file_path: str, write_content: str = None, from_line: int = 1, to_line: Optional[int] = None, content: str = None, **kwargs) -> ToolResult:
        try:
            actual_content = write_content if write_content is not None else content
            if actual_content is None:
                actual_content = ""
                
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Read existing lines if file exists
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                lines = []
                
            # If to_line is not provided, we replace starting from from_line to the end of the file
            end_line = to_line if to_line is not None else max(1, len(lines))
            
            # Apply replacement
            start_idx = from_line - 1
            if start_idx > len(lines):
                # Pad with newlines if from_line is beyond the end of file
                lines.extend(["\n"] * (start_idx - len(lines)))
                
            new_lines = actual_content.splitlines(keepends=True)
            # Add back trailing newline if original content had it
            if actual_content.endswith('\n') and (not new_lines or not new_lines[-1].endswith('\n')):
                if new_lines:
                    new_lines[-1] += '\n'
                else:
                    new_lines.append('\n')
            lines[start_idx:end_line] = new_lines
            
            # Save updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
            # Regenerate blocks for FileWriteResult
            blocks, total_lines = get_file_content_blocks(file_path)
            
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

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
                
            blocks, total_lines = get_file_content_blocks(file_path)
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
        "Updates/Edits a file using line-based chunks or search-and-replace lists. "
        "Input matches FileUpdateTask: 'file_path' (str), 'chunks' (list of ReplacementChunk). "
        "Fallback/legacy input: 'edits' (list of {search, replace})."
    )

    async def execute(self, file_path: str, chunks: List[Union[ReplacementChunk, Dict[str, Any]]] = None, edits: list = None, upsert: bool = True, **kwargs) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("")
                    
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 1. Process block-based chunks (preferred)
            if chunks:
                processed_chunks = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        processed_chunks.append(ReplacementChunk(**chunk))
                    else:
                        processed_chunks.append(chunk)
                        
                # Sort chunks by from_line in descending order to avoid line shifting
                processed_chunks.sort(key=lambda c: c.from_line, reverse=True)
                
                for chunk in processed_chunks:
                    start_idx = chunk.from_line - 1
                    end_idx = chunk.to_line
                    
                    if start_idx > len(lines):
                        lines.extend(["\n"] * (start_idx - len(lines)))
                        
                    new_lines = chunk.replacement_content.splitlines(keepends=True)
                    lines[start_idx:end_idx] = new_lines
                    
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                    
            # 2. Process legacy search-and-replace edits
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
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated)

            # Regenerate blocks for FileUpdateResult
            blocks, total_lines = get_file_content_blocks(file_path)
            
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
    description = "Searches for a query string or regex pattern in a file and returns FileSearchResult."

    async def execute(self, file_path: str = None, search_query: str = None, is_regex: bool = False, case_sensitive: bool = False, path: str = None, pattern: str = None, query: str = None, **kwargs) -> ToolResult:
        try:
            target_path = file_path or path
            final_query = search_query or pattern or query
            
            if not target_path:
                return ToolResult(success=False, output=None, error="No file path provided to search.")
            if not final_query:
                return ToolResult(success=False, output=None, error="No search query/pattern provided.")
                
            if not os.path.exists(target_path):
                return ToolResult(success=False, output=None, error=f"Path not found: {target_path}")

            files_to_search = []
            if os.path.isfile(target_path):
                files_to_search = [target_path]
            else:
                for root, _, files in os.walk(target_path):
                    for file in files:
                        files_to_search.append(os.path.join(root, file))

            matches = []
            
            for f_path in files_to_search:
                try:
                    with open(f_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    continue  # Skip binary files
                except Exception:
                    continue

                for i, line_content in enumerate(lines):
                    match_found = False
                    
                    if is_regex:
                        flags = 0 if case_sensitive else re.IGNORECASE
                        if re.search(query, line_content, flags):
                            match_found = True
                    else:
                        if case_sensitive:
                            if query in line_content:
                                match_found = True
                        else:
                            if query.lower() in line_content.lower():
                                match_found = True
                                
                    if match_found:
                        block_idx = i // 100
                        matches.append(FileSearchMatch(
                            line_number=i + 1,
                            line_content=line_content.strip(),
                            block_index=block_idx
                        ))

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

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, output=None, error=f"File not found: {file_path}")
            os.remove(file_path)
            return ToolResult(success=True, output=f"Successfully deleted {file_path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

