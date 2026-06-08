from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from typing import List, Dict, Union
from phoenix.framework.agent.cognition.schema import MultiBlockUpdateEdit, MultiBlockUpdateResult

class MultiBlockUpdateTool(BaseTool):
    """
    Tool for updating multiple non-contiguous blocks of code in a single file.
    """
    name = "file_update_multi"
    description = (
        "Updates multiple blocks of code in a file using unique search/replace substrings. "
        "Input: 'file_path' (str), 'edits' (list of MultiBlockUpdateEdit)."
    )

    def __init__(self, runtime=None):
        super().__init__()
        from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, edits: List[Union[MultiBlockUpdateEdit, Dict[str, str]]], **kwargs) -> ToolResult:
        try:
            read_result = await self.runtime.execute_io("read", file_path)
            if not read_result.success:
                result = MultiBlockUpdateResult(
                    file_path=file_path,
                    success=False,
                    applied_count=0,
                    output=f"File not found: {file_path}"
                )
                return ToolResult(success=False, output=result.dict(), error=f"File not found: {file_path}")

            new_content = read_result.output or ""
            applied_count = 0
            
            processed_edits = []
            for edit in edits:
                if isinstance(edit, dict):
                    processed_edits.append(MultiBlockUpdateEdit(**edit))
                else:
                    processed_edits.append(edit)

            for edit in processed_edits:
                target = edit.target
                replacement = edit.replacement
                
                if not target:
                    continue
                    
                if target not in new_content:
                    result = MultiBlockUpdateResult(
                        file_path=file_path,
                        success=False,
                        applied_count=applied_count,
                        output=f"Target content not found in file: {target[:100]}..."
                    )
                    return ToolResult(
                        success=False, 
                        output=result.dict(), 
                        error=f"Target content not found in file: {target[:100]}..."
                    )
                
                # Check for multiple occurrences to avoid ambiguity
                if new_content.count(target) > 1:
                    result = MultiBlockUpdateResult(
                        file_path=file_path,
                        success=False,
                        applied_count=applied_count,
                        output=f"Target content is not unique in file: {target[:100]}..."
                    )
                    return ToolResult(
                        success=False, 
                        output=result.dict(), 
                        error=f"Target content is not unique in file (found {new_content.count(target)} occurrences): {target[:100]}..."
                    )

                new_content = new_content.replace(target, replacement)
                applied_count += 1

            write_result = await self.runtime.execute_io("edit", file_path, new_content)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)

            result = MultiBlockUpdateResult(
                file_path=file_path,
                success=True,
                applied_count=applied_count,
                output=f"Successfully applied {applied_count} block updates."
            )
            return ToolResult(success=True, output=result.dict())
            
        except Exception as e:
            result = MultiBlockUpdateResult(
                file_path=file_path,
                success=False,
                applied_count=0,
                output=str(e)
            )
            return ToolResult(success=False, output=result.dict(), error=str(e))
