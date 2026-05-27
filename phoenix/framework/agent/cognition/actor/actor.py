import asyncio
import json
from typing import Dict, Any
from .base import BaseActor
from phoenix.framework.agent.tools.base import ToolResult

class Actor(BaseActor):
    """
    Executes plans by interacting with tools.
    Moved to cognition module for centralized agent logic.
    """
    
    async def execute(self, plan: Dict[str, Any]) -> str:
        actions = plan.get("actions", [])
        
        # Backward compatibility for single tool field
        if not actions and "tool" in plan:
            actions = [{"tool": plan["tool"], "kwargs": plan.get("kwargs", {})} ]
            
        if not actions:
            return "No actions specified in the plan."
            
        # Check for finish
        if any(a.get("tool") == "finish" for a in actions):
            return "Task marked as finished by Planner."
            
        # Execute all actions
        results = []
        for action in actions:
            tool_name = action.get("tool")
            kwargs = action.get("kwargs", {})
            try:
                # Try getting the tool directly from registry to access raw ToolResult objects
                tool = self.tool_manager.registry.get_tool(tool_name)
                res = await tool.execute(**kwargs)
                results.append((tool_name, res))
            except Exception:
                # Fallback to tool_manager's string-based tool execution
                try:
                    res_str = await self.tool_manager.execute_tool(tool_name, kwargs)
                    success = "failed" not in res_str.lower() and "error" not in res_str.lower()
                    results.append((tool_name, ToolResult(
                        success=success,
                        output=res_str,
                        error=res_str if not success else None
                    )))
                except Exception as e:
                    results.append((tool_name, ToolResult(
                        success=False,
                        output=None,
                        error=str(e)
                    )))
        
        # Combine results into a structured, smart Markdown execution report
        report_sections = ["### 🛠️ Action Execution Report"]
        
        # Summary list
        summary = []
        for tool_name, res in results:
            status_emoji = "✅ SUCCESS" if res.success else "❌ FAILED"
            summary.append(f"- **Tool**: `{tool_name}` | **Status**: {status_emoji}")
        report_sections.append("\n".join(summary))
        
        # Detail sections
        for tool_name, res in results:
            report_sections.append(f"\n---\n#### 🔍 `{tool_name}` Detailed Output")
            
            output_val = res.output
            error_val = res.error
            
            # Smart formatting based on schema types
            if isinstance(output_val, dict):
                # 1. CommandExecutionResult
                if "command" in output_val and "exit_code" in output_val:
                    report_sections.append(f"- **Command**: `{output_val.get('command')}`")
                    report_sections.append(f"- **Exit Code**: `{output_val.get('exit_code')}`")
                    if output_val.get("stdout"):
                        report_sections.append(f"- **Stdout**:\n```text\n{output_val.get('stdout').strip()}\n```")
                    if output_val.get("stderr"):
                        report_sections.append(f"- **Stderr**:\n```text\n{output_val.get('stderr').strip()}\n```")
                        
                # 2. CodeCompileResult
                elif "file_path" in output_val and "success" in output_val and "error" in output_val:
                    report_sections.append(f"- **File Checked**: `{output_val.get('file_path')}`")
                    report_sections.append(f"- **Compiles Successfully**: `{output_val.get('success')}`")
                    if output_val.get("error"):
                        report_sections.append(f"- **Compilation Error**:\n```text\n{output_val.get('error').strip()}\n```")
                
                # 3. CodeExecutionResult (REPL)
                elif "success" in output_val and "output" in output_val:
                    report_sections.append(f"- **Executed Successfully**: `{output_val.get('success')}`")
                    if output_val.get("output"):
                        report_sections.append(f"- **Output**:\n```text\n{output_val.get('output').strip()}\n```")
                    if output_val.get("error"):
                        report_sections.append(f"- **Runtime Error**:\n```text\n{output_val.get('error').strip()}\n```")
                
                # 4. Fallback dictionary JSON printing
                else:
                    report_sections.append(f"```json\n{json.dumps(output_val, indent=2)}\n```")
            else:
                # String output
                if output_val:
                    report_sections.append(f"```text\n{str(output_val).strip()}\n```")
                    
            if error_val and not (isinstance(output_val, dict) and ("stderr" in output_val or "error" in output_val)):
                report_sections.append(f"- **Error Details**:\n```text\n{str(error_val).strip()}\n```")
                
        return "\n".join(report_sections)
