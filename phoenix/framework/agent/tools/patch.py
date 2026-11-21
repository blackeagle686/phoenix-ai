"""Deterministic multi-block file patching tool with whitespace recovery and bounded Python syntax rollback."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from phoenix.framework.agent.cognition.schema import MultiBlockUpdateEdit, MultiBlockUpdateResult
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.tools._fs_utils import (
    EditItemSchema,
    SyntaxGuard,
    build_compact_diff,
    resolve_existing_path,
    smart_find_and_replace,
)
from phoenix.framework.agent.tools.base import BaseTool, ToolResult


class _MultiBlockUpdateArgs(BaseModel):
    file_path: str = Field(..., description="Path to the target file inside workspace")
    edits: List[EditItemSchema] = Field(
        ...,
        description="Non-empty list of {'target': ..., 'replacement': ...} blocks",
    )
    verify_syntax: bool = Field(
        default=True,
        description="Validate Python AST syntax before saving .py files (max 3 rollbacks)",
    )


def _dump_model(model: Any) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


class MultiBlockUpdateTool(BaseTool):
    """Tool for updating multiple non-contiguous blocks of code in a single file."""

    name = "file_update_multi"
    args_schema = _MultiBlockUpdateArgs
    param_aliases = {"path": "file_path"}
    description = (
        "Updates multiple blocks of code in a file atomically with exact + whitespace-tolerant unique matching. "
        "Inputs: 'file_path' (str), 'edits' (list of {'target': str, 'replacement': str})."
    )

    def __init__(self, runtime: Optional[Any] = None, max_syntax_rollbacks: int = 3):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()
        self._syntax_guard = SyntaxGuard(max_rollbacks=max_syntax_rollbacks)

    async def execute(
        self,
        file_path: str,
        edits: List[Union[MultiBlockUpdateEdit, Dict[str, str]]],
        verify_syntax: bool = True,
        **kwargs,
    ) -> ToolResult:
        try:
            if not edits:
                res = MultiBlockUpdateResult(
                    file_path=file_path,
                    success=False,
                    applied_count=0,
                    output="No edits provided.",
                )
                return ToolResult(success=False, output=_dump_model(res), error="No edits provided.")

            ok, resolved_path, err, meta = await asyncio.to_thread(
                resolve_existing_path, file_path, self.runtime, False
            )
            if not ok:
                res = MultiBlockUpdateResult(
                    file_path=file_path,
                    success=False,
                    applied_count=0,
                    output=err or f"File not found: {file_path}",
                )
                out_dict = _dump_model(res)
                out_dict.update(meta)
                return ToolResult(success=False, output=out_dict, error=err)

            read_result = await self.runtime.execute_io("read", resolved_path)
            if not read_result.success:
                res = MultiBlockUpdateResult(
                    file_path=resolved_path,
                    success=False,
                    applied_count=0,
                    output=read_result.error or f"File not found: {resolved_path}",
                )
                return ToolResult(success=False, output=_dump_model(res), error=read_result.error)

            old_content = read_result.output or ""
            updated = old_content
            applied_count = 0
            match_modes: List[str] = []

            for idx, item in enumerate(edits, start=1):
                if isinstance(item, dict):
                    edit = MultiBlockUpdateEdit(
                        target=item.get("target", item.get("search", "")),
                        replacement=item.get("replacement", item.get("replace", "")),
                    )
                else:
                    edit = item

                if not edit.target:
                    continue

                new_text, find_err, find_meta = smart_find_and_replace(
                    updated, edit.target, edit.replacement
                )
                if find_err is not None or new_text is None:
                    res = MultiBlockUpdateResult(
                        file_path=resolved_path,
                        success=False,
                        applied_count=0,
                        output=f"Edit #{idx} failed: {find_err}",
                    )
                    out_dict = _dump_model(res)
                    out_dict["rolled_back"] = True
                    out_dict.update(find_meta)
                    return ToolResult(success=False, output=out_dict, error=f"Edit #{idx} failed: {find_err}")

                updated = new_text
                applied_count += 1
                match_modes.append(str(find_meta.get("match_mode", "exact")))

            allow, rb_err, syn_warn, _ = self._syntax_guard.check_python(
                resolved_path, updated, enabled=verify_syntax
            )
            if not allow:
                res = MultiBlockUpdateResult(
                    file_path=resolved_path,
                    success=False,
                    applied_count=0,
                    output=rb_err or "Syntax rollback triggered.",
                )
                out_dict = _dump_model(res)
                out_dict["rolled_back"] = True
                return ToolResult(success=False, output=out_dict, error=rb_err)

            write_result = await self.runtime.execute_io("edit", resolved_path, updated)
            if not write_result.success:
                return ToolResult(success=False, output=None, error=write_result.error)

            res = MultiBlockUpdateResult(
                file_path=resolved_path,
                success=True,
                applied_count=applied_count,
                output=f"Applied {applied_count} block updates.",
            )
            out_dict = _dump_model(res)
            out_dict["match_modes"] = match_modes
            out_dict["diff"] = build_compact_diff(old_content, updated, resolved_path)
            if syn_warn:
                out_dict["syntax_warning"] = syn_warn
            out_dict.update(meta)
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            res = MultiBlockUpdateResult(
                file_path=file_path,
                success=False,
                applied_count=0,
                output=str(e),
            )
            return ToolResult(success=False, output=_dump_model(res), error=str(e))
