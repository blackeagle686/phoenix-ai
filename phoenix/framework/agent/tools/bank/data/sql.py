"""SQL database tool with strict read-only SQL sanitization, row capping, and clean engine disposal."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Literal, Optional

from phoenix.framework.agent.tools.base import BaseTool, ToolResult

_READONLY_START_RE = re.compile(r"^(select|pragma|explain|show|describe)\b", re.IGNORECASE)
_WRITE_KEYWORDS_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|grant|revoke|vacuum|reindex)\b",
    re.IGNORECASE,
)


def _strip_sql_comments_and_strings(sql: str) -> str:
    """Remove SQL comments and string literals safely before checking statement safety."""
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--[^\r\n]*", " ", no_block)
    no_strings = re.sub(r"'(?:''|[^'])*'", "''", no_line)
    return no_strings.strip()


def _is_safe_readonly_sql(query: str) -> tuple[bool, Optional[str]]:
    cleaned = _strip_sql_comments_and_strings(query).rstrip(";").strip()
    if not cleaned:
        return False, "Empty SQL query."
    if ";" in cleaned:
        return False, "Multi-statement SQL queries are blocked in read-only mode."
    if _WRITE_KEYWORDS_RE.search(cleaned):
        return False, "Tool is configured as read-only. Write/DDL keywords are blocked."
    if not (_READONLY_START_RE.match(cleaned) or cleaned.lower().startswith("with ")):
        return False, "Tool is configured as read-only. Only SELECT/WITH/PRAGMA/EXPLAIN queries are allowed."
    return True, None


class SQLDatabaseTool(BaseTool):
    name = "sql_database"
    description = (
        "Inspects SQL database schemas or runs SQL queries (SQLite, PostgreSQL, MySQL). "
        "Inputs: 'action' ('get_schema' | 'execute_query'), optional 'query' (str), optional 'max_rows' (int=100)."
    )

    def __init__(self, connection_uri: Optional[str] = None, read_only: bool = True):
        """
        :param connection_uri: Optional SQLAlchemy connection string.
        :param read_only: If True, blocks write/DDL/multi-statement queries.
        """
        super().__init__(read_only=read_only)
        self.connection_uri = connection_uri

    async def execute(
        self,
        action: Literal["get_schema", "execute_query"],
        query: Optional[str] = None,
        connection_uri: Optional[str] = None,
        max_rows: int = 100,
        **kwargs,
    ) -> ToolResult:
        uri = connection_uri or self.connection_uri or os.environ.get("SQL_CONNECTION_URI")
        if not uri:
            return ToolResult(
                success=False,
                output=None,
                error="SQL_CONNECTION_URI must be provided via arguments or environment variables.",
            )

        try:
            from sqlalchemy import MetaData, create_engine, text
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="SQLAlchemy is not installed. Please install it using: pip install sqlalchemy",
            )

        row_cap = max(1, min(int(max_rows or 100), 1000))

        def _sync_db() -> ToolResult:
            engine = None
            try:
                engine = create_engine(uri)
                if action == "get_schema":
                    metadata = MetaData()
                    metadata.reflect(bind=engine)
                    schema_info = []
                    for table in metadata.sorted_tables:
                        cols = [f"{c.name} ({c.type})" for c in table.columns]
                        schema_info.append(f"Table: {table.name} | Columns: {', '.join(cols)}")
                    output = "\n".join(schema_info)
                    return ToolResult(success=True, output=output or "Database has no tables.")

                if action == "execute_query":
                    if not query:
                        return ToolResult(success=False, output=None, error="Missing 'query' for execute_query.")

                    if self.read_only:
                        safe, err_msg = _is_safe_readonly_sql(query)
                        if not safe:
                            return ToolResult(success=False, output=None, error=err_msg)

                    with engine.connect() as conn:
                        result = conn.execute(text(query))
                        if not result.returns_rows:
                            conn.commit()
                            return ToolResult(success=True, output="Query executed successfully. No results returned.")

                        rows = result.fetchmany(row_cap + 1)
                        if not rows:
                            return ToolResult(success=True, output="Query executed successfully. No rows found.")

                        truncated = len(rows) > row_cap
                        rows = rows[:row_cap]
                        columns = list(result.keys())
                        header = " | ".join(str(c) for c in columns)
                        row_strs = [" | ".join(str(val) for val in row) for row in rows]
                        output = header + "\n" + "-" * len(header) + "\n" + "\n".join(row_strs)
                        if truncated:
                            output += f"\n... (truncated at {row_cap} rows)"
                        return ToolResult(success=True, output=output[:10000])

                return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
            except Exception as e:
                return ToolResult(success=False, output=None, error=f"Database Error: {str(e)}")
            finally:
                if engine is not None:
                    engine.dispose()

        return await asyncio.to_thread(_sync_db)
