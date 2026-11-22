"""User-Modifiable Micro Tool System for CVAgent (Alerts, Local Files, Local SQLite DB).

Zero external dependencies — 100% Python stdlib (`sqlite3`, `os`, `json`, `urllib.request`)
engineered for low-RAM mobile CPUs and edge gateways.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_DC_SLOTS: Dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(DROP|ALTER|TRUNCATE|DELETE|UPDATE|INSERT|REPLACE|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


@dataclass(**_DC_SLOTS)
class MicroToolResult:
    """Compact outcome returned by every CVAgent micro tool."""

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(**_DC_SLOTS)
class MicroTool:
    """Single user-modifiable tool callable (sync or async) for CV event handling."""

    name: str
    fn: Callable[..., Any]
    description: str = ""
    default_args: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tags: Tuple[str, ...] = ()

    async def invoke(self, arguments: Optional[Dict[str, Any]] = None) -> MicroToolResult:
        t0 = time.perf_counter()
        if not self.enabled:
            return MicroToolResult(
                tool_name=self.name,
                success=False,
                error=f"Tool '{self.name}' is disabled.",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        merged = dict(self.default_args)
        if arguments:
            merged.update(arguments)
        try:
            res = self.fn(**merged)
            if inspect.isawaitable(res):
                res = await res
            return MicroToolResult(
                tool_name=self.name,
                success=True,
                output=res,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            return MicroToolResult(
                tool_name=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )


class MicroToolRegistry:
    """Mutable registry enabling users to add, replace, customize, disable, or delete any CV tool."""

    def __init__(self) -> None:
        self._tools: Dict[str, MicroTool] = {}

    def register(
        self,
        tool_or_fn: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        default_args: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        tags: Sequence[str] = (),
        overwrite: bool = True,
    ) -> MicroTool:
        if isinstance(tool_or_fn, MicroTool):
            tool = tool_or_fn
            if name:
                tool.name = name
            if description is not None:
                tool.description = description
            if default_args is not None:
                tool.default_args = dict(default_args)
            tool.enabled = enabled
        elif callable(tool_or_fn):
            t_name = (name or getattr(tool_or_fn, "__name__", "") or "custom_tool").strip()
            t_desc = description if description is not None else (inspect.getdoc(tool_or_fn) or "").strip()
            tool = MicroTool(
                name=t_name,
                fn=tool_or_fn,
                description=t_desc,
                default_args=dict(default_args or {}),
                enabled=enabled,
                tags=tuple(tags),
            )
        else:
            raise TypeError("Expected a MicroTool instance or callable function.")

        if not overwrite and tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already exists; pass overwrite=True or use modify().")
        self._tools[tool.name] = tool
        return tool

    def modify(
        self,
        name: str,
        *,
        fn: Optional[Callable[..., Any]] = None,
        description: Optional[str] = None,
        default_args: Optional[Dict[str, Any]] = None,
        merge_default_args: bool = True,
        enabled: Optional[bool] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> MicroTool:
        """Modify an existing built-in or user tool in place (swap handler, defaults, or enabled flag)."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        tool = self._tools[name]
        if fn is not None:
            if not callable(fn):
                raise TypeError("Replacement `fn` must be callable.")
            tool.fn = fn
        if description is not None:
            tool.description = description
        if default_args is not None:
            if merge_default_args:
                merged = dict(tool.default_args)
                merged.update(default_args)
                tool.default_args = merged
            else:
                tool.default_args = dict(default_args)
        if enabled is not None:
            tool.enabled = bool(enabled)
        if tags is not None:
            tool.tags = tuple(tags)
        return tool

    def replace(self, name: str, fn: Callable[..., Any], *, description: Optional[str] = None) -> MicroTool:
        return self.modify(name, fn=fn, description=description)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def disable(self, name: str) -> None:
        self.modify(name, enabled=False)

    def enable(self, name: str) -> None:
        self.modify(name, enabled=True)

    def get(self, name: str) -> Optional[MicroTool]:
        return self._tools.get(name)

    def enabled_names(self) -> List[str]:
        return [name for name, t in self._tools.items() if t.enabled]

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "default_args": dict(t.default_args),
                "tags": list(t.tags),
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> MicroToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return MicroToolResult(tool_name=name, success=False, error=f"Unknown tool '{name}'.")
        return await tool.invoke(arguments)


class AlertDispatcher:
    """Multi-channel micro alert sender (callback, local JSONL file, optional stdlib webhook)."""

    def __init__(
        self,
        *,
        on_alert: Optional[Callable[[Dict[str, Any]], Any]] = None,
        alert_log_path: Optional[str] = None,
        webhook_url: Optional[str] = None,
        webhook_timeout_sec: float = 3.0,
        max_history: int = 100,
    ):
        self.on_alert = on_alert
        self.alert_log_path = alert_log_path
        self.webhook_url = webhook_url
        self.webhook_timeout_sec = float(webhook_timeout_sec)
        self.max_history = max(10, int(max_history))
        self.history: List[Dict[str, Any]] = []

    async def send(
        self,
        message: str,
        *,
        severity: str = "high",
        camera_id: str = "cam_0",
        zone: str = "default",
        event_id: str = "",
        channel: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "severity": str(severity).lower(),
            "camera_id": camera_id,
            "zone": zone,
            "event_id": event_id,
            "channel": channel,
            "message": str(message),
            "metadata": dict(metadata or {}),
        }
        delivered_to: List[str] = []

        if self.on_alert is not None:
            res = self.on_alert(payload)
            if inspect.isawaitable(res):
                await res
            delivered_to.append("callback")

        if self.alert_log_path:
            parent = os.path.dirname(os.path.abspath(self.alert_log_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.alert_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            delivered_to.append("jsonl_file")

        if self.webhook_url:
            def _post_webhook() -> int:
                req = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.webhook_timeout_sec) as resp:
                    return int(resp.status)

            status = await asyncio.to_thread(_post_webhook)
            delivered_to.append(f"webhook:{status}")

        if not delivered_to:
            delivered_to.append("memory_buffer")

        self.history.append(payload)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        return {"sent": True, "delivered_to": delivered_to, "alert": payload}


class LocalFileStore:
    """Symlink-jailed, RAM-capped local file reader/writer for phone/edge storage."""

    def __init__(
        self,
        workspace_root: str = "./cv_workspace",
        *,
        max_read_bytes: int = 65536,
        max_write_bytes: int = 262144,
    ):
        self.workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        os.makedirs(self.workspace_root, exist_ok=True)
        self.max_read_bytes = max(1024, int(max_read_bytes))
        self.max_write_bytes = max(1024, int(max_write_bytes))

    def _safe_path(self, rel_or_abs: str) -> str:
        if not rel_or_abs or not str(rel_or_abs).strip():
            raise ValueError("File path cannot be empty.")
        raw = str(rel_or_abs).strip()
        cand = raw if os.path.isabs(raw) else os.path.join(self.workspace_root, raw)
        real_target = os.path.realpath(os.path.abspath(cand))
        if os.path.commonpath([self.workspace_root, real_target]) != self.workspace_root:
            raise PermissionError(f"Path '{rel_or_abs}' escapes workspace root '{self.workspace_root}'.")
        return real_target

    def read_file(self, path: str, *, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        target = self._safe_path(path)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"File not found: '{path}'")
        cap = min(int(max_bytes or self.max_read_bytes), self.max_read_bytes)
        size = os.path.getsize(target)
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(cap)
        return {
            "path": os.path.relpath(target, self.workspace_root),
            "bytes_total": size,
            "truncated": size > cap,
            "content": content,
        }

    def write_file(self, path: str, content: str, *, append: bool = False) -> Dict[str, Any]:
        target = self._safe_path(path)
        encoded = str(content or "").encode("utf-8")
        if len(encoded) > self.max_write_bytes:
            raise ValueError(f"Write size {len(encoded)} bytes exceeds cap ({self.max_write_bytes} bytes).")
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(str(content or ""))
        return {
            "path": os.path.relpath(target, self.workspace_root),
            "bytes_written": len(encoded),
            "mode": "append" if append else "write",
        }

    def list_files(self, subdir: str = "", *, max_entries: int = 50) -> List[str]:
        base = self._safe_path(subdir) if subdir else self.workspace_root
        if not os.path.isdir(base):
            return []
        found: List[str] = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
            for fn in sorted(files):
                if fn.startswith("."):
                    continue
                found.append(os.path.relpath(os.path.join(root, fn), self.workspace_root))
                if len(found) >= max_entries:
                    return found
        return found


class LocalSQLiteStore:
    """Zero-dependency local SQLite database connector & safe record reader for edge/phone storage."""

    def __init__(self, *, max_rows: int = 50, allow_writes: bool = True):
        self.max_rows = max(1, min(int(max_rows), 250))
        self.allow_writes = bool(allow_writes)
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._paths: Dict[str, str] = {}

    def connect(self, db_path: str, *, alias: str = "default") -> Dict[str, Any]:
        if not db_path or not str(db_path).strip():
            raise ValueError("Database path cannot be empty.")
        raw = str(db_path).strip()
        if raw != ":memory:":
            abs_p = os.path.abspath(raw)
            parent = os.path.dirname(abs_p)
            if parent:
                os.makedirs(parent, exist_ok=True)
            resolved = abs_p
        else:
            resolved = ":memory:"

        if alias in self._connections:
            try:
                self._connections[alias].close()
            except Exception:
                pass

        conn = sqlite3.connect(resolved, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._connections[alias] = conn
        self._paths[alias] = resolved
        return {"connected": True, "alias": alias, "db_path": resolved, "tables": self.list_tables(alias=alias)}

    def _get_conn(self, alias: str = "default") -> sqlite3.Connection:
        conn = self._connections.get(alias)
        if conn is None:
            available = list(self._connections.keys())
            raise RuntimeError(
                f"No local database connected under alias '{alias}'. Connected aliases: {available}"
            )
        return conn

    @staticmethod
    def _validate_ident(name: str, kind: str = "identifier") -> str:
        s = str(name or "").strip()
        if not _SAFE_IDENT_RE.match(s):
            raise ValueError(f"Unsafe SQL {kind}: '{name}'")
        return s

    def list_tables(self, *, alias: str = "default") -> List[str]:
        conn = self._get_conn(alias)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [str(r[0]) for r in cur.fetchall()]

    def read_records(
        self,
        table: str,
        *,
        where: Optional[Dict[str, Any]] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
        alias: str = "default",
    ) -> Dict[str, Any]:
        """Parameterized, injection-safe record reader with row cap for low-RAM devices."""
        conn = self._get_conn(alias)
        tbl = self._validate_ident(table, "table")
        if columns:
            cols_sql = ", ".join(self._validate_ident(c, "column") for c in columns)
        else:
            cols_sql = "*"

        params: List[Any] = []
        sql = f"SELECT {cols_sql} FROM {tbl}"
        if where:
            clauses = []
            for k, v in where.items():
                col = self._validate_ident(k, "column")
                if v is None:
                    clauses.append(f"{col} IS NULL")
                else:
                    clauses.append(f"{col} = ?")
                    params.append(v)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

        if order_by:
            ord_col = self._validate_ident(order_by, "order_by column")
            sql += f" ORDER BY {ord_col} {'DESC' if descending else 'ASC'}"

        row_cap = max(1, min(int(limit or self.max_rows), self.max_rows))
        sql += f" LIMIT {row_cap}"

        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        return {"alias": alias, "table": tbl, "count": len(rows), "records": rows}

    def query_readonly(
        self,
        sql: str,
        *,
        params: Optional[Sequence[Any]] = None,
        limit: Optional[int] = None,
        alias: str = "default",
    ) -> Dict[str, Any]:
        """Execute a strictly read-only SELECT query against a connected local SQLite database."""
        conn = self._get_conn(alias)
        q = (sql or "").strip().rstrip(";")
        if not q or not re.match(r"^(SELECT|WITH)\b", q, re.IGNORECASE):
            raise PermissionError("Only read-only SELECT/WITH queries are allowed in db_query.")
        if ";" in q or _FORBIDDEN_SQL_RE.search(q):
            raise PermissionError("Blocked unsafe or multi-statement SQL in db_query.")

        row_cap = max(1, min(int(limit or self.max_rows), self.max_rows))
        cur = conn.execute(q, list(params or []))
        fetched = cur.fetchmany(row_cap)
        rows = [dict(r) for r in fetched]
        return {"alias": alias, "count": len(rows), "records": rows}

    def insert_record(
        self,
        table: str,
        record: Dict[str, Any],
        *,
        alias: str = "default",
    ) -> Dict[str, Any]:
        """Insert a parameterized record into a local SQLite table (e.g. CV event audit log)."""
        if not self.allow_writes:
            raise PermissionError("Database writes are disabled on this LocalSQLiteStore.")
        if not record or not isinstance(record, dict):
            raise ValueError("Record must be a non-empty dictionary.")
        conn = self._get_conn(alias)
        tbl = self._validate_ident(table, "table")
        cols = [self._validate_ident(k, "column") for k in record.keys()]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {tbl} ({', '.join(cols)}) VALUES ({placeholders})"
        cur = conn.execute(sql, list(record.values()))
        conn.commit()
        return {"inserted": True, "alias": alias, "table": tbl, "rowid": cur.lastrowid}

    def close_all(self) -> None:
        for conn in list(self._connections.values()):
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()
        self._paths.clear()

    def connected_aliases(self) -> Dict[str, str]:
        return dict(self._paths)


def build_default_cv_tools(
    *,
    registry: MicroToolRegistry,
    alerts: AlertDispatcher,
    files: LocalFileStore,
    db: LocalSQLiteStore,
) -> MicroToolRegistry:
    """Register the core modifiable toolset for CV event handling."""
    registry.register(
        alerts.send,
        name="send_alert",
        description="Send a CV alert (severity, camera_id, zone, message, channel, metadata).",
        tags=("alert", "cv"),
    )
    registry.register(
        files.read_file,
        name="read_file",
        description="Read a local file safely inside workspace_root (path, max_bytes).",
        tags=("file", "read"),
    )
    registry.register(
        files.write_file,
        name="write_file",
        description="Write or append content to a local file inside workspace_root (path, content, append=False).",
        tags=("file", "write"),
    )
    registry.register(
        lambda path, content: files.write_file(path, content, append=True),
        name="append_file",
        description="Append a log line or content to a local file inside workspace_root (path, content).",
        tags=("file", "write"),
    )
    registry.register(
        files.list_files,
        name="list_files",
        description="List files inside workspace_root (subdir='', max_entries=50).",
        tags=("file", "read"),
    )
    registry.register(
        db.connect,
        name="db_connect",
        description="Connect to a local SQLite database file (db_path, alias='default').",
        tags=("database",),
    )
    registry.register(
        db.list_tables,
        name="db_list_tables",
        description="List tables in a connected local SQLite database (alias='default').",
        tags=("database", "read"),
    )
    registry.register(
        db.read_records,
        name="db_read_records",
        description="Read filtered records from a local SQLite table (table, where={}, columns=None, limit=20, alias='default').",
        tags=("database", "read"),
    )
    registry.register(
        db.query_readonly,
        name="db_query",
        description="Run a read-only SELECT query on a local SQLite database (sql, params=[], limit=20, alias='default').",
        tags=("database", "read"),
    )
    registry.register(
        db.insert_record,
        name="db_insert_record",
        description="Insert a record dict into a local SQLite table (table, record={}, alias='default').",
        tags=("database", "write"),
    )
    return registry
