import os
from typing import Optional
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class SQLDatabaseTool(BaseTool):
    name = "sql_database"
    description = (
        "Interacts with SQL databases (SQLite, PostgreSQL, MySQL, etc.). "
        "Inputs: 'action' (str, one of: 'get_schema', 'execute_query'), "
        "'query' (str, the SQL query to execute, required for execute_query). "
        "Database URI is picked up from SQL_CONNECTION_URI environment variable if not provided."
    )

    def __init__(self, connection_uri: Optional[str] = None, read_only: bool = True):
        """
        :param connection_uri: Optional SQLAlchemy connection string.
        :param read_only: If True, blocks UPDATE, INSERT, DELETE, DROP, etc.
        """
        self.connection_uri = connection_uri
        self.read_only = read_only

    async def execute(
        self, 
        action: str, 
        query: Optional[str] = None, 
        connection_uri: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        uri = connection_uri or self.connection_uri or os.environ.get("SQL_CONNECTION_URI")
        if not uri:
            return ToolResult(
                success=False, 
                output=None, 
                error="SQL_CONNECTION_URI must be provided via arguments or environment variables."
            )

        try:
            from sqlalchemy import create_engine, text, MetaData
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="SQLAlchemy is not installed. Please install it using: pip install sqlalchemy"
            )

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
                return ToolResult(success=True, output=output if output else "Database has no tables.")

            elif action == "execute_query":
                if not query:
                    return ToolResult(success=False, output=None, error="Missing 'query' for execute_query.")
                
                if self.read_only:
                    q_lower = query.lower().strip()
                    # A basic safety net. To be truly safe, a read-only DB user should be used.
                    if not (q_lower.startswith("select") or q_lower.startswith("pragma") or q_lower.startswith("explain")):
                        return ToolResult(
                            success=False, 
                            output=None, 
                            error="Tool is configured as read-only. Only SELECT queries are allowed."
                        )
                
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    
                    # For non-returning queries
                    if not result.returns_rows:
                        conn.commit()
                        return ToolResult(success=True, output="Query executed successfully. No results returned.")
                        
                    rows = result.fetchall()
                    if not rows:
                        return ToolResult(success=True, output="Query executed successfully. No rows found.")
                    
                    columns = list(result.keys())
                    header = " | ".join(str(c) for c in columns)
                    row_strs = [" | ".join(str(val) for val in row) for row in rows]
                    
                    output = header + "\n" + "-" * len(header) + "\n" + "\n".join(row_strs)
                    return ToolResult(success=True, output=output[:10000]) # Cap output length
                    
            else:
                return ToolResult(success=False, output=None, error=f"Unknown action: {action}")

        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Database Error: {str(e)}")
