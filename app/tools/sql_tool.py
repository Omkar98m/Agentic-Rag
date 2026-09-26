from sqlalchemy import text
from app.tools.db import engine


def execute_query(query: str):
    """Execute SQL query and return structured data."""

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))

            rows = result.fetchall()

            return [dict(row._mapping) for row in rows]

    except Exception as e:
        return {"error": str(e)}


def sql_tool(query: str) -> str:
    """Execute SQL query and return readable result."""

    result = execute_query(query)

    if isinstance(result, dict) and "error" in result:
        return f"SQL error: {result['error']}"

    if not result:
        return "No results found."

    return str(result)