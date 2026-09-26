import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from app.tools.sql_tool import execute_query


CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


def visualization_tool(
    query: str,
    chart_type: str,
    x: str,
    y: str
):
    """
    Execute SQL query and create a visualization.

    chart_type:
    - bar
    - line
    - scatter
    """

    data = execute_query(query)

    if isinstance(data, dict) and "error" in data:
        return f"SQL error: {data['error']}"

    if not data:
        return "No data available for visualization."

    df = pd.DataFrame(data)

    if x not in df.columns:
        return f"Column '{x}' not found."

    if y not in df.columns:
        return f"Column '{y}' not found. Available columns: {list(df.columns)}"

    plt.figure(figsize=(10, 6))

    if chart_type == "bar":
        plt.bar(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)

    elif chart_type == "line":
        plt.plot(df[x], df[y], marker="o")
        plt.xlabel(x)
        plt.ylabel(y)

    elif chart_type == "scatter":
        plt.scatter(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)

    else:
        return "Unsupported chart type."

    plt.title(f"{y} by {x}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(CHART_DIR, "chart.png")
    plt.savefig(file_path)
    plt.close()

    return {
        
    "message": "Chart created successfully",
    "chart_url": "/charts/chart.png",
    "data": data
}
    