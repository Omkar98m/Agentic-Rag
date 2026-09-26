from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from app.tools.pdf_tool import pdf_rag_tool
from app.tools.sql_tool import sql_tool
from app.tools.viz_tool import visualization_tool


@tool
def company_policy_tool(question: str) -> str:
    """Search the company policy PDF for policy-related questions."""
    return pdf_rag_tool(question)


@tool
def employee_sql_tool(query: str) -> str:
    """Execute a SQL query on the employee database."""
    return sql_tool(query)

@tool
def employee_visualization_tool(
    query: str,
    chart_type: str,
    x: str,
    y: str
):
    """Create a chart from employee SQL data."""
    return visualization_tool(query, chart_type, x, y)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

llm_with_tools = llm.bind_tools([
    company_policy_tool,
    employee_sql_tool,
    employee_visualization_tool
])


def run_agent(question: str):

    response = llm_with_tools.invoke(
        f"""
You are an intelligent company assistant.

Choose the appropriate tool based on the user's question.

Use company_policy_tool for questions about:
- company policies
- leave policy
- rules
- benefits
- working hours
- company procedures

Use employee_sql_tool for questions about:
- employees
- salaries
- departments
- projects
- performance
- employee leaves
- employee status
- employee data

Use employee_visualization_tool when the user asks for:
- charts
- graphs
- visualizations
- bar charts
- line charts
- scatter plots

For visualization questions:
1. Generate a SQL query first.
2. Always give calculated SQL columns an explicit short alias.
3. Use the EXACT SQL result column name for x and y.
4. Do not use natural-language descriptions as column names.

Example:

User:
"Show average salary by department as a bar chart"

Correct:
query =
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department

chart_type = "bar"
x = "department"
y = "avg_salary"

User question:
{question}
"""
    )

    if response.tool_calls:

        tool_call = response.tool_calls[0]

        if tool_call["name"] == "company_policy_tool":
            result = company_policy_tool.invoke(
                tool_call["args"]
            )

        elif tool_call["name"] == "employee_sql_tool":
            result = employee_sql_tool.invoke(
                tool_call["args"]
            )

        elif tool_call["name"] == "employee_visualization_tool":
            result = employee_visualization_tool.invoke(tool_call["args"])

        return result

    return response.content