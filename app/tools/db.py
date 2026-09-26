from sqlalchemy import create_engine
import pandas as pd


engine = create_engine("sqlite:///company.db")

df = pd.read_csv("data/employees.csv")

df.to_sql(
    "employees",
    engine,
    if_exists="replace",
    index=False
)

print("Employee data loaded successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")