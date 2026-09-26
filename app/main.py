from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agents.agent import run_agent

app = FastAPI()

app.mount(
    "/charts",
    StaticFiles(directory="charts"),
    name="charts"
)


@app.get("/")
def home():
    return {"message": "Agentic RAG is running"}


@app.get("/ask")
def ask(question: str):
    result = run_agent(question)

    return {
        "question": question,
        "answer": result
    }