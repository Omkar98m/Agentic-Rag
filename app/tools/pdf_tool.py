from app.rag.rag import retrieve, generate_answer


def pdf_rag_tool(question: str) -> str:
    """
    Search the company policy PDF and answer the question.
    """

    documents = retrieve(question)

    if not documents:
        return "I don't know based on the provided company policy."

    return generate_answer(question, documents)