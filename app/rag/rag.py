import os
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


CHROMA_PATH = "./chroma_db"


# -----------------------------
# 1. Load existing vector DB
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


# -----------------------------
# 2. Initialize Gemini
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# -----------------------------
# 3. Retrieve relevant chunks
# -----------------------------

def retrieve(question):

    documents = vectorstore.similarity_search(
        question,
        k=3
    )

    return documents


# -----------------------------
# 4. Generate answer
# -----------------------------

def generate_answer(question, documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not present in the context, say:

"I don't know based on the provided company policy."

Do not make up information.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# -----------------------------
# 5. Chat loop
# -----------------------------

def main():

    print("\n==============================")
    print("   TechNova Policy Assistant")
    print("==============================")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve
        documents = retrieve(question)

        # Generate
        answer = generate_answer(
            question,
            documents
        )

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()