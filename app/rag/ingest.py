from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


def ingest_documents():
    loader = PyPDFLoader("document\\company_policies_RAG.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

def retrieve(vectorstore, question):
    return vectorstore.similarity_search(
        question,
        k=3
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

def generate_answer(question, documents, llm):

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content