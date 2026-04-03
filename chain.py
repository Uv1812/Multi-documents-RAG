import uuid
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

vector_db_store = {}
chain_store = {}
history_store = {}  # we manage history manually


def get_groq_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant"
    )


def create_session_id() -> str:
    return str(uuid.uuid4())


def add_pdfs_to_vectorstore(session_id: str, pdf_vectors: FAISS):
    if session_id in vector_db_store:
        vector_db_store[session_id].merge_from(pdf_vectors)
        chain_store.pop(session_id, None)
    else:
        vector_db_store[session_id] = pdf_vectors


def get_chain(session_id: str):
    if session_id in chain_store:
        return chain_store[session_id]

    if session_id not in vector_db_store:
        raise KeyError(f"No vectorstore for session: {session_id}")

    retriever = vector_db_store[session_id].as_retriever(
        search_kwargs={"k": 4}
    )

    llm = get_groq_llm()

    # NO memory passed — we handle history ourselves
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )

    chain_store[session_id] = chain
    history_store[session_id] = []  # empty history for new session
    return chain


def ask_question(session_id: str, question: str) -> dict:
    chain = get_chain(session_id)

    # Get this session's history
    history = history_store.get(session_id, [])

    # Run chain with history passed explicitly
    result = chain.invoke({
        "question": question,
        "chat_history": history
    })

    # Manually append to history as (human, ai) tuples
    history.append((question, result["answer"]))
    history_store[session_id] = history

    # Extract sources
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        label = f"{src} (page {page})"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return {"answer": result["answer"], "sources": sources}


def delete_session(session_id: str):
    vector_db_store.pop(session_id, None)
    chain_store.pop(session_id, None)
    history_store.pop(session_id, None)