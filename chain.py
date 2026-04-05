import uuid
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import GROQ_API_KEY

vector_db_store = {}
history_store = {}


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
    else:
        vector_db_store[session_id] = pdf_vectors


def ask_question(session_id: str, question: str) -> dict:
    if session_id not in vector_db_store:
        raise KeyError(f"No vectorstore for session: {session_id}")

    retriever = vector_db_store[session_id].as_retriever(
        search_kwargs={"k": 4}
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    history = history_store.get(session_id, [])
    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions \
based on the provided document context. If the answer is not in the \
context, say 'I don't have information about that in the uploaded documents.'

Context from documents:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    llm = get_groq_llm()
    chain = prompt | llm

    result = chain.invoke({
        "context": context,
        "history": messages,
        "question": question
    })

    answer = result.content

    history.append((question, answer))
    history_store[session_id] = history

    return {"answer": answer}


def delete_session(session_id: str):
    vector_db_store.pop(session_id, None)
    history_store.pop(session_id, None)
