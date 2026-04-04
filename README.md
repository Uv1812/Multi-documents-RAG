📘 Multi-PDF RAG Chatbot

A simple and fast RAG (Retrieval-Augmented Generation) chatbot that allows users to upload multiple PDF files, processes them into a vector store, and answers questions with conversation memory.

Designed with a clean dark-mode UI, smooth chat experience, and accurate retrieval using FAISS.

![Alt text](https://github.com/user-attachments/assets/10be084d-850d-447a-9468-ce326aa7331f)

✨ What This Project Does
Upload multiple PDFs at once
Extract text → split → embed → store in FAISS
Ask questions and get accurate, contextual answers
Follow-up queries work using session-based memory
Shows full chat history
Lightweight & production-ready

Perfect for study notes, research papers, manuals, or multi-document Q&A systems.

🛠 Tech Stack

| Component             | Technology                             |
| --------------------- | -------------------------------------- |
| **Backend Framework** | FastAPI                                |
| **LLM**               | Groq (Llama 3.1 8B Instant)            |
| **Embeddings**        | HuggingFace MiniLM-L6-v2               |
| **Vector Store**      | FAISS (Local Storage)                  |
| **RAG Framework**     | LangChain                              |
| **PDF Processing**    | PyPDFLoader + TextSplitter             |
| **Frontend**          | HTML + CSS + JavaScript (Dark Mode UI) |
| **Memory**            | Session-based ConversationBufferMemory |
| **Extras**            | CORS, dotenv, Pydantic                 |
