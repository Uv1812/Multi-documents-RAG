import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chain import get_chain, create_session_id, add_pdfs_to_vectorstore, delete_session, vector_db_store, ask_question
from ingest import process_pdfs

app = FastAPI(title="Multi-Doc RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.post("/create_session")
def create_new_session():
    session_id = create_session_id()
    return {"session_id": session_id}


@app.post("/upload_pdfs")
async def upload_pdfs(session_id: str, files: list[UploadFile] = File(...)):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    pdf_paths = []

    try:
        # Save uploaded files temporarily
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

            path = f"temp_{session_id}_{file.filename}"
            with open(path, "wb") as f:
                f.write(await file.read())
            pdf_paths.append(path)

        # Process and store vectorstore
        vector_db = process_pdfs(pdf_paths)
        add_pdfs_to_vectorstore(session_id, vector_db)

        filenames = [f.filename for f in files]
        return {
            "status": "success",
            "message": f"{len(files)} PDF(s) uploaded and processed.",
            "files": filenames
        }

    finally:
        # Always clean up temp files
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post("/chat")
async def chat_api(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    if request.session_id not in vector_db_store:
        return {"answer": "Please upload PDFs first.", "sources": []}

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    result = ask_question(request.session_id, request.user_message)
    return result


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    delete_session(session_id)
    return {"status": "Session deleted", "session_id": session_id}