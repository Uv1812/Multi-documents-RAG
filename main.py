import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# This triggers embedding model load at startup — must be before chain/ingest
import embeddings

from chain import create_session_id, add_pdfs_to_vectorstore, delete_session, vector_db_store, ask_question
from ingest import process_pdfs

app = FastAPI(title="Lumina RAG API")
# Add this health check endpoint — Render pings this to confirm service is up
@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok"})

# Fix the HEAD request on /
@app.head("/")
def head_root():
    return JSONResponse({})
    
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
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename} is not a PDF"
                )
            path = f"temp_{session_id}_{file.filename}"
            with open(path, "wb") as f:
                f.write(await file.read())
            pdf_paths.append(path)

        vector_db = process_pdfs(pdf_paths)
        add_pdfs_to_vectorstore(session_id, vector_db)

        filenames = [f.filename for f in files]
        return {
            "status": "success",
            "message": f"{len(files)} PDF(s) uploaded and processed.",
            "files": filenames
        }

    finally:
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post("/chat")
async def chat_api(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    if request.session_id not in vector_db_store:
        return {"answer": "Please upload PDFs first before asking questions."}

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    result = ask_question(request.session_id, request.user_message)
    return result


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    delete_session(session_id)
    return {"status": "Session deleted", "session_id": session_id}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
