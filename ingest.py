import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddings import embedder

DATA_DIR = "data"
VECTOR_DIR = "vectorstore/"


def process_pdfs(pdf_files: list) -> FAISS:
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = filename

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    print(f"  Total chunks created: {len(all_chunks)}")

    vectorstore = FAISS.from_documents(all_chunks, embedder)
    return vectorstore


def ingest_pdfs():
    print("Reading PDFs from:", DATA_DIR)
    all_chunks = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, file)
            print(f"\n Loading: {file}")

            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            for p in pages:
                p.metadata["source"] = file

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    print(f"\n Total Chunks: {len(all_chunks)}")
    print("\n Embedding and saving to FAISS...\n")

    vector_store = FAISS.from_documents(all_chunks, embedding=embedder)
    vector_store.save_local(VECTOR_DIR)

    print("Ingestion complete!")
    print("Vectorstore saved at:", VECTOR_DIR)


if __name__ == "__main__":
    ingest_pdfs()
