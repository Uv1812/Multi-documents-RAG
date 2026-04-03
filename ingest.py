import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import hugging_token

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_token
os.environ["HF_TOKEN"] = hugging_token

DATA_DIR = "data"
VECTOR_DIR = "vectorstore/"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


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

            print("  Sample Metadata:")
            for c in chunks[:2]:
                print("   ", c.metadata)

            all_chunks.extend(chunks)

    print(f"\n Total Chunks: {len(all_chunks)}")
    print("\n Embedding and saving to FAISS...\n")

    embed = get_embeddings()
    vector_store = FAISS.from_documents(all_chunks, embedding=embed)
    vector_store.save_local(VECTOR_DIR)

    print("✅ Ingestion complete!")
    print("📌 Vectorstore saved at:", VECTOR_DIR)


def process_pdfs(pdf_files: list[str]) -> FAISS:
    """
    Takes a list of PDF file paths.
    Loads, tags metadata, splits, embeds, and returns a FAISS vectorstore.
    """
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

        # Tag every page with source filename and page number
        for page in pages:
            page.metadata["source"] = filename

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    print(f"  Total chunks created: {len(all_chunks)}")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore


if __name__ == "__main__":
    ingest_pdfs()