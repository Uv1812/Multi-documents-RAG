import os
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
print("Embedding model ready.")
