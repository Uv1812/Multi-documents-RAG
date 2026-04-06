import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import hugging_token

os.environ["HF_TOKEN"] = hugging_token

embedder = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hugging_token
)

print("Embedder configured — using HuggingFace Inference API.")
