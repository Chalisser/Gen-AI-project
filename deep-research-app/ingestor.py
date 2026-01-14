import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings # New local library
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

COLLECTION_NAME = "local_research"

def ingest_pdf_locally(file_path):
    loader = PyPDFLoader(file_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Use .from_documents WITHOUT creating a separate QdrantClient first
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path="./qdrant_db", # LangChain will open, use, and CLOSE the lock properly
        collection_name=COLLECTION_NAME,
    )
    print(f"Success! {len(chunks)} chunks stored locally.")

if __name__ == "__main__":
    ingest_pdf_locally("DistilBERT.pdf")