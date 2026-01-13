import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
COLLECTION_NAME = "research_documents"

def ingest_pdf(file_path):
    # 1. LOAD: Extract text from the PDF
    print(f"--- Loading {file_path} ---")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. SPLIT: Break long text into smaller 'chunks'
    # Why? LLMs can't read 100 pages at once. 
    # Overlap ensures no context is lost at the cut points.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. INDEX: Store in Qdrant Vector Database
    # We use 'local' mode here (path="./qdrant_db"), which saves the data on your computer.
    client = QdrantClient(path="./qdrant_db")
    
    # Create the collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=OpenAIEmbeddings(),
    )

    vector_store.add_documents(chunks)
    print("--- Ingestion Complete! ---")

# Execute
if __name__ == "__main__":
    ingest_pdf("your_research_paper.pdf")