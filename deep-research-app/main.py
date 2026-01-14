from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from research_agent import research_app
from ingestor import ingest_pdf_locally as index_document
import shutil
import os

app = FastAPI()

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    # 1. Create a temporary path for the file
    temp_file_path = f"temp_{file.filename}"
    
    # 2. Save the uploaded content to that path
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 3. Trigger the ingestion into Qdrant
        index_document(temp_file_path)
        return {"status": "success", "message": f"File {file.filename} indexed."}
    finally:
        # 4. Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/research")
async def run_research(topic: str):
    inputs = {"topic": topic}
    # Since ResearchAgent isn't a Compiled Graph yet, call the method directly
    # Note: run_deep_research is currently a normal 'def', so don't 'await' it 
    # unless you change it to 'async def'
    result = research_app.run_deep_research(topic) 
    return {"report": result}