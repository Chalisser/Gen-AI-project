import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from ingestor import ingest_pdf_locally  # Import your local ingestor
from research_agent import ResearchAgent # Import the agent class

app = FastAPI(title="Deep Research AI")

# Initialize the agent once at startup
agent = ResearchAgent()

# Create a permanent directory for processed files if needed
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    """
    Handles PDF uploads and triggers local indexing.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Define the local path
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 2. Save the uploaded content to that path
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Trigger the ingestion into Qdrant
        ingest_pdf_locally(file_path)
        return {"status": "success", "message": f"File {file.filename} is now ready for research."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/research")
async def run_research(topic: str):
    """
    Runs the multi-step research pipeline: Decompose -> Gather -> Synthesize.
    """
    try:
        # We call the method on the 'agent' instance we created at the top
        # Since this is a CPU-heavy task, FastAPI handles it as a standard sync call
        result = agent.run_deep_research(topic) 
        return {"topic": topic, "report": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)