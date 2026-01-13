from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from research_agent import research_app
from ingestor import index_document

app = FastAPI()

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    # Save file and trigger index_document
    return {"status": "success"}

@app.post("/research")
async def run_research(topic: str):
    # Run the LangGraph workflow
    inputs = {"topic": topic}
    result = await research_app.ainvoke(inputs)
    return result["final_report"]