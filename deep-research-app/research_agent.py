import os
from typing import List
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
COLLECTION_NAME = "local_research"
MODEL_NAME = "llama3.2"  # Ensure you ran 'ollama pull llama3.2'
EMBED_MODEL = "mxbai-embed-large" # Ensure you ran 'ollama pull mxbai-embed-large'

class ResearchAgent:
    def __init__(self):
        # 1. Initialize the Local "Brain"
        self.llm = OllamaLLM(model=MODEL_NAME, temperature=0.2)
        
        # 2. Connect to the Local "Library" (Qdrant)
        self.client = QdrantClient(path="./qdrant_db")
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=COLLECTION_NAME, 
            embedding=self.embeddings)

    def decompose_topic(self, topic: str) -> List[str]:
        """Phase 1: Break a broad topic into 3-5 specific sub-questions."""
        print(f"--- Decomposing Topic: {topic} ---")
        prompt = f"""
        You are a lead researcher. Break down the following topic into 3 specific, 
        distinct sub-questions to investigate within the provided documents.
        Topic: {topic}
        
        Return ONLY the questions, one per line. Do not include numbers or intro text.
        """
        response = self.llm.invoke(prompt)
        # Split by newline and clean up whitespace
        questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return questions[:3] # Limit to top 3 for speed

    def gather_evidence(self, sub_questions: List[str]) -> str:
        """Phase 2: Retrieve relevant text for each sub-question."""
        all_context = ""
        for query in sub_questions:
            print(f"--- Researching: {query} ---")
            # Search Qdrant for the top 3 most relevant chunks per question
            docs = self.vector_store.similarity_search(query, k=3)
            all_context += f"\n\nResults for '{query}':\n" 
            all_context += "\n".join([d.page_content for d in docs])
        return all_context

    def write_report(self, topic: str, context: str) -> str:
        """Phase 3: Synthesize all gathered info into a structured report."""
        print("--- Writing Final Report ---")
        prompt = f"""
        You are a senior analyst. Using ONLY the following research notes, 
        write a professional structured report about '{topic}'.
        
        Structure:
        1. Executive Summary
        2. Key Findings (detailed)
        3. Conclusion
        
        Research Notes:
        {context}
        
        If the notes do not contain information about a specific part of the topic, 
        state that the information was not found in the documents.
        """
        return self.llm.invoke(prompt)

    def run_deep_research(self, topic: str):
        """The main workflow that orchestrates the phases."""
        # Step 1: Planning
        sub_questions = self.decompose_topic(topic)
        
        # Step 2: Investigation
        context = self.gather_evidence(sub_questions)
        
        # Step 3: Synthesis
        report = self.write_report(topic, context)
        
        return report
    
agent = ResearchAgent()

# 2. Define the variable that main.py is looking for
# We use the run_deep_research method as our entry point
research_app = agent

# For testing the file independently
if __name__ == "__main__":
    agent = ResearchAgent()
    print(agent.run_deep_research("What is the main financial outlook mentioned?"))