from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# 1. Define the shared state
class ResearchState(TypedDict):
    topic: str
    sub_questions: List[str]
    context_data: List[str]
    final_report: str

llm = ChatOpenAI(model="gpt-4o")

# 2. Node: Topic Decomposer
def decompose_topic(state: ResearchState):
    prompt = f"Decompose this research topic into 5 specific sub-questions: {state['topic']}"
    response = llm.invoke(prompt)
    # logic to parse response into list
    return {"sub_questions": ["q1", "q2", "q3"]} 

# 3. Node: Gather Information (Retrieval)
def gather_info(state: ResearchState):
    # Here you would loop through sub_questions and query Qdrant
    # For now, we simulate finding data
    return {"context_data": ["Source 1: Findings...", "Source 2: Data..."]}

# 4. Node: Synthesize & Report
def generate_report(state: ResearchState):
    prompt = f"Synthesize these findings: {state['context_data']} into a report for: {state['topic']}"
    report = llm.invoke(prompt)
    return {"final_report": report.content}

# 5. Build the Graph
workflow = StateGraph(ResearchState)
workflow.add_node("decomposer", decompose_topic)
workflow.add_node("gatherer", gather_info)
workflow.add_node("synthesizer", generate_report)

workflow.set_entry_point("decomposer")
workflow.add_edge("decomposer", "gatherer")
workflow.add_edge("gatherer", "synthesizer")
workflow.add_edge("synthesizer", END)

research_app = workflow.compile()