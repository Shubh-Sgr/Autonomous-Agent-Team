import os
import warnings
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore", message="The default value of `allowed_objects` will change")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Initializing model (Falls back to Llama 3.3 if OSS model preview is unassigned)
try:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
except Exception:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

class AgentState(TypedDict):
    task: str
    report: str
    feedback: str
    iterations: int

def researcher_node(state: AgentState):
    prompt = f"Task: {state['task']}\nFeedback: {state.get('feedback', '')}\nWrite a detailed report:"
    response = llm.invoke(prompt)
    return {"report": response.content, "iterations": state.get("iterations", 0) + 1}

def reviewer_node(state: AgentState):
    prompt = f"Review this: {state['report']}\nIf perfect, say 'APPROVED'. Else, list 3 fixes."
    response = llm.invoke(prompt)
    return {"feedback": response.content}

def should_continue(state: AgentState):
    if "APPROVED" in state["feedback"].upper() or state.get("iterations", 0) >= 3:
        return END
    return "researcher"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)

agent_app = workflow.compile()
