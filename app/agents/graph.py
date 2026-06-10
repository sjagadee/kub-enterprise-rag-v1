import os
import logfire
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.grader import grader_node


def route_planner(state: AgentState):
    """
    Route decisions after the planner node.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"


def route_retry(state: AgentState):
    """
    Route decisions after the grader node.
    """
    if state["current_query"] == "RETRY":
        return "retriever"
    return "end"

# --- Graph definition ---
workflow = StateGraph(AgentState)  # type: ignore
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("grader", grader_node)


# --- Edges (flow control) ---
workflow.add_edge(START, "planner")
workflow.add_conditional_edges(
    "planner", 
    route_planner,
    {"retriever": "retriever", "responder": "responder"}
)
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", "grader")
workflow.add_conditional_edges(
    "grader",
    route_retry,
    {"retriever": "retriever", "end": END}
)

# InMemorySaver allows to save conversation in memory
checkpointer = InMemorySaver()

# Compile the graph with checkpointer
rag_agent = workflow.compile(checkpointer=checkpointer)
