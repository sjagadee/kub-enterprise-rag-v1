import os
import logfire
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


def route_planner(state: AgentState):
    """
    Route decisions after the planner node.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"


# --- Graph definition ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)


# --- Edges (flow control) ---
workflow.add_edge(START, "planner")
workflow.add_conditional_edges(
    "planner", 
    route_planner,
    {"retriever": "retriever", "responder": "responder"}
)
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

# InMemorySaver allows to save conversation in memory
checkpointer = InMemorySaver()

# Compile the graph with checkpointer
rag_agent = workflow.compile(checkpointer=checkpointer)
