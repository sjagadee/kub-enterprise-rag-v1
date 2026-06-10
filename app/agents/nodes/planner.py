from langchain.chat_models import init_chat_model
from app.agents.state import AgentState
from app.config import settings
import logfire


# initialize llm model
llm = init_chat_model(
    model=settings.GROQ_MODEL,
    model_provider=settings.GROQ,
    api_key=settings.GROQ_API_KEY,
    temperature=0
)


def planner_node(state: AgentState):
    """
     The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get conversion history
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    NEW USER MESSAGE: {user_message}

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
     
    with logfire.span("Planner Decision"):
        response = llm.invoke(prompt)
        decision = response.content.strip() # type: ignore
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }