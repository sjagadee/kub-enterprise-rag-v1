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


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History. 
    """
    query = state["current_query"]

    # Get conversion history
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        # Conversational prompt creation
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history}

        LATEST MESSAGE:
        "{user_message}"
        """
    else:
        # Technical RAG prompt creation
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history}

        USER QUESTION:
        "{user_message}"
        """

    with logfire.span("LLM Synthesis"):
        try:
            response = llm.invoke(prompt)
            
            logfire.info(f"Response sythesized successfully")
            return {
                "final_answer": response.content,
                "status": "Response Generated",
                "messages": [
                    {
                        "role": "assistant",
                        "content": response.content
                    }
                ]
            }
        except Exception as e:
            logfire.error(f"Failed to generate response: {e}")
            raise e