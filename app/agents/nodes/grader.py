from langchain.chat_models import init_chat_model
from app.agents.state import AgentState
from app.config import settings
import logfire

llm = init_chat_model(
    model=settings.GROQ_FALLBACK_MODEL,
    model_provider=settings.GROQ,
    api_key=settings.GROQ_API_KEY,
    temperature=0
)

def grader_node(state: AgentState):
    """
    This node grades the response from the responder node against the original query
    and returns the final response if the response is good, otherwise it hands over the
    control back to retriever node for retreiving more relevant information. 

    Note: Only if the planner had decided if the query is not conversational
    then only this node gets called.
    """
    query = state["current_query"]
    answer = state["final_answer"]

    prompt = f"""
    You are an intelligent Assistant Grader. 
    Analyze the response and the original query and check if the response is relevant to the query.
    The Grades needs to be in between 1-10, where 10 is the most relevant and 1 is the least relevant. 
    If the grade is less than 5 then it is not relevant.
    

    ORIGINAL QUERY:
    {query}

    RESPONSE:
    {answer}

    Task:
    1. If the response is relevant to the query, the grade needs to be between 5-10. You can decide how relevant the response is to the query.  
    2. If the response is not relevant to the query, the grade needs to be between 1-4.
    
    Respond with ONLY the grade as a number.
    """

    with logfire.span("Grader Decision"):
        response = llm.invoke(prompt)
        decision = int(response.content.strip()) # type: ignore
        logfire.info(f"Grader's decision: {decision}")

    if decision >= 5:
        return {
            "final_answer": answer,
            "status": "Response Generated and Graded as Relevant to the query",
            "messages": [
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
    elif decision < 5 and state["retry_count"] < 3:
        return {
            "current_query": "RETRY",
            "status": f"Retrying - Assistant could not find relevant information, Grade: {decision}",
            "retry_count": state["retry_count"] + 1,
            "documents": []  # Reset documents for fresh search
        }
    else:
        return {
            "final_answer": "Apologies! I'm having trouble finding the exact information you need. Would you like me to try a broader search or rephrase your question?",
            "status": "Conversation Ended - Unable to find relevant information",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Apologies! I'm having trouble finding the exact information you need. Would you like me to try a broader search or rephrase your question?"
                }
            ]
        }
    
        
    