# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent

from pydantic import BaseModel
from typing import Optional


# Initialize application
app = FastAPI(title="Enterprise RAG App")


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_user_id"


@app.get("/")
def home():
    return {"message": "Enterprise RAG App is Online. Use POST /query to chat."}


@app.get("/graph")
def get_graph_image():
    """
    Return the LangGraph workflow as a Mermaid graph image.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.error(f"Failed to generate graph image: {e}")
        return Response(content="Error generating graph image", status_code=500)


@app.post("/query")
def chat_endpoint(request: QueryRequest, response: Response):
    """
    Main API endpoint.
    Executes the LangGraph RAG pipeline.
    """
    with logfire.span("FastAPI Request Handler"):
        # Create a thread-safe session
        session_id = request.thread_id
        query = request.query

        # Setup initial state
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "current_query": query,
            "documents": [],
            "plan": ["Start"],
            "status": "Initializing Graph..."
        }
        
        # config for InMemorySaver
        config = {"configurable": {"thread_id": session_id}}

        try:
            # Run graph synchronusly for LogFire to trace all nodes
            final_output = rag_agent.invoke(initial_state, config=config)
            
            logfire.info("Graph execution completed successfully")
            
            return {
                "question": query,
                "answer": final_output.get("final_answer", "No answer found"),
                "though_process": final_output.get("plan"),
                "status": final_output.get("status"),
                "sources": final_output.get("documents", [])
            }
        except Exception as e:
            logfire.error(f"Failed to process query: {e}")
            response.status_code = 500
            return {
                "question": query,
                "answer": "Failed to process query, please try again later.",
                "though_process": ["Error encountered during graph execution"],
                "status": "error",
                "sources": []
            }
            

       