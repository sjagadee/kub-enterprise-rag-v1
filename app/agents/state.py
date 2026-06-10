from typing import List, Annotated
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """
    Represents the state of the agent during query processing.
    """
    # using Annotated we are telling the state that we are adding to the messages list
    # instead of replacing the messages list
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
    retry_count: int


