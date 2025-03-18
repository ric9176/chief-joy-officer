from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list  # Store retrieved context
    user_memories: Optional[dict]  # Store user memory information
