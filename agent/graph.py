from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from types import TracebackType
from typing import Optional, Type

from agent.utils.state import AgentState
from agent.utils.nodes import call_model, tool_node, should_continue

def create_graph_builder():
    """Create a base graph builder with nodes and edges configured."""
    # Create the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", call_model)
    builder.add_node("action", tool_node)

    # Update edges
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
    )
    builder.add_edge("action", "agent")
    
    return builder

def create_agent_graph_without_memory():
    """Create an agent graph without memory persistence."""
    builder = create_graph_builder()
    return builder.compile()

class SQLiteCheckpointer:
    """Context manager for SQLite checkpointing."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.saver: Optional[AsyncSqliteSaver] = None
    
    async def __aenter__(self) -> AsyncSqliteSaver:
        """Initialize and return the AsyncSqliteSaver."""
        conn = await aiosqlite.connect(self.db_path)
        self.saver = AsyncSqliteSaver(conn)
        return self.saver
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Clean up the SQLite connection."""
        if self.saver and hasattr(self.saver, 'conn'):
            await self.saver.conn.close()
            self.saver = None

def get_checkpointer(db_path: str = "data/short_term.db") -> SQLiteCheckpointer:
    """Create and return a SQLiteCheckpointer instance."""
    return SQLiteCheckpointer(db_path)

async def create_agent_graph(checkpointer: AsyncSqliteSaver):
    """Create an agent graph with SQLite-based memory persistence."""
    builder = create_graph_builder()
    graph = builder.compile(checkpointer=checkpointer)
    return graph

# Export the graph builder functions
__all__ = ["create_agent_graph", "create_agent_graph_without_memory", "get_checkpointer"] 