from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
import aiosqlite
from types import TracebackType
from typing import Optional, Type

from agent.utils.state import AgentState
from agent.utils.nodes import (
    call_model,
    tool_node,
    read_memory,
    write_memory,
    should_continue
)

def create_graph_builder():
    """Create a base graph builder with nodes and edges configured."""
    builder = StateGraph(AgentState)
    

    # Add nodes
    builder.add_node("agent", call_model)
    builder.add_node("action", tool_node)
    builder.add_node("read_memory", read_memory)
    builder.add_node("write_memory", write_memory)

    # Set entry point
    builder.set_entry_point("agent")

    builder.add_edge("agent", "write_memory")
    builder.add_edge("write_memory", END)

    # Add conditional edges from agent
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "read_memory": "read_memory",
            "write_memory": "write_memory",
            END: END
        }
    )
    
    # Connect action back to agent
    builder.add_edge("action", "agent")
    
    # Memory operations should end after completion
    builder.add_edge("read_memory", "agent")
    
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

# Initialize store for across-thread memory
across_thread_memory = InMemoryStore()

async def create_agent_graph(checkpointer: AsyncSqliteSaver):
    """Create an agent graph with memory persistence."""
    builder = create_graph_builder()
    # Compile with both SQLite checkpointer for within-thread memory
    # and InMemoryStore for across-thread memory
    graph = builder.compile(
        checkpointer=checkpointer,
        store=across_thread_memory
    )
    return graph

langgraph_studio_graph = create_agent_graph_without_memory()

# Export the graph builder functions
__all__ = ["create_agent_graph", "create_agent_graph_without_memory", "get_checkpointer"] 