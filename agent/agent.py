from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agent.utils.state import AgentState
from agent.utils.nodes import call_model, tool_node, should_continue

def create_agent_graph():
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

    # Initialize memory saver for conversation persistence
    memory = MemorySaver()

    # Compile the graph with memory
    return builder.compile(checkpointer=memory)

def create_agent_graph_without_memory():
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

    # Compile the graph without memory
    return builder.compile()

# Create both graph variants
graph_with_memory = create_agent_graph()
graph = create_agent_graph_without_memory() 