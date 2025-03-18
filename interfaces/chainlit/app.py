import uuid
import os
import json
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent import create_agent_graph, get_checkpointer
from agent.utils.state import AgentState

SHORT_TERM_MEMORY_DB_PATH = "data/short_term.db"

os.makedirs(os.path.dirname(SHORT_TERM_MEMORY_DB_PATH), exist_ok=True)

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    try:
        # Generate and store a session ID
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        
        welcome_message = cl.Message(
            content="Hello! I'm your chief joy officer, here to help you with finding fun things to do in London!",
            author="Assistant"
        )
        await welcome_message.send()
        
    except Exception as e:
        print(f"Error in chat initialization: {str(e)}")
        await cl.Message(
            content="I apologize, but I encountered an error during initialization. Please try refreshing the page.",
            author="System"
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and stream responses"""
    # Initialize response message
    msg = cl.Message(content="")
    
    try:
        async with cl.Step(type="run"):
            async with get_checkpointer(SHORT_TERM_MEMORY_DB_PATH) as saver:
                # Create graph with memory
                graph = await create_agent_graph(saver)
                
                # Get session ID
                session_id = cl.user_session.get("session_id")
                
                # Process through graph with current message
                async for chunk in graph.astream(
                    {"messages": [HumanMessage(content=message.content)]},
                    {"configurable": {"thread_id": session_id}},
                    stream_mode="messages"
                ):
                    if chunk[1]["langgraph_node"] == "agent" and isinstance(
                        chunk[0], (AIMessageChunk, AIMessage)
                    ):
                        await msg.stream_token(chunk[0].content)
                
                # Get final state
                final_state = await graph.aget_state(
                    config={"configurable": {"thread_id": session_id}}
                )
        
        # Send the final message
        await msg.send()
                
    except Exception as e:
        print(f"Error in message handler: {str(e)}")
        await cl.Message(
            content="I apologize, but I encountered an error processing your message. Please try again.",
            author="System"
        ).send()