import uuid
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from agent import create_agent_graph, get_checkpointer
from agent.utils.state import AgentState
import os
import json

SHORT_TERM_MEMORY_DB_PATH = "data/short_term.db"

os.makedirs(os.path.dirname(SHORT_TERM_MEMORY_DB_PATH), exist_ok=True)

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    try:
        # Generate and store a session ID
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        
        # Initialize empty message history
        cl.user_session.set("message_history", [])
        
        welcome_message = cl.Message(
            content="Hello! I'm your chief joy officer, here to help you with finding fun things to do in London!",
            author="Assistant"
        )
        await welcome_message.send()
        
    except Exception as e:
        print(f"Error in chat initialization: {str(e)}")
        error_message = cl.Message(
            content="I apologize, but I encountered an error during initialization. Please try refreshing the page.",
            author="System"
        )
        await error_message.send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and stream responses"""
    # Get or create session ID
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
    
    # Initialize response message
    msg = cl.Message(content="")
    
    try:
        async with cl.Step(type="run"):
            async with get_checkpointer(SHORT_TERM_MEMORY_DB_PATH) as saver:
                # Create graph with memory
                graph = await create_agent_graph(saver)
                
                # Get message history and add current message
                message_history = cl.user_session.get("message_history", [])
                current_message = HumanMessage(content=message.content)
                message_history.append(current_message)
                
                # Create current state
                current_state = AgentState(
                    messages=message_history,
                    context=cl.user_session.get("last_context", [])
                )
                
                # Stream the response
                async for chunk in graph.astream(
                    current_state,
                    config={"configurable": {"thread_id": session_id}},
                    stream_mode="messages"
                ):
                    # Handle different node outputs
                    if isinstance(chunk[0], AIMessageChunk):
                        await msg.stream_token(chunk[0].content)
                    elif isinstance(chunk[0], AIMessage):
                        if chunk[0] not in message_history:
                            message_history.append(chunk[0])
                
                # Get final state
                final_state = await graph.aget_state(
                    config={"configurable": {"thread_id": session_id}}
                )
                
                # Update session state
                if final_state:
                    cl.user_session.set("message_history", message_history)
                    cl.user_session.set("last_context", final_state.values.get("context", []))
        
        # Send the final message
        await msg.send()
                
    except Exception as e:
        print(f"Error in message handler: {str(e)}")
        await cl.Message(
            content="I apologize, but I encountered an error processing your message. Please try again.",
            author="System"
        ).send()