import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from agent import create_agent_graph, create_agent_graph_without_memory, get_checkpointer
from agent.utils.state import AgentState
import os
import json

# Path to SQLite database for short-term memory
SHORT_TERM_MEMORY_DB_PATH = "data/short_term.db"

# Ensure the data directory exists
os.makedirs(os.path.dirname(SHORT_TERM_MEMORY_DB_PATH), exist_ok=True)

@cl.on_chat_start
async def on_chat_start():
    # Generate and store a session ID
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    
    # Initialize empty message history
    cl.user_session.set("message_history", [])
    
    # Initialize config using stored session ID
    config = RunnableConfig(
        configurable={
            "thread_id": session_id,
            "session_id": session_id,
            "checkpoint_ns": session_id
        }
    )
    
    # Initialize empty state with auth
    try:
        async with get_checkpointer(SHORT_TERM_MEMORY_DB_PATH) as saver:
            graph = await create_agent_graph(saver)
            initial_state = AgentState(
                messages=[], 
                context=[]
            )
            
            await graph.ainvoke(initial_state, config=config)
            
            # Store initial state
            cl.user_session.set("last_state", {
                "messages": [], 
                "context": []
            })
    except Exception as e:
        print(f"Error initializing state: {str(e)}")
    
    await cl.Message(
        content="Hello! I'm your chief joy officer, here to help you with finding fun things to do in London!",
        author="Assistant"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Get or create session ID
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
    
    print(f"Session ID: {session_id}")
    
    # Get message history
    message_history = cl.user_session.get("message_history", [])
    
    # Add new message to history
    current_message = HumanMessage(content=message.content)
    message_history.append(current_message)
    cl.user_session.set("message_history", message_history)
    
    config = RunnableConfig(
        configurable={
            "thread_id": session_id,
            "session_id": session_id,
            "checkpoint_ns": session_id
        }
    )
    
    try:
        async with get_checkpointer(SHORT_TERM_MEMORY_DB_PATH) as saver:
            # Create graph with memory
            graph = await create_agent_graph(saver)
            
            # Get the last state or create new one
            last_state_dict = cl.user_session.get("last_state", {"messages": [], "context": []})
            
            # Create new state with current message history
            current_state = AgentState(
                messages=message_history,
                context=last_state_dict.get("context", [])
            )
            
            # Setup callback handler and final answer message
            cb = cl.LangchainCallbackHandler()
            final_answer = cl.Message(content="")
            await final_answer.send()
            
            loading_msg = None  # Initialize reference to loading message
            last_state = None  # Track the final state
            
            # Stream the response
            async for chunk in graph.astream(
                current_state,
                config=config
            ):
                for node, values in chunk.items():
                    if node == "retrieve":
                        if loading_msg:
                            await loading_msg.remove()
                        loading_msg = cl.Message(content="🔍 Searching knowledge base...", author="System")
                        await loading_msg.send()
                    elif values.get("messages"):
                        last_message = values["messages"][-1]
                        # Check for tool calls in additional_kwargs
                        if hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("tool_calls"):
                            tool_name = last_message.additional_kwargs["tool_calls"][0]["function"]["name"]
                            if loading_msg:
                                await loading_msg.remove()
                            loading_msg = cl.Message(
                                content=f"🔍 Using {tool_name}...",
                                author="Tool"
                            )
                            await loading_msg.send()
                        # Only stream AI messages, skip tool outputs
                        elif isinstance(last_message, AIMessage):
                            if loading_msg:
                                await loading_msg.remove()
                                loading_msg = None
                            await final_answer.stream_token(last_message.content)
                            # Add AI message to history
                            message_history.append(last_message)
                            cl.user_session.set("message_history", message_history)
                        # Update last state
                        last_state = values
            
            # Update the last state as a serializable dict
            if last_state:
                cl.user_session.set("last_state", {
                    "messages": [msg.content for msg in message_history],
                    "context": last_state.get("context", [])
                })
            await final_answer.send()
            
    except Exception as e:
        print(f"Error in message handler: {str(e)}")
        await cl.Message(content="I apologize, but I encountered an error processing your message. Please try again.").send()