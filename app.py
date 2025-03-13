import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from agent import graph_with_memory as graph
from agent.utils.state import AgentState

@cl.on_chat_start
async def on_chat_start():
    # Generate and store a session ID
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    
    # Initialize the conversation state with proper auth
    cl.user_session.set("messages", [])
    
    # Initialize config using stored session ID
    config = RunnableConfig(
        configurable={
            "thread_id": session_id,
            "sessionId": session_id
        }
    )
    
    # Initialize empty state with auth
    try:
        await graph.ainvoke(
            AgentState(messages=[], context=[]),
            config=config
        )
    except Exception as e:
        print(f"Error initializing state: {str(e)}")
    
    await cl.Message(
        content="Hello! I'm your chief joy officer, here to help you with finding fun things to do in London!",
        author="Assistant"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    print(f"Session ID: {session_id}")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
    
    config = RunnableConfig(
        configurable={
            "thread_id": session_id,
            "checkpoint_ns": "default_namespace",
            "sessionId": session_id
        }
    )
    
    # Try to retrieve previous conversation state
    try:
        previous_state = await graph.aget_state(config)
        if previous_state and previous_state.values:
            previous_messages = previous_state.values.get('messages', [])
            print("Found previous state with messages:", len(previous_messages))
        else:
            print("Previous state empty or invalid")
            previous_messages = []
        current_messages = previous_messages + [HumanMessage(content=message.content)]
    except Exception as e:
        print(f"Error retrieving previous state: {str(e)}")
        current_messages = [HumanMessage(content=message.content)]
    
    # Setup callback handler and final answer message
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    await final_answer.send()
    
    loading_msg = None  # Initialize reference to loading message
    
    # Stream the response
    async for chunk in graph.astream(
        AgentState(messages=current_messages, context=[]),
        config=RunnableConfig(
            configurable={
                "thread_id": session_id,
            }
        )
    ):
        for node, values in chunk.items():
            if node == "retrieve":
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

    await final_answer.send()