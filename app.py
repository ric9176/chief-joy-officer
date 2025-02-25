import uuid
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from langchain_core.tools import tool

import chainlit as cl
from rag import create_rag_pipeline, add_urls_to_vectorstore

# Initialize RAG pipeline
rag_components = create_rag_pipeline(collection_name="london_events")

# Add some initial URLs to the vector store
urls = [
    "https://www.timeout.com/london/things-to-do-in-london-this-weekend",
    "https://www.timeout.com/london/london-events-in-march"
]
add_urls_to_vectorstore(
    rag_components["vector_store"],
    rag_components["text_splitter"],
    urls
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list  # Store retrieved context

# Create a retrieve tool
@tool
def retrieve_context(query: str) -> list[str]:
    """Searches the knowledge base for relevant information about events and activities. Use this when you need specific details about events."""
    return [doc.page_content for doc in rag_components["retriever"].get_relevant_documents(query)]

tavily_tool = TavilySearchResults(max_results=5)
tool_belt = [tavily_tool, retrieve_context]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = llm.bind_tools(tool_belt)

# Define system prompt
SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful AI assistant that answers questions clearly and concisely.
If you don't know something, simply say you don't know.
Be engaging and professional in your responses.
Use the retrieve_context tool when you need specific information about events and activities.
Use the tavily_search tool for general web searches.
""")

def call_model(state: AgentState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tool_belt)

# Simple flow control - always go to final
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    return END

# Create the graph
builder = StateGraph(AgentState)

# Remove retrieve node and modify graph structure
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
graph = builder.compile(checkpointer=memory)

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
            {"messages": [], "context": []},
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
        {"messages": current_messages, "context": []},
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