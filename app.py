from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults

import chainlit as cl

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

tavily_tool = TavilySearchResults(max_results=5)
tool_belt = [tavily_tool]
# Initialize the language models
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# final_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0).with_config(tags=["final_node"])
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

# Define system prompt
SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful AI assistant that answers questions clearly and concisely.
If you don't know something, simply say you don't know.
Be engaging and professional in your responses.
""")


def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages" : [response]}

tool_node = ToolNode(tool_belt)


# Simple flow control - always go to final
def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

# Create the graph
builder = StateGraph(AgentState)

builder.set_entry_point("agent")
builder.add_node("agent", call_model)
builder.add_node("action", tool_node)
# Add edges
builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("action", "agent")

# Compile the graph
graph = builder.compile()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Hello! I'm your AI assistant. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Create configuration with thread ID
    config = {
        "configurable": {
            "thread_id": cl.context.session.id,
            "checkpoint_ns": "default_namespace"
        }
    }
    
    # Setup callback handler and final answer message
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    await final_answer.send()
    
    loading_msg = None  # Initialize reference to loading message
    
    # Stream the response
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=message.content)]},
        config=RunnableConfig(callbacks=[cb], **config)
    ):
        for node, values in chunk.items():
            if values.get("messages"):
                last_message = values["messages"][-1]
                # Check for tool calls in additional_kwargs
                if hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("tool_calls"):
                    tool_name = last_message.additional_kwargs["tool_calls"][0]["function"]["name"]
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