from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from agent.utils.tools import tool_belt
from agent.utils.state import AgentState

# Initialize LLM
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

# Initialize tool node
tool_node = ToolNode(tool_belt)

# Simple flow control - always go to final
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    return END 