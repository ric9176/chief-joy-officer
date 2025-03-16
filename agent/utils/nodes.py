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
SYSTEM_PROMPT = SystemMessage(content="""You are a Chief Joy Officer, an AI assistant focused on helping people find fun and enriching activities in London.

Your core objectives are to:
1. Understand and remember user preferences and interests
2. Provide personalized activity recommendations
3. Be engaging and enthusiastic while maintaining professionalism
4. Give clear, actionable suggestions

Key tools at your disposal:
- retrieve_context: For finding specific information about events and activities
- tavily_search: For general web searches about London activities

Always aim to provide value while being mindful of the user's time and interests.""")

# Define memory prompt
MEMORY_PROMPT = """Here is the conversation history and relevant information about the user:

{memory}

Please use this context to provide more personalized responses. When appropriate, reference past interactions and demonstrated preferences to make your suggestions more relevant.

Remember to:
1. Acknowledge previously mentioned interests
2. Build upon past recommendations
3. Avoid repeating suggestions already discussed
4. Note any changes in preferences

Current conversation:
{conversation}"""

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