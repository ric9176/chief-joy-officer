from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from typing import Literal
# from chainlit.logger import logger


from agent.utils.tools import tool_belt
from agent.utils.state import AgentState

# Initialize LLM for memory operations
model = ChatOpenAI(model="gpt-4", temperature=0)

# Define system prompt with memory
SYSTEM_PROMPT = """You are a Chief Joy Officer, an AI assistant focused on helping people find fun and enriching activities in London.
You have access to memory about the user's preferences and past interactions.

Here is what you remember about this user:
{memory}

Your core objectives are to:
1. Understand and remember user preferences and interests
2. Provide personalized activity recommendations based on their interests
3. Be engaging and enthusiastic while maintaining professionalism
4. Give clear, actionable suggestions

Key tools at your disposal:
- retrieve_context: For finding specific information about events and activities
- tavily_search: For general web searches about London activities

Always aim to provide value while being mindful of the user's time and interests."""

# Define memory creation/update prompt
MEMORY_UPDATE_PROMPT = """You are analyzing the conversation to update the user's profile and preferences.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history carefully
2. Identify new information about the user, such as:
   - Activity preferences (indoor/outdoor, cultural/sports, etc.)
   - Specific interests (art, music, food, etc.)
   - Location preferences in London
   - Time/schedule constraints
   - Past experiences with activities
   - Budget considerations
3. Merge new information with existing memory
4. Format as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent

Remember: Only include factual information directly stated by the user. Do not make assumptions.

Based on the conversation, please update the user information:"""

def get_last_human_message(state: AgentState):
    """Get the last human message from the state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message
    return None

def call_model(state: AgentState, config: RunnableConfig, store: BaseStore):
    """Process messages using memory from the store."""
    # Get the user ID from the config
    user_id = config["configurable"].get("session_id", "default")
    
    # Retrieve memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
    
    # Extract memory content or use default
    memory_content = existing_memory.value.get('memory') if existing_memory else "No previous information about this user."
    
    # Create messages list with system prompt including memory
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(memory=memory_content))
    ] + state["messages"]
    
    response = model.invoke(messages)
    return {"messages": [response]}

def update_memory(state: AgentState, config: RunnableConfig, store: BaseStore):
    """Update user memory based on conversation."""
    user_id = config["configurable"].get("session_id", "default")
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
    
    memory_content = existing_memory.value.get('memory') if existing_memory else "No previous information about this user."
    
    update_prompt = MEMORY_UPDATE_PROMPT.format(memory=memory_content)
    new_memory = model.invoke([
        SystemMessage(content=update_prompt)
    ] + state["messages"])
    
    store.put(namespace, "user_memory", {"memory": new_memory.content})
    return state

def should_continue(state: AgentState) -> Literal["action", "write_memory", END]:
    """Determine the next node in the graph."""
    if not state["messages"]:
        return END
        
    last_message = state["messages"][-1]
    if isinstance(last_message, list):
        last_message = last_message[-1]
        
    last_human_message = get_last_human_message(state)

    # Handle tool calls
    if hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("tool_calls"):
        return "action"
    
    # Handle memory operations for human messages
    if last_human_message:
            
        # Write memory for longer messages that might contain personal information
        if len(last_human_message.content.split()) > 3:
            return "write_memory"
            
    return END

# Define the memory creation prompt
MEMORY_CREATION_PROMPT = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

async def write_memory(state: AgentState, config: RunnableConfig, store: BaseStore) -> AgentState:
    """Reflect on the chat history and save a memory to the store."""

    # Get the session ID from config
    session_id = config["configurable"].get("session_id", "default")
    
    # Define the namespace for this user's memory
    namespace = ("memory", session_id)
    
    # Get existing memory using async interface
    existing_memory = await store.aget(namespace, "user_memory")
    memory_content = existing_memory.value.get('memory') if existing_memory else "No previous information about this user."
    
    # Create system message with memory context
    system_msg = SystemMessage(content=MEMORY_CREATION_PROMPT.format(memory=memory_content))
    
    # Get messages and ensure we're working with the correct format
    messages = state.get("messages", [])
    if not messages:
        return state
        
    # Create memory using the model
    new_memory = await model.ainvoke([system_msg] + messages)
    
    # Store the updated memory using async interface
    await store.aput(namespace, "user_memory", {"memory": new_memory.content})
    
    
    return state

# Initialize tool node
tool_node = ToolNode(tool_belt)

# def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    
#     """Reflect on the memories and chat history to decide whether to update the memory collection."""
#     message = state['messages'][-1]
#     if len(message.tool_calls) ==0:
#         return END
#     else:
#         tool_call = message.tool_calls[0]
#         if tool_call['args']['update_type'] == "user":
#             return "update_profile"
#         elif tool_call['args']['update_type'] == "todo":
#             return "update_todos"
#         elif tool_call['args']['update_type'] == "instructions":
#             return "update_instructions"
#         else:
#             raise ValueError