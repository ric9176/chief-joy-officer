from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
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

@tool
def retrieve_context(query: str) -> list[str]:
    """Searches the knowledge base for relevant information about events and activities. Use this when you need specific details about events."""
    return [doc.page_content for doc in rag_components["retriever"].get_relevant_documents(query)]

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)

# Create tool belt
tool_belt = [tavily_tool, retrieve_context] 