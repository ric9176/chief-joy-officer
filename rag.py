from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import WebBaseLoader
import tiktoken

def load_web_documents(urls: List[str]) -> List[Document]:
    """
    Load documents from web URLs
    
    Args:
        urls: List of URLs to load
        
    Returns:
        List of loaded documents
    """
    loader = WebBaseLoader(urls)
    return loader.load()

def create_rag_pipeline(collection_name: str = "rag_collection"):
    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_dim = 1536  # Dimension for text-embedding-3-small

    # Initialize Qdrant client (in-memory for development)
    client = QdrantClient(":memory:")
    
    # Create collection for vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    # Create text splitter for chunking
    def tiktoken_len(text):
        tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust based on your needs
        chunk_overlap=50,
        length_function=tiktoken_len,
    )

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    return {
        "vector_store": vector_store,
        "text_splitter": text_splitter,
        "retriever": retriever
    }

def add_documents(vector_store, text_splitter, documents: List[Document]):
    """
    Add documents to the vector store
    
    Args:
        vector_store: The initialized vector store
        text_splitter: The text splitter for chunking
        documents: List of Document objects to add
    """
    # Split documents into chunks
    chunks = []
    for doc in documents:
        # Split the page content of each document
        doc_chunks = text_splitter.split_text(doc.page_content)
        chunks.extend(doc_chunks)
    
    # Add chunks to vector store
    vector_store.add_texts(texts=chunks)

def add_urls_to_vectorstore(vector_store, text_splitter, urls: List[str]):
    """
    Load documents from URLs and add them to the vector store
    
    Args:
        vector_store: The initialized vector store
        text_splitter: The text splitter for chunking
        urls: List of URLs to load and add
    """
    # Load documents from URLs
    documents = load_web_documents(urls)
    
    # Add documents to vector store
    add_documents(vector_store, text_splitter, documents)

def get_relevant_context(retriever, question: str) -> List[Document]:
    """
    Get relevant context for a question
    
    Args:
        retriever: The initialized retriever
        question: The question to find context for
        
    Returns:
        List of relevant documents
    """
    return retriever.get_relevant_documents(question) 