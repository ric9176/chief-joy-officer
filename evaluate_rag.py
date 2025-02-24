import os
from dotenv import load_dotenv
import nltk
from typing import List
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from ragas import evaluate, RunConfig, EvaluationDataset

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_APP_TOKEN")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate

# Initialize the URLs (same as app.py)
urls = [
    "https://www.timeout.com/london/things-to-do-in-london-this-weekend",
    "https://www.timeout.com/london/london-events-in-march"
]

# Load documents
loader = WebBaseLoader(urls)
docs = loader.load()

# Initialize generator models for RAGAS
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Generate synthetic test dataset
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

print(dataset.upload())

# Print the generated test questions
print("\nGenerated Test Questions:")
for i, test_row in enumerate(dataset):
    print(f"{i+1}. {test_row.eval_sample.user_input}")

# Set up the RAG pipeline for testing
# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)

# Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="london_events",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="london_events",
    embedding=embeddings,
)

# Add documents to vector store
vector_store.add_documents(documents=split_documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create RAG prompt
RAG_PROMPT = """
You are a helpful assistant who answers questions about events and activities in London.
Answer based only on the provided context. If you cannot find the answer, say so.

Question: {question}

Context: {context}

Answer:"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
llm = ChatOpenAI(model="gpt-4o-mini")

# Process each test question through the RAG pipeline
for test_row in dataset:
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(test_row.eval_sample.user_input)
    
    # Format context and generate response
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = rag_prompt.format_messages(question=test_row.eval_sample.user_input, context=context)
    response = llm.invoke(messages)
    
    # Store results in dataset
    test_row.eval_sample.response = response.content
    test_row.eval_sample.retrieved_contexts = [doc.page_content for doc in retrieved_docs]

# Convert to evaluation dataset
evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())


# Set up evaluator
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# Run evaluation with all metrics
custom_run_config = RunConfig(timeout=360)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ],
    llm=evaluator_llm,
    run_config=custom_run_config
)

print("RAW RESULT: ", result)
print("Type of result: ", type(result))

# Convert to pandas DataFrame for better formatting
df = result.to_pandas()
print("\nEvaluation Results as DataFrame:")
print(df)

try:
    # Try to save as markdown
    print("Attempting to save as markdown...")
    import tabulate  # Try to import explicitly to verify installation
    df.to_markdown("docs/evaluation_results.md", index=False)
    print("Successfully saved as markdown!")
except ImportError as e:
    # Print detailed error message
    print(f"Import Error: {e}")
    print("Note: Install 'tabulate' package for markdown output. Falling back to CSV format.")
    df.to_csv("docs/evaluation_results.csv", index=False)

# Save test questions
with open("docs/test_questions.md", "w") as f:
    f.write("# Test Questions\n\n")
    for i, test_row in enumerate(dataset):
        f.write(f"{i+1}. {test_row.eval_sample.user_input}\n")