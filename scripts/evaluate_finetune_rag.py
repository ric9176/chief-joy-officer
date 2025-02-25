import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)

# Load environment variables
load_dotenv()

# Initialize URLs and load documents
urls = [
    "https://www.timeout.com/london/things-to-do-in-london-this-weekend",
    "https://www.timeout.com/london/london-events-in-march"
]

loader = WebBaseLoader(urls)
docs = loader.load()

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=50,
    length_function=len
)
split_documents = text_splitter.split_documents(docs)

# Initialize embedding models
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
base_embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l")
finetuned_embeddings = HuggingFaceEmbeddings(model_name="ric9176/cjo-ft-v0")

def create_rag_chain(documents: List[Document], embeddings, k: int = 6):
    """Create a RAG chain with specified embeddings"""
    # Create vector store and retriever
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # Create RAG prompt
    rag_prompt = ChatPromptTemplate.from_template("""
    Given a provided context and a question, you must answer the question. 
    If you do not know the answer, you must state that you do not know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create RAG chain
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm | StrOutputParser(), "context": itemgetter("context")}
    )
    
    return rag_chain

def evaluate_embeddings(documents, test_questions):
    """Evaluate different embedding models"""
    results = {}
    
    # Create RAG chains for each embedding model
    chains = {
        "OpenAI": create_rag_chain(documents, openai_embeddings),
        "Base Arctic": create_rag_chain(documents, base_embeddings),
        "Fine-tuned Arctic": create_rag_chain(documents, finetuned_embeddings)
    }
    
    # Generate test dataset using RAGAS
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    
    # Evaluate each model
    for model_name, chain in chains.items():
        print(f"\nEvaluating {model_name}...")
        
        # Generate dataset
        dataset = generator.generate_with_langchain_docs(documents, testset_size=10)
        
        # Process questions through RAG pipeline
        for test_row in dataset:
            response = chain.invoke({"question": test_row.eval_sample.user_input})
            test_row.eval_sample.response = response["response"]
            test_row.eval_sample.retrieved_contexts = [
                context.page_content for context in response["context"]
            ]
        
        # Convert to evaluation dataset
        evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
        
        # Run RAGAS evaluation
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
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
            llm=evaluator_llm
        )
        
        results[model_name] = result
    
    return results

# Run evaluation
print("Starting evaluation of embedding models...")
results = evaluate_embeddings(split_documents, None)

# Save results
print("\nSaving results...")
os.makedirs("docs", exist_ok=True)

# Save detailed results for each model
for model_name, result in results.items():
    df = result.to_pandas()
    filename = f"docs/evaluation_{model_name.lower().replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved results for {model_name} to {filename}")

# Create comparison table
comparison = pd.DataFrame()
for model_name, result in results.items():
    comparison[model_name] = pd.Series(result.scores)

# Save comparison
comparison.to_csv("docs/embedding_comparison.csv")
print("\nSaved comparison to docs/embedding_comparison.csv")

# Print comparison
print("\nEmbedding Models Comparison:")
print(comparison) 