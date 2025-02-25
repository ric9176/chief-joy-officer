# Chief Joy Officer (CJO)

## Overview

Chief Joy Officer (CJO) is an AI-powered agent designed to enhance social experiences by seamlessly integrating into group chats. It assists users in discovering and organizing social activities based on their interests, preferences, and locality.

## Demo

Check out the demo: [Loom Video](https://www.loom.com/share/3a8d6318c9d346b7a14681a0980c5aaa)

Full write up here: [Google Doc](https://docs.google.com/document/d/18een8Siwt-5lXyCZ79ovjxDOLFBP4eT2_aBlhFxUVTI/edit?usp=sharing)

## Problem Statement

### The Challenge

In group chats, organizing social events can be challenging as no one has the time to research activities, check schedules, and make bookings. Typically, one person takes on the responsibility of planning, which can be a burden.

### The User Perspective

- Social connection is vital, but busy schedules make it difficult to plan engaging activities.
- Some groups have a "social architect," but if they get too busy, the quality of social time diminishes.
- Our target users are individuals aged 25-40 who are busy with work but value quality time with friends.

## Proposed Solution

CJO is an AI assistant that acts as a group chat participant, proactively suggesting events and activities. It:

- Learns group preferences through chat interactions.
- Provides personalized recommendations.
- Assists with booking and event details.
- Enables effortless planning and decision-making within the group chat.

## Technology Stack

| Component               | Technology                                            |
| ----------------------- | ----------------------------------------------------- |
| **Orchestration Layer** | Langgraph                                             |
| **Observability**       | Langsmith                                             |
| **LLM**                 | 4o-mini                                               |
| **Web Search Tool**     | Tavily (Firecrawl for advanced scraping)              |
| **Embeddings Model**    | snowflake-arctic-embed-l                              |
| **Vector Database**     | Qdrant                                                |
| **Frontend UI**         | Chainlit (for POC), WhatsApp API integration (future) |
| **Evaluations**         | Langsmith, Ragas                                      |

## Agentic Reasoning

CJO will use agentic reasoning to:

- Enhance contextual understanding using Qdrant’s vector store.
- Retrieve live event data via Tavily.
- Implement long-term memory for personalized recommendations.

## Data Sources & APIs

| Source                                             | Purpose                                      |
| -------------------------------------------------- | -------------------------------------------- |
| [Time Out London](https://www.timeout.com/london/) | Provides event data for POC                  |
| Tavily                                             | Web search for additional context            |
| Firecrawl                                          | Scraping multiple sources (future expansion) |
| Long-term Memory Storage                           | Tracks user preferences over time            |

## Chunking Strategy

Using **RecursiveCharacterTextSplitter** (Langchain) for structured text splitting:

- Keeps paragraphs intact for better semantic coherence.
- Adapts dynamically to text structure for optimized retrieval.

## Prototype Development

CJO was built by refactoring an existing Chainlit app **pythonic-rag** to integrate Langgraph and a ReAct pattern for **Agentic RAG + web search**.

- **App Link**: [Pythonic-RAG on Hugging Face](https://huggingface.co/spaces/ric9176/pythonic-rag)
- **GitHub Repo**: [Pythonic-RAG Repository](https://github.com/ric9176/pythonic-rag)

## Performance Evaluation

### Initial RAG Evaluation Metrics

| Metric                | Score |
| --------------------- | ----- |
| Context Recall        | 0.620 |
| Faithfulness          | 0.885 |
| Factual Correctness   | 0.310 |
| Answer Relevancy      | 0.762 |
| Context Entity Recall | 0.393 |
| Noise Sensitivity     | 0.300 |

#### Key Takeaways

- **Good**: Faithfulness and answer relevancy.
- **Needs Improvement**: Factual correctness, entity recall, and context recall.
- **Action Items**:
  - Increase retrieval context (adjust `k` value).
  - Optimize chunking strategy.
  - Fine-tune the embedding model.
  - Improve the system prompt.

### Fine-Tuned Embeddings Model

- **Model:** [Fine-tuned Arctic Embeddings](https://huggingface.co/ric9176/cjo-ft-v0)
- **Implementation:** Integrated into the data ingestion pipeline.

#### Performance Comparison (RAGAS Metrics)

| Metric                | OpenAI | Base Arctic | Fine-tuned Arctic |
| --------------------- | ------ | ----------- | ----------------- |
| Context Recall        | 1.0    | 0.0         | 0.0               |
| Faithfulness          | 0.0    | 0.0         | 1.0               |
| Factual Correctness   | 0.67   | 0.0         | 0.0               |
| Answer Relevancy      | 0.98   | 0.0         | 0.81              |
| Context Entity Recall | 1.0    | 0.2         | 0.0               |
| Noise Sensitivity     | 0.5    | 0.0         | 0.0               |

#### Next Steps

- Improve dataset size and periodic data ingestion.
- Enhance short-term and long-term memory capabilities.
- Enable passive observation and proactive suggestions.
- Implement WhatsApp API integration.
- Improve codebase modularity and maintainability.
- Add voice interaction capabilities (time permitting).
- Set up auto evaluations in Langsmith for benchmarking improvements.

## Future Improvements

- **Scalability:** Move to a cloud-hosted Qdrant instance.
- **Memory Handling:** Implement SQL-based short-term memory and Qdrant for long-term.
- **Automation:** Schedule ingestion pipelines and improve scraping capabilities.
- **Agent Evaluations:** Benchmark tool selection accuracy.
- **Testing:** Use structured SDG datasets for continuous evaluation.

## Contributing

Contributions are welcome! Please check out the GitHub repository for issue tracking and future enhancements.

---

### Stay Connected

- **GitHub**: [Pythonic-RAG Repository](https://github.com/ric9176/pythonic-rag)
- **Hugging Face**: [Fine-Tuned Embeddings](https://huggingface.co/ric9176/cjo-ft-v0)
- **Demo**: [Loom Video](https://www.loom.com/share/3a8d6318c9d346b7a14681a0980c5aaa)

Let's make social planning effortless with AI! 🚀
