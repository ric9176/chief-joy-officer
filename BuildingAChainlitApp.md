# Building a Chainlit App

What if we want to take our Week 1 Day 2 assignment - [Pythonic RAG](https://github.com/AI-Maker-Space/AIE4/tree/main/Week%201/Day%202) - and bring it out of the notebook?

Well - we'll cover exactly that here!

## Anatomy of a Chainlit Application

[Chainlit](https://docs.chainlit.io/get-started/overview) is a Python package similar to Streamlit that lets users write a backend and a front end in a single (or multiple) Python file(s). It is mainly used for prototyping LLM-based Chat Style Applications - though it is used in production in some settings with 1,000,000s of MAUs (Monthly Active Users).

The primary method of customizing and interacting with the Chainlit UI is through a few critical [decorators](https://blog.hubspot.com/website/decorators-in-python).

> NOTE: Simply put, the decorators (in Chainlit) are just ways we can "plug-in" to the functionality in Chainlit. 

We'll be concerning ourselves with three main scopes:

1. On application start - when we start the Chainlit application with a command like `chainlit run app.py`
2. On chat start - when a chat session starts (a user opens the web browser to the address hosting the application)
3. On message - when the users sends a message through the input text box in the Chainlit UI

Let's dig into each scope and see what we're doing!

## On Application Start:

The first thing you'll notice is that we have the traditional "wall of imports" this is to ensure we have everything we need to run our application. 

```python
import os
from typing import List
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
```

Next up, we have some prompt templates. As all sessions will use the same prompt templates without modification, and we don't need these templates to be specific per template - we can set them up here - at the application scope. 

```python
system_template = """\
Use the following context to answer a users question. If you cannot find the answer in the context, say you don't know the answer."""
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = """\
Context:
{context}

Question:
{question}
"""
user_role_prompt = UserRolePrompt(user_prompt_template)
```

> NOTE: You'll notice that these are the exact same prompt templates we used from the Pythonic RAG Notebook in Week 1 Day 2!

Following that - we can create the Python Class definition for our RAG pipeline - or *chain*, as we'll refer to it in the rest of this walkthrough. 

Let's look at the definition first:

```python
class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        ### RETRIEVAL
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        ### AUGMENTED
        formatted_system_prompt = system_role_prompt.create_message()

        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)


        ### GENERATION
        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}
```

Notice a few things:

1. We have modified this `RetrievalAugmentedQAPipeline` from the initial notebook to support streaming. 
2. In essence, our pipeline is *chaining* a few events together:
    1. We take our user query, and chain it into our Vector Database to collect related chunks
    2. We take those contexts and our user's questions and chain them into the prompt templates
    3. We take that prompt template and chain it into our LLM call
    4. We chain the response of the LLM call to the user
3. We are using a lot of `async` again!

#### QUESTION #1:

Why do we want to support streaming? What about streaming is important, or useful?





