# LangChain Cheat Sheet

LangChain is a framework for developing applications powered by Large Language Models (LLMs). It simplifies the entire LLM application lifecycle with open-source components, third-party integrations, and tools for building complex AI workflows.

## Installation

```bash
# Core LangChain library
pip install langchain

# Specific integrations
pip install langchain-openai        # OpenAI models
pip install langchain-anthropic     # Claude models
pip install langchain-community     # Community integrations
pip install langchain-experimental  # Experimental features

# Vector stores
pip install langchain-chroma        # ChromaDB
pip install langchain-pinecone      # Pinecone
pip install faiss-cpu               # FAISS

# All common packages
pip install langchain[all]

# Development installation
git clone https://github.com/langchain-ai/langchain.git
cd langchain
pip install -e .[all]
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Create chain using LCEL (LangChain Expression Language)
chain = prompt | llm | StrOutputParser()

# Invoke the chain
result = chain.invoke({"input": "What is LangChain?"})
print(result)
```

## Core Concepts

### 1. LangChain Expression Language (LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Basic chain composition
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# Chain components with | operator
chain = prompt | model | output_parser

# Invoke chain
result = chain.invoke({"topic": "programming"})

# Batch processing
results = chain.batch([
    {"topic": "cats"}, 
    {"topic": "dogs"}
])

# Streaming
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async support
import asyncio
async_result = await chain.ainvoke({"topic": "python"})
```

### 2. Prompts and Templates

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

# Basic prompt template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

# Few-shot prompting
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)

# Partial prompts
partial_prompt = prompt.partial(product="smartphones")
result = partial_prompt.format()

# Prompt composition
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    chat_prompt,
    ("human", "Please also explain why this translation is correct.")
])
```

### 3. LLM Integration

```python
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# OpenAI models
openai_chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key"
)

# Anthropic Claude
anthropic = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=1000
)

# Local Ollama models
ollama = Ollama(
    model="llama2",
    base_url="http://localhost:11434"
)

# Model with callbacks for monitoring
from langchain_core.callbacks import BaseCallbackHandler

class TokenCountCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and 'token_usage' in response.llm_output:
            self.total_tokens += response.llm_output['token_usage']['total_tokens']

callback = TokenCountCallback()
llm_with_callback = ChatOpenAI(callbacks=[callback])
```

## Common Patterns

### 1. Sequential Chains

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import PromptTemplate

# First chain: generate synopsis
synopsis_template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
Title: {title}
Playwright: This is a synopsis for the above play:"""

synopsis_prompt = PromptTemplate(
    input_variables=["title"], 
    template=synopsis_template
)
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt)

# Second chain: write review
review_template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
Synopsis: {synopsis}
Review from a New York Times play critic of the above play:"""

review_prompt = PromptTemplate(
    input_variables=["synopsis"], 
    template=review_template
)
review_chain = LLMChain(llm=llm, prompt=review_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain],
    verbose=True
)

# Run the chain
review = overall_chain.invoke({"input": "Tragedy at sunset on the beach"})
```

### 2. Parallel Processing

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Define parallel tasks
joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model | StrOutputParser()
poem_chain = ChatPromptTemplate.from_template("write a short poem about {topic}") | model | StrOutputParser()

# Run in parallel
parallel_chain = RunnableParallel({
    "joke": joke_chain,
    "poem": poem_chain,
    "original_topic": RunnablePassthrough()
})

result = parallel_chain.invoke({"topic": "artificial intelligence"})
print(result["joke"])
print(result["poem"])
print(result["original_topic"])
```

### 3. Conditional Logic and Routing

```python
from langchain_core.runnables import RunnableBranch

def route_question(info):
    if "math" in info["question"].lower():
        return math_chain
    elif "history" in info["question"].lower():
        return history_chain
    else:
        return general_chain

# Math chain
math_chain = ChatPromptTemplate.from_template(
    "You are a math expert. Answer this question: {question}"
) | model | StrOutputParser()

# History chain
history_chain = ChatPromptTemplate.from_template(
    "You are a history expert. Answer this question: {question}"
) | model | StrOutputParser()

# General chain
general_chain = ChatPromptTemplate.from_template(
    "Answer this question: {question}"
) | model | StrOutputParser()

# Route based on question content
routing_chain = RunnableBranch(
    (lambda x: "math" in x["question"].lower(), math_chain),
    (lambda x: "history" in x["question"].lower(), history_chain),
    general_chain  # default
)

result = routing_chain.invoke({"question": "What is 2+2?"})
```

## Memory Management

### 1. Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Basic conversation memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have a conversation
response1 = conversation.invoke({"input": "Hi, I'm John"})
response2 = conversation.invoke({"input": "What's my name?"})

# Access memory
print(memory.buffer)
print(memory.chat_memory.messages)
```

### 2. Conversation Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

# Memory that summarizes conversation
summary_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

# Long conversation will be summarized
for i in range(5):
    response = conversation_with_summary.invoke({
        "input": f"Tell me a fact about number {i}"
    })
```

### 3. Conversation Buffer Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last k interactions
window_memory = ConversationBufferWindowMemory(
    k=3,  # Keep last 3 exchanges
    return_messages=True
)

windowed_conversation = ConversationChain(
    llm=llm,
    memory=window_memory
)
```

### 4. Custom Memory with LCEL

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage
from typing import List

# Custom memory implementation
class CustomMemory:
    def __init__(self):
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage):
        self.messages.append(message)
    
    def get_context(self) -> str:
        return "\n".join([f"{msg.type}: {msg.content}" for msg in self.messages[-6:]])

memory = CustomMemory()

# Chain with custom memory
def add_memory(inputs):
    # Add user input to memory
    memory.add_message(HumanMessage(content=inputs["input"]))
    inputs["chat_history"] = memory.get_context()
    return inputs

def save_response(response):
    # Save AI response to memory
    memory.add_message(AIMessage(content=response.content))
    return response

chat_with_memory = (
    RunnableLambda(add_memory) |
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Here's the chat history:\n{chat_history}"),
        ("human", "{input}")
    ]) |
    model |
    RunnableLambda(save_response) |
    StrOutputParser()
)
```

## Document Loading and Processing

### 1. Document Loaders

```python
from langchain_community.document_loaders import (
    TextLoader, PDFLoader, WebBaseLoader, 
    DirectoryLoader, CSVLoader, UnstructuredHTMLLoader
)

# Text files
text_loader = TextLoader("path/to/file.txt")
docs = text_loader.load()

# PDF files
pdf_loader = PDFLoader("path/to/document.pdf")
pdf_docs = pdf_loader.load()

# Web pages
web_loader = WebBaseLoader("https://example.com")
web_docs = web_loader.load()

# Directory of files
directory_loader = DirectoryLoader(
    "path/to/directory",
    glob="**/*.txt",
    loader_cls=TextLoader
)
all_docs = directory_loader.load()

# CSV files
csv_loader = CSVLoader("path/to/data.csv")
csv_docs = csv_loader.load()

# Custom loader
from langchain_core.documents import Document

def custom_loader(file_path: str) -> List[Document]:
    # Custom loading logic
    with open(file_path, 'r') as f:
        content = f.read()
    
    return [Document(
        page_content=content,
        metadata={"source": file_path, "custom_field": "value"}
    )]
```

### 2. Text Splitting

```python
from langchain.text_splitter import (
    CharacterTextSplitter, RecursiveCharacterTextSplitter,
    TokenTextSplitter, MarkdownHeaderTextSplitter
)

# Character-based splitting
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)
char_chunks = char_splitter.split_documents(docs)

# Recursive character splitting (recommended)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
recursive_chunks = recursive_splitter.split_documents(docs)

# Token-based splitting
token_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
token_chunks = token_splitter.split_documents(docs)

# Markdown-aware splitting
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)
markdown_chunks = markdown_splitter.split_text(markdown_text)

# Custom splitter
from langchain.text_splitter import TextSplitter

class CustomSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        # Custom splitting logic
        return text.split("---")  # Split on custom separator

custom_splitter = CustomSplitter()
custom_chunks = custom_splitter.split_documents(docs)
```

## Vector Stores and Embeddings

### 1. Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

# OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Hugging Face embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Local Ollama embeddings
ollama_embeddings = OllamaEmbeddings(
    model="llama2",
    base_url="http://localhost:11434"
)

# Test embeddings
text = "This is a test document"
embedding_vector = openai_embeddings.embed_query(text)
print(f"Embedding dimension: {len(embedding_vector)}")

# Batch embeddings
texts = ["Document 1", "Document 2", "Document 3"]
batch_embeddings = openai_embeddings.embed_documents(texts)
```

### 2. Vector Store Operations

```python
from langchain_community.vectorstores import Chroma, FAISS, Pinecone
from langchain_core.documents import Document

# Create documents
docs = [
    Document(page_content="The sky is blue", metadata={"source": "fact1"}),
    Document(page_content="Grass is green", metadata={"source": "fact2"}),
    Document(page_content="Fire is hot", metadata={"source": "fact3"}),
]

# ChromaDB vector store
chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=openai_embeddings,
    persist_directory="./chroma_db"
)

# FAISS vector store (in-memory)
faiss_db = FAISS.from_documents(
    documents=docs,
    embedding=openai_embeddings
)

# Save/load FAISS
faiss_db.save_local("./faiss_index")
loaded_faiss = FAISS.load_local("./faiss_index", openai_embeddings)

# Pinecone vector store
import pinecone
pinecone.init(api_key="your-api-key", environment="your-env")

pinecone_db = Pinecone.from_documents(
    documents=docs,
    embedding=openai_embeddings,
    index_name="your-index"
)

# Search operations
query = "What color is the sky?"
similar_docs = chroma_db.similarity_search(query, k=2)

# Search with scores
docs_with_scores = chroma_db.similarity_search_with_score(query, k=2)
for doc, score in docs_with_scores:
    print(f"Score: {score}, Content: {doc.page_content}")

# Filtered search
filtered_docs = chroma_db.similarity_search(
    query, 
    k=2, 
    filter={"source": "fact1"}
)

# Add more documents
new_docs = [Document(page_content="Water is wet", metadata={"source": "fact4"})]
chroma_db.add_documents(new_docs)

# Delete documents
chroma_db.delete(ids=["doc_id_to_delete"])
```

## Retrieval-Augmented Generation (RAG)

### 1. Basic RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Set up vector store as retriever
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Traditional approach with RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is the color of the sky?"})
print(result["result"])
print(result["source_documents"])

# Modern approach with LCEL
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the color of the sky?")
```

### 2. Advanced RAG with Multiple Retrievers

```python
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 retriever (keyword-based)
texts = [doc.page_content for doc in docs]
bm25_retriever = BM25Retriever.from_texts(texts)

# Ensemble retriever (combines multiple retrievers)
ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_db.as_retriever(), bm25_retriever],
    weights=[0.7, 0.3]  # Weight vector search more than keyword search
)

# Multi-query retriever (generates multiple queries)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=chroma_db.as_retriever(),
    llm=llm
)

# Use with RAG chain
advanced_rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

### 3. RAG with Chat History

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Contextualized retrieval
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer generation
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Full RAG chain with history
rag_chain_with_history = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

# Usage with session history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_with_history,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Chat with history
response = conversational_rag_chain.invoke(
    {"input": "What is the sky color?"},
    config={"configurable": {"session_id": "session_1"}},
)

follow_up = conversational_rag_chain.invoke(
    {"input": "Why is that?"},  # References previous question
    config={"configurable": {"session_id": "session_1"}},
)
```

## Agents and Tool Usage

### 1. Basic Agent Setup

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool
from typing import Type

# Define tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()

# Custom tool
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for mathematical calculations"
    
    def _run(self, query: str) -> str:
        try:
            return str(eval(query))  # Be careful with eval in production
        except Exception as e:
            return f"Error: {str(e)}"

calculator = CalculatorTool()

# Available tools
tools = [wikipedia, search, calculator]

# Create agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use agent
result = agent_executor.invoke({
    "input": "What is the population of Tokyo? Then calculate 10% of that number."
})
```

### 2. Custom Tools

```python
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
import requests

# Decorator-based tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    # Mock weather API call
    return f"The weather in {city} is sunny and 75°F"

# Class-based tool with input schema
class WeatherInput(BaseModel):
    city: str = Field(description="The city to get weather for")
    units: str = Field(default="fahrenheit", description="Temperature units")

class WeatherTool(BaseTool):
    name = "weather_tool"
    description = "Get current weather information"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str, units: str = "fahrenheit") -> str:
        # Weather API integration
        return f"Weather in {city}: 72°{units[0].upper()}, partly cloudy"

# API-based tool
@tool
def search_api(query: str) -> str:
    """Search the web using a custom API."""
    # Example API call
    response = requests.get(
        "https://api.example.com/search",
        params={"q": query},
        headers={"Authorization": "Bearer your-token"}
    )
    return response.json().get("results", "No results found")

# File system tool
@tool
def read_file(file_path: str) -> str:
    """Read contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Database tool
@tool
def query_database(sql_query: str) -> str:
    """Execute SQL query on database."""
    import sqlite3
    
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        conn.close()
        return f"Database error: {str(e)}"
```

### 3. Agent Types and Strategies

```python
from langchain.agents import create_react_agent, create_structured_chat_agent

# ReAct agent (Reasoning and Acting)
react_prompt = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

react_agent = create_react_agent(llm, tools, react_prompt)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Structured chat agent
structured_agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

structured_executor = AgentExecutor(
    agent=structured_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Custom agent with error handling
class CustomAgentExecutor(AgentExecutor):
    def _handle_error(self, error: Exception) -> str:
        return f"I encountered an error: {str(error)}. Let me try a different approach."

custom_executor = CustomAgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate"
)
```

## Output Parsing

### 1. Built-in Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser, JsonOutputParser, ListOutputParser,
    CommaSeparatedListOutputParser, DatetimeOutputParser
)
from langchain_core.pydantic_v1 import BaseModel, Field

# String parser (default)
str_parser = StrOutputParser()

# JSON parser
json_parser = JsonOutputParser()

# List parser
list_parser = ListOutputParser()

# Comma-separated list parser
csv_parser = CommaSeparatedListOutputParser()

# Datetime parser
datetime_parser = DatetimeOutputParser()

# Usage with chains
json_chain = (
    ChatPromptTemplate.from_template(
        "Generate a JSON object with name and age for a person named {name}"
    )
    | llm
    | json_parser
)

result = json_chain.invoke({"name": "Alice"})
```

### 2. Pydantic Parser

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define data model
class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age")
    occupation: str = Field(description="person's job")
    skills: List[str] = Field(description="list of skills")

# Create parser
person_parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt with format instructions
person_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person information from the following text."),
    ("human", "{text}\n{format_instructions}"),
])

# Chain with structured output
person_chain = (
    person_prompt.partial(format_instructions=person_parser.get_format_instructions())
    | llm
    | person_parser
)

# Usage
text = "John Doe is a 30-year-old software engineer who knows Python, JavaScript, and SQL."
structured_data = person_chain.invoke({"text": text})
print(f"Name: {structured_data.name}")
print(f"Skills: {structured_data.skills}")
```

### 3. Custom Output Parsers

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import Dict, Any
import re

class CustomEmailParser(BaseOutputParser[Dict[str, Any]]):
    """Parse email components from LLM output."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        # Extract email components using regex
        subject_match = re.search(r'Subject:\s*(.*)', text)
        body_match = re.search(r'Body:\s*(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        recipient_match = re.search(r'To:\s*(.*)', text)
        
        return {
            "subject": subject_match.group(1) if subject_match else "",
            "body": body_match.group(1).strip() if body_match else "",
            "recipient": recipient_match.group(1) if recipient_match else ""
        }
    
    @property
    def _type(self) -> str:
        return "custom_email_parser"

# Usage
email_parser = CustomEmailParser()

email_prompt = ChatPromptTemplate.from_template(
    """Write a professional email with the following format:
Subject: [subject line]
To: [recipient email]
Body: [email content]

Topic: {topic}
Recipient: {recipient}"""
)

email_chain = email_prompt | llm | email_parser

parsed_email = email_chain.invoke({
    "topic": "Meeting followup",
    "recipient": "john@example.com"
})

print(parsed_email)
```

## Advanced Features

### 1. Callbacks and Monitoring

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks import CallbackManager
import time

class TimingCallback(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.start_time = time.time()
        print(f"Chain started with inputs: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"Chain completed in {duration:.2f} seconds")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM called with prompts: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            print(f"Token usage: {token_usage}")

# Use callback
timing_callback = TimingCallback()
callback_manager = CallbackManager([timing_callback])

chain_with_callbacks = (
    prompt 
    | llm.with_config(callbacks=[timing_callback])
    | StrOutputParser()
)

result = chain_with_callbacks.invoke({"input": "Hello, world!"})
```

### 2. Streaming and Async Operations

```python
import asyncio
from langchain_core.callbacks import AsyncCallbackHandler

# Streaming responses
def stream_response(chain, input_data):
    for chunk in chain.stream(input_data):
        print(chunk, end="", flush=True)

# Async operations
async def async_chain_example():
    async_llm = ChatOpenAI(temperature=0)
    async_chain = prompt | async_llm | StrOutputParser()
    
    # Async invoke
    result = await async_chain.ainvoke({"input": "Hello async world!"})
    print(result)
    
    # Async streaming
    async for chunk in async_chain.astream({"input": "Stream this async"}):
        print(chunk, end="", flush=True)
    
    # Async batch processing
    batch_results = await async_chain.abatch([
        {"input": "First query"},
        {"input": "Second query"},
        {"input": "Third query"}
    ])
    
    return batch_results

# Run async example
# results = asyncio.run(async_chain_example())

# Async callback
class AsyncTimingCallback(AsyncCallbackHandler):
    async def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Async chain started: {inputs}")
    
    async def on_chain_end(self, outputs, **kwargs):
        print(f"Async chain completed: {outputs}")
```

### 3. Configuration and Environment Management

```python
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration management
config = RunnableConfig(
    tags=["production", "chat"],
    metadata={"user_id": "123", "session_id": "abc"},
    recursion_limit=10
)

# Chain with config
result = chain.invoke(
    {"input": "Hello"},
    config=config
)

# Environment-specific setup
class EnvironmentConfig:
    def __init__(self, env: str = "development"):
        self.env = env
        if env == "production":
            self.llm_model = "gpt-4"
            self.temperature = 0.1
            self.max_tokens = 1000
        else:
            self.llm_model = "gpt-3.5-turbo"
            self.temperature = 0.7
            self.max_tokens = 500
    
    def get_llm(self):
        return ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

# Usage
env_config = EnvironmentConfig("production")
production_llm = env_config.get_llm()
```

### 4. Custom Chain Classes

```python
from langchain_core.runnables import Runnable
from typing import Dict, Any

class CustomProcessingChain(Runnable):
    """Custom chain with preprocessing and postprocessing."""
    
    def __init__(self, llm, preprocessor=None, postprocessor=None):
        self.llm = llm
        self.preprocessor = preprocessor or self._default_preprocess
        self.postprocessor = postprocessor or self._default_postprocess
    
    def _default_preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Default preprocessing
        if "text" in input_data:
            input_data["text"] = input_data["text"].strip().lower()
        return input_data
    
    def _default_postprocess(self, output: str) -> str:
        # Default postprocessing
        return output.strip().capitalize()
    
    def invoke(self, input_data: Dict[str, Any], config=None) -> str:
        # Preprocess
        processed_input = self.preprocessor(input_data)
        
        # LLM call
        prompt = ChatPromptTemplate.from_template("Process this text: {text}")
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke(processed_input, config=config)
        
        # Postprocess
        final_result = self.postprocessor(result)
        
        return final_result

# Custom preprocessing/postprocessing functions
def custom_preprocessor(data):
    data["text"] = f"IMPORTANT: {data['text']}"
    return data

def custom_postprocessor(output):
    return f"[PROCESSED] {output}"

# Usage
custom_chain = CustomProcessingChain(
    llm=llm,
    preprocessor=custom_preprocessor,
    postprocessor=custom_postprocessor
)

result = custom_chain.invoke({"text": "hello world"})
```

## Integration Examples

### 1. FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Initialize chain
chat_chain = prompt | llm | StrOutputParser()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await chat_chain.ainvoke({"input": request.message})
        return ChatResponse(
            response=response,
            session_id=request.session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
```

### 2. Streamlit Chat Interface

```python
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = prompt | llm | StrOutputParser()

st.title("LangChain Chatbot")

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"input": user_input})
            st.write(response)
            st.session_state.messages.append(AIMessage(content=response))
```

### 3. Gradio Interface

```python
import gradio as gr

def chat_interface(message, history):
    # Convert history to proper format
    context = "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in history])
    
    # Add current message
    full_prompt = f"{context}\nHuman: {message}\nAssistant:"
    
    # Get response
    response = chain.invoke({"input": full_prompt})
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="LangChain Chat",
    description="Chat with LangChain-powered AI assistant"
)

# Launch interface
if __name__ == "__main__":
    demo.launch(share=True)
```

## Performance Optimization

### 1. Caching

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# In-memory caching
set_llm_cache(InMemoryCache())

# SQLite caching (persistent)
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Redis caching
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)
set_llm_cache(RedisCache(redis_client))

# Custom cache
from langchain_core.caches import BaseCache

class CustomCache(BaseCache):
    def __init__(self):
        self._cache = {}
    
    def lookup(self, prompt, llm_string):
        return self._cache.get((prompt, llm_string))
    
    def update(self, prompt, llm_string, return_val):
        self._cache[(prompt, llm_string)] = return_val

set_llm_cache(CustomCache())
```

### 2. Batch Processing

```python
# Batch LLM calls
batch_prompts = [
    {"input": f"Summarize topic {i}"} for i in range(10)
]

# Sequential processing
results = []
for prompt in batch_prompts:
    result = chain.invoke(prompt)
    results.append(result)

# Parallel batch processing
batch_results = chain.batch(batch_prompts, config={"max_concurrency": 5})

# Async batch processing
async def async_batch_processing():
    return await chain.abatch(batch_prompts)

# Streaming batch
for result in chain.batch(batch_prompts):
    print(f"Completed: {result}")
```

### 3. Memory Management

```python
# Efficient memory usage
from langchain.memory import ConversationSummaryBufferMemory

# Summary + buffer memory (hybrid approach)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)

# Custom memory with cleanup
class EfficientMemory:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # Keep only recent messages
            self.messages = self.messages[-self.max_messages:]
    
    def clear_old_messages(self):
        # Keep only last 5 messages
        self.messages = self.messages[-5:]

# Periodic cleanup
efficient_memory = EfficientMemory()

# Use context managers for cleanup
from contextlib import contextmanager

@contextmanager
def managed_chain(chain):
    try:
        yield chain
    finally:
        # Cleanup operations
        if hasattr(chain, 'memory') and chain.memory:
            chain.memory.clear()
```

## Best Practices

### 1. Error Handling

```python
from langchain_core.exceptions import LangChainException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_chain_invoke(chain, input_data, max_retries=3):
    """Invoke chain with error handling and retries."""
    for attempt in range(max_retries):
        try:
            result = chain.invoke(input_data)
            logger.info(f"Chain invocation successful on attempt {attempt + 1}")
            return result
        
        except LangChainException as e:
            logger.error(f"LangChain error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
        
        # Exponential backoff
        time.sleep(2 ** attempt)

# Usage
try:
    result = robust_chain_invoke(chain, {"input": "test"})
except Exception as e:
    logger.error(f"All retry attempts failed: {str(e)}")
    # Fallback behavior
    result = "I apologize, but I'm having trouble processing your request right now."
```

### 2. Testing

```python
import pytest
from unittest.mock import Mock, patch

class TestLangChainComponents:
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock = Mock()
        mock.invoke.return_value = "Mocked response"
        return mock
    
    @pytest.fixture
    def sample_chain(self, mock_llm):
        """Sample chain for testing."""
        prompt = ChatPromptTemplate.from_template("Test: {input}")
        return prompt | mock_llm | StrOutputParser()
    
    def test_chain_invoke(self, sample_chain):
        """Test basic chain invocation."""
        result = sample_chain.invoke({"input": "hello"})
        assert result == "Mocked response"
    
    def test_prompt_formatting(self):
        """Test prompt template formatting."""
        prompt = ChatPromptTemplate.from_template("Hello {name}")
        formatted = prompt.format(name="Alice")
        assert "Alice" in str(formatted)
    
    @patch('langchain_openai.ChatOpenAI')
    def test_with_real_llm_mock(self, mock_openai):
        """Test with mocked OpenAI client."""
        mock_openai.return_value.invoke.return_value = "Test response"
        
        llm = ChatOpenAI()
        result = llm.invoke("test prompt")
        
        assert result == "Test response"
        mock_openai.assert_called_once()

# Run tests with: pytest test_langchain.py
```

### 3. Production Deployment

```python
from langchain_core.runnables import RunnableConfig
import os
from typing import Dict, Any

class ProductionChainWrapper:
    """Production-ready chain wrapper with monitoring and error handling."""
    
    def __init__(self, chain, environment="production"):
        self.chain = chain
        self.environment = environment
        self.request_count = 0
        self.error_count = 0
    
    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke chain with production safeguards."""
        self.request_count += 1
        
        try:
            # Input validation
            self._validate_input(input_data)
            
            # Rate limiting check
            if self.request_count > 1000:  # Example limit
                raise Exception("Rate limit exceeded")
            
            # Invoke chain with timeout
            config = RunnableConfig(
                tags=[self.environment],
                metadata={"request_id": self.request_count}
            )
            
            result = self.chain.invoke(input_data, config=config)
            
            # Output validation
            self._validate_output(result)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Chain error: {str(e)}")
            
            # Fallback response
            return self._get_fallback_response(input_data)
    
    def _validate_input(self, input_data: Dict[str, Any]):
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
        
        if "input" not in input_data:
            raise ValueError("Input must contain 'input' key")
        
        if len(input_data["input"]) > 10000:  # Character limit
            raise ValueError("Input too long")
    
    def _validate_output(self, output: str):
        """Validate output."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")
        
        if len(output) == 0:
            raise ValueError("Empty output")
    
    def _get_fallback_response(self, input_data: Dict[str, Any]) -> str:
        """Provide fallback response on error."""
        return "I apologize, but I'm unable to process your request at the moment."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "environment": self.environment
        }

# Usage
production_chain = ProductionChainWrapper(chain, "production")
result = production_chain.invoke({"input": "Hello world"})
metrics = production_chain.get_metrics()
```

### 4. Monitoring and Observability

```python
from langchain_community.callbacks import LangChainTracer
from langchain.callbacks.manager import tracing_v2_enabled

# LangSmith tracing (if available)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"

# Use tracing context
with tracing_v2_enabled():
    result = chain.invoke({"input": "traced request"})

# Custom monitoring
class MonitoringCallback(BaseCallbackHandler):
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "average_response_time": 0,
            "error_count": 0
        }
        self.start_times = {}
    
    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        self.metrics["total_requests"] += 1
        self.start_times[run_id] = time.time()
    
    def on_chain_end(self, outputs, run_id, **kwargs):
        if run_id in self.start_times:
            duration = time.time() - self.start_times[run_id]
            # Update average response time
            current_avg = self.metrics["average_response_time"]
            new_avg = ((current_avg * (self.metrics["total_requests"] - 1)) + duration) / self.metrics["total_requests"]
            self.metrics["average_response_time"] = new_avg
            del self.start_times[run_id]
    
    def on_chain_error(self, error, run_id, **kwargs):
        self.metrics["error_count"] += 1
        if run_id in self.start_times:
            del self.start_times[run_id]

# Use monitoring
monitoring = MonitoringCallback()
monitored_chain = chain.with_config(callbacks=[monitoring])

# Health check endpoint
def health_check():
    return {
        "status": "healthy",
        "metrics": monitoring.metrics,
        "timestamp": time.time()
    }
```

## Troubleshooting

### Common Issues and Solutions

1. **API Key Issues**
   ```python
   # Check environment variables
   import os
   print("OpenAI API Key:", "SET" if os.getenv("OPENAI_API_KEY") else "NOT SET")
   
   # Set programmatically
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   ```

2. **Memory Issues with Large Documents**
   ```python
   # Use streaming for large documents
   def process_large_document(doc, chunk_size=1000):
       chunks = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)]
       results = []
       
       for chunk in chunks:
           result = chain.invoke({"input": chunk})
           results.append(result)
       
       return results
   ```

3. **Rate Limiting**
   ```python
   import time
   from tenacity import retry, wait_exponential, stop_after_attempt
   
   @retry(
       wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3)
   )
   def invoke_with_retry(chain, input_data):
       return chain.invoke(input_data)
   ```

4. **Token Limits**
   ```python
   import tiktoken
   
   def count_tokens(text, model="gpt-4"):
       encoding = tiktoken.encoding_for_model(model)
       return len(encoding.encode(text))
   
   def truncate_text(text, max_tokens=4000, model="gpt-4"):
       encoding = tiktoken.encoding_for_model(model)
       tokens = encoding.encode(text)
       if len(tokens) > max_tokens:
           truncated_tokens = tokens[:max_tokens]
           return encoding.decode(truncated_tokens)
       return text
   ```

---

*For the latest documentation and updates, visit the [LangChain Documentation](https://python.langchain.com/) and [GitHub Repository](https://github.com/langchain-ai/langchain).*