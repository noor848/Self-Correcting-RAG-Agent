# Self-Correcting RAG Agent

An AI agent that retrieves documents, checks their quality, and rewrites queries when needed.

## What It Does

Most AI chatbots use whatever documents they find, even if they're not relevant. This agent:

1. **Retrieves** documents from a vector database
2. **Grades** each document for relevance
3. **Decides** whether to answer or search again
4. **Rewrites** the query if results are poor
5. **Prevents loops** by tracking what it's tried

## Quick Start

```bash
# Install dependencies
pip install langchain langgraph langchain-community langchain-huggingface faiss-cpu transformers torch bitsandbytes

# Run the agent
python rag_agent.py
```

## How It Works

```
User Question
    ↓
Retrieve Documents
    ↓
Grade Relevance
    ↓
Good Results? → Generate Answer ✓
    ↓
Bad Results? → Rewrite Query → Try Again
```

## Example

**Vague Question:** "What about infrastructure?"
- Agent realizes this is too vague
- Rewrites to: "infrastructure spending federal investment"
- Gets better results
- Generates accurate answer

## Built With

- **LangGraph** - Agent orchestration
- **LangChain** - RAG pipeline
- **FAISS** - Vector search
- **DeepSeek Coder** - Language model (4-bit quantized)
