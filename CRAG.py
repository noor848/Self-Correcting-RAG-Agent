"""
Self-Correcting RAG Agent with LangGraph
A production-ready RAG system with intelligent document grading and query rewriting.
"""

import os
from typing import List, Dict
from typing_extensions import TypedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_URL = "https://raw.githubusercontent.com/KxSystems/kdbai-samples/main/retrieval_augmented_generation/data/state_of_the_union.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_REWRITE_ATTEMPTS = 2


# ============================================================================
# STATE DEFINITION
# ============================================================================

class GraphState(TypedDict):
    """Represents the state of our graph.
    
    Attributes:
        question: The user's question
        documents: The list of retrieved documents
        generation: The LLM's final answer
        question_history: A list of all questions asked, to prevent loops
    """
    question: str
    documents: List[str]
    generation: str
    question_history: List[str]


# ============================================================================
# SETUP: LOAD DATA AND BUILD RETRIEVER
# ============================================================================

def setup_retriever():
    """Load documents, create embeddings, and build the retriever."""
    print("Loading and preparing knowledge base...")
    
    # Load documents
    loader = WebBaseLoader(web_paths=(DATA_URL,))
    docs = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    
    # Define retriever
    retriever = vectorstore.as_retriever()
    
    print(f"✅ Knowledge base ready. Split into {len(splits)} chunks.")
    return retriever


# ============================================================================
# SETUP: LOAD LLM
# ============================================================================

def setup_llm():
    """Load the generator LLM with 4-bit quantization."""
    print(f"Loading Generator LLM ({LLM_MODEL})...")
    
    # Define 4-bit config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print(f"✅ Generator LLM loaded.")
    return llm


# ============================================================================
# GRAPH NODES
# ============================================================================

def retrieve_node(state: GraphState, retriever) -> Dict:
    """Retrieve documents based on the question."""
    print("---NODE: RETRIEVE---")
    question = state["question"]
    
    # Retrieve documents
    documents = retriever.invoke(question)
    doc_texts = [doc.page_content for doc in documents]
    
    return {
        "documents": doc_texts,
        "question": question,
        "question_history": state.get("question_history", [])
    }


def grade_documents_node(state: GraphState, llm) -> Dict:
    """Grade retrieved documents for relevance."""
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    # Grading prompt
    prompt_template = """You are a grader. Your job is to check if a retrieved document is relevant to a user question.
Respond with a *single word*: 'yes' if relevant, 'no' if not.

Document: {document}
Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    grader_chain = prompt | llm | StrOutputParser()
    
    # Grade each document
    relevant_docs = []
    for doc in documents:
        try:
            result = grader_chain.invoke({"question": question, "document": doc})
            score = result.strip().lower()
            if "yes" in score:
                print("  -> Document: Relevant")
                relevant_docs.append(doc)
            else:
                print("  -> Document: NOT Relevant")
        except Exception as e:
            print(f"  -> Grader Error: {e}")
            continue
    
    print(f"  -> Kept {len(relevant_docs)}/{len(documents)} documents")
    
    return {
        "documents": relevant_docs,
        "question": question,
        "question_history": state.get("question_history", [])
    }


def generate_node(state: GraphState, llm) -> Dict:
    """Generate an answer from the filtered documents."""
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG prompt
    prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Be concise.

Question: {question}
Context: {context}

Helpful Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    rag_chain = prompt | llm | StrOutputParser()
    
    # Generate answer
    generation = rag_chain.invoke({
        "context": "\n".join(documents),
        "question": question
    })
    
    print("  -> Answer generated")
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "question_history": state.get("question_history", [])
    }


def rewrite_node(state: GraphState, llm) -> Dict:
    """Rewrite the query for better retrieval, with loop prevention."""
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    question_history = state.get("question_history", [])
    
    # Loop prevention
    if question in question_history:
        print("  -> Loop detected. Ending.")
        return {
            "documents": [],
            "generation": "Could not find relevant information after multiple attempts.",
            "question_history": question_history,
            "question": question
        }
    
    # Add to history
    question_history.append(question)
    
    # Rewrite prompt
    prompt_template = """You are a query rewriter. Rewrite the following question to be a concise and specific search query for a vector database.
Respond ONLY with the rewritten query, nothing else.

Original Question: {question}

Rewritten Query:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    rewrite_chain = prompt | llm | StrOutputParser()
    
    # Generate new question
    new_question = rewrite_chain.invoke({"question": question}).strip()
    print(f"  -> Rewritten: '{new_question}'")
    
    return {
        "question": new_question,
        "question_history": question_history,
        "documents": []
    }


# ============================================================================
# CONDITIONAL EDGE
# ============================================================================

def decide_to_generate_or_rewrite(state: GraphState) -> str:
    """Decide whether to generate an answer, rewrite query, or end."""
    print("---CONDITIONAL EDGE---")
    documents = state["documents"]
    question_history = state.get("question_history", [])
    generation = state.get("generation", "")
    
    # If we already generated (from loop prevention), end
    if generation:
        print("  -> Decision: END (generation exists)")
        return "end"
    
    # If we have relevant documents, generate
    if documents:
        print("  -> Decision: GENERATE (found relevant docs)")
        return "generate"
    
    # If we've tried too many times, give up
    if len(question_history) > MAX_REWRITE_ATTEMPTS:
        print("  -> Decision: END (max attempts reached)")
        return "end"
    
    # Otherwise, rewrite the query
    print("  -> Decision: REWRITE (no relevant docs)")
    return "rewrite"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_graph(retriever, llm):
    """Build and compile the LangGraph workflow."""
    print("Building the graph...")
    
    # Create workflow
    workflow = StateGraph(GraphState)
    
    # Add nodes with dependencies
    workflow.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents_node(state, llm))
    workflow.add_node("generate", lambda state: generate_node(state, llm))
    workflow.add_node("rewrite", lambda state: rewrite_node(state, llm))
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite", "retrieve")  # The loop!
    workflow.add_edge("generate", END)
    
    # Add conditional edge
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_or_rewrite,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "end": END
        }
    )
    
    # Compile
    app = workflow.compile()
    print("✅ Graph compiled successfully!")
    
    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(app, query: str):
    """Run the agent with a given query."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    # Initial state
    inputs = {
        "question": query,
        "documents": [],
        "generation": "",
        "question_history": []
    }
    
    # Stream outputs
    final_answer = None
    for output in app.stream(inputs):
        for key, value in output.items():
            if key == "generate":
                final_answer = value.get("generation", "")
    
    # Print result
    print(f"\n{'='*60}")
    print(f"🤖 FINAL ANSWER:")
    print(f"{'='*60}")
    print(final_answer)
    print(f"{'='*60}\n")
    
    return final_answer


if __name__ == "__main__":
    # Setup
    retriever = setup_retriever()
    llm = setup_llm()
    app = build_graph(retriever, llm)
    
    # Test 1: Good query
    run_agent(app, "What did the President say about the PRO Act?")
    
    # Test 2: Vague query (should trigger rewrite)
    run_agent(app, "What about infrastructure?")
