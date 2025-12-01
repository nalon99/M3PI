"""
Multi-Agent System with Orchestrator and Specialized RAG Agents
This is the main entry point for the multi-agent system.
"""

# ============================================================================
# (1) Setup & Imports
# ============================================================================

import sys
import os
from pathlib import Path

# Add project root to Python path to enable absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
)
from langchain_community.document_loaders.pdf import PyPDFLoader
from langfuse import Langfuse
from src.agents.orchestrator import OrchestratorAgent
from src.agents.finance_agent import FinanceAgent
from src.agents.hr_agent import HRAgent
from src.agents.tech_agent import TechAgent
from dotenv import load_dotenv
import json
import os

# Optional: Import evaluator for quality monitoring
try:
    from src.evaluator import EvaluatorAgent
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    EvaluatorAgent = None


# Set up environment variables:

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Vector Search Configuration
# Read USE_EXACT_KNN_SEARCH from environment, default to False (A-NN)
USE_EXACT_KNN_SEARCH = os.getenv("USE_EXACT_KNN_SEARCH", "false").lower() == "true"

# ============================================================================
# (2) Document Loading & Vector Stores
# ============================================================================

def _helper_load_docs(path: str):
    """
    Helper function to load documents from a given path.
    It will detect the file type and load it accordingly.
    Args:
        path: The path to the documents to load.
    Returns:
        A list of documents.
    """
    docs_dir = os.path.dirname(os.path.abspath(path))
    path_end = path.split("/")[-1]
    full_path = os.path.join(docs_dir, path_end) # TODO: check if it works in windows, and with *.txt, *.csv, *.pdf, *.md
    print(f"Loading documents from: {full_path}")
    if path_end.endswith(".txt"):
        specialized_loader = TextLoader
    elif path_end.endswith(".csv"):
        specialized_loader = CSVLoader
    elif path_end.endswith(".pdf"):
        specialized_loader = PyPDFLoader
    elif path_end.endswith(".md"):
        # Use TextLoader for structured markdown files (simpler, no extra dependencies)
        specialized_loader = TextLoader
    else:
        raise ValueError(f"Unsupported file type: {full_path}")

    loader = DirectoryLoader(
                            docs_dir,
                            glob=path_end,  # Pattern to match all .txt files
                            loader_cls=specialized_loader,
                            show_progress=True  # Optional: shows loading progress
                        )
    documents = loader.load()
    # does it make sense to apply a RecursiveCharacterTextSplitter to the documents?
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # TODO: Apply a text splitter to the documents?
    # documents = text_splitter.split_documents(documents)
    return documents


def load_finance_docs():
    # Load all documents from data/finance_docs/
    # return _helper_load_docs("data/finance_docs/*.txt")
    return _helper_load_docs("data/finance_docs/*.csv")


def load_hr_docs():
    # Load all documents from data/hr_docs/
    return _helper_load_docs("data/hr_docs/*.pdf")


def load_tech_docs():
    # Load all documents from data/tech_docs/
    return _helper_load_docs("data/tech_docs/*.md")


def initialize_vector_stores(embeddings: HuggingFaceEmbeddings):
    """
    Initialize vector stores for each domain.
    
    Args:
        embeddings: HuggingFaceEmbeddings instance to use for vectorization
    
    Returns:
        A dictionary with the vector stores for each domain where embeddings are normalized to L2 norm.
    """
    finance_vector_store = FAISS.from_documents(
        load_finance_docs(),
        embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )
    hr_vector_store = FAISS.from_documents(
        load_hr_docs(),
        embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )
    tech_vector_store = FAISS.from_documents(
        load_tech_docs(),
        embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return {
        "finance": finance_vector_store,
        "hr": hr_vector_store,
        "tech": tech_vector_store,
    }


# ============================================================================
# (3) Agent Definitions
# ============================================================================

def initialize_specialized_agents():
    """
    Initialize specialized agents for each domain.
    
    Returns:
        A dictionary with the specialized agents for each domain.
    """
    finance_agent = FinanceAgent(
        vector_store=vector_stores["finance"],
        agent_name="finance",
        llm=llm,
        embeddings=embeddings,
        langfuse_client=langfuse_client,
        use_exact_knn=USE_EXACT_KNN_SEARCH
    )
    print("✓ Finance agent initialized")
    hr_agent = HRAgent(
        vector_store=vector_stores["hr"],
        agent_name="hr",
        llm=llm,
        embeddings=embeddings,
        langfuse_client=langfuse_client,
        use_exact_knn=USE_EXACT_KNN_SEARCH
    )
    print("✓ HR agent initialized")
    
    tech_agent = TechAgent(
        vector_store=vector_stores["tech"],
        agent_name="tech",
        llm=llm,
        embeddings=embeddings,
        langfuse_client=langfuse_client,
        use_exact_knn=USE_EXACT_KNN_SEARCH
    )
    print("✓ Tech agent initialized")

    if USE_EXACT_KNN_SEARCH:
        print("Using exact k-nearest neighbors (K-NN) search for all agents")
    else:
        print("Using approximate nearest neighbors (A-NN) search for all agents")

    # Initialize evaluator (optional, for quality monitoring)
    evaluator = None
    enable_evaluation = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
    
    if enable_evaluation and EVALUATOR_AVAILABLE:
        print("\n[6/7] Initializing evaluator agent...")
        evaluator = EvaluatorAgent(
            llm=llm,
            langfuse_client=langfuse_client,
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "6.0"))
        )
        print("✓ Evaluator agent initialized")
        print(f"  Quality threshold: {evaluator.quality_threshold}/10")
    else:
        print("\n[6/7] Skipping evaluator (optional feature)")
        if not EVALUATOR_AVAILABLE:
            print("  Note: Evaluator module not available")

    return {
        "finance": finance_agent,
        "hr": hr_agent,
        "tech": tech_agent,
        "evaluator": evaluator
    }


# ============================================================================
# (4) Orchestrator & Routing
# ============================================================================
def initialize_orchestrator(agents_dict: dict) -> OrchestratorAgent:
    """
    Initialize the orchestrator agent.
    Args:
        agents_dict: A dictionary with the specialized agents for each domain.
    Returns:
        An OrchestratorAgent instance.
    """
    if agents_dict["evaluator"] is not None:
        enable_evaluation = True
    else:
        enable_evaluation = False
    orchestrator = OrchestratorAgent(
        finance_agent=agents_dict["finance"],
        hr_agent=agents_dict["hr"],
        tech_agent=agents_dict["tech"],
        evaluator=agents_dict["evaluator"],
        enable_evaluation=enable_evaluation
    )
    return orchestrator


# ============================================================================
# (5) Testing & Examples
# ============================================================================
# TODO: Create main function or class to run the system

# TODO: Implement query processing function:
#   - Accept user query
#   - Pass to orchestrator
#   - Handle orchestrator's routing decision
#   - Receive response from specialized agent
#   - Return final answer to user

# TODO: Load test queries from test_queries.json

# TODO: Create test runner that:
#   - Iterates through test queries
#   - Processes each query through the system
#   - Logs results
#   - Optionally runs evaluator for each response (bonus)

# TODO: Add example usage:
#   - Simple query example
#   - Multi-turn conversation example
#   - Error handling example

# ============================================================================
# (6) Langfuse Integration
# ============================================================================
# TODO: Initialize Langfuse client:
#   - Set up Langfuse tracing
#   - Configure callback handlers for LangChain

# TODO: Add tracing to:
#   - Orchestrator classification and routing decisions
#   - Each agent's retrieval and generation steps
#   - End-to-end query processing flow

# TODO: Implement evaluator agent (BONUS):
#   - Create EvaluatorAgent that scores responses (1-10)
#   - Integrate with Langfuse evaluation system
#   - Automatically evaluate each RAG response based on:
#     * Original query
#     * Final answer quality
#     * Relevance, accuracy, completeness

# TODO: Set up Langfuse dashboard configuration:
#   - Project settings
#   - Evaluation metrics
#   - Trace visualization

# ============================================================================
# Main Execution
# ============================================================================
# TODO: Add if __name__ == "__main__": block
# TODO: Initialize all components
# TODO: Run test suite or interactive mode
# TODO: Display results and traces in Langfuse

if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Agent RAG System - Initialization")
    print("=" * 80)
    
    # Initialize Langfuse client
    print("\n[1/6] Initializing Langfuse client...")
    langfuse_client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )
    print("✓ Langfuse client initialized")
    
    # Initialize embeddings
    print("\n[2/6] Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✓ Embeddings initialized")
    
    # Initialize LLM
    print("\n[3/6] Initializing LLM...")
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini"
    )
    print("✓ LLM initialized")
    
    # Initialize vector stores
    print("\n[4/6] Loading documents and initializing vector stores...")
    vector_stores = initialize_vector_stores(embeddings)
    print(f"✓ Vector stores initialized:")
    # Get document counts from docstore
    try:
        finance_count = len(vector_stores['finance'].docstore._dict) if hasattr(vector_stores['finance'], 'docstore') else "unknown"
        hr_count = len(vector_stores['hr'].docstore._dict) if hasattr(vector_stores['hr'], 'docstore') else "unknown"
        tech_count = len(vector_stores['tech'].docstore._dict) if hasattr(vector_stores['tech'], 'docstore') else "unknown"
    except:
        finance_count = hr_count = tech_count = "loaded"
    print(f"  - Finance: {finance_count} documents")
    print(f"  - HR: {hr_count} documents")
    print(f"  - Tech: {tech_count} documents")
    
    # Initialize specialized agents
    print("\n[5/6] Initializing specialized RAG agents...")
    agents_dict = initialize_specialized_agents()

    # Initialize orchestrator
    print("\n[6/6] Initializing orchestrator agent...")
    orchestrator = initialize_orchestrator(agents_dict)
    print("✓ Orchestrator agent initialized")
    
    print("\n" + "=" * 80)
    print("System Ready! Running test queries...")
    print("=" * 80)
    
    # Test queries covering different domains
    test_queries = [
        {
            "query": "What is my vacation accrual rate?",
            "expected_agent": "hr_agent",
            "description": "HR query - vacation benefits"
        },
        {
            "query": "What is the maximum meal reimbursement?",
            "expected_agent": "finance_agent",
            "description": "Finance query - expense policy"
        },
        {
            "query": "I cannot connect to the internet, what should I do?",
            "expected_agent": "tech_agent",
            "description": "Tech query - network troubleshooting"
        }
    ]
    
    # Process each test query
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'=' * 80}")
        print(f"Query: {test['query']}")
        print(f"Expected Agent: {test['expected_agent']}")
        print(f"Description: {test['description']}")
        print(f"\nProcessing...")
        
        try:
            # Process query through orchestrator
            response = orchestrator.process_query(test['query'])
            
            # Display response
            print(f"\n✓ Response received:")
            print(f"{'-' * 80}")
            print(response)
            print(f"{'-' * 80}")
            
            # Note: If evaluator is enabled, quality scores are automatically logged to Langfuse
            
        except Exception as e:
            print(f"\n✗ Error processing query:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("Test queries completed!")
    print("=" * 80)
    print("\n💡 Check Langfuse dashboard for detailed traces and debugging information:")
    print(f"   {LANGFUSE_HOST}")
    print("\n")