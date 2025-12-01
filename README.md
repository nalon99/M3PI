# Multi-Agent RAG System

## Quick Start

### Requirements
- Python 3.13 (compatible with Python 3.8+)
- LangChain 0.0.350 (pinned for AgentExecutor compatibility)
- OpenAI API key or Open Router API key

### Installation
```bash
# Install dependencies
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with:
```bash
# Required: API Key (use OpenAI key or Open Router key depending on USE_OPEN_ROUTER setting)
OPENAI_API_KEY=your_api_key_here

# Required: Provider selection
# Set USE_OPEN_ROUTER=true if using Open Router (set OPENAI_API_KEY to your Open Router API key)
# Set USE_OPEN_ROUTER=false if using OpenAI (set OPENAI_API_KEY to your OpenAI API key)
USE_OPEN_ROUTER=false

# Optional: Langfuse for tracing
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Vector search configuration
USE_EXACT_KNN_SEARCH=false  # Set to true for exact k-NN (slower, 100% accurate)

# Optional: Quality evaluation
ENABLE_EVALUATION=true
QUALITY_THRESHOLD=6.0
```

### Running
```bash
python src/multi_agent_system.py
```

## Document Structure

The system expects documents organized by domain in the `data/` directory:

- **`data/finance_docs/`**: CSV files (e.g., `Finance_Test_Data.csv`)
- **`data/hr_docs/`**: PDF files (e.g., `Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf`)
- **`data/tech_docs/`**: Markdown files (e.g., `Technical_Support_FAQ.md`)

Each domain uses specialized document loaders optimized for the respective file format.

## System Workflow

The multi-agent system follows a streamlined workflow from user query to final response:

### 1. Query Reception
User submits a query → `OrchestratorAgent.process_query()` or `route_query()`

### 2. Orchestrator Processing
The Orchestrator Agent performs several checks and routing steps:

- **Duplicate Detection**: Checks if the query was recently processed (last 10 queries)
- **Cycle Prevention**: Verifies the query hasn't exceeded maximum handoff depth (default: 3)
- **Agent Selection**: Uses LangChain's `AgentExecutor` with `OPENAI_FUNCTIONS` to analyze the query and select the appropriate specialized agent tool
  - The LLM receives a system prompt describing each agent's domain
  - The LLM outputs a function call (e.g., `finance_agent(query="...")`)
  - LangChain automatically invokes the selected tool

### 3. Specialized Agent Processing
The selected RAG agent (Finance, HR, or Tech) processes the query:

- **RetrievalQA Chain**: Uses LangChain's `RetrievalQA` chain which handles both retrieval and generation in a single call
  - **Retrieval**: Searches the domain-specific vector store using cosine similarity
  - **Generation**: Generates an answer using the retrieved context and domain-specific prompt
- **Response Formatting**: Formats the answer with source documents and metadata

### 4. Response Return
The agent's response flows back through the orchestrator to the user, with full tracing in Langfuse.

### Architecture Components

```
User Query
    ↓
OrchestratorAgent (LangChain AgentExecutor + OPENAI_FUNCTIONS)
    ├─ Duplicate/Cycle Checks
    ├─ LLM-based Agent Selection
    └─ Tool Invocation
        ↓
Specialized RAG Agent (Finance/HR/Tech)
    ├─ RetrievalQA Chain (LangChain)
    │   ├─ Vector Store Retrieval (FAISS)
    │   └─ LLM Generation
    └─ Response Formatting
        ↓
Final Response
```

### Key Features

- **Production-Grade Components**: Uses LangChain's `RetrievalQA` chain, `AgentExecutor`, and `Tool` interfaces
- **Single Retrieval**: No duplicate retrieval - the chain handles retrieval and generation efficiently
- **Full Observability**: Every step is traced in Langfuse with spans and metadata for complete debugging
- **Error Handling**: Comprehensive error handling at each stage with user-friendly messages

### Langfuse Tracing for Debugging

The entire workflow is fully traced in Langfuse, enabling engineers to debug issues by inspecting the complete execution path:

**Orchestrator Level:**
- `orchestrator_route_query` - Top-level trace with query, run_id, and available agents
- `agent_executor_invoke` - Span for the agent executor call
- `{agent_name}_selected` - Span showing which agent was chosen (e.g., `finance_agent_selected`)
- Metadata: selected_agent, response_length, handoff_count

**RAG Agent Level:**
- `{agent_name}_retrieval_and_generation` - Span for the RetrievalQA chain invocation
- Metadata: retrieved_docs_count, answer_length, doc_sources, similarity_threshold
- Error tracking: Failed retrievals and generation errors are captured with detailed metadata

**Debugging Capabilities:**
- **Misclassifications**: See which agent was selected and why (via orchestrator trace)
- **Failed Retrievals**: Inspect retrieval span to see if documents were found, count, and sources
- **Incorrect Answers**: Review generation metadata, retrieved context, and answer quality
- **Full Execution Path**: Trace the complete flow from user query → orchestrator → agent → retrieval → generation → response

## Vector Search Methods

This system supports two types of vector search for retrieving relevant documents:

### Approximate Nearest Neighbors (A-NN)

**Default method** - Uses FAISS approximate search algorithms (e.g., HNSW, IVF).

**Characteristics:**
- **Speed**: Very fast, sub-linear time complexity
- **Scalability**: Efficient for large datasets (millions+ vectors)
- **Accuracy**: High-quality approximate results (typically 95-99% accuracy)
- **Use Case**: Production systems with large document collections

**When to use**: Large-scale deployments where speed is critical and slight approximation is acceptable.

### Exact k-Nearest Neighbors (K-NN)

**Optional method** - Uses FAISS `IndexFlatIP` for exact cosine similarity search.

**Characteristics:**
- **Speed**: Linear time complexity (O(n×d))
- **Scalability**: Best for small to medium datasets (<10,000 vectors)
- **Accuracy**: 100% accurate - finds the true k nearest neighbors
- **Use Case**: Small datasets where accuracy is paramount

**When to use**: Small document collections where exact results are required, or for testing/validation.

**Configuration**: Set `use_exact_knn=True` when initializing agents to enable exact k-NN search.

### Cosine Similarity

Both search methods use **cosine similarity** as the distance metric, which is optimal for semantic text search.

**Why Cosine Similarity?**
- **Semantic Focus**: Measures the angle between vectors (direction), not magnitude, capturing meaning rather than document length
- **Text Embeddings**: OpenAI embeddings are normalized, making inner product equivalent to cosine similarity
- **Robustness**: Less sensitive to document length variations
- **Industry Standard**: The preferred metric for NLP and semantic search tasks

**Implementation**: 
- **Vector Normalization**: All document vectors are normalized (L2 norm = 1.0) before indexing
- **Query Normalization**: Query vectors are automatically normalized before search via a custom embedding wrapper
- **Index Type**: Uses FAISS `IndexFlatIP` (Inner Product) which equals cosine similarity for normalized vectors
- **Consistent Behavior**: Both A-NN and exact k-NN use the same normalized vector store, ensuring consistent cosine similarity search

## Orchestrator Features

The Orchestrator Agent manages query routing and includes built-in safeguards:

### Duplicate Query Detection

Prevents processing the same query multiple times by checking recent query history (last 10 queries). Duplicate queries are rejected with a user-friendly message.

### Handoff Chain Tracking

Tracks which agents have handled each query, maintaining a history of agent handoffs for observability and debugging.

### Cycle Prevention

Prevents infinite loops and ping-pong scenarios between agents:

- **Maximum Handoff Depth**: Limits queries to a maximum of 3 handoffs (configurable via `max_handoff_depth`)
- **Ping-Pong Detection**: Detects when a query is routed back to the same agent that previously handled it
- **Automatic Rejection**: Cycles are automatically detected and rejected with clear error messages

**Configuration**: Adjust `max_handoff_depth` in the `OrchestratorAgent.__init__()` method to change the maximum allowed handoffs.

### ChatPromptTemplate for Multi-Turn Conversations

The Orchestrator Agent uses `ChatPromptTemplate` instead of `PromptTemplate` to support:

- **Conversation History**: Enables multi-turn conversations by maintaining context across user interactions via `MessagesPlaceholder` for chat history
- **Agent Scratchpad**: Provides a dedicated space for the agent's reasoning and tool call decisions, essential for `OPENAI_FUNCTIONS` agent type
- **LangChain Agent Integration**: Required by `create_openai_functions_agent`, which expects chat-based prompts with message roles (system, human, AI)
- **Memory Integration**: Works seamlessly with `ConversationBufferMemory`, which stores and retrieves chat messages for context continuity

This design enables the orchestrator to handle follow-up questions, maintain conversation context, and provide coherent multi-turn interactions with users.

## Automated Quality Evaluation (Bonus Feature)

The system includes an optional **Evaluator Agent** that automatically scores each response on quality dimensions to catch low-quality answers and provide continuous quality metrics.

### Evaluation Dimensions

Each response is scored (1-10) on three dimensions:

1. **Relevance**: Does the answer directly address the user's query?
2. **Accuracy**: Is the information factually correct and reliable?
3. **Completeness**: Does the answer fully address all parts of the query?

### Overall Score

The overall score is calculated as a weighted average:
- Relevance: 40%
- Accuracy: 40%
- Completeness: 20%

### Quality Threshold

Responses below the quality threshold (default: 6.0/10) are flagged as low-quality. The system logs warnings for low-quality responses, enabling proactive quality monitoring.

### Langfuse Integration

All evaluations are automatically logged to Langfuse with:
- Individual dimension scores
- Overall quality score
- Evaluation reasoning
- Link to original trace

This enables:
- **Quality Dashboards**: Track quality metrics over time in Langfuse
- **Score Analytics**: Analyze score distributions and trends
- **Continuous Monitoring**: Identify quality issues before customers see them

### Configuration

Set environment variables to control evaluation:
- `ENABLE_EVALUATION=true` - Enable/disable automatic evaluation (default: true)
- `QUALITY_THRESHOLD=6.0` - Minimum acceptable score (default: 6.0)

**Note**: The evaluator is optional. If the `evaluator.py` module is not available, the system will continue to work without evaluation.

## Open Router Support

The system supports Open Router as an alternative to OpenAI API:

**To use Open Router:**
- Set `USE_OPEN_ROUTER=true` in your `.env` file
- Set `OPENAI_API_KEY` to your Open Router API key

**To use OpenAI:**
- Set `USE_OPEN_ROUTER=false` in your `.env` file
- Set `OPENAI_API_KEY` to your OpenAI API key

**Technical Details:**
- Automatically uses `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` agent type when Open Router is enabled (compatible with Open Router's tools format)
- Model format: `openai/gpt-4o-mini` or other Open Router model identifiers
- Open Router requires the newer `tools` format instead of deprecated `functions` format, so the orchestrator automatically switches agent types when Open Router is enabled

## Known Limitations

### File Format Constraints

The system has strict file format requirements for each domain:

- **Finance Documents**: Must be CSV files (`.csv`). Other formats in `data/finance_docs/` will not be processed.
- **HR Documents**: Must be PDF files (`.pdf`). Other formats in `data/hr_docs/` will not be processed.
- **Tech Documents**: Must be Markdown files (`.md`). Other formats in `data/tech_docs/` will not be processed.

**Note**: The `_helper_load_docs()` function uses file extension matching, so files must have the correct extension to be loaded.

### Other Limitations

- **Document Loaders**: Currently supports only specific file types (CSV, PDF, Markdown, Text). Additional formats require extending the loader function.
- **Vector Store**: The system rebuilds vector stores with normalized vectors, which may take time for large document collections.
- **Exact k-NN**: Best suited for small to medium datasets (<10,000 vectors) due to linear time complexity.
- **LangChain Version**: Requires LangChain 0.0.350 (AgentExecutor was removed in 1.0.0+)
- **Python 3.13**: Multiprocessing is disabled to prevent semaphore leaks (set via environment variables)

## Technical Decisions

### LangChain Components

**LangChain 0.0.350**: Pinned to this version because `AgentExecutor` and `OpenAIFunctionsAgent` were removed in 1.0.0+. These components are essential for the orchestrator's function-calling routing mechanism.

**AgentExecutor with max_iterations=5**: Limits agent execution to 5 iterations to prevent infinite loops when the LLM can't decide on a tool. Default was 15, which caused excessive retries.

**RetrievalQA Chain**: Used instead of separate retrieval + generation steps to avoid duplicate retrieval and improve efficiency. The chain handles both operations atomically.

**ChatPromptTemplate**: Chosen over `PromptTemplate` to support conversation history and multi-turn interactions via `MessagesPlaceholder`, required for `OPENAI_FUNCTIONS` agent type.

### Routing Strategy

**OpenAI Functions (OPENAI_FUNCTIONS)**: Used for OpenAI API to leverage native function-calling capabilities. The LLM outputs structured function calls that LangChain executes automatically.

**Structured Chat ReAct (STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)**: Used for Open Router because it supports the newer `tools` format (Open Router doesn't support deprecated `functions` format).

**max_handoff_depth=3**: Prevents agent-to-agent ping-pong scenarios by limiting query handoffs to 3 maximum. This prevents infinite routing loops while allowing legitimate multi-agent collaboration.

### RAG Configuration

**FAISS with Cosine Similarity**: Uses cosine similarity (via normalized embeddings) for semantic search. Normalized embeddings (L2 norm=1) make inner product equivalent to cosine similarity, optimal for text similarity.

**Normalized Embeddings**: All embeddings are normalized to L2 norm=1.0 before indexing, ensuring consistent cosine similarity calculations and eliminating the need for explicit distance strategy configuration.

**Approximate Nearest Neighbors (A-NN) by Default**: Fast and scalable for production. Exact k-NN is available for small datasets where 100% accuracy is required, but A-NN provides 95-99% accuracy with much better performance.

**Retriever with top_k=4-5**: Retrieves 4-5 documents per query, balancing context richness with response quality. HR agent uses 5 for complex policy queries, others use 4.

