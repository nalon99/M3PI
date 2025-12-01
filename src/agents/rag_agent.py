"""
Base RAG Agent
Base class for all specialized RAG agents with shared functionality.
Contains common RAG logic: retrieval, generation, formatting, and error handling.
"""

# ============================================================================
# (1) Setup & Imports
# ============================================================================
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
# Import HuggingFaceEmbeddings - use direct module import to avoid __init__.py issues
import importlib.util
import sys
import os

# Find langchain package path
langchain_path = None
for p in sys.path:
    test_path = os.path.join(p, 'langchain', 'embeddings', 'huggingface.py')
    if os.path.exists(test_path):
        langchain_path = p
        break

if langchain_path:
    # Direct import from file to bypass __init__.py
    spec = importlib.util.spec_from_file_location(
        "huggingface_embeddings",
        os.path.join(langchain_path, 'langchain', 'embeddings', 'huggingface.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    HuggingFaceEmbeddings = module.HuggingFaceEmbeddings
else:
    # Fallback: try normal import
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("Could not import HuggingFaceEmbeddings. Please install langchain or langchain-huggingface.")
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langfuse import observe
# langfuse_context may not be available in this version, will handle if needed
from typing import Dict, List, Optional, Any, Literal
import json
import logging
from datetime import datetime
import faiss

# ============================================================================
# (2) Base RAG Agent Class Definition
# ============================================================================
class RAGAgent(ABC):
    """
    Base class for specialized RAG agents.
    Provides common functionality for retrieval, generation, and response formatting.
    Subclasses should override domain-specific methods like create_prompt_template().
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        agent_name: str,
        llm: ChatOpenAI,
        embeddings: HuggingFaceEmbeddings,
        top_k: int = 4,
        similarity_threshold: float = 0.7,
        langfuse_client: Any = None,
        use_exact_knn: bool = False
    ):
        """
        Initialize the base RAG agent.
        
        Args:
            vector_store: FAISS vector store containing domain documents and uses DistanceStrategy.COSINE for cosine similarity search with normalized embeddings
            agent_name: Name identifier for this agent (e.g., "finance", "hr", "tech")
            llm: LangChain LLM instance (required)
            embeddings: Embeddings model (required). Embeddings must be normalized (set normalize_embeddings=True in the HuggingFaceEmbeddings object)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score for retrieval
            langfuse_client: Langfuse client for tracing (required)
            use_exact_knn: If True, use exact k-NN search. If False, use approximate nearest neighbors (default: False)
        """
        self.vector_store = vector_store
        self.agent_name = agent_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_exact_knn = use_exact_knn
        
        # Initialize LLM (required)
        self.llm = llm
        
        # Initialize embeddings (required)
        self.embeddings = embeddings
        
        # Initialize Langfuse client (required)
        self.langfuse_client = langfuse_client
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_name}")
        
        # Create retriever based on search type
        # Note: Embeddings are already normalized (normalize_embeddings=True) and 
        # vector store uses DistanceStrategy.COSINE, so cosine similarity is already configured
        if use_exact_knn:
            # Use exact k-NN search with IndexFlatIP
            self.retriever = self._create_exact_knn_retriever()
            self.logger.info(f"Using exact k-NN search with cosine similarity for {self.agent_name} agent")
        else:
            # Use approximate nearest neighbors (A-NN) with cosine similarity
            # Vector store already configured with DistanceStrategy.COSINE and normalized embeddings
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
            self.logger.info(f"Using A-NN search with cosine similarity for {self.agent_name} agent")
        
        # Get domain-specific prompt template
        self.prompt_template = self.create_prompt_template()
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    # ============================================================================
    # (3) Abstract Methods - To be implemented by subclasses
    # ============================================================================
    
    @abstractmethod
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create domain-specific prompt template.
        Must be implemented by each subclass.
        
        Returns:
            PromptTemplate: Domain-specific prompt for the agent
        """
        pass
    
    @abstractmethod
    def create_domain_guidance(self) -> str:
        """
        Create domain-specific guidance for the prompt.
        Must be implemented by each subclass.
        
        Returns:
            str: Domain-specific instructions and examples
        """
        pass
    
    # ============================================================================
    # (3.5) Exact k-NN Retriever Creation (Simplified)
    # ============================================================================
    
    def _create_exact_knn_retriever(self) -> BaseRetriever:
        """
        Create a retriever that uses exact k-NN search with IndexFlatIP.
        Since embeddings are already normalized, we only need to rebuild the index.
        
        Returns:
            BaseRetriever: Retriever configured for exact k-NN search with cosine similarity
        """
        import numpy as np
        
        # Check if index is already IndexFlatIP (exact search)
        if isinstance(self.vector_store.index, faiss.IndexFlatIP):
            # Already using exact index, no need to rebuild
            self.logger.info(f"Index is already IndexFlatIP, using existing index for {self.agent_name} agent")
            return self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
        
        # Need to rebuild with IndexFlatIP for exact search
        self.logger.info(f"Rebuilding index with IndexFlatIP for exact k-NN search ({self.agent_name} agent)")
        
        # Get embedding dimension
        try:
            test_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(test_embedding)
        except:
            embedding_dim = 1536
        
        # Get all documents from the existing vector store
        docstore = self.vector_store.docstore
        all_embeddings = []
        doc_ids = []
        
        # Extract documents and their embeddings (already normalized)
        if hasattr(self.vector_store, 'index_to_docstore_id') and self.vector_store.index_to_docstore_id:
            for idx, doc_id in self.vector_store.index_to_docstore_id.items():
                doc = docstore._dict.get(doc_id)
                if doc is not None:
                    doc_ids.append(doc_id)
                    try:
                        if hasattr(self.vector_store.index, 'reconstruct'):
                            # Reconstruct embedding from existing index (already normalized)
                            doc_embedding = self.vector_store.index.reconstruct(int(idx))
                            all_embeddings.append(doc_embedding)
                        else:
                            # Fallback: re-embed (will be normalized by HuggingFaceEmbeddings)
                            all_embeddings.append(self.embeddings.embed_query(doc.page_content))
                    except:
                        # Fallback: re-embed (will be normalized by HuggingFaceEmbeddings)
                        all_embeddings.append(self.embeddings.embed_query(doc.page_content))
        else:
            # Fallback: iterate through docstore directly
            for doc_id, doc in docstore._dict.items():
                doc_ids.append(doc_id)
                # Re-embed (will be normalized by HuggingFaceEmbeddings)
                all_embeddings.append(self.embeddings.embed_query(doc.page_content))
        
        # Convert to numpy array (embeddings are already normalized)
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index with IndexFlatIP for exact cosine similarity search
        # Since embeddings are already normalized, inner product = cosine similarity
        exact_index = faiss.IndexFlatIP(embedding_dim)
        exact_index.add(embeddings_array)
        
        # Create mapping from index position to docstore_id
        index_to_docstore_id = {i: doc_ids[i] for i in range(len(doc_ids))}
        
        # Create new FAISS vector store with exact index
        # Use original embeddings (already normalized) - no need for wrapper
        exact_vector_store = FAISS(
            embedding_function=self.embeddings,  # Already normalized
            index=exact_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        # Update the vector store reference
        self.vector_store = exact_vector_store
        
        # Create retriever from the exact index vector store
        return exact_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
    
    # ============================================================================
    # (4) Query Processing - Common Implementation
    # ============================================================================
    
    @observe()
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query from the orchestrator using RetrievalQA chain.
        The chain handles both retrieval and generation in a single call.
        Full execution path is traced in Langfuse for debugging.
        
        Args:
            query_data: JSON formatted query from orchestrator
                {
                    "query": "user query text",
                    "context": {...},
                    "conversation_id": "...",
                    "timestamp": "..."
                }
        
        Returns:
            Dict: Formatted response in JSON format
        """
        try:
            # Extract query text
            query_text = query_data.get("query", "")
            if not query_text:
                return self._create_error_response("Empty query received")
            
            # Log query processing start
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            # Create span for retrieval and generation (RetrievalQA does both)
            # Note: In Langfuse 3.x, use start_span() instead of span()
            retrieval_span = self.langfuse_client.start_span(
                name=f"{self.agent_name}_retrieval_and_generation",
                metadata={
                    "agent": self.agent_name,
                    "query": query_text[:200],
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold
                }
            )
            
            try:
                # Use RetrievalQA chain - handles both retrieval and generation
                # This avoids duplicate retrieval (chain already does retrieval internally)
                # Note: In Langfuse 3.x, spans don't support context manager protocol
                # We'll manually track and end the span
                result = self.qa_chain.invoke({"query": query_text})
                
                # Extract answer and source documents from chain result
                answer = result.get("result", "")
                retrieved_docs = result.get("source_documents", [])
                
                # Check if we got results
                if not answer:
                    # Note: In Langfuse 3.x, use update() to set output and level, then end()
                    retrieval_span.update(
                        output="No answer generated",
                        level="ERROR",
                        metadata={"error": "Empty answer from chain"}
                    )
                    retrieval_span.end()
                    return self.handle_retrieval_error(query_text)
                
                if not retrieved_docs:
                    self.logger.warning(f"No documents retrieved for query: {query_text[:100]}...")
                    retrieval_span.update(
                        output=answer[:200],
                        level="WARNING",
                        metadata={"warning": "No documents retrieved", "answer_length": len(answer)}
                    )
                    retrieval_span.end()
                else:
                    # Log successful retrieval and generation
                    self.logger.info(f"Retrieved {len(retrieved_docs)} documents for {self.agent_name} agent")
                    self.logger.info(f"Generated answer for {self.agent_name} agent")
                    
                    # Extract document metadata for debugging
                    doc_sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs[:3]]
                    
                    # Update span with success details, then end it
                    retrieval_span.update(
                        output=answer[:200],
                        level="DEFAULT",
                        metadata={
                            "retrieved_docs_count": len(retrieved_docs),
                            "answer_length": len(answer),
                            "doc_sources": doc_sources,
                            "agent": self.agent_name
                        }
                    )
                    retrieval_span.end()
                
            except Exception as chain_error:
                # Log chain invocation error
                retrieval_span.update(
                    output=f"Error: {str(chain_error)}",
                    level="ERROR",
                    metadata={"error": str(chain_error), "agent": self.agent_name}
                )
                retrieval_span.end()
                raise
            
            # Update trace with overall metadata
            # Note: In Langfuse 3.x, use langfuse_client.update_current_trace() instead of langfuse_context
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
            if self.langfuse_client:
                self.langfuse_client.update_current_trace(
                    metadata={
                        "retrieved_docs_count": len(retrieved_docs),
                        "answer_length": len(answer),
                        "context_length": len(context_text),
                        "agent": self.agent_name,
                        "query": query_text[:200]
                    }
                )
            
            # Format response
            response = self.format_response(
                answer=answer,
                sources=retrieved_docs,
                query=query_text
            )
            
            # Log success
            self.logger.info(f"Successfully processed query for {self.agent_name} agent")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self.handle_generation_error(str(e))
    
    # ============================================================================
    # (5) Response Formatting - Common Implementation
    # ============================================================================
    
    def format_response(
        self,
        answer: str,
        sources: List[Any],
        query: str,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Format response in agreed JSON structure.
        Common implementation with optional domain-specific customization.
        
        Args:
            answer: Generated answer text
            sources: Retrieved source documents
            query: Original query
            confidence: Optional confidence score
        
        Returns:
            Dict: Formatted JSON response
        """
        # Extract source metadata
        source_list = []
        for doc in sources:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            source_list.append(source_info)
        
        # Calculate confidence if not provided
        if confidence is None:
            confidence = self._calculate_confidence(sources)
        
        # Base response structure
        response = {
            "answer": answer,
            "sources": source_list,
            "confidence": confidence,
            "agent": self.agent_name,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "retrieved_docs_count": len(sources)
            }
        }
        
        # Allow subclasses to add domain-specific metadata
        response = self._add_domain_metadata(response)
        
        return response
    
    def _calculate_confidence(self, sources: List[Any]) -> float:
        """
        Calculate confidence score based on retrieved sources.
        Can be overridden by subclasses for domain-specific logic.
        
        Args:
            sources: Retrieved documents
        
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        # Simple heuristic: more sources = higher confidence
        # Subclasses can override with more sophisticated logic
        if not sources:
            return 0.0
        return min(1.0, len(sources) / self.top_k)
    
    def _add_domain_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add domain-specific metadata to response.
        Can be overridden by subclasses.
        
        Args:
            response: Base response dictionary
        
        Returns:
            Dict: Response with domain-specific metadata added
        """
        # Base implementation does nothing
        # Subclasses can override to add domain-specific fields
        return response
    
    # ============================================================================
    # (6) Error Handling - Common Implementation
    # ============================================================================
    
    def handle_retrieval_error(self, query: str) -> Dict[str, Any]:
        """
        Handle cases where no relevant documents are found.
        Common implementation for all agents.
        
        Args:
            query: Original query text
        
        Returns:
            Dict: Error response in JSON format
        """
        error_message = f"I couldn't find relevant information in the {self.agent_name} documentation to answer your query: '{query}'. Please try rephrasing your question or contact support for assistance."
        
        self.logger.warning(f"No documents retrieved for query: {query}")
        
        return {
            "answer": error_message,
            "sources": [],
            "confidence": 0.0,
            "agent": self.agent_name,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "error": "retrieval_error",
                "error_type": "no_documents_found"
            }
        }
    
    def handle_generation_error(self, error_message: str) -> Dict[str, Any]:
        """
        Handle LLM generation errors.
        Common implementation for all agents.
        
        Args:
            error_message: Error description
        
        Returns:
            Dict: Error response in JSON format
        """
        fallback_message = f"I encountered an error while processing your request. Please try again later or contact support if the issue persists."
        
        self.logger.error(f"Generation error: {error_message}")
        
        return {
            "answer": fallback_message,
            "sources": [],
            "confidence": 0.0,
            "agent": self.agent_name,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "error": "generation_error",
                "error_message": error_message
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message: Error description
        
        Returns:
            Dict: Error response in JSON format
        """
        return {
            "answer": f"Error: {error_message}",
            "sources": [],
            "confidence": 0.0,
            "agent": self.agent_name,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "error": "processing_error",
                "error_message": error_message
            }
        }

