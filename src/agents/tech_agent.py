"""
Tech Agent
Specialized RAG agent for handling technical support queries.
Inherits from RAGAgent base class and adds tech-specific customization.
"""

# ============================================================================
# (1) Setup & Imports
# ============================================================================
import sys
from pathlib import Path

# Add project root to Python path to enable absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, Any, Optional
from src.agents.rag_agent import RAGAgent

# ============================================================================
# (2) Tech Agent Class Definition
# ============================================================================
class TechAgent(RAGAgent):
    """
    Tech-specific RAG agent.
    Handles queries about network, VPN, security, software, troubleshooting, etc.
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        llm: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        top_k: int = 4,
        similarity_threshold: float = 0.7,
        langfuse_client: Any = None,
        use_exact_knn: bool = False
    ):
        """
        Initialize Tech Agent.
        
        Args:
            vector_store: FAISS vector store containing tech documents
            llm: LangChain LLM instance (required)
            embeddings: Embeddings model (required)
            top_k: Number of documents to retrieve (default: 4)
            similarity_threshold: Minimum similarity score (default: 0.7)
            langfuse_client: Langfuse client for tracing (required)
            use_exact_knn: If True, use exact k-NN search. If False, use approximate nearest neighbors (default: False)
        """
        super().__init__(
            vector_store=vector_store,
            agent_name="tech",
            llm=llm,
            embeddings=embeddings,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            langfuse_client=langfuse_client,
            use_exact_knn=use_exact_knn
        )
    
    # ============================================================================
    # (3) Domain-Specific Prompt Template
    # ============================================================================
    
    def create_domain_guidance(self) -> str:
        """
        Create tech-specific guidance for the prompt.
        
        Returns:
            str: Tech domain instructions and examples
        """
        return """
You are a technical support expert assistant. Your role is to answer questions about:
- Network connectivity and troubleshooting
- VPN access and configuration
- Security policies (passwords, account lockouts)
- Software procurement and licensing
- Technical procedures and step-by-step instructions

Key guidelines:
1. For troubleshooting queries, provide clear, sequential steps
2. Use technical terminology accurately (e.g., "FQDN", "Active Directory")
3. For multi-step procedures, number each step clearly
4. Include specific naming conventions, formats, or requirements when mentioned
5. For policy queries (e.g., password requirements), state exact numeric values
6. For troubleshooting, list steps in the order they should be performed

Example queries you handle:
- "I cannot access the internet. What are the troubleshooting steps?"
- "How do I access a file share via VPN?"
- "What are the password policy requirements?"
- "What ticket category should I use for software procurement?"
"""
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create tech-specific prompt template.
        
        Returns:
            PromptTemplate: Tech domain prompt
        """
        domain_guidance = self.create_domain_guidance()
        
        template = f"""You are a technical support expert. Use the following pieces of context from technical documentation to answer the question.

If you don't know the answer based on the provided context, say that you don't know. Do not make up information.

Context from technical documentation:
{{context}}

Question: {{question}}

Instructions:
{domain_guidance}

Provide a clear, step-by-step answer based solely on the context provided. For troubleshooting queries, format your answer as numbered steps. For policy queries, include exact values and requirements.

Answer (in JSON format with 'answer' field):"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    # ============================================================================
    # (4) Domain-Specific Customization (Optional)
    # ============================================================================
    
    def _add_domain_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add tech-specific metadata to response.
        
        Args:
            response: Base response dictionary
        
        Returns:
            Dict: Response with tech-specific metadata
        """
        # Add tech-specific metadata
        response["metadata"]["domain"] = "tech"
        response["metadata"]["query_type"] = self._classify_tech_query_type(
            response["metadata"].get("query", "")
        )
        return response
    
    def _classify_tech_query_type(self, query: str) -> str:
        """
        Classify the type of tech query.
        
        Args:
            query: User query text
        
        Returns:
            str: Query type classification
        """
        query_lower = query.lower()
        if any(term in query_lower for term in ["network", "internet", "connect", "vpn"]):
            return "network_connectivity"
        elif any(term in query_lower for term in ["password", "security", "lockout", "account"]):
            return "security_account"
        elif any(term in query_lower for term in ["software", "license", "procurement", "ticket"]):
            return "software_procurement"
        elif any(term in query_lower for term in ["troubleshoot", "fix", "error", "problem"]):
            return "troubleshooting"
        else:
            return "general_tech"
