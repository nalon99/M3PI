"""
Finance Agent
Specialized RAG agent for handling finance-related queries.
Inherits from RAGAgent base class and adds finance-specific customization.
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
# (2) Finance Agent Class Definition
# ============================================================================
class FinanceAgent(RAGAgent):
    """
    Finance-specific RAG agent.
    Handles queries about expense policies, tax forms, reimbursement, etc.
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
        Initialize Finance Agent.
        
        Args:
            vector_store: FAISS vector store containing finance documents
            llm: LangChain LLM instance (required)
            embeddings: Embeddings model (required)
            top_k: Number of documents to retrieve (default: 4)
            similarity_threshold: Minimum similarity score (default: 0.7)
            langfuse_client: Langfuse client for tracing (required)
            use_exact_knn: If True, use exact k-NN search. If False, use approximate nearest neighbors (default: False)
        """
        super().__init__(
            vector_store=vector_store,
            agent_name="finance",
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
        Get finance-specific guidance for the prompt.
        
        Returns:
            str: Finance domain instructions and examples
        """
        return """
You are a finance policy expert assistant. Your role is to answer questions about:
- Expense policies and reimbursement procedures
- Tax forms and requirements (e.g., Form 1099-NEC)
- Reimbursement amounts and thresholds
- Vendor payment policies
- Compliance and gift policies

Key guidelines:
1. Always provide exact numeric values when available (e.g., "$75.00", "$600 threshold")
2. Cite specific policy names and document sources
3. For tax-related queries, mention the exact form name and number
4. For reimbursement queries, include specific dates and procedures
5. If a query involves a threshold (e.g., "$600"), clearly state what happens above and below that threshold
6. Be precise with monetary amounts and dates

Example queries you handle:
- "What is the maximum reimbursement amount for meals in high-cost cities?"
- "What tax form is needed for vendors over $600?"
- "What is the policy for gifts from vendors valued at $150?"
"""
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create finance-specific prompt template.
        
        Returns:
            PromptTemplate: Finance domain prompt
        """
        domain_guidance = self.create_domain_guidance()
        
        template = f"""You are a finance policy expert. Use the following pieces of context from finance documents to answer the question.

If you don't know the answer based on the provided context, say that you don't know. Do not make up information.

Context from finance documents:
{{context}}

Question: {{question}}

Instructions:
{domain_guidance}

Provide a clear, accurate answer based solely on the context provided. Include specific numbers, dates, and policy names when available.

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
        Add finance-specific metadata to response.
        
        Args:
            response: Base response dictionary
        
        Returns:
            Dict: Response with finance-specific metadata
        """
        # Add finance-specific metadata
        response["metadata"]["domain"] = "finance"
        response["metadata"]["query_type"] = self._classify_finance_query_type(
            response["metadata"].get("query", "")
        )
        return response
    
    def _classify_finance_query_type(self, query: str) -> str:
        """
        Classify the type of finance query.
        
        Args:
            query: User query text
        
        Returns:
            str: Query type classification
        """
        query_lower = query.lower()
        if any(term in query_lower for term in ["tax", "1099", "form"]):
            return "tax_compliance"
        elif any(term in query_lower for term in ["reimbursement", "expense", "reimburse"]):
            return "expense_policy"
        elif any(term in query_lower for term in ["vendor", "payment", "gift"]):
            return "vendor_policy"
        else:
            return "general_finance"
