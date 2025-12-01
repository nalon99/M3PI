"""
HR Agent
Specialized RAG agent for handling HR-related queries.
Inherits from RAGAgent base class and adds HR-specific customization.
"""

# ============================================================================
# (1) Setup & Imports
# ============================================================================
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, Any, Optional
from src.agents.rag_agent import RAGAgent

# ============================================================================
# (2) HR Agent Class Definition
# ============================================================================
class HRAgent(RAGAgent):
    """
    HR-specific RAG agent.
    Handles queries about benefits, vacation, payroll, compliance, etc.
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        llm: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        top_k: int = 5,  # HR queries may need more context for complex policies
        similarity_threshold: float = 0.7,
        langfuse_client: Any = None,
        use_exact_knn: bool = False
    ):
        """
        Initialize HR Agent.
        
        Args:
            vector_store: FAISS vector store containing HR documents
            llm: LangChain LLM instance (required)
            embeddings: Embeddings model (required)
            top_k: Number of documents to retrieve (default: 5 for complex policies)
            similarity_threshold: Minimum similarity score (default: 0.7)
            langfuse_client: Langfuse client for tracing (required)
            use_exact_knn: If True, use exact k-NN search. If False, use approximate nearest neighbors (default: False)
        """
        super().__init__(
            vector_store=vector_store,
            agent_name="hr",
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
        Get HR-specific guidance for the prompt.
        
        Returns:
            str: HR domain instructions and examples
        """
        return """
You are an HR policy expert assistant. Your role is to answer questions about:
- Employee benefits and vacation accrual
- Payroll and overtime calculations
- Meal and rest break requirements
- Compliance and reporting procedures
- Personnel file access
- Employment policies and legal requirements

Key guidelines:
1. For tiered structures (e.g., vacation accrual by years of service), clearly explain which tier applies
2. For conditional logic (e.g., meal break waivers), explain all conditions and exceptions
3. For calculations (e.g., overtime), show the breakdown step-by-step
4. For compliance queries, mention specific reporting procedures and contacts
5. Always cite specific policy sections when available
6. For multi-part queries, address each part systematically

Example queries you handle:
- "What is my vacation accrual rate after 7 years of service?"
- "What are my meal break requirements for an 11-hour shift?"
- "How is overtime calculated for 13 hours in one day?"
- "Who should I report pay stub errors to?"
"""
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create HR-specific prompt template.
        
        Returns:
            PromptTemplate: HR domain prompt
        """
        domain_guidance = self.create_domain_guidance()
        
        template = f"""You are an HR policy expert. Use the following pieces of context from HR documents to answer the question.

If you don't know the answer based on the provided context, say that you don't know. Do not make up information.

Context from HR documents:
{{context}}

Question: {{question}}

Instructions:
{domain_guidance}

Provide a clear, comprehensive answer based solely on the context provided. For complex queries involving calculations or conditional logic, break down your answer step-by-step.

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
        Add HR-specific metadata to response.
        
        Args:
            response: Base response dictionary
        
        Returns:
            Dict: Response with HR-specific metadata
        """
        # Add HR-specific metadata
        response["metadata"]["domain"] = "hr"
        response["metadata"]["query_type"] = self._classify_hr_query_type(
            response["metadata"].get("query", "")
        )
        return response
    
    def _classify_hr_query_type(self, query: str) -> str:
        """
        Classify the type of HR query.
        
        Args:
            query: User query text
        
        Returns:
            str: Query type classification
        """
        query_lower = query.lower()
        if any(term in query_lower for term in ["vacation", "accrual", "carryover", "pto"]):
            return "benefits_vacation"
        elif any(term in query_lower for term in ["overtime", "pay", "payroll", "wage"]):
            return "payroll_overtime"
        elif any(term in query_lower for term in ["break", "meal", "rest", "shift"]):
            return "scheduling_breaks"
        elif any(term in query_lower for term in ["report", "compliance", "error", "file"]):
            return "compliance_reporting"
        else:
            return "general_hr"
