"""
Orchestrator Agent
Classifies user queries and routes them to appropriate specialized agents.
Manages delegation, prevents duplicate handling, and tracks conversation handoffs.
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

from abc import ABC
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI
import uuid
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Import AgentExecutor and OpenAIFunctionsAgent
# Note: In langchain 0.0.350, we use OpenAIFunctionsAgent.from_llm_and_tools() instead of create_openai_functions_agent()
try:
    from langchain.agents import AgentExecutor
    from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
except ImportError:
    raise ImportError(
        "Could not import AgentExecutor or OpenAIFunctionsAgent. "
        "Please ensure you have a compatible version of LangChain installed (langchain==0.0.350). "
        "Try: uv pip install 'langchain==0.0.350'"
    )

from langchain.memory import ConversationBufferMemory

# Import Langfuse for tracing
from langfuse import Langfuse

from src.agents.rag_agent import RAGAgent
from src.agents.finance_agent import FinanceAgent
from src.agents.hr_agent import HRAgent
from src.agents.tech_agent import TechAgent

# Optional: Import evaluator for quality monitoring
try:
    from src.evaluator import EvaluatorAgent
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    EvaluatorAgent = None

# Load environment variables from .env file
# This ensures the orchestrator can access API keys regardless of where it's executed from
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# ============================================================================
# (2) Tool Wrapper Functions for RAG Agents
# ============================================================================
# Note: These functions are now implemented as methods inside OrchestratorAgent
# to simplify signatures and improve encapsulation.


# ============================================================================
# (3) Orchestrator System Prompt Template
# ============================================================================

def create_orchestrator_prompt() -> ChatPromptTemplate:
    """
    Create the system prompt template for the Orchestrator Agent.
    This prompt guides the LLM on how to select the appropriate specialized agent.
    
    Returns:
        ChatPromptTemplate: Prompt template for the orchestrator
    """
    system_message = """You are an intelligent Orchestrator Agent responsible for routing user queries to the most appropriate specialized agent.

Your task is to analyze the user's question and determine which specialized agent should handle it:

1. **finance_agent**: Use for questions about:
   - Expense policies, reimbursement, and expense reports
   - Tax forms (W-9, 1099, etc.) and tax requirements
   - Budgeting, procurement, and financial compliance
   - Financial policies and procedures
   - Examples: "What is the maximum meal reimbursement?", "What tax form do I need for vendors over $600?"

2. **hr_agent**: Use for questions about:
   - Employee benefits, vacation accrual, and time off
   - Payroll, overtime calculations, and pay stubs
   - Meal breaks, rest breaks, and work schedule policies
   - Compliance reporting and personnel file access
   - Examples: "What is my vacation accrual rate?", "How is overtime calculated?", "What are my meal break requirements?"

3. **tech_agent**: Use for questions about:
   - Network connectivity issues and troubleshooting
   - VPN access, configuration, and file share access
   - Security policies, passwords, and authentication
   - Software procurement, licensing, and technical procedures
   - Step-by-step technical instructions
   - Examples: "I cannot connect to the internet, what should I do?", "How do I access file shares via VPN?"

**Routing Guidelines:**
- Analyze the query carefully to identify the primary domain
- If a query spans multiple domains, choose the most relevant one
- If the query is unclear or doesn't fit any domain, you may need to ask for clarification
- Always use the appropriate tool to get the answer from the specialized agent
- After receiving the agent's response, present it clearly to the user

**Important:**
- You MUST use one of the available tools to answer the user's question
- Do not make up answers - always delegate to the appropriate specialized agent
- Be concise and helpful in your responses"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    return prompt


# ============================================================================
# (4) Orchestrator Agent Class Definition
# ============================================================================
class OrchestratorAgent(ABC):
    def __init__(
        self,
        finance_agent: FinanceAgent,
        hr_agent: HRAgent,
        tech_agent: TechAgent,
        evaluator: Optional[EvaluatorAgent] = None,
        enable_evaluation: bool = False
    ):
        """
        Initialize the Orchestrator Agent with specialized RAG agents.
        
        Args:
            finance_agent: FinanceAgent instance
            hr_agent: HRAgent instance
            tech_agent: TechAgent instance
            evaluator: Optional EvaluatorAgent instance for quality monitoring
            enable_evaluation: Whether to enable automatic evaluation (default: False)
        """
        # Initialize logging first (needed for LLM initialization logging)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        # Ensure .env is loaded (in case it wasn't loaded at module import time)
        # Use override=True to ensure .env values take precedence over shell environment variables
        load_dotenv(dotenv_path=project_root / ".env", override=True)
        
        # Get API key (user sets OPENAI_API_KEY to the appropriate key for their chosen provider)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")
        
        use_open_router = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"
        if use_open_router:
            # Open Router configuration
            self.llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model_name="openai/gpt-4o-mini"
            )
            self.logger.info(f"[ORCHESTRATOR] Using Open Router")
        else:
            # Standard OpenAI API
            self.llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                model_name="gpt-4o-mini"
            )
            self.logger.info(f"[ORCHESTRATOR] Using OpenAI")
        
        # Store specialized agents for tool creation
        self.finance_agent = finance_agent
        self.hr_agent = hr_agent
        self.tech_agent = tech_agent
        
        # Initialize Langfuse tracing (must be before tool creation)
        self.langfuse = langfuse
        
        # Create tools from specialized agents with logging and tracing features
        self.tools = self._create_agent_tools()
        
        # Create orchestrator prompt
        self.prompt = create_orchestrator_prompt()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the agent and executor
        # Note: Open Router doesn't support the deprecated "functions" format used by OpenAIFunctionsAgent
        # When using Open Router, we use STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION instead
        use_open_router = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"
        
        if use_open_router:
            # For Open Router: Use ReAct agent with structured chat (supports tools format, not functions)
            from langchain.agents import initialize_agent, AgentType
            
            # initialize_agent returns an AgentExecutor directly
            # This uses the tools format which Open Router supports
            self.agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=self.memory,
                handle_parsing_errors=True,
                max_iterations=5  # Limit iterations to prevent infinite loops
            )
            # Extract the agent for reference (though we use agent_executor directly)
            self.agent = self.agent_executor.agent if hasattr(self.agent_executor, 'agent') else None
        else:
            # For OpenAI: Use OpenAIFunctionsAgent (supports functions format)
            # In langchain 0.0.350, we use OpenAIFunctionsAgent.from_llm_and_tools()
            # Extract system message from prompt for OpenAIFunctionsAgent
            system_message = None
            if hasattr(self.prompt, 'messages') and len(self.prompt.messages) > 0:
                # Get system message from ChatPromptTemplate
                for msg in self.prompt.messages:
                    if isinstance(msg, tuple) and msg[0] == "system":
                        from langchain_core.messages import SystemMessage
                        system_message = SystemMessage(content=msg[1])
                        break
            
            self.agent = OpenAIFunctionsAgent.from_llm_and_tools(
                llm=self.llm,
                tools=self.tools,
                system_message=system_message
            )
            
            # Create agent executor for OpenAI
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5  # Limit iterations to prevent infinite loops
            )
        
        # Track delegation history (to prevent duplicate handling)
        self.delegation_history = []
        
        # Track handoff history (to prevent ping-pong scenarios)
        # Format: [{"query": normalized_query, "agent": agent_name, "timestamp": ...}, ...]
        self.handoff_history = []
        
        # Maximum number of handoffs allowed for a query (prevents infinite loops)
        self.max_handoff_depth = 3
        
        # Initialize evaluator (optional, for quality monitoring)
        self.evaluator = evaluator
        self.enable_evaluation = enable_evaluation and evaluator is not None
        if self.enable_evaluation:
            self.logger.info("[ORCHESTRATOR] Quality evaluation enabled")

    def _create_agent_tool_definition(self, agent: RAGAgent):
        """
        Create a reusable tool function for any agent with logging and tracing.
        
        This is a factory function that returns a closure (inner function) that will
        be called by LangChain's AgentExecutor when the tool is invoked.
        
        Args:
            agent: RAGAgent instance to process queries
        
        Returns:
            function: A function that takes query: str and returns str (the agent's response)
        """
        def agent_tool(query: str) -> str:
            """
            Process queries using the specified agent with full logging and tracing.
            
            Args:
                query: User's query string
            
            Returns:
                str: Answer from the agent
            """
            # Track agent selection in Langfuse
            # Note: In Langfuse 3.x, use start_span() instead of span()
            tool_span = self.langfuse.start_span(
                name=f"{agent.agent_name}_selected",
                metadata={
                    "agent": agent.agent_name,
                    "query": query[:200],  # Truncate for metadata
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Log agent selection
            self.logger.info(f"[ROUTING DECISION] {agent.agent_name} selected for query: {query[:100]}...")
            
            try:
                # Format query as JSON for agent
                query_data = {
                    "query": query,
                    "context": {},
                    "conversation_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Process query through agent
                # Note: In Langfuse 3.x, spans don't support context manager protocol
                # We'll manually track and end the span
                try:
                    response = agent.process_query(query_data)
                    
                    # Extract answer from response
                    answer = response.get("answer", f"I couldn't process your {agent.agent_name} query.")
                    
                    # Update span with success, then end it
                    # Note: In Langfuse 3.x, use update() to set output and level, then end()
                    tool_span.update(output=answer[:200], level="DEFAULT")
                    tool_span.end()
                    self.logger.info(f"[ROUTING RESULT] {agent.agent_name} completed successfully")
                    
                    return answer
                except Exception as e:
                    error_msg = f"Error processing {agent.agent_name} query: {str(e)}"
                    # Update span with error, then end it
                    tool_span.update(output=error_msg, level="ERROR")
                    tool_span.end()
                    self.logger.error(f"[ROUTING ERROR] {agent.agent_name} failed: {str(e)}")
                    raise  # Re-raise to be caught by outer exception handler
            except Exception as e:
                # Ensure span is ended even if there's an error
                if tool_span:
                    try:
                        tool_span.update(output=f"Error: {str(e)}", level="ERROR")
                        tool_span.end()
                    except:
                        pass  # Ignore errors ending span
                error_msg = f"Error processing {agent.agent_name} query: {str(e)}"
                self.logger.error(f"[ROUTING ERROR] {agent.agent_name} failed: {str(e)}")
                return error_msg
        
        return agent_tool

    def _create_finance_tool_function(self, finance_agent: FinanceAgent):
        """
        Create a tool function wrapper for the Finance Agent with logging and tracing.
        
        Args:
            finance_agent: FinanceAgent instance
        
        Returns:
            function: Tool function that processes finance queries with tracking
        """
        return self._create_agent_tool_definition(finance_agent)

    def _create_hr_tool_function(self, hr_agent: HRAgent):
        """
        Create a tool function wrapper for the HR Agent with logging and tracing.
        
        Args:
            hr_agent: HRAgent instance
        
        Returns:
            function: Tool function that processes HR queries with tracking
        """
        return self._create_agent_tool_definition(hr_agent)

    def _create_tech_tool_function(self, tech_agent: TechAgent):
        """
        Create a tool function wrapper for the Tech Agent with logging and tracing.
        
        Args:
            tech_agent: TechAgent instance
        
        Returns:
            function: Tool function that processes tech queries with tracking
        """
        return self._create_agent_tool_definition(tech_agent)

    def _create_agent_tools(self) -> list[Tool]:
        """
        Create LangChain Tool objects for each specialized agent with logging and tracing.
        
        The tools are used by the Orchestrator Agent to delegate queries to the appropriate
        specialized agent. Each tool invocation is tracked in Langfuse and logged for
        observability.
        
        Returns:
            list[Tool]: List of Tool objects for available agents, ready for use with
                       AgentExecutor and OPENAI_FUNCTIONS agent type
        """
        tools = []
        
        # Create Finance Agent tool with tracking
        finance_tool_func = self._create_finance_tool_function(self.finance_agent)
        finance_tool = Tool(
            name="finance_agent",
            func=finance_tool_func,
            description=(
                "Use this tool to answer questions about finance, expenses, reimbursement, "
                "tax forms, budgeting, procurement, and financial policies. "
                "Examples: 'What is the maximum meal reimbursement?', "
                "'What tax form do I need for vendors over $600?', "
                "'What is the expense report submission deadline?'"
            )
        )
        tools.append(finance_tool)
        
        # Create HR Agent tool with tracking
        hr_tool_func = self._create_hr_tool_function(self.hr_agent)
        hr_tool = Tool(
            name="hr_agent",
            func=hr_tool_func,
            description=(
                "Use this tool to answer questions about HR policies, employee benefits, "
                "vacation accrual, payroll, overtime, meal breaks, compliance, and personnel matters. "
                "Examples: 'What is my vacation accrual rate?', "
                "'How is overtime calculated?', "
                "'What are my meal break requirements?', "
                "'Who should I report pay stub errors to?'"
            )
        )
        tools.append(hr_tool)
        
        # Create Tech Agent tool with tracking
        tech_tool_func = self._create_tech_tool_function(self.tech_agent)
        tech_tool = Tool(
            name="tech_agent",
            func=tech_tool_func,
            description=(
                "Use this tool to answer technical support questions about network connectivity, "
                "VPN access, security policies, passwords, software procurement, and troubleshooting. "
                "Examples: 'I cannot connect to the internet, what should I do?', "
                "'How do I access file shares via VPN?', "
                "'What are the password policy requirements?', "
                "'What ticket category should I use for software procurement?'"
            )
        )
        tools.append(tech_tool)
        
        return tools

    def route_query_old(self, query: str) -> str:
        """Classifies the query into HR or TECH or FINANCE."""
        result = self.llm.invoke(
            f"Classify this query into 'HR' or 'TECH' or 'FINANCE' or 'UNKNOWN'. Respond with only one word:\n{query}",
            config={"run_id": str(uuid.uuid4())},
        )
        category = result.content.strip().lower()
        # return one of: "finance", "hr", "tech", or "unknown"
        out_value = "finance" if "finance" in category else "hr" if "hr" in category else "tech" if "tech" in category else "unknown"
        return out_value.upper()

    def route_query(self, query: str) -> str:
        """
        Route and execute a user query through the appropriate specialized agent.
        
        This method performs the complete routing and execution flow:
        1. **Routing Decision**: The LLM (via AgentExecutor with OPENAI_FUNCTIONS) analyzes
           the query and selects the appropriate specialized agent tool
        2. **Tool Execution**: The selected tool is automatically invoked, which calls the
           corresponding specialized agent's process_query() method
        3. **Response Return**: The agent's response is returned to the user
        
        The entire process is traced in Langfuse with:
        - Top-level trace: "orchestrator_route_query"
        - Agent selection span: "{agent_name}_agent_selected" (created by tool wrapper)
        - Agent execution span: "agent_execution"
        
        Args:
            query: User's query string to route and process
        
        Returns:
            str: Response from the specialized agent that handled the query
        
        Raises:
            No exceptions are raised; errors are caught and returned as user-friendly messages
        
        Example:
            >>> orchestrator = OrchestratorAgent(finance_agent, hr_agent, tech_agent)
            >>> response = orchestrator.route_query("What is my vacation accrual rate?")
            >>> # Langfuse will show: hr_agent_selected span within orchestrator_route_query trace
        """
        run_id = str(uuid.uuid4())
        
        try:
            # Create top-level Langfuse span for this routing operation
            # Note: In Langfuse 3.x, use start_span() instead of trace()
            trace = self.langfuse.start_span(
                name="orchestrator_route_query",
                metadata={
                    "query": query[:200],  # Truncate for metadata
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "available_agents": ["finance_agent", "hr_agent", "tech_agent"]
                }
            )
            
            # Log routing attempt
            self.logger.info(f"[ORCHESTRATOR] Starting query routing - Run ID: {run_id}")
            self.logger.info(f"[ORCHESTRATOR] Query: {query[:100]}...")
            
            # Check for duplicate query
            if self._is_duplicate_query(query):
                self.logger.warning(f"[ORCHESTRATOR] Duplicate query detected: {query[:100]}...")
                return "This query was recently processed. Please wait a moment or rephrase your question."
            
            # Execute routing and agent invocation (atomic operation)
            # The AgentExecutor will:
            # 1. Use LLM to analyze query and select tool
            # 2. Invoke the selected tool (which creates a span in Langfuse)
            # 3. Return the agent's response
            # Note: In Langfuse 3.x, start_as_current_span() returns a context manager
            # Use it with 'with' statement - it automatically handles span lifecycle
            with trace.start_as_current_span(
                name="agent_executor_invoke",
                metadata={
                    "run_id": run_id,
                    "query": query[:200]
                }
            ):
                # process query through agent executor
                result = self.agent_executor.invoke(
                    {"input": query},
                    config={"run_id": run_id}
                )
            
            # Extract the answer from the result
            answer = result.get("output", "I couldn't process your query.")
            
            # Determine which agent was selected by checking the result
            # First try to extract from intermediate steps, then fallback to other methods
            selected_agent = self._extract_selected_agent_from_result(result)
            
            # If still unknown, try deep inspection as fallback
            if selected_agent == "unknown":
                selected_agent = self._extract_selected_agent_from_result(result, use_deep_inspection=True)
                if selected_agent != "unknown":
                    self.logger.info(f"[ORCHESTRATOR] Agent determined via deep inspection: {selected_agent}")
            
            # Check for handoff cycles before tracking
            is_cycle, cycle_reason = self._check_handoff_cycle(query, selected_agent)
            if is_cycle:
                self.logger.warning(f"[ORCHESTRATOR] Cycle detected: {cycle_reason}")
                return f"I detected a potential loop in processing your query. {cycle_reason}. Please rephrase your question or contact support."
            
            # Track the handoff
            self._track_handoff(query, selected_agent, run_id)
            
            # Track delegation in history
            self.delegation_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "selected_agent": selected_agent
            })
            
            # Keep only last 20 delegations to prevent memory growth
            if len(self.delegation_history) > 20:
                self.delegation_history = self.delegation_history[-20:]
            
            # Update trace with routing decision
            trace.update(
                metadata={
                    "selected_agent": selected_agent,
                    "query": query[:200],
                    "run_id": run_id,
                    "response_length": len(answer),
                    "handoff_count": len([h for h in self.handoff_history 
                                         if h.get("query") == self._normalize_query(query)])
                }
            )
            
            # Log successful routing with agent selection
            self.logger.info(f"[ORCHESTRATOR] Query routing completed successfully")
            self.logger.info(f"[ORCHESTRATOR] Selected agent: {selected_agent}")
            
            # Evaluate response quality (if evaluator is enabled)
            if self.enable_evaluation and self.evaluator:
                try:
                    self.logger.info("[ORCHESTRATOR] Running quality evaluation...")
                    evaluation = self.evaluator.evaluate_response(
                        query=query,
                        answer=answer,
                        trace_id=trace.id if hasattr(trace, 'id') else None,
                        metadata={
                            "selected_agent": selected_agent,
                            "run_id": run_id
                        }
                    )
                    
                    # Log evaluation results
                    overall_score = evaluation.get("overall_score", 0)
                    relevance = evaluation.get("relevance", {}).get("score", 0)
                    accuracy = evaluation.get("accuracy", {}).get("score", 0)
                    completeness = evaluation.get("completeness", {}).get("score", 0)
                    is_acceptable = evaluation.get("is_acceptable", False)
                    
                    status = "✓ ACCEPTABLE" if is_acceptable else "✗ LOW QUALITY"
                    eval_message = (
                        f"[ORCHESTRATOR] Evaluation Result: {status} - "
                        f"Overall: {overall_score:.1f}/10 "
                        f"(Relevance: {relevance}, Accuracy: {accuracy}, Completeness: {completeness})"
                    )
                    self.logger.info(eval_message)
                    print(eval_message)  # Also print to console for visibility
                    
                    # Update trace with evaluation results
                    existing_metadata = trace.metadata if hasattr(trace, 'metadata') else {}
                    evaluation_metadata = {
                        "evaluation_overall_score": overall_score,
                        "evaluation_relevance": relevance,
                        "evaluation_accuracy": accuracy,
                        "evaluation_completeness": completeness,
                        "evaluation_acceptable": is_acceptable
                    }
                    trace.update(metadata={**existing_metadata, **evaluation_metadata})
                    
                    # Check if response should be rejected due to low quality
                    if self.evaluator.should_reject_response(evaluation):
                        self.logger.warning(
                            f"[ORCHESTRATOR] Low quality response detected (score: {overall_score:.1f}/10). "
                            f"Consider improving the answer or retrieval."
                        )
                        # Optionally: You could return a modified response or flag here
                        # For now, we log the warning but still return the answer
                    
                except Exception as eval_error:
                    self.logger.warning(f"[ORCHESTRATOR] Evaluation failed: {str(eval_error)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    # Continue even if evaluation fails
            
            # End the trace successfully
            trace.end()
            
            return answer
            
        except Exception as e:
            error_msg = f"Error routing query: {str(e)}"
            self.logger.error(f"[ORCHESTRATOR ERROR] {error_msg}")
            self.logger.exception("Full error traceback:")
            
            # Log error to Langfuse
            # Note: In Langfuse 3.x, use start_span() instead of trace()
            error_trace = self.langfuse.start_span(
                name="orchestrator_route_query_error",
                metadata={
                    "query": query[:200],
                    "error": str(e),
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat()
                },
                level="ERROR"
            )
            error_trace.end()
            
            return f"I encountered an error while processing your query. Please try again or rephrase your question. Error: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        Main entry point for processing queries (alias for route_query).
        
        This method provides a consistent interface with the RAGAgent pattern,
        where all agents have a `process_query()` method. It delegates to
        `route_query()` which handles the actual routing and execution.
        
        Args:
            query: User's query string to route and process
        
        Returns:
            str: Response from the specialized agent that handled the query
        
        Example:
            >>> orchestrator = OrchestratorAgent(finance_agent, hr_agent, tech_agent)
            >>> response = orchestrator.process_query("What is my vacation accrual rate?")
        """
        return self.route_query(query)
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for duplicate detection (simple approach).
        
        Args:
            query: Original query string
        
        Returns:
            str: Normalized query (lowercase, stripped)
        """
        return query.lower().strip()
    
    def _is_duplicate_query(self, query: str) -> bool:
        """
        Check if this query was already processed recently.
        
        Args:
            query: Query string to check
        
        Returns:
            bool: True if duplicate found, False otherwise
        """
        normalized = self._normalize_query(query)
        
        # Check last 10 entries in delegation history
        recent_queries = [self._normalize_query(entry.get("query", "")) 
                         for entry in self.delegation_history[-10:]]
        
        return normalized in recent_queries
    
    def _check_handoff_cycle(self, query: str, agent_name: str) -> Tuple[bool, str]:
        """
        Check if this query+agent combination creates a cycle.
        
        Args:
            query: Query string
            agent_name: Name of the agent being selected
        
        Returns:
            tuple: (is_cycle, reason) - True if cycle detected, False otherwise
        """
        normalized = self._normalize_query(query)
        
        # Count how many times this query has been handled
        query_handoffs = [h for h in self.handoff_history 
                         if h.get("query") == normalized]
        
        # Check 1: Too many handoffs for this query
        if len(query_handoffs) >= self.max_handoff_depth:
            return True, f"Maximum handoff depth ({self.max_handoff_depth}) reached for this query"
        
        # Check 2: Same query going to same agent again (ping-pong)
        if len(query_handoffs) > 0:
            # If the last handoff for this query was to the same agent, it's a ping-pong
            last_agent = query_handoffs[-1].get("agent")
            if last_agent == agent_name:
                return True, f"Query already handled by {agent_name} (ping-pong detected)"
        
        return False, ""
    
    def _track_handoff(self, query: str, agent_name: str, run_id: str):
        """
        Record a handoff in the history.
        
        Args:
            query: Query string
            agent_name: Name of the agent that handled it
            run_id: Unique run identifier
        """
        self.handoff_history.append({
            "query": self._normalize_query(query),
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id
        })
        
        # Keep only last 50 handoffs to prevent memory growth
        if len(self.handoff_history) > 50:
            self.handoff_history = self.handoff_history[-50:]
    
    def _extract_selected_agent_from_result(self, result: Dict[str, Any], use_deep_inspection: bool = False) -> str:
        """
        Extract which agent was selected from the AgentExecutor result.
        
        This is a helper method to identify which tool was invoked by checking
        the intermediate steps in the agent execution result.
        
        Args:
            result: Result dictionary from AgentExecutor.invoke()
            use_deep_inspection: If True, use more aggressive inspection methods
        
        Returns:
            str: Name of the selected agent ("finance_agent", "hr_agent", "tech_agent", or "unknown")
        """
        # Check intermediate steps if available
        if "intermediate_steps" in result and result["intermediate_steps"]:
            for step in result["intermediate_steps"]:
                if len(step) >= 2:
                    # Try multiple ways to extract tool name
                    tool_name = None
                    action_obj = step[0]  # First element is usually the action
                    
                    # Method 1: Direct tool attribute (most common for AgentAction)
                    if hasattr(action_obj, 'tool'):
                        tool_name = action_obj.tool
                    # Method 2: Check if it's an AgentAction with tool_input
                    elif hasattr(action_obj, 'tool_input'):
                        # This is likely an AgentAction - check for tool name
                        if hasattr(action_obj, 'tool'):
                            tool_name = action_obj.tool
                    # Method 3: Check if it's a tuple with (action, observation)
                    elif isinstance(action_obj, tuple) and len(action_obj) > 0:
                        first_item = action_obj[0]
                        if isinstance(first_item, str):
                            tool_name = first_item
                        elif hasattr(first_item, 'tool'):
                            tool_name = first_item.tool
                        elif hasattr(first_item, 'name'):
                            tool_name = first_item.name
                    # Method 4: Check if it's a dict-like object
                    elif isinstance(action_obj, dict):
                        tool_name = action_obj.get('tool') or action_obj.get('name') or action_obj.get('action')
                    # Method 5: Try to get from the action object's name attribute
                    elif hasattr(action_obj, 'name'):
                        tool_name = action_obj.name
                    
                    # Normalize and check tool name
                    if tool_name:
                        tool_name = str(tool_name).strip()
                        # Direct match
                        if tool_name in ["finance_agent", "hr_agent", "tech_agent"]:
                            return tool_name
                        # Partial match (in case of variations)
                        if "finance" in tool_name.lower():
                            return "finance_agent"
                        elif "hr" in tool_name.lower() or "human" in tool_name.lower():
                            return "hr_agent"
                        elif "tech" in tool_name.lower():
                            return "tech_agent"
        
        # Fallback: check if we can infer from the output content
        output = result.get("output", "").lower()
        if "finance" in output or "expense" in output or "reimbursement" in output or "tax" in output or "vendor" in output:
            return "finance_agent"
        elif "hr" in output or "vacation" in output or "payroll" in output or "employee" in output or "benefit" in output or "gift" in output:
            return "hr_agent"
        elif "tech" in output or "network" in output or "vpn" in output or "internet" in output or "computer" in output or "password" in output:
            return "tech_agent"
        
        # Deep inspection: search the entire result structure as string
        if use_deep_inspection:
            result_str = str(result).lower()
            if "finance_agent" in result_str or '"finance_agent"' in result_str:
                return "finance_agent"
            elif "hr_agent" in result_str or '"hr_agent"' in result_str:
                return "hr_agent"
            elif "tech_agent" in result_str or '"tech_agent"' in result_str:
                return "tech_agent"
        
        # Log warning if we couldn't determine the agent
        self.logger.warning(
            "[ORCHESTRATOR] Could not determine selected agent from result. "
            f"Intermediate steps available: {bool(result.get('intermediate_steps'))}"
        )
        
        return "unknown"


    