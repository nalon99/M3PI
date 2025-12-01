"""
Evaluator Agent (BONUS)
Automatically evaluates RAG responses and assigns quality scores (1-10).
Integrated with Langfuse for automatic evaluation and quality monitoring.
"""

# ============================================================================
# (1) Setup & Imports
# ============================================================================
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse


# ============================================================================
# (2) Evaluation Prompt Template
# ============================================================================

def create_evaluation_prompt() -> PromptTemplate:
    """
    Create the evaluation prompt template for scoring responses.
    
    Returns:
        PromptTemplate: Prompt template for evaluation
    """
    evaluation_prompt = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Evaluate the following response to a user query based on three quality dimensions:

**Original Query:**
{query}

**Generated Answer:**
{answer}

**Evaluation Criteria:**

1. **Relevance (1-10)**: Does the answer directly address the user's query?
   - 9-10: Perfectly relevant, directly answers the question
   - 7-8: Mostly relevant, addresses main question
   - 5-6: Somewhat relevant, partially addresses question
   - 3-4: Limited relevance, tangentially related
   - 1-2: Not relevant, doesn't address the question

2. **Accuracy (1-10)**: Is the information factually correct and reliable?
   - 9-10: Highly accurate, factually correct
   - 7-8: Mostly accurate, minor issues
   - 5-6: Somewhat accurate, some errors
   - 3-4: Limited accuracy, significant errors
   - 1-2: Inaccurate, factually incorrect

3. **Completeness (1-10)**: Does the answer fully address all parts of the query?
   - 9-10: Complete, addresses all aspects
   - 7-8: Mostly complete, minor gaps
   - 5-6: Partially complete, some gaps
   - 3-4: Incomplete, major gaps
   - 1-2: Very incomplete, doesn't address query

**Instructions:**
- Evaluate each dimension independently
- Provide a score (1-10) for each dimension
- Provide brief reasoning for each score
- Calculate overall score as weighted average: Relevance (40%) + Accuracy (40%) + Completeness (20%)

**Output Format (JSON only):**
{{
    "relevance": {{
        "score": <1-10>,
        "reasoning": "<brief explanation>"
    }},
    "accuracy": {{
        "score": <1-10>,
        "reasoning": "<brief explanation>"
    }},
    "completeness": {{
        "score": <1-10>,
        "reasoning": "<brief explanation>"
    }},
    "overall_score": <1-10>,
    "overall_reasoning": "<summary of evaluation>"
}}"""

    return PromptTemplate.from_template(evaluation_prompt)


# ============================================================================
# (3) Evaluator Agent Class Definition
# ============================================================================

class EvaluatorAgent:
    """
    Evaluator Agent for scoring RAG responses on quality dimensions.
    Integrates with Langfuse for continuous quality monitoring.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        langfuse_client: Langfuse,
        quality_threshold: float = 6.0
    ):
        """
        Initialize the Evaluator Agent.
        
        Args:
            llm: LangChain LLM instance for evaluation
            langfuse_client: Langfuse client for logging evaluations
            quality_threshold: Minimum overall score to consider response acceptable (default: 6.0)
        """
        self.llm = llm
        self.langfuse_client = langfuse_client
        self.quality_threshold = quality_threshold
        self.evaluation_prompt = create_evaluation_prompt()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_response(
        self,
        query: str,
        answer: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response on quality dimensions.
        
        Args:
            query: Original user query
            answer: Generated answer to evaluate
            trace_id: Optional Langfuse trace ID to link evaluation
            metadata: Optional metadata (e.g., agent_name, retrieved_docs_count)
        
        Returns:
            Dict containing:
                - relevance: {score, reasoning}
                - accuracy: {score, reasoning}
                - completeness: {score, reasoning}
                - overall_score: float (1-10)
                - overall_reasoning: str
                - is_acceptable: bool (based on quality_threshold)
        """
        try:
            # Format evaluation prompt
            prompt_text = self.evaluation_prompt.format(
                query=query,
                answer=answer
            )
            
            # Get LLM evaluation
            response = self.llm.invoke(prompt_text)
            evaluation_text = response.content.strip()
            
            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in evaluation_text:
                json_start = evaluation_text.find("```json") + 7
                json_end = evaluation_text.find("```", json_start)
                evaluation_text = evaluation_text[json_start:json_end].strip()
            elif "```" in evaluation_text:
                json_start = evaluation_text.find("```") + 3
                json_end = evaluation_text.find("```", json_start)
                evaluation_text = evaluation_text[json_start:json_end].strip()
            
            evaluation_result = json.loads(evaluation_text)
            
            # Extract scores
            relevance_score = evaluation_result.get("relevance", {}).get("score", 0)
            accuracy_score = evaluation_result.get("accuracy", {}).get("score", 0)
            completeness_score = evaluation_result.get("completeness", {}).get("score", 0)
            overall_score = evaluation_result.get("overall_score", 0)
            
            # Determine if response is acceptable
            is_acceptable = overall_score >= self.quality_threshold
            
            # Prepare evaluation result
            evaluation_data = {
                "relevance": {
                    "score": relevance_score,
                    "reasoning": evaluation_result.get("relevance", {}).get("reasoning", "")
                },
                "accuracy": {
                    "score": accuracy_score,
                    "reasoning": evaluation_result.get("accuracy", {}).get("reasoning", "")
                },
                "completeness": {
                    "score": completeness_score,
                    "reasoning": evaluation_result.get("completeness", {}).get("reasoning", "")
                },
                "overall_score": overall_score,
                "overall_reasoning": evaluation_result.get("overall_reasoning", ""),
                "is_acceptable": is_acceptable,
                "quality_threshold": self.quality_threshold
            }
            
            # Log evaluation to Langfuse
            self._log_evaluation_to_langfuse(
                query=query,
                answer=answer,
                evaluation=evaluation_data,
                trace_id=trace_id,
                metadata=metadata
            )
            
            # Log evaluation result
            status = "✓ ACCEPTABLE" if is_acceptable else "✗ LOW QUALITY"
            self.logger.info(
                f"[EVALUATOR] {status} - Overall Score: {overall_score:.1f}/10 "
                f"(Relevance: {relevance_score}, Accuracy: {accuracy_score}, Completeness: {completeness_score})"
            )
            
            return evaluation_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[EVALUATOR] Failed to parse evaluation JSON: {str(e)}")
            self.logger.error(f"[EVALUATOR] LLM response: {evaluation_text[:500]}")
            return self._create_error_evaluation(str(e))
        except Exception as e:
            self.logger.error(f"[EVALUATOR] Error evaluating response: {str(e)}")
            return self._create_error_evaluation(str(e))
    
    def _log_evaluation_to_langfuse(
        self,
        query: str,
        answer: str,
        evaluation: Dict[str, Any],
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log evaluation results to Langfuse.
        
        Args:
            query: Original query
            answer: Generated answer
            evaluation: Evaluation results dictionary
            trace_id: Optional trace ID to link evaluation
            metadata: Optional metadata
        """
        try:
            # Create score in Langfuse
            score_data = {
                "name": "response_quality",
                "value": evaluation["overall_score"],
                "comment": evaluation["overall_reasoning"][:500],  # Truncate for comment
                "trace_id": trace_id,
                "metadata": {
                    "relevance_score": evaluation["relevance"]["score"],
                    "accuracy_score": evaluation["accuracy"]["score"],
                    "completeness_score": evaluation["completeness"]["score"],
                    "is_acceptable": evaluation["is_acceptable"],
                    "quality_threshold": self.quality_threshold,
                    "query": query[:200],
                    "answer_length": len(answer),
                    **(metadata or {})
                }
            }
            
            # Create score via Langfuse SDK
            self.langfuse_client.score(**score_data)
            
        except Exception as e:
            self.logger.warning(f"[EVALUATOR] Failed to log evaluation to Langfuse: {str(e)}")
    
    def _create_error_evaluation(self, error_msg: str) -> Dict[str, Any]:
        """
        Create an error evaluation result when evaluation fails.
        
        Args:
            error_msg: Error message
        
        Returns:
            Dict: Error evaluation result
        """
        return {
            "relevance": {"score": 0, "reasoning": f"Evaluation error: {error_msg}"},
            "accuracy": {"score": 0, "reasoning": f"Evaluation error: {error_msg}"},
            "completeness": {"score": 0, "reasoning": f"Evaluation error: {error_msg}"},
            "overall_score": 0,
            "overall_reasoning": f"Evaluation failed: {error_msg}",
            "is_acceptable": False,
            "error": True
        }
    
    def should_reject_response(self, evaluation: Dict[str, Any]) -> bool:
        """
        Determine if a response should be rejected based on quality threshold.
        
        Args:
            evaluation: Evaluation result dictionary
        
        Returns:
            bool: True if response should be rejected (low quality)
        """
        if evaluation.get("error"):
            return True  # Reject on evaluation errors
        
        return not evaluation.get("is_acceptable", False)

