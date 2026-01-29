"""
Data models for HybridASAGGrader.

This module contains all data classes used in the grading system:
- MetricScores: Container for feature extraction scores
- LLMFeedback: Container for LLM-generated feedback
- GradingResult: Complete grading result
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


# Default weights for final grade calculation (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "semantic": 0.20,
    "coverage": 0.20,
    "formality": 0.20,
    "grammar": 0.20,
    "logic": 0.20
}


@dataclass
class MetricScores:
    """Container for all feature extraction scores."""
    semantic_score: float = 0.0
    coverage_score: float = 0.0
    missing_keywords: List[str] = field(default_factory=list)
    formality_score: float = 0.0
    grammar_score: float = 0.0
    logic_score: float = 0.0
    logic_details: Dict[str, float] = field(default_factory=dict)
    final_grade: float = 0.0  # Final grade on 0-100 scale


def calculate_final_grade(
    metrics: MetricScores,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate final grade on 0-100 scale based on weighted combination of metrics.
    
    PRODUCTION-READY VERSION with:
    - Penalty for high contradiction (factual errors)
    - Proper handling of edge cases
    
    Args:
        metrics: MetricScores object containing all individual scores (0-1 scale)
        weights: Dictionary with weights for each criterion. Keys should be:
                 'semantic', 'coverage', 'formality', 'grammar', 'logic'
                 Weights should sum to 1.0. Default is 20% for each.
    
    Returns:
        Final grade on 0-100 scale
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Validate weights sum to 1.0 (with small tolerance for floating point)
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        # Normalize weights if they don't sum to 1.0
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted sum (all scores are 0-1, result is 0-100)
    base_grade = (
        metrics.semantic_score * weights.get("semantic", 0.20) +
        metrics.coverage_score * weights.get("coverage", 0.20) +
        metrics.formality_score * weights.get("formality", 0.20) +
        metrics.grammar_score * weights.get("grammar", 0.20) +
        metrics.logic_score * weights.get("logic", 0.20)
    ) * 100
    
    # PRODUCTION FIX: Apply penalty for high contradiction (factual/logical errors)
    # This ensures answers with wrong facts don't get inflated scores
    if metrics.logic_details:
        fwd_contradiction = metrics.logic_details.get("contradiction", 0)
        bwd_contradiction = metrics.logic_details.get("backward_contradiction", 0)
        max_contradiction = max(fwd_contradiction, bwd_contradiction)
        
        if max_contradiction > 0.90:
            # Severe factual error - cap grade at 40
            base_grade = min(base_grade, 40.0)
        elif max_contradiction > 0.70:
            # Significant factual error - cap grade at 55
            base_grade = min(base_grade, 55.0)
        elif max_contradiction > 0.50:
            # Moderate contradiction - apply percentage penalty
            penalty = (max_contradiction - 0.50) * 30  # Up to 15 point penalty
            base_grade = max(0, base_grade - penalty)
    
    # PRODUCTION FIX: Bonus for high-quality answers (semantic > 0.95, good grammar)
    # This helps "almost correct" answers get appropriate recognition
    if (metrics.semantic_score >= 0.95 and 
        metrics.grammar_score >= 0.90 and 
        metrics.logic_score >= 0.50 and
        len(metrics.missing_keywords) <= 1):
        # Near-perfect answer - ensure minimum grade of 85
        base_grade = max(base_grade, 85.0)
    
    return round(min(100.0, max(0.0, base_grade)), 2)


@dataclass
class LLMFeedback:
    """Container for LLM-generated feedback."""
    tags: List[str] = field(default_factory=list)
    explanation: str = ""
    suggestion: str = ""
    raw_response: str = ""
    parse_success: bool = False


@dataclass
class GradingResult:
    """Complete grading result containing all scores and feedback."""
    metrics: MetricScores = field(default_factory=MetricScores)
    feedback: LLMFeedback = field(default_factory=LLMFeedback)
    is_valid: bool = True
    skip_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": asdict(self.metrics),
            "feedback": asdict(self.feedback),
            "is_valid": self.is_valid,
            "skip_reason": self.skip_reason
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def __repr__(self) -> str:
        return self.to_json()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradingResult':
        """Create GradingResult from dictionary."""
        result = cls()
        
        if "metrics" in data:
            metrics_data = data["metrics"]
            result.metrics = MetricScores(
                semantic_score=metrics_data.get("semantic_score", 0.0),
                coverage_score=metrics_data.get("coverage_score", 0.0),
                missing_keywords=metrics_data.get("missing_keywords", []),
                formality_score=metrics_data.get("formality_score", 0.0),
                grammar_score=metrics_data.get("grammar_score", 0.0),
                logic_score=metrics_data.get("logic_score", 0.0),
                logic_details=metrics_data.get("logic_details", {}),
                final_grade=metrics_data.get("final_grade", 0.0)
            )
        
        if "feedback" in data:
            feedback_data = data["feedback"]
            result.feedback = LLMFeedback(
                tags=feedback_data.get("tags", []),
                explanation=feedback_data.get("explanation", ""),
                suggestion=feedback_data.get("suggestion", ""),
                raw_response=feedback_data.get("raw_response", ""),
                parse_success=feedback_data.get("parse_success", False)
            )
        
        result.is_valid = data.get("is_valid", True)
        result.skip_reason = data.get("skip_reason")
        
        return result
