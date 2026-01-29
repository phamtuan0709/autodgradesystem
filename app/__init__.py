"""
Hybrid ASAG Grader - Web Application Module

A modular Python package for Automated Short Answer Grading (ASAG)
with customizable weights for web deployment.

Author: HybridASAG Team
Version: 1.0.0
"""

from .models import MetricScores, LLMFeedback, GradingResult, DEFAULT_WEIGHTS
from .grader import HybridASAGGrader
from .config import GraderConfig

__all__ = [
    "MetricScores",
    "LLMFeedback", 
    "GradingResult",
    "DEFAULT_WEIGHTS",
    "HybridASAGGrader",
    "GraderConfig",
]

__version__ = "1.0.0"
