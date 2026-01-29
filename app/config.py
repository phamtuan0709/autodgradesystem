"""
Configuration module for HybridASAGGrader.

This module contains all configurable settings including:
- Model identifiers
- Threshold values
- Default weights
- Device settings
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class GraderConfig:
    """
    Configuration class for HybridASAGGrader.
    
    Attributes:
        weights: Custom weights for final grade calculation (sum to 1.0)
        thresholds: Custom threshold values for tag assignment
        verbose: Enable/disable verbose logging
        device: PyTorch device (auto-detected if None)
    """
    
    # Model identifiers
    SEMANTIC_MODEL: str = "princeton-nlp/sup-simcse-roberta-large"
    KEYBERT_MODEL: str = "all-MiniLM-L6-v2"
    FORMALITY_MODEL: str = "cointegrated/roberta-base-formality"
    GRAMMAR_MODEL: str = "textattack/roberta-base-CoLA"
    LOGIC_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    REASONING_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # Default weights for final grade calculation
    weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.20,
        "coverage": 0.20,
        "formality": 0.20,
        "grammar": 0.20,
        "logic": 0.20
    })
    
    # Thresholds for tag assignment
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        # Correctness thresholds
        "semantic_correct": 0.85,
        "semantic_partial": 0.55,
        "semantic_off_topic": 0.35,
        
        # Coverage thresholds
        "coverage_correct": 0.80,
        "coverage_good": 0.60,
        "coverage_missing": 0.60,
        
        # Grammar thresholds
        "grammar_good": 0.50,
        "grammar_poor": 0.35,
        
        # Logic thresholds
        "logic_correct": 0.50,
        "logic_good": 0.40,
        "logic_error": 0.20,
        
        # Contradiction thresholds
        "contradiction_high": 0.60,
        "contradiction_moderate": 0.35,
        
        # Formality threshold
        "formality_poor": 0.10,
    })
    
    # Other settings
    verbose: bool = True
    device: Optional[torch.device] = None
    
    # Generation settings
    max_new_tokens: int = 500
    max_explanation_length: int = 500
    max_suggestion_length: int = 400
    
    # Keyword settings
    keyword_similarity_threshold: float = 0.50
    min_words_for_full_analysis: int = 5
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0 (with tolerance)."""
        total = sum(self.weights.values())
        return abs(total - 1.0) < 0.001
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update weights with validation and normalization."""
        valid_keys = {"semantic", "coverage", "formality", "grammar", "logic"}
        for key, value in new_weights.items():
            if key in valid_keys:
                self.weights[key] = value
        self.normalize_weights()
    
    def update_threshold(self, key: str, value: float) -> bool:
        """Update a single threshold value."""
        if key in self.thresholds:
            self.thresholds[key] = value
            return True
        return False


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (No GPU acceleration available)")
    return device


# Weight presets for common use cases
WEIGHT_PRESETS = {
    "balanced": {
        "semantic": 0.20,
        "coverage": 0.20,
        "formality": 0.20,
        "grammar": 0.20,
        "logic": 0.20
    },
    "content_focused": {
        "semantic": 0.40,
        "coverage": 0.30,
        "formality": 0.05,
        "grammar": 0.10,
        "logic": 0.15
    },
    "academic_writing": {
        "semantic": 0.20,
        "coverage": 0.15,
        "formality": 0.25,
        "grammar": 0.25,
        "logic": 0.15
    },
    "logic_heavy": {
        "semantic": 0.25,
        "coverage": 0.15,
        "formality": 0.10,
        "grammar": 0.15,
        "logic": 0.35
    },
    "quick_check": {
        "semantic": 0.50,
        "coverage": 0.30,
        "formality": 0.05,
        "grammar": 0.05,
        "logic": 0.10
    }
}
