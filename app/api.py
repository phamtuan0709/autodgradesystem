"""
FastAPI Web API for HybridASAGGrader.

This module provides a RESTful API for grading short answers
with customizable weights for each evaluation criterion.

Run with: uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .grader import HybridASAGGrader
from .models import GradingResult, calculate_final_grade, DEFAULT_WEIGHTS
from .config import GraderConfig, WEIGHT_PRESETS, get_device


# =========================================================================
# PYDANTIC MODELS FOR API
# =========================================================================

class WeightsInput(BaseModel):
    """Weight configuration for grading criteria."""
    semantic: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for semantic similarity")
    coverage: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for keyword coverage")
    formality: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for formality")
    grammar: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for grammar")
    logic: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for logical coherence")
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "semantic": self.semantic,
            "coverage": self.coverage,
            "formality": self.formality,
            "grammar": self.grammar,
            "logic": self.logic
        }
    
    def normalized(self) -> Dict[str, float]:
        """Return normalized weights that sum to 1.0."""
        weights = self.to_dict()
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return DEFAULT_WEIGHTS.copy()


class GradeRequest(BaseModel):
    """Request body for grading a single answer."""
    context: str = Field(default="", description="Background context for the question")
    question: str = Field(..., min_length=1, description="The question being answered")
    reference: str = Field(..., min_length=1, description="The reference/expected answer")
    student: str = Field(..., description="The student's answer to grade")
    weights: Optional[WeightsInput] = Field(
        default=None, 
        description="Custom weights for grading criteria. If not provided, uses default (20% each)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "context": "Photosynthesis is the process by which plants make food.",
                    "question": "Explain the process of photosynthesis.",
                    "reference": "Photosynthesis is a biological process where plants convert sunlight, CO2, and water into glucose and oxygen using chlorophyll.",
                    "student": "Plants use sunlight and water to make food and release oxygen.",
                    "weights": {
                        "semantic": 0.30,
                        "coverage": 0.25,
                        "formality": 0.10,
                        "grammar": 0.15,
                        "logic": 0.20
                    }
                }
            ]
        }
    }


class BatchGradeRequest(BaseModel):
    """Request body for batch grading multiple answers."""
    items: List[Dict[str, str]] = Field(
        ..., 
        min_length=1,
        description="List of items to grade. Each item must have 'question', 'reference', 'student' keys."
    )
    weights: Optional[WeightsInput] = Field(
        default=None, 
        description="Custom weights applied to all items"
    )


class RecalculateRequest(BaseModel):
    """Request body for recalculating grade with new weights."""
    metrics: Dict[str, Any] = Field(..., description="Metrics from a previous grading result")
    weights: WeightsInput = Field(..., description="New weights to apply")


class GradeResponse(BaseModel):
    """Response containing grading result."""
    success: bool = True
    data: Dict[str, Any]
    message: Optional[str] = None


class WeightPresetsResponse(BaseModel):
    """Response containing available weight presets."""
    presets: Dict[str, Dict[str, float]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    device: str
    models_loaded: bool


# =========================================================================
# GLOBAL STATE
# =========================================================================

grader_instance: Optional[HybridASAGGrader] = None


def get_grader() -> HybridASAGGrader:
    """Get or create the grader instance."""
    global grader_instance
    if grader_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Grader not initialized. Please wait for models to load."
        )
    return grader_instance


# =========================================================================
# FASTAPI APP
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize grader on startup, cleanup on shutdown."""
    global grader_instance
    print("\n" + "="*60)
    print("Starting HybridASAGGrader API...")
    print("="*60)
    
    try:
        config = GraderConfig(verbose=True)
        grader_instance = HybridASAGGrader(config=config)
        print("\n✅ Grader initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize grader: {e}")
        grader_instance = None
    
    yield
    
    # Cleanup
    print("\nShutting down HybridASAGGrader API...")
    grader_instance = None


app = FastAPI(
    title="HybridASAG Grader API",
    description="""
    ## Automated Short Answer Grading System
    
    This API provides automated grading for short answer questions using a hybrid approach:
    - **Semantic Analysis**: SimCSE for meaning similarity
    - **Keyword Coverage**: KeyBERT for concept coverage  
    - **Formality**: RoBERTa classifier for writing style
    - **Grammar**: CoLA model for grammatical correctness
    - **Logic**: DeBERTa NLI for logical coherence
    - **Reasoning**: Qwen2.5-3B for detailed feedback generation
    
    ### Key Features
    - Customizable weights for each grading criterion
    - Batch processing support
    - Pre-defined weight presets for common use cases
    - Real-time grade recalculation with new weights
    
    ### Weight Presets Available
    - `balanced`: Equal weight (20%) for all criteria
    - `content_focused`: Prioritizes semantic and coverage
    - `academic_writing`: Emphasizes formality and grammar
    - `logic_heavy`: Prioritizes logical coherence
    - `quick_check`: Fast evaluation focusing on content
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HybridASAG Grader API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    global grader_instance
    device = get_device()
    
    return HealthResponse(
        status="healthy" if grader_instance else "initializing",
        device=str(device),
        models_loaded=grader_instance is not None
    )


@app.get("/weights/presets", response_model=WeightPresetsResponse, tags=["Weights"])
async def get_weight_presets():
    """Get available weight presets."""
    return WeightPresetsResponse(presets=WEIGHT_PRESETS)


@app.get("/weights/default", tags=["Weights"])
async def get_default_weights():
    """Get default weights."""
    return {"weights": DEFAULT_WEIGHTS}


@app.post("/grade", response_model=GradeResponse, tags=["Grading"])
async def grade_answer(request: GradeRequest):
    """
    Grade a single student answer.
    
    Provide the question, reference answer, and student answer.
    Optionally specify custom weights for each grading criterion.
    
    Returns:
    - Metrics (semantic, coverage, formality, grammar, logic scores)
    - Final grade (0-100)
    - Tags (Correct/Partially Correct/Incorrect + issue tags)
    - Explanation and suggestion
    """
    grader = get_grader()
    
    try:
        # Get weights
        weights = request.weights.normalized() if request.weights else None
        
        # Grade the answer
        result = grader.grade_answer(
            context=request.context,
            question=request.question,
            reference=request.reference,
            student=request.student,
            weights=weights
        )
        
        return GradeResponse(
            success=True,
            data=result.to_dict(),
            message="Grading completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.post("/grade/batch", response_model=GradeResponse, tags=["Grading"])
async def grade_batch(request: BatchGradeRequest):
    """
    Grade multiple student answers in batch.
    
    All items will use the same weights (if provided).
    
    Returns a list of grading results.
    """
    grader = get_grader()
    
    try:
        weights = request.weights.normalized() if request.weights else None
        
        results = grader.grade_batch(request.items, weights=weights)
        
        return GradeResponse(
            success=True,
            data={
                "count": len(results),
                "results": [r.to_dict() for r in results]
            },
            message=f"Batch grading completed: {len(results)} items processed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch grading failed: {str(e)}")


@app.post("/grade/recalculate", tags=["Grading"])
async def recalculate_grade(request: RecalculateRequest):
    """
    Recalculate final grade with new weights.
    
    Use this to experiment with different weight configurations
    without re-running the full grading process.
    
    Provide the metrics from a previous grading result and new weights.
    """
    try:
        from .models import MetricScores
        
        # Reconstruct MetricScores from dict
        metrics = MetricScores(
            semantic_score=request.metrics.get("semantic_score", 0.0),
            coverage_score=request.metrics.get("coverage_score", 0.0),
            missing_keywords=request.metrics.get("missing_keywords", []),
            formality_score=request.metrics.get("formality_score", 0.0),
            grammar_score=request.metrics.get("grammar_score", 0.0),
            logic_score=request.metrics.get("logic_score", 0.0),
            logic_details=request.metrics.get("logic_details", {})
        )
        
        # Calculate new grade
        weights = request.weights.normalized()
        new_grade = calculate_final_grade(metrics, weights)
        
        return {
            "success": True,
            "original_grade": request.metrics.get("final_grade", 0.0),
            "new_grade": new_grade,
            "weights_applied": weights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recalculation failed: {str(e)}")


@app.get("/tags", tags=["Reference"])
async def get_available_tags():
    """Get available tags for grading."""
    grader = get_grader()
    return grader.get_available_tags()


@app.get("/thresholds", tags=["Reference"])
async def get_thresholds():
    """Get current threshold values used for tag assignment."""
    grader = get_grader()
    return grader.get_thresholds()


# =========================================================================
# ERROR HANDLERS
# =========================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "status_code": 500
        }
    )


# =========================================================================
# RUN DIRECTLY
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
