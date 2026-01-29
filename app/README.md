# HybridASAG Grader - Web Application

A modular Python package for **Automated Short Answer Grading (ASAG)** with customizable weights, designed for web deployment.

## 🚀 Features

- **Multi-Model Architecture**: Uses 6 specialized ML models for comprehensive grading
- **Customizable Weights**: Adjust importance of each grading criterion (semantic, coverage, formality, grammar, logic)
- **RESTful API**: FastAPI-based web service for easy integration
- **Production-Ready**: Includes contradiction penalty and edge case handling
- **Weight Presets**: Pre-configured weight profiles for common use cases
- **Batch Processing**: Grade multiple answers in a single request

## 📦 Installation

```bash
# Navigate to the app directory
cd app

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon (MPS) acceleration, ensure PyTorch is properly configured
pip install torch torchvision torchaudio
```

## 🏃 Quick Start

### Option 1: Run as API Server

```bash
# From the parent directory
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python -m app.api
```

Then open `http://localhost:8000/docs` for interactive API documentation.

### Option 2: Use as Python Module

```python
from app import HybridASAGGrader, GraderConfig

# Initialize with default settings
grader = HybridASAGGrader()

# Grade an answer with default weights (20% each)
result = grader.grade_answer(
    context="Photosynthesis is the process by which plants make food.",
    question="Explain photosynthesis.",
    reference="Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.",
    student="Plants use sunlight to make food and release oxygen."
)

print(f"Final Grade: {result.metrics.final_grade}/100")
print(f"Tags: {result.feedback.tags}")
print(f"Explanation: {result.feedback.explanation}")
```

### Option 3: Use with Custom Weights

```python
from app import HybridASAGGrader

grader = HybridASAGGrader()

# Custom weights (prioritize content over style)
custom_weights = {
    "semantic": 0.35,   # 35% - meaning similarity
    "coverage": 0.25,   # 25% - keyword coverage
    "formality": 0.10,  # 10% - writing style
    "grammar": 0.15,    # 15% - grammatical correctness
    "logic": 0.15       # 15% - logical coherence
}

result = grader.grade_answer(
    context="...",
    question="...",
    reference="...",
    student="...",
    weights=custom_weights
)
```

## 📊 Weight Presets

| Preset | Semantic | Coverage | Formality | Grammar | Logic | Use Case |
|--------|----------|----------|-----------|---------|-------|----------|
| `balanced` | 20% | 20% | 20% | 20% | 20% | Default, general purpose |
| `content_focused` | 40% | 30% | 5% | 10% | 15% | Prioritize meaning & concepts |
| `academic_writing` | 20% | 15% | 25% | 25% | 15% | Emphasize writing quality |
| `logic_heavy` | 25% | 15% | 10% | 15% | 35% | Focus on logical reasoning |
| `quick_check` | 50% | 30% | 5% | 5% | 10% | Fast content validation |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check and model status |
| `GET` | `/weights/presets` | Get available weight presets |
| `GET` | `/weights/default` | Get default weights |
| `POST` | `/grade` | Grade a single answer |
| `POST` | `/grade/batch` | Grade multiple answers |
| `POST` | `/grade/recalculate` | Recalculate grade with new weights |
| `GET` | `/tags` | Get available grading tags |
| `GET` | `/thresholds` | Get threshold values |

### Example API Request

```bash
curl -X POST "http://localhost:8000/grade" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is photosynthesis?",
    "reference": "Photosynthesis is how plants convert sunlight into food.",
    "student": "Plants use light to make food.",
    "weights": {
      "semantic": 0.35,
      "coverage": 0.25,
      "formality": 0.10,
      "grammar": 0.15,
      "logic": 0.15
    }
  }'
```

## 🏷️ Tag System

### Correctness Tags (mutually exclusive)
- **Correct**: Fully correct answer
- **Partially Correct**: On topic but incomplete
- **Incorrect**: Wrong or contradictory answer

### Issue Tags (can have multiple)
- **Missing Concepts**: Key terms/ideas missing
- **Factual Error**: Contains wrong information
- **Logical Error**: Contradicts reference/logic
- **Vague Expression**: Too general, lacks specificity
- **Grammar Error**: Poor grammar/spelling
- **Off-Topic**: Doesn't address the question
- **Incomplete**: Answer too brief

## 🛠️ Configuration

```python
from app import GraderConfig, HybridASAGGrader

# Custom configuration
config = GraderConfig(
    verbose=True,
    max_new_tokens=500,
    max_explanation_length=500,
    max_suggestion_length=400,
    keyword_similarity_threshold=0.50,
    min_words_for_full_analysis=5,
)

# Custom thresholds
config.thresholds["semantic_correct"] = 0.85
config.thresholds["grammar_good"] = 0.50

grader = HybridASAGGrader(config=config)
```

## 🔧 Models Used

| Metric | Model | Purpose |
|--------|-------|---------|
| Semantic | SimCSE (RoBERTa-Large) | Meaning similarity |
| Keywords | KeyBERT (MiniLM) | Concept coverage |
| Formality | RoBERTa-base-formality | Academic style |
| Grammar | RoBERTa-base-CoLA | Grammatical correctness |
| Logic | DeBERTa-v3 NLI | Logical coherence |
| Reasoning | Qwen2.5-3B-Instruct | Feedback generation |

## 📁 Project Structure

```
app/
├── __init__.py      # Package exports
├── config.py        # Configuration and settings
├── models.py        # Data classes (MetricScores, LLMFeedback, GradingResult)
├── grader.py        # HybridASAGGrader main class
├── api.py           # FastAPI web service
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## 🔒 Production Notes

1. **Contradiction Penalty**: Answers with high contradiction (>0.90) are capped at 40 points
2. **Quality Bonus**: Near-perfect answers (semantic ≥0.95, good grammar) get minimum 85 points
3. **Weight Normalization**: Weights are automatically normalized to sum to 1.0
4. **Error Handling**: Robust JSON parsing with fallback feedback generation

## 📄 License

This project is part of the NCKH research initiative.

## 🙏 Acknowledgments

- Princeton NLP for SimCSE
- Hugging Face for Transformers
- MaartenGr for KeyBERT
- Alibaba for Qwen2.5
