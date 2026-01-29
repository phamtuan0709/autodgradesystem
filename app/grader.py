"""
HybridASAGGrader - Main grading engine.

This module contains the HybridASAGGrader class which implements
a hybrid automated short answer grading system using:
- SimCSE for semantic similarity
- KeyBERT for keyword extraction
- RoBERTa for formality and grammar
- DeBERTa NLI for logical coherence
- Qwen2.5-3B for reasoning and feedback generation
"""

import json
import re
import warnings
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from scipy.spatial.distance import cosine

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from keybert import KeyBERT

from .models import MetricScores, LLMFeedback, GradingResult, calculate_final_grade, DEFAULT_WEIGHTS
from .config import GraderConfig, get_device

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HybridASAGGrader:
    """
    Hybrid Automated Short Answer Grading System - PRODUCTION VERSION
    
    Uses specialized models for feature extraction and Qwen2.5-3B for reasoning.
    Optimized for Apple Silicon (MPS) with fallback to CUDA/CPU.
    
    Tag System (Hierarchical - mutually exclusive levels):
    ┌─────────────────────────────────────────────────────────┐
    │ CORRECTNESS LEVEL (pick one):                           │
    │   • Correct          - Fully correct answer             │
    │   • Partially Correct - On topic but incomplete         │
    │   • Incorrect        - Wrong/contradictory answer       │
    ├─────────────────────────────────────────────────────────┤
    │ ISSUE TAGS (can have multiple, only if not "Correct"):  │
    │   • Missing Concepts  - Key terms/ideas missing         │
    │   • Factual Error     - Contains wrong information      │
    │   • Logical Error     - Contradicts reference/logic     │
    │   • Vague Expression  - Too general, lacks specificity  │
    │   • Grammar Error     - Poor grammar/spelling           │
    │   • Off-Topic         - Doesn't address the question    │
    │   • Incomplete        - Answer cut off/too brief        │
    └─────────────────────────────────────────────────────────┘
    """
    
    # Tag definitions
    CORRECTNESS_TAGS = ["Correct", "Partially Correct", "Incorrect"]
    ISSUE_TAGS = ["Missing Concepts", "Factual Error", "Logical Error", 
                  "Vague Expression", "Grammar Error", "Off-Topic", "Incomplete"]
    ALL_TAGS = CORRECTNESS_TAGS + ISSUE_TAGS
    
    def __init__(self, config: GraderConfig = None, verbose: bool = True):
        """
        Initialize the grader with all required models.
        
        Args:
            config: GraderConfig object with custom settings. If None, uses defaults.
            verbose: Enable/disable verbose logging (overrides config if provided)
        """
        self.config = config or GraderConfig()
        self.verbose = verbose if verbose is not None else self.config.verbose
        self.device = self.config.device or get_device()
        
        # Copy thresholds from config
        self.THRESHOLDS = self.config.thresholds.copy()
        self.KEYWORD_SIMILARITY_THRESHOLD = self.config.keyword_similarity_threshold
        self.MIN_WORDS_FOR_FULL_ANALYSIS = self.config.min_words_for_full_analysis
        
        # Load models
        self._load_models()
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _load_models(self):
        """Load all required models with memory optimization."""
        self._log("\n" + "="*60)
        self._log("Loading Models...")
        self._log("="*60)
        
        # 1. Semantic Similarity Model
        self._log("\n[1/6] Loading Semantic Model (SimCSE)...")
        self.semantic_model = SentenceTransformer(self.config.SEMANTIC_MODEL)
        if self.device.type != "cpu":
            self.semantic_model = self.semantic_model.to(self.device)
        self._log(f"      Done: {self.config.SEMANTIC_MODEL}")
        
        # 2. KeyBERT
        self._log("\n[2/6] Loading KeyBERT Model...")
        self.keybert_model = KeyBERT(model=self.config.KEYBERT_MODEL)
        self._log(f"      Done: {self.config.KEYBERT_MODEL} (KeyBERT backend)")
        
        # 3. Formality Classifier
        self._log("\n[3/6] Loading Formality Model...")
        self.formality_tokenizer = AutoTokenizer.from_pretrained(self.config.FORMALITY_MODEL)
        self.formality_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.FORMALITY_MODEL
        ).to(self.device)
        self.formality_model.eval()
        self._log(f"      Done: {self.config.FORMALITY_MODEL}")
        
        # 4. Grammar Classifier
        self._log("\n[4/6] Loading Grammar Model...")
        self.grammar_tokenizer = AutoTokenizer.from_pretrained(self.config.GRAMMAR_MODEL)
        self.grammar_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.GRAMMAR_MODEL
        ).to(self.device)
        self.grammar_model.eval()
        self._log(f"      Done: {self.config.GRAMMAR_MODEL}")
        
        # 5. Logic/NLI CrossEncoder
        self._log("\n[5/6] Loading Logic Model (NLI)...")
        self.logic_model = CrossEncoder(self.config.LOGIC_MODEL)
        self._log(f"      Done: {self.config.LOGIC_MODEL}")
        
        # 6. Qwen2.5-3B for reasoning
        self._log("\n[6/6] Loading Reasoning Model (Qwen2.5-3B-Instruct)...")
        self.reasoning_tokenizer = AutoTokenizer.from_pretrained(
            self.config.REASONING_MODEL,
            trust_remote_code=True
        )
        
        # Determine optimal dtype for device
        if self.device.type == "mps":
            model_dtype = torch.float16
        elif self.device.type == "cuda":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(
            self.config.REASONING_MODEL,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.reasoning_model.eval()
        self._log(f"      Done: {self.config.REASONING_MODEL}")
        
        self._log("\n" + "="*60)
        self._log("All models loaded successfully!")
        self._log("="*60 + "\n")
    
    # =========================================================================
    # FEATURE EXTRACTION METHODS
    # =========================================================================
    
    def get_semantic_score(self, reference: str, student: str) -> float:
        """Calculate semantic similarity using SimCSE embeddings."""
        with torch.no_grad():
            embeddings = self.semantic_model.encode(
                [reference, student],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            ref_emb = embeddings[0].cpu().numpy()
            stu_emb = embeddings[1].cpu().numpy()
            similarity = 1 - cosine(ref_emb, stu_emb)
            return float(max(0.0, min(1.0, similarity)))
    
    def get_keyword_coverage(
        self, 
        question: str, 
        reference: str, 
        student: str
    ) -> Tuple[float, List[str]]:
        """Calculate keyword coverage using KeyBERT + semantic matching."""
        context_doc = reference
        word_count = len(reference.split())
        num_keywords = max(3, min(8, word_count // 15 + 2))
        
        # Extract unigrams
        unigram_keywords = self.keybert_model.extract_keywords(
            context_doc,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=num_keywords,
            use_mmr=True,
            diversity=0.5
        )
        
        # Extract bigrams (filtered)
        bigram_keywords = self.keybert_model.extract_keywords(
            context_doc,
            keyphrase_ngram_range=(2, 2),
            stop_words='english',
            top_n=max(2, num_keywords // 2),
            use_mmr=True,
            diversity=0.7
        )
        good_bigrams = [(kw, score) for kw, score in bigram_keywords if score > 0.35]
        
        all_keywords = unigram_keywords + good_bigrams
        if not all_keywords:
            return 1.0, []
        
        keyword_list = [kw[0] for kw in all_keywords if len(kw[0]) > 2]
        seen = set()
        keyword_list = [x for x in keyword_list if not (x.lower() in seen or seen.add(x.lower()))]
        
        if not keyword_list:
            return 1.0, []
        
        student_lower = student.lower()
        student_words = set(student_lower.split())
        
        with torch.no_grad():
            all_texts = keyword_list + [student]
            embeddings = self.semantic_model.encode(
                all_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            keyword_embeddings = embeddings[:-1].cpu().numpy()
            student_embedding = embeddings[-1].cpu().numpy()
        
        found_keywords = []
        missing_keywords = []
        
        for i, keyword in enumerate(keyword_list):
            keyword_lower = keyword.lower()
            
            # Method 1: Exact/substring match
            if keyword_lower in student_lower:
                found_keywords.append(keyword)
                continue
            
            # Method 2: Word-level match for unigrams
            if ' ' not in keyword and keyword_lower in student_words:
                found_keywords.append(keyword)
                continue
            
            # Method 3: Semantic similarity matching
            similarity = 1 - cosine(keyword_embeddings[i], student_embedding)
            if similarity >= self.KEYWORD_SIMILARITY_THRESHOLD:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        coverage_score = len(found_keywords) / len(keyword_list) if keyword_list else 1.0
        return float(coverage_score), missing_keywords
    
    def get_formality_score(self, text: str) -> float:
        """Calculate formality score using RoBERTa classifier."""
        with torch.no_grad():
            inputs = self.formality_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            outputs = self.formality_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return float(probs[0][1].item())
    
    def get_grammar_score(self, text: str) -> float:
        """Calculate grammar acceptability score using CoLA model."""
        with torch.no_grad():
            inputs = self.grammar_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            outputs = self.grammar_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return float(probs[0][1].item())
    
    def get_logic_score(
        self, 
        reference: str, 
        student: str,
        grammar_score: float = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate logical coherence using bidirectional NLI.
        
        IMPROVED: Considers grammar quality - poor grammar can confuse NLI models.
        """
        scores = self.logic_model.predict(
            [
                (reference, student),
                (student, reference)
            ],
            apply_softmax=True
        )
        
        forward_probs = scores[0]
        backward_probs = scores[1]
        
        prob_dict = {
            "contradiction": float(forward_probs[0]),
            "entailment": float(forward_probs[1]),
            "neutral": float(forward_probs[2]),
            "backward_entailment": float(backward_probs[1]),
            "backward_contradiction": float(backward_probs[0]),
            "backward_neutral": float(backward_probs[2])
        }
        
        fwd_contradiction = float(forward_probs[0])
        fwd_entailment = float(forward_probs[1])
        fwd_neutral = float(forward_probs[2])
        
        bwd_entailment = float(backward_probs[1])
        bwd_contradiction = float(backward_probs[0])
        bwd_neutral = float(backward_probs[2])
        
        max_contradiction = max(fwd_contradiction, bwd_contradiction)
        avg_entailment = (fwd_entailment + bwd_entailment) / 2
        avg_neutral = (fwd_neutral + bwd_neutral) / 2
        
        # Calculate base logic score
        if max_contradiction > self.THRESHOLDS["contradiction_high"]:
            logic_score = (1 - max_contradiction) * 0.15
        elif max_contradiction > self.THRESHOLDS["contradiction_moderate"]:
            base_score = avg_entailment
            penalty = max_contradiction * 0.3
            logic_score = base_score * (1 - penalty) + 0.25
        elif avg_neutral > 0.8:
            logic_score = 0.55 + (avg_entailment * 0.25) + ((1 - max_contradiction) * 0.15)
        else:
            entailment_score = (fwd_entailment * 0.55 + bwd_entailment * 0.35)
            neutral_bonus = min(fwd_neutral, bwd_neutral) * 0.15
            logic_score = entailment_score + neutral_bonus + 0.15
        
        # GRAMMAR ADJUSTMENT: If grammar is poor, NLI may be unreliable
        if grammar_score is not None and grammar_score < self.THRESHOLDS["grammar_good"]:
            if max_contradiction < self.THRESHOLDS["contradiction_high"]:
                grammar_boost = (self.THRESHOLDS["grammar_good"] - grammar_score) * 0.3
                logic_score = min(1.0, logic_score + grammar_boost)
        
        logic_score = max(0.0, min(1.0, logic_score))
        return float(logic_score), prob_dict
    
    def get_simple_word_match_score(self, reference: str, student: str) -> float:
        """Simple word overlap score for very short answers."""
        ref_words = set(reference.lower().split())
        stu_words = set(student.lower().split())
        if not ref_words:
            return 0.0
        overlap = ref_words.intersection(stu_words)
        return len(overlap) / len(ref_words)
    
    # =========================================================================
    # TAG ASSIGNMENT LOGIC
    # =========================================================================
    
    def _assign_tags(self, metrics: MetricScores) -> Tuple[List[str], str]:
        """
        Assign tags using optimized multi-factor scoring.
        
        Uses weighted combination of metrics for more accurate classification.
        IMPROVED: Adjusts semantic score when high contradiction is detected.
        """
        tags = []
        correctness = None
        
        # Get contradiction info
        fwd_contradiction = metrics.logic_details.get("contradiction", 0)
        bwd_contradiction = metrics.logic_details.get("backward_contradiction", 0)
        max_contradiction = max(fwd_contradiction, bwd_contradiction)
        has_high_contradiction = fwd_contradiction > self.THRESHOLDS["contradiction_high"] or \
                                 bwd_contradiction > self.THRESHOLDS["contradiction_high"]
        has_moderate_contradiction = fwd_contradiction > self.THRESHOLDS["contradiction_moderate"] or \
                                     bwd_contradiction > self.THRESHOLDS["contradiction_moderate"]
        
        # IMPROVED: Adjust semantic score when contradiction is high
        adjusted_semantic = metrics.semantic_score
        if has_high_contradiction:
            adjusted_semantic = metrics.semantic_score * (1 - max_contradiction * 0.6)
        elif has_moderate_contradiction:
            adjusted_semantic = metrics.semantic_score * (1 - max_contradiction * 0.3)
        
        # Calculate composite score
        composite_score = (
            adjusted_semantic * 0.45 +
            metrics.coverage_score * 0.20 +
            metrics.logic_score * 0.20 +
            metrics.grammar_score * 0.15
        )
        
        # Check if grammar is the primary issue
        grammar_is_primary_issue = (
            metrics.semantic_score >= self.THRESHOLDS["semantic_partial"] and
            metrics.grammar_score < self.THRESHOLDS["grammar_good"] and
            not has_high_contradiction
        )
        
        # STEP 1: Determine correctness level
        if has_high_contradiction:
            correctness = "Incorrect"
        elif adjusted_semantic >= self.THRESHOLDS["semantic_correct"] and \
             metrics.logic_score >= self.THRESHOLDS["logic_correct"] and \
             not has_moderate_contradiction and \
             metrics.grammar_score >= self.THRESHOLDS["grammar_good"]:
            correctness = "Correct"
        elif adjusted_semantic >= self.THRESHOLDS["semantic_correct"] and \
             not has_moderate_contradiction and \
             composite_score >= 0.65:
            correctness = "Correct"
        elif adjusted_semantic >= self.THRESHOLDS["semantic_partial"] or composite_score >= 0.45:
            correctness = "Partially Correct"
        elif adjusted_semantic < self.THRESHOLDS["semantic_off_topic"]:
            correctness = "Incorrect"
        else:
            correctness = "Partially Correct"
        
        tags.append(correctness)
        
        # STEP 2: If "Correct", check for missing keywords
        if correctness == "Correct":
            num_missing = len(metrics.missing_keywords) if metrics.missing_keywords else 0
            should_downgrade = (
                (num_missing >= 2 and metrics.coverage_score < self.THRESHOLDS["coverage_correct"]) or
                (num_missing >= 3) or
                (metrics.coverage_score < 0.50)
            )
            
            if should_downgrade:
                tags[0] = "Partially Correct"
                correctness = "Partially Correct"
                tags.append("Missing Concepts")
            return tags, correctness
        
        # STEP 3: Add issue tags (only for non-Correct)
        
        if metrics.semantic_score < self.THRESHOLDS["semantic_off_topic"]:
            tags.append("Off-Topic")
        
        if has_high_contradiction:
            tags.append("Logical Error")
            if metrics.semantic_score >= self.THRESHOLDS["semantic_off_topic"]:
                tags.append("Factual Error")
        elif metrics.logic_score < self.THRESHOLDS["logic_error"] and not grammar_is_primary_issue:
            if "Off-Topic" not in tags:
                tags.append("Factual Error")
        
        if metrics.missing_keywords and (metrics.coverage_score < self.THRESHOLDS["coverage_missing"] or len(metrics.missing_keywords) >= 2):
            if "Missing Concepts" not in tags:
                tags.append("Missing Concepts")
        
        if adjusted_semantic >= self.THRESHOLDS["semantic_partial"] and \
           metrics.coverage_score < self.THRESHOLDS["coverage_good"] and \
           "Missing Concepts" not in tags and \
           "Grammar Error" not in tags and \
           not metrics.missing_keywords:
            tags.append("Vague Expression")
        
        if metrics.grammar_score < self.THRESHOLDS["grammar_good"]:
            tags.append("Grammar Error")
        
        return tags, correctness
    
    # =========================================================================
    # REASONING LAYER (QWEN2)
    # =========================================================================
    
    def _build_prompt(self, 
                      context: str,
                      question: str,
                      reference: str,
                      student: str,
                      metrics: MetricScores) -> str:
        """Build optimized prompt for Qwen2."""
        missing_list = ", ".join(metrics.missing_keywords[:3]) if metrics.missing_keywords else "None"
        
        tags, correctness = self._assign_tags(metrics)
        tags_json = json.dumps(tags)
        
        error_details = []
        if "Missing Concepts" in tags and metrics.missing_keywords:
            error_details.append(f"Missing key terms: {', '.join(metrics.missing_keywords[:3])}")
        if "Logical Error" in tags or "Factual Error" in tags:
            error_details.append("Contains factual/logical errors that contradict the reference")
        if "Vague Expression" in tags:
            error_details.append("Answer is too general, lacks specific details")
        if "Grammar Error" in tags:
            error_details.append("Has grammar/spelling issues")
        
        error_analysis = "; ".join(error_details) if error_details else "Minor issues only"
        
        prompt = f'''<|im_start|>system
You are a grading assistant. Provide specific, actionable feedback. Output ONLY valid JSON.
<|im_end|>
<|im_start|>user
Grade this student answer with SPECIFIC feedback.

Question: "{question[:100]}"
Reference Answer: "{reference[:150]}"
Student Answer: "{student[:150]}"

Analysis:
- Grade: {correctness}
- Issues: {error_analysis}
- Missing Keywords: {missing_list}

Requirements for your response:
1. explanation: Explain what is wrong or right, referencing specific parts of the student's answer
2. suggestion: Give SPECIFIC advice on how to improve, including what words/concepts to add or correct

Output JSON:
{{"tags": {tags_json}, "explanation": "specific explanation referencing student answer", "suggestion": "specific improvement advice"}}
<|im_end|>
<|im_start|>assistant
{{"tags": {tags_json}, "explanation": "'''
        return prompt
    
    def _parse_llm_response(self, response: str, metrics: MetricScores = None) -> LLMFeedback:
        """Parse JSON response with robust fallback."""
        feedback = LLMFeedback(raw_response=response)
        max_explanation = self.config.max_explanation_length
        max_suggestion = self.config.max_suggestion_length
        
        try:
            response = response.strip()
            
            for prefix in ["```json", "```"]:
                if response.startswith(prefix):
                    response = response[len(prefix):]
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            if not response.startswith('{'):
                tags, correctness = self._assign_tags(metrics) if metrics else (["Partially Correct"], "Partially Correct")
                tags_json = json.dumps(tags)
                
                if '"suggestion"' in response or "'suggestion'" in response:
                    reconstructed = '{"tags": ' + tags_json + ', "explanation": "' + response
                    
                    try:
                        reconstructed = reconstructed.replace("'", '"')
                        reconstructed = re.sub(r',\s*}', '}', reconstructed)
                        
                        last_brace = reconstructed.rfind('}')
                        if last_brace != -1:
                            reconstructed = reconstructed[:last_brace + 1]
                        
                        data = json.loads(reconstructed)
                        feedback.tags = self._validate_tags(data.get("tags", []), metrics)
                        feedback.explanation = str(data.get("explanation", ""))[:max_explanation]
                        feedback.suggestion = str(data.get("suggestion", ""))[:max_suggestion]
                        feedback.parse_success = True
                        return feedback
                    except:
                        pass
                
                exp_match = re.search(r'^([^"]+)', response)
                sug_match = re.search(r'"suggestion"\s*:\s*"([^"]+)"', response)
                
                if exp_match:
                    explanation = exp_match.group(1).strip().rstrip('",')
                    suggestion = sug_match.group(1) if sug_match else ""
                    
                    feedback.tags = self._validate_tags([], metrics)
                    feedback.explanation = explanation[:max_explanation]
                    feedback.suggestion = suggestion[:max_suggestion]
                    feedback.parse_success = True
                    return feedback
            
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r'"\s*\n\s*"', '" "', json_str)
                
                data = json.loads(json_str)
                
                raw_tags = data.get("tags", [])
                feedback.tags = self._validate_tags(raw_tags, metrics)
                feedback.explanation = str(data.get("explanation", ""))[:max_explanation]
                feedback.suggestion = str(data.get("suggestion", ""))[:max_suggestion]
                feedback.parse_success = True
            else:
                feedback = self._generate_fallback_feedback(response, metrics)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            feedback = self._generate_fallback_feedback(response, metrics)
        
        return feedback
    
    def _validate_tags(self, tags: List[str], metrics: MetricScores = None) -> List[str]:
        """Validate and fix tag conflicts."""
        if not tags:
            if metrics:
                tags, _ = self._assign_tags(metrics)
            else:
                tags = ["Partially Correct"]
        
        valid_tags = []
        for tag in tags:
            if isinstance(tag, str):
                for valid in self.ALL_TAGS:
                    if tag.lower().strip() == valid.lower():
                        valid_tags.append(valid)
                        break
        
        if not valid_tags:
            if metrics:
                return self._assign_tags(metrics)[0]
            return ["Partially Correct"]
        
        if "Correct" in valid_tags:
            return ["Correct"]
        
        correctness_in_tags = [t for t in valid_tags if t in self.CORRECTNESS_TAGS]
        issue_tags = [t for t in valid_tags if t in self.ISSUE_TAGS]
        issue_tags = list(dict.fromkeys(issue_tags))
        
        if len(correctness_in_tags) == 0:
            if any(t in ["Logical Error", "Factual Error", "Off-Topic"] for t in issue_tags):
                return ["Incorrect"] + issue_tags
            else:
                return ["Partially Correct"] + issue_tags
        elif len(correctness_in_tags) > 1:
            if "Incorrect" in correctness_in_tags:
                return ["Incorrect"] + issue_tags
            else:
                return ["Partially Correct"] + issue_tags
        
        return [correctness_in_tags[0]] + issue_tags
    
    def _generate_fallback_feedback(self, raw_response: str, metrics: MetricScores = None) -> LLMFeedback:
        """Generate high-quality rule-based feedback."""
        feedback = LLMFeedback(raw_response=raw_response, parse_success=False)
        
        if metrics is None:
            feedback.explanation = raw_response[:400] if raw_response else "Unable to evaluate."
            feedback.tags = ["Partially Correct"]
            return feedback
        
        tags, correctness = self._assign_tags(metrics)
        explanations = []
        suggestions = []
        
        if correctness == "Correct":
            explanations.append("Excellent answer that demonstrates strong understanding of the topic.")
            suggestions.append("Great work! Keep up the quality.")
        else:
            if "Off-Topic" in tags:
                explanations.append(f"The answer does not address the question (semantic similarity: {metrics.semantic_score:.0%}).")
                suggestions.append("Please re-read the question carefully and provide a response that directly addresses what is being asked.")
            
            if "Logical Error" in tags:
                explanations.append("The answer contains statements that contradict the reference material.")
                suggestions.append("Check your understanding of the core concepts. Make sure your statements align with established facts.")
            
            if "Factual Error" in tags and "Logical Error" not in tags:
                explanations.append("The answer contains factual inaccuracies.")
                suggestions.append("Review the source material and verify your facts before answering.")
            
            if "Missing Concepts" in tags and metrics.missing_keywords:
                missing_str = ", ".join(metrics.missing_keywords[:3])
                explanations.append(f"Key concepts are missing from your answer: {missing_str}.")
                suggestions.append(f"Include these important terms in your answer: '{missing_str}'. Explain how they relate to the topic.")
            
            if "Vague Expression" in tags:
                explanations.append("Your answer is on topic but too general.")
                suggestions.append("Be more specific. Instead of general statements, use precise terms and provide concrete examples or details.")
            
            if "Grammar Error" in tags:
                explanations.append(f"Writing quality needs improvement (grammar score: {metrics.grammar_score:.0%}).")
                suggestions.append("Proofread your answer. Check for complete sentences, proper capitalization, and correct verb forms.")
            
            if len(explanations) == 0:
                explanations.append("The answer is partially correct but could be improved.")
                suggestions.append("Expand your answer with more details and make sure to cover all key points from the topic.")
        
        feedback.tags = tags
        feedback.explanation = " ".join(explanations)
        feedback.suggestion = " ".join(suggestions)
        
        return feedback
    
    def generate_feedback(self,
                          context: str,
                          question: str,
                          reference: str,
                          student: str,
                          metrics: MetricScores) -> LLMFeedback:
        """Generate feedback using Qwen2."""
        prompt = self._build_prompt(context, question, reference, student, metrics)
        
        with torch.no_grad():
            inputs = self.reasoning_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=800
            ).to(self.device)
            
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.reasoning_tokenizer.eos_token_id
            )
            
            response = self.reasoning_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        
        return self._parse_llm_response(response, metrics)
    
    # =========================================================================
    # MAIN GRADING METHOD
    # =========================================================================
    
    def grade_answer(
        self,
        context: str,
        question: str,
        reference: str,
        student: str,
        weights: Dict[str, float] = None
    ) -> GradingResult:
        """
        Grade a student answer against a reference answer.
        
        Args:
            context: Background context for the question
            question: The question being answered
            reference: The reference/expected answer
            student: The student's answer
            weights: Optional custom weights for final grade calculation.
                     Keys: 'semantic', 'coverage', 'formality', 'grammar', 'logic'
                     Values should sum to 1.0. Default is 20% for each criterion.
        
        Returns:
            GradingResult with metrics (including final_grade on 0-100 scale) and feedback
        """
        result = GradingResult()
        
        if not student or not student.strip():
            result.is_valid = False
            result.skip_reason = "Empty student answer"
            return result
        
        word_count = len(student.split())
        
        if word_count < self.MIN_WORDS_FOR_FULL_ANALYSIS:
            result.metrics.semantic_score = self.get_simple_word_match_score(reference, student)
            result.metrics.coverage_score = result.metrics.semantic_score
            result.metrics.formality_score = 0.5
            result.metrics.grammar_score = 0.5
            result.metrics.logic_score = result.metrics.semantic_score
            result.metrics.final_grade = calculate_final_grade(result.metrics, weights)
            
            result.feedback = LLMFeedback(
                tags=["Incomplete", "Vague Expression"],
                explanation=f"Answer too short ({word_count} words). Cannot fully evaluate.",
                suggestion="Please provide a more detailed and complete answer.",
                parse_success=True
            )
            return result
        
        self._log("\nCalculating metrics...")
        
        self._log("   - Semantic similarity...")
        result.metrics.semantic_score = self.get_semantic_score(reference, student)
        
        self._log("   - Keyword coverage...")
        coverage, missing = self.get_keyword_coverage(question, reference, student)
        result.metrics.coverage_score = coverage
        result.metrics.missing_keywords = missing
        
        self._log("   - Formality...")
        result.metrics.formality_score = self.get_formality_score(student)
        
        self._log("   - Grammar...")
        result.metrics.grammar_score = self.get_grammar_score(student)
        
        self._log("   - Logic/coherence...")
        logic_score, logic_details = self.get_logic_score(
            reference, student, result.metrics.grammar_score
        )
        result.metrics.logic_score = logic_score
        result.metrics.logic_details = logic_details
        
        self._log("   - Final grade...")
        result.metrics.final_grade = calculate_final_grade(result.metrics, weights)
        
        self._log("\nGenerating feedback with Qwen2...")
        
        result.feedback = self.generate_feedback(
            context, question, reference, student, result.metrics
        )
        
        self._log("Grading complete!\n")
        
        return result
    
    def grade_batch(
        self, 
        items: List[Dict[str, str]], 
        weights: Dict[str, float] = None
    ) -> List[GradingResult]:
        """
        Grade multiple answers.
        
        Args:
            items: List of dicts with 'context', 'question', 'reference', 'student' keys
            weights: Optional custom weights for final grade calculation
        
        Returns:
            List of GradingResult objects
        """
        results = []
        total = len(items)
        
        for i, item in enumerate(items, 1):
            self._log(f"\n{'='*60}")
            self._log(f"Grading answer {i}/{total}")
            self._log(f"{'='*60}")
            
            result = self.grade_answer(
                context=item.get("context", ""),
                question=item["question"],
                reference=item["reference"],
                student=item["student"],
                weights=weights
            )
            results.append(result)
        
        return results
    
    def recalculate_grade(
        self, 
        result: GradingResult, 
        weights: Dict[str, float]
    ) -> float:
        """
        Recalculate final grade with new weights without re-grading.
        
        Useful for experimenting with different weight configurations.
        
        Args:
            result: Existing GradingResult object
            weights: New weights to apply
        
        Returns:
            New final grade (0-100)
        """
        return calculate_final_grade(result.metrics, weights)
    
    def get_available_tags(self) -> Dict[str, List[str]]:
        """Return available tags for reference."""
        return {
            "correctness_tags": self.CORRECTNESS_TAGS,
            "issue_tags": self.ISSUE_TAGS,
            "all_tags": self.ALL_TAGS
        }
    
    def get_thresholds(self) -> Dict[str, float]:
        """Return current thresholds for debugging."""
        return self.THRESHOLDS
    
    def get_weights(self) -> Dict[str, float]:
        """Return current default weights."""
        return self.config.weights.copy()
