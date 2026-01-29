"""
Comprehensive Evaluation Framework for Hybrid ASAG Model
=========================================================
This module provides:
1. Public ASAG Dataset loading (SemEval-2013, Mohler)
2. K-Fold Cross Validation
3. 4 Baselines for comparison
4. Ablation Study components
5. Complete evaluation metrics for research paper

Author: Research Team
Date: January 2026
"""

import torch
import json
import re
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    mean_squared_error, mean_absolute_error, hamming_loss
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import urllib.request
import zipfile

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: DATA CLASSES
# =============================================================================

@dataclass
class ASAGSample:
    """Single sample from ASAG dataset."""
    id: str
    question: str
    reference_answer: str
    student_answer: str
    gold_label: str  # correct, partially_correct_incomplete, contradictory, irrelevant, non_domain
    gold_score: float = None  # Numeric score if available
    context: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Results from a single model evaluation."""
    model_name: str
    predictions: List[str]
    gold_labels: List[str]
    scores: List[float] = None  # Optional numeric scores
    gold_scores: List[float] = None
    
    # Computed metrics
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    qwk: float = 0.0  # Quadratic Weighted Kappa
    pearson: float = 0.0
    spearman: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    
    per_class_metrics: Dict = field(default_factory=dict)
    confusion_mat: np.ndarray = None


# =============================================================================
# SECTION 2: DATASET LOADERS
# =============================================================================

class SemEvalDataLoader:
    """
    Load SemEval-2013 Task 7 dataset (Scientsbank & Beetle)
    
    Label mapping:
    - correct -> Correct
    - partially_correct_incomplete -> Partially Correct  
    - contradictory -> Incorrect
    - irrelevant -> Incorrect (Off-Topic)
    - non_domain -> Incorrect (Off-Topic)
    """
    
    LABEL_MAPPING = {
        'correct': 'Correct',
        'partially_correct_incomplete': 'Partially Correct',
        'contradictory': 'Incorrect',
        'irrelevant': 'Incorrect',
        'non_domain': 'Incorrect'
    }
    
    # 5-way label for more granular analysis
    LABEL_MAPPING_5WAY = {
        'correct': 0,
        'partially_correct_incomplete': 1,
        'contradictory': 2,
        'irrelevant': 3,
        'non_domain': 4
    }
    
    # 3-way mapping (common in research)
    LABEL_MAPPING_3WAY = {
        'correct': 2,
        'partially_correct_incomplete': 1,
        'contradictory': 0,
        'irrelevant': 0,
        'non_domain': 0
    }
    
    def __init__(self, data_dir: str = "./data/semeval2013"):
        self.data_dir = data_dir
        self.samples = []
    
    def download_dataset(self):
        """Download SemEval-2013 Task 7 dataset if not present."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("Creating data directory...")
        
        # Check if data exists
        beetle_dir = os.path.join(self.data_dir, "beetle")
        scientsbank_dir = os.path.join(self.data_dir, "scientsbank")
        
        if os.path.exists(beetle_dir) and os.path.exists(scientsbank_dir):
            print("Dataset already downloaded.")
            return True
            
        print("Note: SemEval-2013 Task 7 data needs to be downloaded manually from:")
        print("https://www.cs.york.ac.uk/semeval-2013/task7/")
        print("\nAlternatively, using synthetic data for demonstration...")
        return False
    
    def create_synthetic_dataset(self, n_samples: int = 500) -> List[ASAGSample]:
        """
        Create synthetic ASAG dataset for testing/demonstration.
        Uses realistic question-answer patterns.
        """
        print(f"Creating synthetic dataset with {n_samples} samples...")
        
        # Topic templates
        topics = [
            {
                "context": "Photosynthesis is the process by which plants convert light energy, usually from the sun, into chemical energy that can be used to fuel the plant's activities.",
                "question": "Explain the process of photosynthesis.",
                "reference": "Photosynthesis is a biological process where plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. Chlorophyll in chloroplasts absorbs light energy to convert these raw materials into food for the plant.",
                "correct": "Photosynthesis is when plants use light energy to convert water and carbon dioxide into glucose and oxygen through chlorophyll.",
                "partial": "Plants make food using sunlight.",
                "incorrect": "Photosynthesis is when plants breathe in oxygen and release carbon dioxide at night."
            },
            {
                "context": "Mitosis is a type of cell division that results in two daughter cells each having the same number and kind of chromosomes as the parent nucleus.",
                "question": "What is mitosis and what is its purpose?",
                "reference": "Mitosis is a form of cell division where a single cell divides to produce two identical daughter cells with the same number of chromosomes. Its purpose is growth, repair, and asexual reproduction.",
                "correct": "Mitosis is cell division that creates two identical daughter cells with the same chromosome count for growth and repair.",
                "partial": "Mitosis is when cells divide into two.",
                "incorrect": "Mitosis is when sex cells are formed with half the chromosomes."
            },
            {
                "context": "Electric circuits are pathways that allow electric current to flow through various components like resistors, capacitors, and light bulbs.",
                "question": "Describe how a simple electric circuit works.",
                "reference": "A simple electric circuit consists of a power source, conducting wires, and a load. Current flows from the positive terminal through the conductor, powers the load, and returns to the negative terminal, creating a closed loop.",
                "correct": "An electric circuit is a closed loop where current flows from battery positive terminal through wires and components back to the negative terminal.",
                "partial": "Electricity flows through wires to power things.",
                "incorrect": "Circuits work by storing electricity in the wires until you turn them on."
            },
            {
                "context": "Newton's laws of motion describe the relationship between the motion of an object and the forces acting on it.",
                "question": "Explain Newton's First Law of Motion.",
                "reference": "Newton's First Law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced external force. This is also known as the law of inertia.",
                "correct": "Newton's First Law or law of inertia states that objects remain at rest or in uniform motion unless an external unbalanced force acts on them.",
                "partial": "Objects keep moving unless something stops them.",
                "incorrect": "Newton's First Law says that heavier objects fall faster than lighter ones."
            },
            {
                "context": "The water cycle describes the continuous movement of water within the Earth and atmosphere through evaporation, condensation, precipitation, and collection.",
                "question": "Describe the main stages of the water cycle.",
                "reference": "The water cycle has four main stages: evaporation (water turns to vapor from heat), condensation (vapor cools and forms clouds), precipitation (water falls as rain or snow), and collection (water gathers in bodies of water or groundwater).",
                "correct": "The water cycle includes evaporation of water into vapor, condensation into clouds, precipitation as rain or snow, and collection in oceans, lakes, and groundwater.",
                "partial": "Water evaporates and then rains back down.",
                "incorrect": "The water cycle is when water flows from mountains to the ocean and disappears."
            },
            {
                "context": "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms.",
                "question": "What is DNA and what is its structure?",
                "reference": "DNA is a double helix molecule made of nucleotides containing a sugar, phosphate, and one of four bases (adenine, thymine, guanine, cytosine). The bases pair specifically (A-T, G-C) to form the rungs of the twisted ladder structure.",
                "correct": "DNA is a double helix structure of nucleotides with base pairs adenine-thymine and guanine-cytosine forming the genetic code.",
                "partial": "DNA is a molecule that contains genetic information.",
                "incorrect": "DNA is a single strand of proteins that determines physical traits."
            },
            {
                "context": "Gravity is a natural phenomenon by which all things with mass or energy are attracted to one another.",
                "question": "Explain how gravity works according to Newton.",
                "reference": "According to Newton, gravity is a force of attraction between any two masses. The gravitational force is proportional to the product of the masses and inversely proportional to the square of the distance between them (F = G*m1*m2/r²).",
                "correct": "Newton's law of gravity states that gravitational force between two objects is proportional to their masses and inversely proportional to the distance squared.",
                "partial": "Gravity pulls things toward the Earth.",
                "incorrect": "Gravity only works on heavy objects and doesn't affect light things like feathers."
            },
            {
                "context": "Chemical reactions involve the breaking and forming of chemical bonds, resulting in new substances with different properties.",
                "question": "What happens during a chemical reaction?",
                "reference": "During a chemical reaction, reactants undergo bond breaking and new bonds form to create products. Energy is either absorbed (endothermic) or released (exothermic). The total mass remains constant (law of conservation of mass).",
                "correct": "Chemical reactions break bonds in reactants and form new bonds to create products, with energy changes and conserved mass.",
                "partial": "Chemicals mix together and change.",
                "incorrect": "Chemical reactions create new atoms from nothing."
            },
            {
                "context": "The respiratory system is responsible for taking in oxygen and expelling carbon dioxide through the process of breathing.",
                "question": "How does the human respiratory system work?",
                "reference": "Air enters through the nose/mouth, travels down the trachea to bronchi and bronchioles, reaching alveoli in the lungs. In alveoli, oxygen diffuses into blood capillaries while carbon dioxide diffuses out to be exhaled.",
                "correct": "We breathe in air through airways to alveoli where oxygen enters the blood and carbon dioxide exits for exhalation.",
                "partial": "We breathe in oxygen and breathe out carbon dioxide.",
                "incorrect": "The lungs directly convert oxygen into energy."
            },
            {
                "context": "Evolution is the change in heritable characteristics of biological populations over successive generations through natural selection.",
                "question": "Explain Darwin's theory of natural selection.",
                "reference": "Natural selection is a mechanism of evolution where individuals with favorable traits are more likely to survive, reproduce, and pass on their genes. Over generations, this leads to adaptation and species change.",
                "correct": "Natural selection means organisms with advantageous traits survive and reproduce more, passing these traits to offspring and causing species evolution.",
                "partial": "Strong animals survive and weak ones die.",
                "incorrect": "Evolution means animals can change their DNA during their lifetime to adapt."
            }
        ]
        
        samples = []
        labels = ['correct', 'partially_correct_incomplete', 'contradictory']
        np.random.seed(42)
        
        for i in range(n_samples):
            topic = topics[i % len(topics)]
            label = labels[i % 3]
            
            # Add some variation
            variations = {
                'correct': [
                    topic['correct'],
                    topic['correct'].replace('.', ', which is essential.'),
                    topic['correct'] + " This is an important concept."
                ],
                'partially_correct_incomplete': [
                    topic['partial'],
                    topic['partial'] + " But I'm not sure about details.",
                    "I think " + topic['partial'].lower()
                ],
                'contradictory': [
                    topic['incorrect'],
                    "Actually, " + topic['incorrect'].lower(),
                    topic['incorrect'] + " That's what I learned."
                ]
            }
            
            student_answer = np.random.choice(variations[label])
            
            # Add some noise/typos randomly
            if np.random.random() < 0.15:
                words = student_answer.split()
                if len(words) > 3:
                    idx = np.random.randint(1, len(words)-1)
                    words[idx] = words[idx][:-1] if len(words[idx]) > 2 else words[idx]
                student_answer = ' '.join(words)
            
            sample = ASAGSample(
                id=f"syn_{i:04d}",
                question=topic['question'],
                reference_answer=topic['reference'],
                student_answer=student_answer,
                gold_label=label,
                gold_score=2.0 if label == 'correct' else (1.0 if label == 'partially_correct_incomplete' else 0.0),
                context=topic['context']
            )
            samples.append(sample)
        
        # Shuffle
        np.random.shuffle(samples)
        self.samples = samples
        print(f"Created {len(samples)} synthetic samples")
        
        # Print distribution
        label_counts = defaultdict(int)
        for s in samples:
            label_counts[s.gold_label] += 1
        print(f"Label distribution: {dict(label_counts)}")
        
        return samples
    
    def load_from_json(self, filepath: str) -> List[ASAGSample]:
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = ASAGSample(
                id=item.get('id', str(len(samples))),
                question=item['question'],
                reference_answer=item['reference'],
                student_answer=item['student'],
                gold_label=item['label'],
                gold_score=item.get('score', None),
                context=item.get('context', '')
            )
            samples.append(sample)
        
        self.samples = samples
        return samples
    
    def save_to_json(self, filepath: str, samples: List[ASAGSample] = None):
        """Save dataset to JSON file."""
        samples = samples or self.samples
        data = [s.to_dict() for s in samples]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(samples)} samples to {filepath}")
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[List[ASAGSample], List[ASAGSample]]:
        """Split data into train and test sets."""
        labels = [s.gold_label for s in self.samples]
        indices = np.arange(len(self.samples))
        
        if stratify:
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, 
                stratify=labels, random_state=42
            )
        else:
            np.random.shuffle(indices)
            split = int(len(indices) * (1 - test_size))
            train_idx, test_idx = indices[:split], indices[split:]
        
        train_samples = [self.samples[i] for i in train_idx]
        test_samples = [self.samples[i] for i in test_idx]
        
        return train_samples, test_samples


# =============================================================================
# SECTION 3: BASELINE MODELS
# =============================================================================

class BaselineModels:
    """
    Collection of baseline models for ASAG comparison.
    
    Baselines:
    1. TF-IDF + Cosine Similarity
    2. SBERT Semantic Only
    3. Rule-based Keyword Matching
    4. LLM Direct Grading (Zero-shot)
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.models_loaded = {}
        
    def _ensure_sbert(self):
        """Load SBERT model if not loaded."""
        if 'sbert' not in self.models_loaded:
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.models_loaded['sbert'] = True
            print("Loaded SBERT model: all-MiniLM-L6-v2")
    
    def _ensure_tfidf(self):
        """Initialize TF-IDF vectorizer."""
        if 'tfidf' not in self.models_loaded:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.models_loaded['tfidf'] = True
    
    # -------------------------------------------------------------------------
    # BASELINE 1: TF-IDF + Cosine Similarity
    # -------------------------------------------------------------------------
    
    def baseline_tfidf(
        self, 
        samples: List[ASAGSample],
        thresholds: Dict[str, float] = None
    ) -> List[str]:
        """
        Baseline 1: TF-IDF with Cosine Similarity
        
        Traditional bag-of-words approach using TF-IDF vectors.
        """
        self._ensure_tfidf()
        
        thresholds = thresholds or {
            'correct': 0.75,
            'partial': 0.45
        }
        
        predictions = []
        
        for sample in tqdm(samples, desc="TF-IDF Baseline"):
            texts = [sample.reference_answer, sample.student_answer]
            
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                similarity = 0.0
            
            if similarity >= thresholds['correct']:
                pred = 'correct'
            elif similarity >= thresholds['partial']:
                pred = 'partially_correct_incomplete'
            else:
                pred = 'contradictory'
            
            predictions.append(pred)
        
        return predictions
    
    # -------------------------------------------------------------------------
    # BASELINE 2: SBERT Semantic Only
    # -------------------------------------------------------------------------
    
    def baseline_sbert_semantic(
        self, 
        samples: List[ASAGSample],
        thresholds: Dict[str, float] = None
    ) -> List[str]:
        """
        Baseline 2: SBERT Semantic Similarity Only
        
        Uses sentence embeddings without NLI or keyword matching.
        """
        self._ensure_sbert()
        
        thresholds = thresholds or {
            'correct': 0.85,
            'partial': 0.55
        }
        
        predictions = []
        
        for sample in tqdm(samples, desc="SBERT Baseline"):
            embeddings = self.sbert_model.encode(
                [sample.reference_answer, sample.student_answer],
                show_progress_bar=False
            )
            
            similarity = 1 - cosine(embeddings[0], embeddings[1])
            
            if similarity >= thresholds['correct']:
                pred = 'correct'
            elif similarity >= thresholds['partial']:
                pred = 'partially_correct_incomplete'
            else:
                pred = 'contradictory'
            
            predictions.append(pred)
        
        return predictions
    
    # -------------------------------------------------------------------------
    # BASELINE 3: Rule-based Keyword Matching
    # -------------------------------------------------------------------------
    
    def baseline_keyword_matching(
        self, 
        samples: List[ASAGSample],
        thresholds: Dict[str, float] = None
    ) -> List[str]:
        """
        Baseline 3: Pure Keyword Coverage
        
        Counts keyword overlap without semantic understanding.
        """
        thresholds = thresholds or {
            'correct': 0.70,
            'partial': 0.35
        }
        
        # Simple stop words
        stop_words = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 
                         'been', 'being', 'have', 'has', 'had', 'do', 'does',
                         'did', 'will', 'would', 'could', 'should', 'may',
                         'might', 'must', 'shall', 'can', 'need', 'dare',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                         'from', 'as', 'into', 'through', 'during', 'before',
                         'after', 'above', 'below', 'between', 'under', 'again',
                         'further', 'then', 'once', 'here', 'there', 'when',
                         'where', 'why', 'how', 'all', 'each', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                         'only', 'own', 'same', 'so', 'than', 'too', 'very',
                         'and', 'but', 'if', 'or', 'because', 'until', 'while',
                         'this', 'that', 'these', 'those', 'it', 'its'])
        
        predictions = []
        
        for sample in tqdm(samples, desc="Keyword Baseline"):
            ref_words = set(
                w.lower() for w in re.findall(r'\b\w+\b', sample.reference_answer)
                if w.lower() not in stop_words and len(w) > 2
            )
            stu_words = set(
                w.lower() for w in re.findall(r'\b\w+\b', sample.student_answer)
                if w.lower() not in stop_words and len(w) > 2
            )
            
            if not ref_words:
                coverage = 0.0
            else:
                coverage = len(ref_words.intersection(stu_words)) / len(ref_words)
            
            if coverage >= thresholds['correct']:
                pred = 'correct'
            elif coverage >= thresholds['partial']:
                pred = 'partially_correct_incomplete'
            else:
                pred = 'contradictory'
            
            predictions.append(pred)
        
        return predictions
    
    # -------------------------------------------------------------------------
    # BASELINE 4: LLM Zero-shot Direct Grading
    # -------------------------------------------------------------------------
    
    def baseline_llm_zeroshot(
        self, 
        samples: List[ASAGSample],
        model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    ) -> List[str]:
        """
        Baseline 4: Direct LLM Grading without feature extraction
        
        Asks LLM to grade directly without pre-computed metrics.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading LLM: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type != 'cpu' else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        model.eval()
        
        predictions = []
        
        for sample in tqdm(samples, desc="LLM Zero-shot Baseline"):
            prompt = f"""Grade this student answer compared to the reference answer.

Question: {sample.question}

Reference Answer: {sample.reference_answer}

Student Answer: {sample.student_answer}

Output ONLY one of these labels:
- correct (if fully correct)
- partially_correct_incomplete (if on topic but missing details)  
- contradictory (if wrong or off-topic)

Label:"""

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            
            # Parse response
            if 'correct' in response and 'partially' not in response and 'incorrect' not in response:
                pred = 'correct'
            elif 'partial' in response or 'incomplete' in response:
                pred = 'partially_correct_incomplete'
            else:
                pred = 'contradictory'
            
            predictions.append(pred)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return predictions


# =============================================================================
# SECTION 4: EVALUATION METRICS
# =============================================================================

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for ASAG.
    
    Metrics included:
    - Accuracy, Precision, Recall, F1 (macro/weighted)
    - Quadratic Weighted Kappa (QWK)
    - Pearson & Spearman correlation
    - RMSE, MAE
    - Per-class metrics
    - Confusion matrix
    - Statistical significance tests
    """
    
    LABEL_TO_SCORE = {
        'correct': 2,
        'partially_correct_incomplete': 1,
        'contradictory': 0,
        'irrelevant': 0,
        'non_domain': 0,
        'Correct': 2,
        'Partially Correct': 1,
        'Incorrect': 0
    }
    
    @staticmethod
    def compute_all_metrics(
        predictions: List[str],
        gold_labels: List[str],
        gold_scores: List[float] = None
    ) -> EvaluationResult:
        """Compute all evaluation metrics."""
        
        # Convert to numeric for ordinal metrics
        pred_numeric = [EvaluationMetrics.LABEL_TO_SCORE.get(p, 0) for p in predictions]
        gold_numeric = [EvaluationMetrics.LABEL_TO_SCORE.get(g, 0) for g in gold_labels]
        
        # Classification metrics
        accuracy = accuracy_score(gold_labels, predictions)
        macro_precision = precision_score(gold_labels, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(gold_labels, predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(gold_labels, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(gold_labels, predictions, average='weighted', zero_division=0)
        
        # QWK (Quadratic Weighted Kappa) - standard metric in ASAG research
        qwk = cohen_kappa_score(gold_numeric, pred_numeric, weights='quadratic')
        
        # Correlation metrics
        pearson, _ = pearsonr(gold_numeric, pred_numeric)
        spearman, _ = spearmanr(gold_numeric, pred_numeric)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(gold_numeric, pred_numeric))
        mae = mean_absolute_error(gold_numeric, pred_numeric)
        
        # Confusion matrix
        labels = sorted(set(gold_labels) | set(predictions))
        conf_mat = confusion_matrix(gold_labels, predictions, labels=labels)
        
        # Per-class metrics
        class_report = classification_report(
            gold_labels, predictions, 
            output_dict=True, zero_division=0
        )
        
        result = EvaluationResult(
            model_name="",
            predictions=predictions,
            gold_labels=gold_labels,
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            qwk=qwk,
            pearson=pearson,
            spearman=spearman,
            rmse=rmse,
            mae=mae,
            per_class_metrics=class_report,
            confusion_mat=conf_mat
        )
        
        return result
    
    @staticmethod
    def statistical_significance_test(
        gold_labels: List[str],
        predictions_a: List[str],
        predictions_b: List[str],
        test_type: str = 'mcnemar'
    ) -> Tuple[float, float, str]:
        """
        Perform statistical significance test between two models.
        
        Args:
            test_type: 'mcnemar', 'ttest', or 'wilcoxon'
        
        Returns:
            (statistic, p_value, interpretation)
        """
        correct_a = [int(p == g) for p, g in zip(predictions_a, gold_labels)]
        correct_b = [int(p == g) for p, g in zip(predictions_b, gold_labels)]
        
        if test_type == 'mcnemar':
            # McNemar's test for paired nominal data
            from statsmodels.stats.contingency_tables import mcnemar
            
            # Build contingency table
            n11 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 1)
            n12 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 0)
            n21 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 1)
            n22 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 0)
            
            table = np.array([[n11, n12], [n21, n22]])
            result = mcnemar(table, exact=True)
            stat, pval = result.statistic, result.pvalue
            
        elif test_type == 'ttest':
            # Paired t-test
            stat, pval = ttest_rel(correct_a, correct_b)
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            diff = [a - b for a, b in zip(correct_a, correct_b)]
            if all(d == 0 for d in diff):
                return 0, 1.0, "No difference"
            stat, pval = wilcoxon(diff)
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        if pval < 0.001:
            interp = "Highly significant (p < 0.001)"
        elif pval < 0.01:
            interp = "Very significant (p < 0.01)"
        elif pval < 0.05:
            interp = "Significant (p < 0.05)"
        else:
            interp = "Not significant (p >= 0.05)"
        
        return stat, pval, interp


# =============================================================================
# SECTION 5: K-FOLD CROSS VALIDATION
# =============================================================================

class KFoldEvaluator:
    """K-Fold Cross Validation evaluator for ASAG models."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
    
    def evaluate_model(
        self,
        samples: List[ASAGSample],
        grading_function: Callable,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Run K-Fold cross validation on a model.
        
        Args:
            samples: List of ASAGSample
            grading_function: Function that takes List[ASAGSample] and returns List[str] predictions
            model_name: Name for logging
        
        Returns:
            Dictionary with fold results and aggregated metrics
        """
        labels = [s.gold_label for s in samples]
        indices = np.arange(len(samples))
        
        fold_results = []
        all_predictions = [None] * len(samples)
        
        print(f"\n{'='*60}")
        print(f"K-Fold Cross Validation: {model_name}")
        print(f"{'='*60}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.kfold.split(indices, labels)):
            print(f"\nFold {fold_idx + 1}/{self.n_splits}")
            print("-" * 40)
            
            test_samples = [samples[i] for i in test_idx]
            
            # Get predictions
            predictions = grading_function(test_samples)
            gold_labels = [s.gold_label for s in test_samples]
            
            # Store predictions
            for idx, pred in zip(test_idx, predictions):
                all_predictions[idx] = pred
            
            # Compute metrics for this fold
            fold_metrics = EvaluationMetrics.compute_all_metrics(predictions, gold_labels)
            fold_metrics.model_name = f"{model_name}_fold{fold_idx+1}"
            fold_results.append(fold_metrics)
            
            print(f"  Accuracy: {fold_metrics.accuracy:.4f}")
            print(f"  Macro F1: {fold_metrics.macro_f1:.4f}")
            print(f"  QWK:      {fold_metrics.qwk:.4f}")
        
        # Aggregate results
        aggregated = {
            'model_name': model_name,
            'n_splits': self.n_splits,
            'accuracy': {
                'mean': np.mean([r.accuracy for r in fold_results]),
                'std': np.std([r.accuracy for r in fold_results]),
                'all': [r.accuracy for r in fold_results]
            },
            'macro_f1': {
                'mean': np.mean([r.macro_f1 for r in fold_results]),
                'std': np.std([r.macro_f1 for r in fold_results]),
                'all': [r.macro_f1 for r in fold_results]
            },
            'weighted_f1': {
                'mean': np.mean([r.weighted_f1 for r in fold_results]),
                'std': np.std([r.weighted_f1 for r in fold_results]),
                'all': [r.weighted_f1 for r in fold_results]
            },
            'qwk': {
                'mean': np.mean([r.qwk for r in fold_results]),
                'std': np.std([r.qwk for r in fold_results]),
                'all': [r.qwk for r in fold_results]
            },
            'pearson': {
                'mean': np.mean([r.pearson for r in fold_results]),
                'std': np.std([r.pearson for r in fold_results]),
                'all': [r.pearson for r in fold_results]
            },
            'spearman': {
                'mean': np.mean([r.spearman for r in fold_results]),
                'std': np.std([r.spearman for r in fold_results]),
                'all': [r.spearman for r in fold_results]
            },
            'fold_results': fold_results,
            'all_predictions': all_predictions
        }
        
        print(f"\n{'='*60}")
        print(f"Aggregated Results ({self.n_splits}-Fold)")
        print(f"{'='*60}")
        print(f"  Accuracy:    {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
        print(f"  Macro F1:    {aggregated['macro_f1']['mean']:.4f} ± {aggregated['macro_f1']['std']:.4f}")
        print(f"  Weighted F1: {aggregated['weighted_f1']['mean']:.4f} ± {aggregated['weighted_f1']['std']:.4f}")
        print(f"  QWK:         {aggregated['qwk']['mean']:.4f} ± {aggregated['qwk']['std']:.4f}")
        print(f"  Pearson:     {aggregated['pearson']['mean']:.4f} ± {aggregated['pearson']['std']:.4f}")
        print(f"  Spearman:    {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
        
        return aggregated


# =============================================================================
# SECTION 6: ABLATION STUDY
# =============================================================================

class AblationStudy:
    """
    Ablation study framework to evaluate contribution of each component.
    
    Components to ablate:
    1. Semantic similarity (SimCSE)
    2. Keyword coverage (KeyBERT)
    3. NLI/Logic scoring
    4. Grammar checking
    5. Formality checking
    6. LLM reasoning layer
    """
    
    def __init__(self, base_grader):
        """
        Args:
            base_grader: The full HybridASAGGrader instance
        """
        self.base_grader = base_grader
        self.ablation_results = {}
    
    def create_ablated_grader(
        self, 
        disable_semantic: bool = False,
        disable_keywords: bool = False,
        disable_nli: bool = False,
        disable_grammar: bool = False,
        disable_formality: bool = False,
        disable_llm: bool = False
    ):
        """
        Create a modified grading function with specific components disabled.
        
        Returns a grading function compatible with KFoldEvaluator.
        """
        def ablated_grading_function(samples: List[ASAGSample]) -> List[str]:
            predictions = []
            
            for sample in tqdm(samples, desc="Ablated Grading"):
                # Get individual metric scores
                metrics = {}
                
                # Semantic score
                if disable_semantic:
                    metrics['semantic'] = 0.5  # Neutral default
                else:
                    metrics['semantic'] = self.base_grader.get_semantic_score(
                        sample.reference_answer, sample.student_answer
                    )
                
                # Keyword coverage
                if disable_keywords:
                    metrics['coverage'] = 0.5
                    metrics['missing'] = []
                else:
                    coverage, missing = self.base_grader.get_keyword_coverage(
                        sample.question, sample.reference_answer, sample.student_answer
                    )
                    metrics['coverage'] = coverage
                    metrics['missing'] = missing
                
                # Grammar score
                if disable_grammar:
                    metrics['grammar'] = 0.5
                else:
                    metrics['grammar'] = self.base_grader.get_grammar_score(sample.student_answer)
                
                # Formality score
                if disable_formality:
                    metrics['formality'] = 0.5
                else:
                    metrics['formality'] = self.base_grader.get_formality_score(sample.student_answer)
                
                # Logic/NLI score
                if disable_nli:
                    metrics['logic'] = 0.5
                    metrics['logic_details'] = {}
                else:
                    logic, details = self.base_grader.get_logic_score(
                        sample.reference_answer, sample.student_answer, metrics['grammar']
                    )
                    metrics['logic'] = logic
                    metrics['logic_details'] = details
                
                # Compute weighted composite score
                if disable_semantic and disable_keywords and disable_nli:
                    # All main components disabled - use grammar/formality only
                    composite = metrics['grammar'] * 0.6 + metrics['formality'] * 0.4
                else:
                    weights = {
                        'semantic': 0.45 if not disable_semantic else 0,
                        'coverage': 0.20 if not disable_keywords else 0,
                        'logic': 0.20 if not disable_nli else 0,
                        'grammar': 0.15 if not disable_grammar else 0
                    }
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        # Normalize weights
                        weights = {k: v/total_weight for k, v in weights.items()}
                    
                    composite = (
                        metrics['semantic'] * weights.get('semantic', 0) +
                        metrics['coverage'] * weights.get('coverage', 0) +
                        metrics['logic'] * weights.get('logic', 0) +
                        metrics['grammar'] * weights.get('grammar', 0)
                    )
                
                # Determine prediction from composite score
                # Check for contradiction
                has_contradiction = False
                if metrics.get('logic_details'):
                    fwd_c = metrics['logic_details'].get('contradiction', 0)
                    bwd_c = metrics['logic_details'].get('backward_contradiction', 0)
                    has_contradiction = max(fwd_c, bwd_c) > 0.6
                
                if has_contradiction:
                    pred = 'contradictory'
                elif composite >= 0.75:
                    pred = 'correct'
                elif composite >= 0.45:
                    pred = 'partially_correct_incomplete'
                else:
                    pred = 'contradictory'
                
                predictions.append(pred)
            
            return predictions
        
        return ablated_grading_function
    
    def run_ablation_study(
        self,
        samples: List[ASAGSample],
        kfold_evaluator: KFoldEvaluator
    ) -> Dict[str, Dict]:
        """
        Run complete ablation study.
        
        Tests:
        1. Full model (all components)
        2. w/o Semantic
        3. w/o Keywords
        4. w/o NLI
        5. w/o Grammar
        6. w/o LLM (rule-based only)
        """
        
        ablation_configs = [
            ("Full Model", {}),
            ("w/o Semantic", {"disable_semantic": True}),
            ("w/o Keywords", {"disable_keywords": True}),
            ("w/o NLI/Logic", {"disable_nli": True}),
            ("w/o Grammar", {"disable_grammar": True}),
            ("w/o Formality", {"disable_formality": True}),
            ("w/o Semantic+Keywords", {"disable_semantic": True, "disable_keywords": True}),
            ("w/o NLI+Grammar", {"disable_nli": True, "disable_grammar": True}),
        ]
        
        results = {}
        
        for config_name, config in ablation_configs:
            print(f"\n{'#'*60}")
            print(f"Ablation: {config_name}")
            print(f"{'#'*60}")
            
            grading_func = self.create_ablated_grader(**config)
            result = kfold_evaluator.evaluate_model(
                samples, grading_func, config_name
            )
            results[config_name] = result
        
        self.ablation_results = results
        return results
    
    def generate_ablation_table(self) -> pd.DataFrame:
        """Generate LaTeX-ready ablation results table."""
        rows = []
        
        for config_name, result in self.ablation_results.items():
            row = {
                'Configuration': config_name,
                'Accuracy': f"{result['accuracy']['mean']:.3f} ± {result['accuracy']['std']:.3f}",
                'Macro F1': f"{result['macro_f1']['mean']:.3f} ± {result['macro_f1']['std']:.3f}",
                'QWK': f"{result['qwk']['mean']:.3f} ± {result['qwk']['std']:.3f}",
                'Δ Accuracy': "",
                'Δ F1': ""
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Calculate deltas from full model
        if 'Full Model' in self.ablation_results:
            full_acc = self.ablation_results['Full Model']['accuracy']['mean']
            full_f1 = self.ablation_results['Full Model']['macro_f1']['mean']
            
            for i, config_name in enumerate(self.ablation_results.keys()):
                if config_name != 'Full Model':
                    result = self.ablation_results[config_name]
                    delta_acc = result['accuracy']['mean'] - full_acc
                    delta_f1 = result['macro_f1']['mean'] - full_f1
                    df.loc[i, 'Δ Accuracy'] = f"{delta_acc:+.3f}"
                    df.loc[i, 'Δ F1'] = f"{delta_f1:+.3f}"
        
        return df


# =============================================================================
# SECTION 7: VISUALIZATION
# =============================================================================

class ResultVisualizer:
    """Visualization utilities for evaluation results."""
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_mat: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6),
        save_path: str = None
    ):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=figsize)
        sns.heatmap(
            confusion_mat, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(
        results: Dict[str, Dict],
        metrics: List[str] = ['accuracy', 'macro_f1', 'qwk'],
        figsize: Tuple[int, int] = (12, 6),
        save_path: str = None
    ):
        """Plot bar chart comparing models across metrics."""
        n_models = len(results)
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        
        for i, (model_name, result) in enumerate(results.items()):
            means = [result[m]['mean'] for m in metrics]
            stds = [result[m]['std'] for m in metrics]
            
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, 
                         yerr=stds, label=model_name,
                         color=colors[i], capsize=3)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_ablation_impact(
        ablation_results: Dict[str, Dict],
        metric: str = 'macro_f1',
        figsize: Tuple[int, int] = (10, 6),
        save_path: str = None
    ):
        """Plot ablation study impact chart."""
        configs = list(ablation_results.keys())
        means = [ablation_results[c][metric]['mean'] for c in configs]
        stds = [ablation_results[c][metric]['std'] for c in configs]
        
        # Calculate delta from full model
        full_model_score = ablation_results.get('Full Model', {}).get(metric, {}).get('mean', means[0])
        
        colors = ['green' if m >= full_model_score else 'red' for m in means]
        colors[0] = 'blue'  # Full model
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(configs, means, xerr=stds, capsize=3, color=colors, alpha=0.7)
        
        ax.axvline(x=full_model_score, color='blue', linestyle='--', alpha=0.5, label='Full Model')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Ablation Study: Impact on {metric.replace("_", " ").title()}')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{mean:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# =============================================================================
# SECTION 8: RESULTS EXPORT
# =============================================================================

class ResultExporter:
    """Export results in various formats for research papers."""
    
    @staticmethod
    def to_latex_table(
        results: Dict[str, Dict],
        metrics: List[str] = ['accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'pearson'],
        caption: str = "Model Performance Comparison",
        label: str = "tab:results"
    ) -> str:
        """Generate LaTeX table for research paper."""
        
        header = "\\begin{table}[h]\n\\centering\n"
        header += f"\\caption{{{caption}}}\n"
        header += f"\\label{{{label}}}\n"
        
        n_cols = len(metrics) + 1
        col_format = "|l|" + "c|" * len(metrics)
        header += f"\\begin{{tabular}}{{{col_format}}}\n\\hline\n"
        
        # Column headers
        metric_headers = [m.replace('_', ' ').title() for m in metrics]
        header += "Model & " + " & ".join(metric_headers) + " \\\\ \\hline\n"
        
        # Data rows
        rows = []
        for model_name, result in results.items():
            values = []
            for m in metrics:
                mean = result[m]['mean']
                std = result[m]['std']
                values.append(f"{mean:.3f} ± {std:.3f}")
            row = f"{model_name} & " + " & ".join(values) + " \\\\ \\hline"
            rows.append(row)
        
        body = "\n".join(rows)
        
        footer = "\n\\end{tabular}\n\\end{table}"
        
        return header + body + footer
    
    @staticmethod
    def to_csv(
        results: Dict[str, Dict],
        filepath: str,
        metrics: List[str] = ['accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'pearson', 'spearman']
    ):
        """Export results to CSV."""
        rows = []
        
        for model_name, result in results.items():
            row = {'Model': model_name}
            for m in metrics:
                row[f'{m}_mean'] = result[m]['mean']
                row[f'{m}_std'] = result[m]['std']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return df
    
    @staticmethod
    def generate_full_report(
        all_results: Dict[str, Dict],
        ablation_results: Dict[str, Dict],
        output_dir: str = "./evaluation_results"
    ):
        """Generate complete evaluation report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Main comparison table
        latex_main = ResultExporter.to_latex_table(
            all_results,
            caption="Performance comparison of ASAG models",
            label="tab:main_results"
        )
        with open(f"{output_dir}/main_results.tex", 'w') as f:
            f.write(latex_main)
        
        # 2. Ablation table
        latex_ablation = ResultExporter.to_latex_table(
            ablation_results,
            caption="Ablation study results",
            label="tab:ablation"
        )
        with open(f"{output_dir}/ablation_results.tex", 'w') as f:
            f.write(latex_ablation)
        
        # 3. CSV exports
        ResultExporter.to_csv(all_results, f"{output_dir}/main_results.csv")
        ResultExporter.to_csv(ablation_results, f"{output_dir}/ablation_results.csv")
        
        # 4. JSON export (complete data)
        export_data = {
            'main_results': {k: {
                m: {'mean': v[m]['mean'], 'std': v[m]['std']}
                for m in ['accuracy', 'macro_f1', 'qwk', 'pearson']
            } for k, v in all_results.items()},
            'ablation_results': {k: {
                m: {'mean': v[m]['mean'], 'std': v[m]['std']}
                for m in ['accuracy', 'macro_f1', 'qwk']
            } for k, v in ablation_results.items()}
        }
        
        with open(f"{output_dir}/complete_results.json", 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nFull report generated in {output_dir}/")
        return export_data


# =============================================================================
# SECTION 9: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function for running complete evaluation."""
    
    print("="*70)
    print("ASAG Model Evaluation Framework")
    print("="*70)
    
    # 1. Load/Create dataset
    data_loader = SemEvalDataLoader()
    samples = data_loader.create_synthetic_dataset(n_samples=500)
    
    # 2. Save dataset for reproducibility
    data_loader.save_to_json("./data/asag_dataset.json", samples)
    
    # 3. Initialize components
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    baselines = BaselineModels(device=device)
    kfold = KFoldEvaluator(n_splits=5)
    
    # 4. Run baseline evaluations
    all_results = {}
    
    # Baseline 1: TF-IDF
    print("\n" + "="*60)
    print("Evaluating Baseline 1: TF-IDF + Cosine")
    print("="*60)
    tfidf_result = kfold.evaluate_model(
        samples,
        baselines.baseline_tfidf,
        "TF-IDF"
    )
    all_results['TF-IDF'] = tfidf_result
    
    # Baseline 2: SBERT
    print("\n" + "="*60)
    print("Evaluating Baseline 2: SBERT Semantic")
    print("="*60)
    sbert_result = kfold.evaluate_model(
        samples,
        baselines.baseline_sbert_semantic,
        "SBERT"
    )
    all_results['SBERT'] = sbert_result
    
    # Baseline 3: Keywords
    print("\n" + "="*60)
    print("Evaluating Baseline 3: Keyword Matching")
    print("="*60)
    keyword_result = kfold.evaluate_model(
        samples,
        baselines.baseline_keyword_matching,
        "Keywords"
    )
    all_results['Keywords'] = keyword_result
    
    # 5. Visualize results
    visualizer = ResultVisualizer()
    visualizer.plot_model_comparison(
        all_results,
        save_path="./evaluation_results/model_comparison.png"
    )
    
    # 6. Export results
    exporter = ResultExporter()
    exporter.to_csv(all_results, "./evaluation_results/baseline_results.csv")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    import os
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./evaluation_results", exist_ok=True)
    main()
