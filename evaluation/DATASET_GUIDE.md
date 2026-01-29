# Hướng dẫn sử dụng Public ASAG Datasets

## 1. SemEval-2013 Task 7 (Recommended)

### Mô tả
- **Beetle**: 4,000+ student answers về điện tử cơ bản
- **Scientsbank**: 10,000+ student answers về khoa học
- **5-way labels**: correct, partially_correct_incomplete, contradictory, irrelevant, non_domain
- **3-way labels**: correct, partially_correct_incomplete, incorrect (gộp 3 loại cuối)

### Download
1. Truy cập: https://www.cs.york.ac.uk/semeval-2013/task7/
2. Download `semeval2013-task7.zip`
3. Giải nén vào `./data/semeval2013/`

### Cấu trúc thư mục
```
data/
└── semeval2013/
    ├── beetle/
    │   ├── train/
    │   │   └── Core/
    │   ├── test-unseen-answers/
    │   └── test-unseen-questions/
    └── scientsbank/
        ├── train/
        └── test/
```

### Code load SemEval XML
```python
import xml.etree.ElementTree as ET
from pathlib import Path

def load_semeval_beetle(data_dir: str) -> List[ASAGSample]:
    """Load Beetle dataset from SemEval-2013."""
    samples = []
    
    train_dir = Path(data_dir) / "beetle" / "train" / "Core"
    
    for xml_file in train_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get question info
        question_elem = root.find(".//questionText")
        question = question_elem.text if question_elem is not None else ""
        
        ref_elem = root.find(".//referenceAnswer")
        reference = ref_elem.text if ref_elem is not None else ""
        
        # Get student answers
        for answer in root.findall(".//studentAnswer"):
            student_text = answer.text or ""
            accuracy = answer.get("accuracy", "unknown")
            answer_id = answer.get("id", str(len(samples)))
            
            sample = ASAGSample(
                id=answer_id,
                question=question,
                reference_answer=reference,
                student_answer=student_text,
                gold_label=accuracy,
                context=""
            )
            samples.append(sample)
    
    return samples
```

---

## 2. Mohler Dataset (2011)

### Mô tả
- 2,273 student answers
- Computer Science domain (Data Structures)
- **Numeric scores**: 0-5 scale (có thể dùng cho regression)
- 2 annotators với inter-annotator agreement

### Download
- Paper: "Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments" (Mohler et al., 2011)
- Dataset: Liên hệ tác giả hoặc tìm trên GitHub mirrors

### Format
```
QuestionID | Question | Reference | StudentAnswer | Score1 | Score2
```

### Code load Mohler
```python
def load_mohler_dataset(filepath: str) -> List[ASAGSample]:
    """Load Mohler dataset from TSV file."""
    import pandas as pd
    
    df = pd.read_csv(filepath, sep='\t', header=0)
    samples = []
    
    for idx, row in df.iterrows():
        # Average of two annotators
        avg_score = (row['Score1'] + row['Score2']) / 2
        
        # Convert to 3-way label
        if avg_score >= 4.0:
            label = 'correct'
        elif avg_score >= 2.5:
            label = 'partially_correct_incomplete'
        else:
            label = 'contradictory'
        
        sample = ASAGSample(
            id=str(idx),
            question=row['Question'],
            reference_answer=row['Reference'],
            student_answer=row['StudentAnswer'],
            gold_label=label,
            gold_score=avg_score
        )
        samples.append(sample)
    
    return samples
```

---

## 3. ASAP-SAS (Kaggle)

### Mô tả
- Automated Student Assessment Prize - Short Answer Scoring
- 10 prompts với 17,000+ responses
- Science domain
- **Numeric scores**: 0-2 hoặc 0-3 tùy prompt

### Download
- Kaggle: https://www.kaggle.com/c/asap-sas/data
- Yêu cầu đăng ký competition

### Code load ASAP-SAS
```python
def load_asap_sas(filepath: str, prompt_id: int = 1) -> List[ASAGSample]:
    """Load ASAP-SAS dataset for a specific prompt."""
    import pandas as pd
    
    df = pd.read_csv(filepath, sep='\t')
    df = df[df['EssaySet'] == prompt_id]
    
    samples = []
    max_score = df['Score1'].max()
    
    for idx, row in df.iterrows():
        score = row['Score1']
        
        # Normalize to 3-way
        if score == max_score:
            label = 'correct'
        elif score >= max_score / 2:
            label = 'partially_correct_incomplete'
        else:
            label = 'contradictory'
        
        sample = ASAGSample(
            id=str(row['Id']),
            question=f"Prompt {prompt_id}",  # Actual prompt text available separately
            reference_answer="",  # Need to add from rubric
            student_answer=row['EssayText'],
            gold_label=label,
            gold_score=score
        )
        samples.append(sample)
    
    return samples
```

---

## 4. Texas Dataset (Dzikovska et al., 2013)

### Mô tả
- Subset of SemEval data
- Focus on science tutoring dialogs
- Good for evaluation of explanation quality

---

## Recommended Setup for Your Research

### Option 1: SemEval-2013 Only (Simplest)
```python
# Load both Beetle and Scientsbank
beetle_samples = load_semeval_beetle("./data/semeval2013")
scientsbank_samples = load_semeval_scientsbank("./data/semeval2013")

# Combine
all_samples = beetle_samples + scientsbank_samples
print(f"Total: {len(all_samples)} samples")

# Run evaluation
kfold = KFoldEvaluator(n_splits=5)
results = kfold.evaluate_model(all_samples, hybrid_grader_function, "Hybrid-ASAG")
```

### Option 2: Multi-Dataset (More Comprehensive)
```python
# Train on SemEval, test on Mohler (cross-domain)
train_samples = load_semeval_beetle("./data/semeval2013")
test_samples = load_mohler_dataset("./data/mohler/data.tsv")

# This shows generalization ability
```

### Option 3: Synthetic + Real (For Development)
```python
# Use synthetic for rapid iteration
synthetic = data_loader.create_synthetic_dataset(n_samples=500)

# Final evaluation on real data
real_samples = load_semeval_beetle("./data/semeval2013")
```

---

## Metrics Comparison với State-of-the-Art

### SemEval-2013 Leaderboard (3-way)
| System | Accuracy | QWK |
|--------|----------|-----|
| ETS (Best) | 71.2% | 0.72 |
| CoMiC | 66.8% | 0.65 |
| CU | 59.9% | 0.58 |

### Mohler Dataset SOTA
| System | Pearson | RMSE |
|--------|---------|------|
| BERT Fine-tuned | 0.78 | 0.95 |
| SBERT | 0.72 | 1.05 |
| TF-IDF | 0.52 | 1.45 |

### Your Goal
- Beat SBERT baseline by 5-10% on F1
- Show improvement from each component (ablation)
- Demonstrate statistical significance (p < 0.05)

---

## Tips for Research Paper

1. **Report both 3-way and 5-way results** để so sánh với nhiều papers
2. **Use QWK** (Quadratic Weighted Kappa) - metric chuẩn trong ASAG
3. **Cross-domain evaluation** - train on Beetle, test on Scientsbank
4. **Error analysis** - phân tích các trường hợp sai để explain
5. **Ablation study** - chứng minh từng component đều cần thiết

---

## File Structure cho Project hoàn chỉnh

```
autograde_final/
├── data/
│   ├── semeval2013/
│   │   ├── beetle/
│   │   └── scientsbank/
│   ├── mohler/
│   └── asag_evaluation_dataset.json
├── evaluation_results/
│   ├── figures/
│   │   ├── model_comparison.png
│   │   ├── ablation_impact.png
│   │   └── confusion_matrices.png
│   ├── main_results.csv
│   ├── ablation_results.csv
│   ├── main_results.tex
│   └── complete_results.json
├── hybrid_asag_grader.ipynb
├── evaluation_framework.py
├── run_evaluation.ipynb
└── README.md
```
