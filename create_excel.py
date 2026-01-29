import pandas as pd
import json

# Test cases from the notebook
test_cases = [
    {
        "name": "Good Answer",
        "context": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
        "question": "Explain the process of photosynthesis and its importance.",
        "student": "Photosynthesis is when plants use sunlight, CO2, and water to make glucose and oxygen. Chlorophyll helps capture the light energy. This process is important because it provides food for plants and oxygen for animals to breathe."
    },
    {
        "name": "Partial Answer (Missing Keywords)",
        "context": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
        "question": "Explain the process of photosynthesis and its importance.",
        "student": "Plants make their own food using light. This is important for the environment."
    },
    {
        "name": "Incorrect Answer",
        "context": "Photosynthesis is the process by which green plants use sunlight to synthesize foods.",
        "question": "Explain the process of photosynthesis.",
        "student": "Photosynthesis is when animals eat plants to get energy. The plants die and become fertilizer for other plants."
    },
    {
        "name": "Grammar Issues",
        "context": "The water cycle describes how water evaporates from the surface of the earth.",
        "question": "Describe the water cycle.",
        "student": "water cycle is when water go up to sky and then it rain down again and again this happen many time"
    },
    {
        "name": "Very Short Answer",
        "context": "The water cycle describes continuous movement of water.",
        "question": "Describe the water cycle.",
        "student": "water evaporation"
    }
]

# Load grading results
with open('grading_results.json', 'r', encoding='utf-8') as f:
    grading_results = json.load(f)

# Create data for Excel
data = []
for i, (test_case, result) in enumerate(zip(test_cases, grading_results), 1):
    # Get tags as string
    tags = result['feedback']['tags']
    tags_str = ', '.join(tags)
    
    # Get final grade
    final_grade = result['metrics']['final_grade']
    
    # Create Gold Score / Tag column
    gold_score_tag = f"{final_grade}/100 | {tags_str}"
    
    row = {
        'Case': i,
        'Context (C)': test_case['context'],
        'Question (Q)': test_case['question'],
        'Student Answer (A_stu)': test_case['student'],
        'Gold Score / Tag': gold_score_tag,
        'Explanation': result['feedback']['explanation'],
        'Suggestion': result['feedback']['suggestion']
    }
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
excel_path = 'grading_samples.xlsx'
df.to_excel(excel_path, index=False, engine='openpyxl')

print(f"Excel file created: {excel_path}")
print(f"\nPreview:")
print(df[['Case', 'Gold Score / Tag']].to_string(index=False))
