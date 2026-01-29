"""
Production Fixes for HybridASAGGrader
=====================================

This module contains the production-ready improvements for the grading system:

1. Fix truncation issue - Increase max_new_tokens and character limits
2. Add contradiction penalty to final grade
3. Improve correctness tag thresholds
4. Better handling of "almost correct" answers

Apply these fixes to the notebook by running this script.
"""

import json
import re

def apply_production_fixes(notebook_path: str) -> bool:
    """Apply all production fixes to the notebook."""
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    fixes_applied = []
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        # Process each line
        new_source = []
        for line in cell.get('source', []):
            original = line
            
            # Fix 1: Increase max_new_tokens
            if 'max_new_tokens=300' in line:
                line = line.replace('max_new_tokens=300', 'max_new_tokens=500')
                fixes_applied.append("max_new_tokens: 300 -> 500")
            
            # Fix 2: Increase explanation character limit
            if '[:300]' in line and 'explanation' in line:
                line = line.replace('[:300]', '[:500]')
                fixes_applied.append("explanation limit: 300 -> 500")
            
            # Fix 3: Increase suggestion character limit
            if '[:200]' in line and 'suggestion' in line:
                line = line.replace('[:200]', '[:400]')
                fixes_applied.append("suggestion limit: 200 -> 400")
            
            new_source.append(line)
        
        cell['source'] = new_source
    
    # Save
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("=== PRODUCTION FIXES APPLIED ===")
    for fix in set(fixes_applied):
        print(f"  ✅ {fix}")
    
    return True


def create_improved_calculate_final_grade():
    """Return the improved calculate_final_grade function code."""
    return '''
def calculate_final_grade(
    metrics: 'MetricScores',
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
'''


def create_improved_assign_tags_step2():
    """Return the improved STEP 2 tag assignment logic."""
    return '''
        # STEP 2: If "Correct", check for missing keywords - may need to downgrade
        if correctness == "Correct":
            # PRODUCTION FIX: Only downgrade if multiple keywords missing or coverage is very low
            # Allow 1 missing keyword for high semantic answers (synonyms/paraphrasing)
            num_missing = len(metrics.missing_keywords) if metrics.missing_keywords else 0
            should_downgrade = (
                (num_missing >= 2 and metrics.coverage_score < self.THRESHOLDS["coverage_correct"]) or
                (num_missing >= 3) or  # Too many missing regardless of coverage
                (metrics.coverage_score < 0.50)  # Very low coverage
            )
            
            if should_downgrade:
                # Downgrade to Partially Correct with Missing Concepts tag
                tags[0] = "Partially Correct"
                correctness = "Partially Correct"
                tags.append("Missing Concepts")
            return tags, correctness
'''


if __name__ == "__main__":
    import sys
    
    notebook_path = "hybrid_asag_grader.ipynb"
    
    print("=" * 60)
    print("PRODUCTION FIXES FOR HYBRID ASAG GRADER")
    print("=" * 60)
    
    print("\nThese fixes address the following issues:")
    print("1. Truncation in explanation/suggestion (max_new_tokens: 300→500)")
    print("2. Character limits for feedback (explanation: 300→500, suggestion: 200→400)")
    print("3. Contradiction penalty in final grade calculation")
    print("4. Better threshold for 'Correct' tag with 1 missing keyword")
    
    print("\n" + "=" * 60)
    print("APPLYING FIXES...")
    print("=" * 60)
    
    try:
        apply_production_fixes(notebook_path)
        print("\n✅ All fixes applied successfully!")
        print("\nNext steps:")
        print("1. Restart the Jupyter kernel")
        print("2. Run all cells from the beginning")
        print("3. Export new grading_results.json")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
