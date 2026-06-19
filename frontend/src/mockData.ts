import type { GradingResult, Weights, MetricScores, SystemThresholds } from './types';

export const DEFAULT_WEIGHTS: Weights = {
  semantic: 0.20,
  coverage: 0.20,
  formality: 0.20,
  grammar: 0.20,
  logic: 0.20,
};

export const WEIGHT_PRESETS: Record<string, Weights> = {
  balanced: { semantic: 0.20, coverage: 0.20, formality: 0.20, grammar: 0.20, logic: 0.20 },
  content_focused: { semantic: 0.40, coverage: 0.30, formality: 0.05, grammar: 0.10, logic: 0.15 },
  academic_writing: { semantic: 0.20, coverage: 0.15, formality: 0.25, grammar: 0.25, logic: 0.15 },
  logic_heavy: { semantic: 0.25, coverage: 0.15, formality: 0.10, grammar: 0.15, logic: 0.35 },
  quick_check: { semantic: 0.50, coverage: 0.30, formality: 0.05, grammar: 0.05, logic: 0.10 },
};

export const DEFAULT_THRESHOLDS: SystemThresholds = {
  semantic_correct: 0.85,
  semantic_partial: 0.55,
  semantic_off_topic: 0.35,
  coverage_correct: 0.80,
  coverage_good: 0.60,
  coverage_missing: 0.60,
  grammar_good: 0.50,
  grammar_poor: 0.35,
  logic_correct: 0.50,
  logic_good: 0.40,
  logic_error: 0.20,
  contradiction_high: 0.60,
  contradiction_moderate: 0.35,
  formality_poor: 0.10,
};

export function calculateLocalGrade(metrics: Omit<MetricScores, 'final_grade'>, weights: Weights): number {
  const totalWeight = weights.semantic + weights.coverage + weights.formality + weights.grammar + weights.logic;
  const normWeights = totalWeight > 0 
    ? {
        semantic: weights.semantic / totalWeight,
        coverage: weights.coverage / totalWeight,
        formality: weights.formality / totalWeight,
        grammar: weights.grammar / totalWeight,
        logic: weights.logic / totalWeight,
      }
    : DEFAULT_WEIGHTS;

  let baseGrade = (
    metrics.semantic_score * normWeights.semantic +
    metrics.coverage_score * normWeights.coverage +
    metrics.formality_score * normWeights.formality +
    metrics.grammar_score * normWeights.grammar +
    metrics.logic_score * normWeights.logic
  ) * 100;

  // Apply contradiction penalties
  if (metrics.logic_details) {
    const fwd = metrics.logic_details.contradiction || 0;
    const bwd = metrics.logic_details.backward_contradiction || 0;
    const maxContradiction = Math.max(fwd, bwd);

    if (maxContradiction > 0.90) {
      baseGrade = Math.min(baseGrade, 40.0);
    } else if (maxContradiction > 0.70) {
      baseGrade = Math.min(baseGrade, 55.0);
    } else if (maxContradiction > 0.50) {
      const penalty = (maxContradiction - 0.50) * 30;
      baseGrade = Math.max(0, baseGrade - penalty);
    }
  }

  // Quality bonus
  if (
    metrics.semantic_score >= 0.95 &&
    metrics.grammar_score >= 0.90 &&
    metrics.logic_score >= 0.50 &&
    metrics.missing_keywords.length <= 1
  ) {
    baseGrade = Math.max(baseGrade, 85.0);
  }

  return Math.round(Math.min(100.0, Math.max(0.0, baseGrade)) * 100) / 100;
}

export function simulateGrading(
  _context: string,
  _question: string,
  reference: string,
  student: string,
  weights: Weights = DEFAULT_WEIGHTS
): GradingResult {
  if (!student || !student.trim()) {
    return {
      metrics: {
        semantic_score: 0,
        coverage_score: 0,
        missing_keywords: [],
        formality_score: 0,
        grammar_score: 0,
        logic_score: 0,
        logic_details: {},
        final_grade: 0,
      },
      feedback: {
        tags: ["Incorrect"],
        explanation: "Empty student answer.",
        suggestion: "Please provide an answer to grade.",
        raw_response: "",
        parse_success: false,
      },
      is_valid: false,
      skip_reason: "Empty student answer",
    };
  }

  const wordCount = student.trim().split(/\s+/).length;
  if (wordCount < 3) {
    const semantic = 0.15;
    const metrics: Omit<MetricScores, 'final_grade'> = {
      semantic_score: semantic,
      coverage_score: semantic,
      missing_keywords: [],
      formality_score: 0.5,
      grammar_score: 0.5,
      logic_score: semantic,
      logic_details: {},
    };
    const final_grade = calculateLocalGrade(metrics, weights);

    return {
      metrics: { ...metrics, final_grade },
      feedback: {
        tags: ["Incomplete", "Vague Expression"],
        explanation: `Answer is too short (${wordCount} words) to perform a detailed evaluation.`,
        suggestion: "Please expand your answer to include core concepts, verb forms, and supporting information.",
        raw_response: "",
        parse_success: true,
      },
      is_valid: true,
      skip_reason: null,
    };
  }

  // Basic keyword analysis
  const refWords = reference.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "").split(/\s+/);
  const stuWords = student.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "").split(/\s+/);
  
  // Extract key vocabulary (excluding short common words)
  const stopWords = new Set(["the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in", "that", "it", "for", "with", "by", "on", "as", "at", "by", "an", "this"]);
  const refKeywords = Array.from(new Set(refWords.filter(w => w.length > 3 && !stopWords.has(w)))).slice(0, 5);

  const foundKeywords = refKeywords.filter(keyword => {
    // Exact or partial match
    if (student.toLowerCase().includes(keyword)) return true;
    // Simple edit distance / similarity simulation
    return stuWords.some(w => {
      if (w === keyword) return true;
      if (w.startsWith(keyword.substring(0, Math.max(3, keyword.length - 2)))) return true;
      return false;
    });
  });

  const missingKeywords = refKeywords.filter(k => !foundKeywords.includes(k));
  const coverage_score = refKeywords.length > 0 ? foundKeywords.length / refKeywords.length : 1.0;

  // Semantic Similarity Simulation based on keyword overlap and string matching
  const commonWordsCount = stuWords.filter(w => refWords.includes(w)).length;
  const jaccard = commonWordsCount / (new Set([...refWords, ...stuWords]).size || 1);
  let semantic_score = Math.min(1.0, Math.max(0.1, jaccard * 1.5 + coverage_score * 0.4));

  // Grammar Simulation: Check capitalization and common errors
  let grammar_score = 0.95;
  if (student[0] !== student[0].toUpperCase()) grammar_score -= 0.15;
  if (!student.trim().endsWith('.') && !student.trim().endsWith('?') && !student.trim().endsWith('!')) grammar_score -= 0.10;
  // Look for lowercase i contractions or obvious bad grammar
  if (/\b(i|dont|cant|wont|isnt|aint|go to|happen many)\b/.test(student.toLowerCase())) {
    grammar_score -= 0.30;
  }
  grammar_score = Math.max(0.1, grammar_score);

  // Formality Simulation: Check length and academic words
  const academicTerms = ["process", "convert", "transform", "synthesize", "cycle", "absorb", "consequence", "mechanism", "structure", "function", "system", "biological"];
  const academicCount = academicTerms.filter(term => student.toLowerCase().includes(term)).length;
  let formality_score = Math.min(1.0, 0.4 + (academicCount * 0.15) + (wordCount / 40));
  if (/\b(like|stuff|thing|cool|guy|guyz|wanna|gonna)\b/.test(student.toLowerCase())) {
    formality_score -= 0.25;
  }
  formality_score = Math.max(0.1, formality_score);

  // Logic & Contradiction Simulation:
  // Detect if the student says something contradicts reference (e.g. "not", "no", "never", "opposite", "destroy" when reference doesn't have it)
  const negationWords = ["not", "never", "no", "opposite", "contradict", "instead of", "unlike"];
  const refHasNegation = negationWords.some(w => reference.toLowerCase().includes(w));
  const stuHasNegation = negationWords.some(w => student.toLowerCase().includes(w));

  let contradiction = 0.02;
  if (stuHasNegation !== refHasNegation) {
    // If student answer has negative word and reference doesn't, or vice-versa, simulate contradiction
    contradiction = 0.72;
  }
  
  // Specific checks
  if (student.toLowerCase().includes("animal") && reference.toLowerCase().includes("plant")) {
    contradiction = 0.96; // Severe error (e.g. "animals do photosynthesis")
  }

  const backward_contradiction = contradiction > 0.5 ? contradiction - 0.05 : 0.01;
  const entailment = contradiction > 0.5 ? 0.02 : Math.max(0.1, semantic_score - 0.1);
  const neutral = 1.0 - contradiction - entailment;

  const logic_details = {
    contradiction,
    entailment,
    neutral,
    backward_entailment: entailment * 0.9,
    backward_contradiction,
    backward_neutral: neutral * 1.1,
  };

  const logic_score = contradiction > 0.60 
    ? (1 - contradiction) * 0.15 
    : (entailment * 0.7 + neutral * 0.3);

  // Put metrics together
  const metrics: Omit<MetricScores, 'final_grade'> = {
    semantic_score,
    coverage_score,
    missing_keywords: missingKeywords,
    formality_score,
    grammar_score,
    logic_score,
    logic_details,
  };

  const final_grade = calculateLocalGrade(metrics, weights);

  // Assign tags
  const tags: string[] = [];
  let correctness = "Partially Correct";

  if (contradiction > 0.6) {
    correctness = "Incorrect";
    tags.push("Incorrect", "Logical Error", "Factual Error");
  } else if (semantic_score >= 0.85 && logic_score >= 0.50 && grammar_score >= 0.50) {
    correctness = "Correct";
    tags.push("Correct");
  } else if (semantic_score < 0.35) {
    correctness = "Incorrect";
    tags.push("Incorrect", "Off-Topic");
  } else {
    correctness = "Partially Correct";
    tags.push("Partially Correct");
  }

  if (correctness !== "Correct") {
    if (missingKeywords.length >= 2 || coverage_score < 0.60) {
      tags.push("Missing Concepts");
    }
    if (grammar_score < 0.50) {
      tags.push("Grammar Error");
    }
    if (semantic_score >= 0.55 && coverage_score < 0.60 && missingKeywords.length === 0) {
      tags.push("Vague Expression");
    }
  } else {
    // Correct tags post-processing: check if missing too many keywords
    if (missingKeywords.length >= 2) {
      tags[0] = "Partially Correct";
      correctness = "Partially Correct";
      tags.push("Missing Concepts");
    }
  }

  // Explanations & Suggestions
  let explanation = "";
  let suggestion = "";

  if (correctness === "Correct") {
    explanation = "Excellent response! The student's answer accurately captures the core concepts and meaning of the reference answer, demonstrating a high-level understanding.";
    suggestion = "Excellent work. You can further improve by making your explanation even more concise.";
  } else {
    const explanationsList: string[] = [];
    const suggestionsList: string[] = [];

    if (tags.includes("Off-Topic")) {
      explanationsList.push(`The answer deviates significantly from the target question (semantic matching: ${Math.round(semantic_score * 100)}%).`);
      suggestionsList.push("Focus on the specific question topic. Re-read the question prompt and make sure you address the exact concepts requested.");
    }
    if (tags.includes("Logical Error")) {
      explanationsList.push("The answer contains statements that directly contradict the reference key facts.");
      suggestionsList.push("Check your facts. Ensure you are not asserting the opposite of the reference answer's statements (e.g. confusing plants and animals).");
    }
    if (tags.includes("Missing Concepts") && missingKeywords.length > 0) {
      explanationsList.push(`Key terms from the reference answer are missing: ${missingKeywords.join(", ")}.`);
      suggestionsList.push(`Try to explicitly include the missing concepts: '${missingKeywords.slice(0, 3).join("', '")}'. Explain how they connect to the answer.`);
    }
    if (tags.includes("Grammar Error")) {
      explanationsList.push(`The answer has grammar/acceptability issues (grammar rating: ${Math.round(grammar_score * 100)}%).`);
      suggestionsList.push("Proofread your work. Fix capitalizations, write in complete sentences, and use proper punctuation (periods).");
    }
    if (tags.includes("Vague Expression")) {
      explanationsList.push("The wording is somewhat vague and lacks the technical precision found in the reference.");
      suggestionsList.push("Use more specific, academic vocabulary. Avoid generic descriptions like 'make stuff' and instead use precise terms like 'synthesize' or 'convert'.");
    }

    if (explanationsList.length === 0) {
      explanationsList.push("The answer is on-topic but incomplete or slightly flawed in logic and structure.");
      suggestionsList.push("Review the reference response and expand on the details. Connect the steps more clearly.");
    }

    explanation = explanationsList.join(" ");
    suggestion = suggestionsList.join(" ");
  }

  const raw_response = JSON.stringify({ tags, explanation, suggestion });

  return {
    metrics: { ...metrics, final_grade },
    feedback: {
      tags,
      explanation,
      suggestion,
      raw_response,
      parse_success: true,
    },
    is_valid: true,
    skip_reason: null,
  };
}
