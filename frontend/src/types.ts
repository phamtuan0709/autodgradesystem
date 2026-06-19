export interface MetricScores {
  semantic_score: number;
  coverage_score: number;
  missing_keywords: string[];
  formality_score: number;
  grammar_score: number;
  logic_score: number;
  logic_details: {
    contradiction?: number;
    entailment?: number;
    neutral?: number;
    backward_entailment?: number;
    backward_contradiction?: number;
    backward_neutral?: number;
    [key: string]: number | undefined;
  };
  final_grade: number;
}

export interface LLMFeedback {
  tags: string[];
  explanation: string;
  suggestion: string;
  raw_response: string;
  parse_success: boolean;
}

export interface GradingResult {
  metrics: MetricScores;
  feedback: LLMFeedback;
  is_valid: boolean;
  skip_reason: string | null;
}

export interface Weights {
  semantic: number;
  coverage: number;
  formality: number;
  grammar: number;
  logic: number;
}

export interface GradeRequest {
  context: string;
  question: string;
  reference: string;
  student: string;
  weights: Weights | null;
}

export interface BatchItem {
  id: string;
  context: string;
  question: string;
  reference: string;
  student: string;
  status: 'idle' | 'grading' | 'success' | 'failed';
  result?: GradingResult;
  error?: string;
}

export interface SystemThresholds {
  semantic_correct: number;
  semantic_partial: number;
  semantic_off_topic: number;
  coverage_correct: number;
  coverage_good: number;
  coverage_missing: number;
  grammar_good: number;
  grammar_poor: number;
  logic_correct: number;
  logic_good: number;
  logic_error: number;
  contradiction_high: number;
  contradiction_moderate: number;
  formality_poor: number;
}
