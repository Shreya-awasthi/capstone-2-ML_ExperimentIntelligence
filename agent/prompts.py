from jinja2 import Template
from pydantic import BaseModel
from typing import List, Literal, Optional

class ExperimentCard(BaseModel):
    model_name:   str
    algorithm:    str
    task:         str
    train_acc:    Optional[float]
    val_acc:      Optional[float]
    test_acc:     Optional[float]
    roc_auc:      Optional[float]
    features:     List[str]
    train_size:   int
    test_size:    int

class ExperimentRisk(BaseModel):
    risk_type:    Literal["overfit","leakage","bias","underperformance","statistical_power","other"]
    severity:     Literal["low","medium","high","critical"]
    evidence:     str          # specific numeric evidence
    blocks_deployment: bool
    recommendation:    str

class ReviewReport(BaseModel):
    experiment_id:      str
    model_name:         str
    overall_verdict:    Literal["approve","approve_with_monitoring","reject","escalate"]
    verdict_reasoning:  str    # CoT trace for human reviewer
    risks:              List[ExperimentRisk]
    similar_experiments: List[str]  # EXP-XXX IDs
    deployment_conditions: List[str]
    estimated_fix_effort: Literal["none","hours","days","weeks"]

# ── PROMPT 1: Risk Synthesiser (Role + CoT) ───────────────────────────────────
REVIEW_TMPL = Template("""
    You are a senior ML safety reviewer at a regulated AI deployment team.
    Your task is to produce a structured experiment review that a human approver can act on directly.

    ═══════════════════════════════════════════════
    EXPERIMENT UNDER REVIEW
    ═══════════════════════════════════════════════
    Model      : {{ experiment.model_name }}
    Algorithm  : {{ experiment.algorithm }}
    Task       : {{ experiment.task }}
    Features   : {{ experiment.features | join(', ') }}

    Metrics:
      train_acc  = {{ experiment.train_acc }}
      val_acc    = {{ experiment.val_acc }}
      test_acc   = {{ experiment.test_acc }}
      roc_auc    = {{ experiment.roc_auc }}
      train_size = {{ experiment.train_size }}
      test_size  = {{ experiment.test_size }}

    ═══════════════════════════════════════════════
    DIAGNOSTIC TOOL RESULTS
    ═══════════════════════════════════════════════
    [Metric Analyser]
      risk_level    : {{ metric_result.risk_level }}
      overfit_gap   : {{ metric_result.overfit_gap }}
      val_test_delta: {{ metric_result.val_test_delta }}
      findings      : {{ metric_result.findings | join('; ') or 'none' }}

    [Leakage Detector]
      risk_level    : {{ leakage_result.risk_level }}
      leakage_type  : {{ leakage_result.leakage_type }}
      blocks_deploy : {{ leakage_result.blocks_deployment }}
      findings      : {{ leakage_result.findings | join('; ') or 'none' }}

    [Bias Checker]
      risk_level    : {{ bias_result.risk_level }}
      blocks_deploy : {{ bias_result.blocks_deployment }}
      requires_mon  : {{ bias_result.requires_monitoring }}
      findings      : {{ bias_result.findings | join('; ') or 'none' }}

    ═══════════════════════════════════════════════
    SIMILAR PAST EXPERIMENTS (for precedent)
    ═══════════════════════════════════════════════
    {% for r in similar_experiments %}
      {{ r.experiment.id }} — {{ r.experiment.model }} ({{ r.experiment.outcome }}) leakage={{ r.experiment.leakage_risk }} [rrf={{ r.rrf_score }}]
    {% else %}
      No similar experiments found.
    {% endfor %}

    {% if human_feedback %}
    ═══════════════════════════════════════════════
    HUMAN REVIEWER FEEDBACK (REVISION REQUEST)
    ═══════════════════════════════════════════════
    A human reviewer has already seen a prior draft of this review and provided the following feedback.
    Revise your assessment to specifically address their concerns:

    {{ human_feedback }}

    {% endif %}
    ═══════════════════════════════════════════════
    REASONING FRAMEWORK — work through all 5 points in your verdict_reasoning
    ═══════════════════════════════════════════════
    1. METRIC QUALITY    : Is the overfit_gap acceptable? Is val_acc / roc_auc above the deployment threshold?
    2. LEAKAGE           : Is there target, temporal, or preprocessing leakage? Does it block deployment?
    3. BIAS / FAIRNESS   : Are there demographic parity issues or coverage gaps? Are protected groups affected?
    4. STATISTICAL POWER : Is test_size large enough to trust the reported metrics?
    5. PRECEDENT         : What do similar past experiments tell us? Do approved ones provide confidence?

    HARD RULES (non-negotiable — override any other reasoning):
    - leakage_type is "target" OR leakage risk_level is "high"  →  verdict MUST be "reject"
    - bias findings contain a fairness/regulatory risk           →  verdict MUST be "reject" or "escalate"
    - no blocks_deployment flags and all risks low/medium        →  "approve" or "approve_with_monitoring"

    ═══════════════════════════════════════════════
    OUTPUT — respond ONLY with valid JSON, no markdown fences
    ═══════════════════════════════════════════════
    {
      "experiment_id":        "<string>",
      "model_name":           "<string>",
      "overall_verdict":      "approve" | "approve_with_monitoring" | "reject" | "escalate",
      "verdict_reasoning":    "<CoT trace covering all 5 points above>",
      "risks": [
        {
          "risk_type":          "overfit" | "leakage" | "bias" | "underperformance" | "statistical_power" | "other",
          "severity":           "low" | "medium" | "high" | "critical",
          "evidence":           "<specific numeric or textual evidence>",
          "blocks_deployment":  true | false,
          "recommendation":     "<concrete actionable fix>"
        }
      ],
      "similar_experiments":    ["EXP-XXX"],
      "deployment_conditions":  ["<condition if approve_with_monitoring, else empty>"],
      "estimated_fix_effort":   "none" | "hours" | "days" | "weeks"
    }
    """)

# ── PROMPT 2: Remediation Advisor (Few-shot) ─────────────────────────────────
REMEDIATION_TMPL = Template("""
    You are an ML remediation advisor. Given a rejected experiment and its review findings,
    produce a numbered, prioritised action plan that the team can start executing today.

    ═══════════════════════════════════════════════
    FEW-SHOT EXAMPLES
    ═══════════════════════════════════════════════

    --- EXAMPLE 1: Target Leakage ---
    Experiment : churn_xgb_v1 | XGBoost | binary_classification
    Features   : recency, frequency, monetary, future_purchase_flag
    Findings   : target leakage — future_purchase_flag derives directly from the label
    Fix plan:
      1. [TODAY]   Remove future_purchase_flag from the feature set entirely.
      2. [TODAY]   Retrain XGBoost on the cleaned feature set (recency, frequency, monetary).
      3. [THIS WEEK] Re-run the full evaluation pipeline; confirm val_acc and roc_auc are stable.
      4. [ONGOING] Add a pre-training leakage gate that rejects features with correlation > 0.95 to the label.

    --- EXAMPLE 2: Overfitting ---
    Experiment : price_rf_v3 | RandomForest | regression
    Metrics    : train_acc=0.97, val_acc=0.71 (overfit_gap=0.26)
    Findings   : high overfit gap; model has memorised training data
    Fix plan:
      1. [TODAY]   Reduce max_depth to 8 and increase min_samples_leaf to 20.
      2. [TODAY]   Enable early stopping on validation loss.
      3. [THIS WEEK] Add L2 regularisation or switch to a GradientBoosting model with shrinkage.
      4. [THIS WEEK] Collect 20% more training data to reduce variance.

    --- EXAMPLE 3: Demographic Bias ---
    Experiment : credit_lgbm_v2 | LightGBM | binary_classification
    Findings   : demographic_parity — approval rate 18pp lower for Black applicants vs. White
    Fix plan:
      1. [TODAY]   Halt deployment immediately; do not proceed until bias is resolved.
      2. [THIS WEEK] Apply re-weighting: upsample underrepresented group in training data.
      3. [THIS WEEK] Add a fairness constraint (e.g. equalized odds) to the training objective.
      4. [THIS WEEK] Audit all features for proxy discrimination (zip code, name encoding, etc.).
      5. [NEXT SPRINT] Run a full disparate impact analysis and document for regulatory review.

    ═══════════════════════════════════════════════
    EXPERIMENT TO REMEDIATE
    ═══════════════════════════════════════════════
    Model      : {{ experiment.model_name }}
    Algorithm  : {{ experiment.algorithm }}
    Task       : {{ experiment.task }}
    Features   : {{ experiment.features | join(', ') }}

    Metrics:
      train_acc  = {{ experiment.train_acc }}
      val_acc    = {{ experiment.val_acc }}
      test_acc   = {{ experiment.test_acc }}
      roc_auc    = {{ experiment.roc_auc }}
      test_size  = {{ experiment.test_size }}

    ═══════════════════════════════════════════════
    REVIEW FINDINGS
    ═══════════════════════════════════════════════
    Metric risks  : {{ metric_result.findings | join('; ') or 'none' }}
    Leakage risks : {{ leakage_result.findings | join('; ') or 'none' }}
    Bias risks    : {{ bias_result.findings | join('; ') or 'none' }}
    Verdict       : {{ overall_verdict }}
    Reasoning     : {{ verdict_reasoning }}

    {% if human_feedback %}
    ═══════════════════════════════════════════════
    HUMAN REVIEWER GUIDANCE
    ═══════════════════════════════════════════════
    The reviewer has provided the following specific guidance — prioritise these in your plan:

    {{ human_feedback }}

    {% endif %}
    ═══════════════════════════════════════════════
    INSTRUCTIONS
    ═══════════════════════════════════════════════
    Using the examples above as a style guide, produce a numbered fix plan for THIS experiment.
    Each step must:
      - Reference the experiment's actual features, algorithm, or metrics (not generic advice)
      - Be tagged with timeline: [TODAY], [THIS WEEK], or [NEXT SPRINT]
      - Be specific enough for an engineer to act on without further clarification

    Respond with the numbered plan only. No preamble, no JSON.
    """)

# if __name__ == "__main__":
#     print("✓ Prompt templates defined")
#     print(f"  ReviewReport fields: {list(ReviewReport.model_fields.keys())}")