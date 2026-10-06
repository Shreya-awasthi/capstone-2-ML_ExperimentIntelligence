from dotenv import load_dotenv
load_dotenv()

import os, json
from openai import OpenAI

endpoint        = os.getenv("endpoint")
deployment_name = os.getenv("deployment_name")
api_key         = os.getenv("OPENAI_API_KEY")
client = OpenAI(base_url=endpoint, api_key=api_key)

def get_llm_response(prompt: str) -> str:
    response = client.chat.completions.create(
    model=deployment_name,
    messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── TOOL 1: metric_analyser ──────────────────────────────────────────────────
def metric_analyser(experiment: dict) -> dict:
    """
    Analyses model metrics for overfitting, underfitting, and distribution shift.
    Returns structured risk assessment with numeric evidence.
    """
    # ── train-val gap (overfitting signal) ───────────────────────────────────
    train_acc = experiment.get("train_acc")
    val_acc   = experiment.get("val_acc")

    if train_acc is not None and val_acc is not None:
        overfit_gap  = round(train_acc - val_acc, 4)
        overfit_risk = "high" if overfit_gap > 0.15 else "medium" if overfit_gap > 0.08 else "low"
    else:
        overfit_gap  = None
        overfit_risk = "low"   # can't evaluate without both values

    # ── val-test delta (distribution shift signal) ───────────────────────────
    test_acc = experiment.get("test_acc")
    if val_acc is not None and test_acc is not None:
        val_test_delta = round(val_acc - test_acc, 4)
        shift_risk     = "high" if abs(val_test_delta) > 0.05 else "low"
    else:
        val_test_delta = None
        shift_risk     = "low"

    # ── underperformance + discriminative power checks ───────────────────────
    findings = []
    if val_acc is not None and val_acc < 0.65:
        findings.append("underperformance: val_acc below acceptable threshold (0.65)")
    roc_auc = experiment.get("roc_auc")
    if roc_auc is not None and roc_auc < 0.70:
        findings.append("low discriminative power: roc_auc below threshold (0.70)")

    # ── combine all signals into final risk_level and return ─────────────────
    if overfit_risk == "high" or shift_risk == "high":
        findings.append(f"overfit_gap={overfit_gap} (overfit risk: {overfit_risk})")
        findings.append(f"val_test_delta={val_test_delta} (shift risk: {shift_risk})")

    risk_level = "high" if (overfit_risk == "high" or shift_risk == "high") \
            else "medium" if overfit_risk == "medium" \
            else "low"

    return {
        "tool":          "metric_analyser",
        "risk_level":    risk_level,
        "overfit_gap":   overfit_gap,
        "val_test_delta": val_test_delta,
        "findings":      findings
    }

# ── TOOL 2: leakage_detector ─────────────────────────────────────────────────
def leakage_detector(experiment: dict) -> dict:
    """
    Uses rule-based checks + LLM reasoning to detect data leakage patterns.
    Covers target leakage, temporal leakage, and preprocessing leakage.
    """
    # ── Rule-based signal: read explicit leakage_risk field ──────────────────
    explicit_risk = experiment.get("leakage_risk", "none")


    # ── LLM signal: reason over feature names vs. task type ──────────────────
    prompt = f"""You are an ML safety auditor. Analyse these experiment details for data leakage.

    Task type : {experiment["task"]}
    Features  : {experiment["features"]}
    Notes     : {experiment.get("notes", "")}

    Check for:
    1. Target leakage   — a feature that directly encodes or derives from the label
    2. Temporal leakage — a feature that uses future data not available at prediction time
    3. Preprocessing leakage — statistics (mean, std) computed on the full dataset before splitting

    Respond ONLY with valid JSON in this exact format:
    {{
    "llm_risk": "none" | "medium" | "high",
    "leakage_type": "none" | "target" | "temporal" | "preprocessing",
    "reasoning": "one sentence explanation"
    }}"""

    try:
        raw = get_llm_response(prompt)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        llm_output = json.loads(raw)
        llm_risk      = llm_output.get("llm_risk", "none")
        leakage_type  = llm_output.get("leakage_type", "none")
        llm_reasoning = llm_output.get("reasoning", "")
    except Exception as e:
        print(f"[DEBUG] leakage_detector LLM error: {type(e).__name__}: {e}")
        llm_risk      = "none"
        leakage_type  = "none"
        llm_reasoning = "LLM response could not be parsed"

    # ── combine both signals into final verdict ───────────────────────────────
    risk_levels = {"none": 0, "medium": 1, "high": 2}
    final_risk  = max(explicit_risk, llm_risk, key=lambda r: risk_levels[r])

    return {
        "tool":               "leakage_detector",
        "risk_level":         final_risk,
        "leakage_type":       leakage_type,
        "blocks_deployment":  final_risk == "high",
        "findings":           [llm_reasoning] if llm_reasoning else []
    }

# ── TOOL 3: bias_checker ──────────────────────────────────────────────────────
def bias_checker(experiment: dict) -> dict:
    """
    Checks for demographic bias, domain coverage gaps, and statistical power issues.
    Returns risk level and actionable flags.
    """
    # ── statistical power check ───────────────────────────────────────────────
    findings = []
    test_size = experiment.get("test_size", 0)

    if test_size < 100:
        findings.append(f"critical: test_size={test_size} is too small for any reliable evaluation")
        power_risk = "high"
    elif test_size < 1000:
        findings.append(f"low statistical power: test_size={test_size} may produce unreliable metrics")
        power_risk = "medium"
    else:
        power_risk = "low"

    # ── classify each bias_flag entry via LLM ────────────────────────────────
    bias_risk = "low"
    for flag in experiment.get("bias_flags", []):
        prompt = f"""Classify this ML bias flag into exactly one of three categories:
        - "fairness_regulatory" : involves protected attributes, demographic groups, or legal fairness concerns
        - "coverage_gap"        : data underrepresentation, cold-start, or domain imbalance  
        - "generic"             : any other bias concern

        Bias flag: "{flag}"

        Respond ONLY with valid JSON in this exact format:
        {{"category": "fairness_regulatory" | "coverage_gap" | "generic", "reasoning": "one sentence"}}"""

        try:
            raw = get_llm_response(prompt)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed   = json.loads(raw)
            category = parsed.get("category", "generic")
            reasoning = parsed.get("reasoning", "")
        except Exception:
            category  = "generic"
            reasoning = "LLM classification failed"

        if category == "fairness_regulatory":
            findings.append(f"[fairness/regulatory risk] {flag} — {reasoning}")
            bias_risk = "high"
        elif category == "coverage_gap":
            findings.append(f"[coverage gap] {flag} — {reasoning}")
            bias_risk = max(bias_risk, "medium", key=lambda r: {"low":0,"medium":1,"high":2}[r])
        else:
            findings.append(f"[generic bias flag] {flag} — {reasoning}")

    # ── combine signals and return ────────────────────────────────────────────
    rank = {"low": 0, "medium": 1, "high": 2}
    risk_level = max(power_risk, bias_risk, key=lambda r: rank[r])

    return {
        "tool":                "bias_checker",
        "risk_level":          risk_level,
        "blocks_deployment":   bias_risk == "high",
        "findings":            findings,
        "requires_monitoring": risk_level in ("medium", "high")
    }

# if __name__ == "__main__":
#     # ── Smoke test all 3 tools ────────────────────────────────────────────────
#     experiments = json.load(open("data/experiments.json"))
#     test_exp = experiments[1]  # EXP-002: known leakage case
#     print(f"Testing on: {test_exp['id']} — {test_exp['model']}")
#     print("\n── metric_analyser ──")
#     print(json.dumps(metric_analyser(test_exp), indent=2))
#     print("\n── leakage_detector ──")
#     print(json.dumps(leakage_detector(test_exp), indent=2))
#     print("\n── bias_checker ──")
#     print(json.dumps(bias_checker(experiments[7]), indent=2))  # EXP-008: credit bias