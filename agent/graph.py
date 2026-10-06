from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag'))

from rag.retriever import hybrid_retrieve_experiments
sys.path.insert(0, os.path.dirname(__file__))

from tools import metric_analyser, leakage_detector, bias_checker, get_llm_response
from prompts import REVIEW_TMPL, ReviewReport

class ExperimentState(TypedDict):
    raw_experiment:      dict
    parsed_card:         dict
    similar_experiments: list
    metric_result:       dict
    leakage_result:      dict
    bias_result:         dict
    review_report:       dict
    hitl_decision:       str
    human_feedback:      str
    approval_attempts:   int
    deployment_record:   dict
    audit_trail:         list

# ── NODE 1: Ingest ────────────────────────────────────────────────────────────
def node_ingest(state: ExperimentState) -> dict:
    """Parse and validate incoming experiment card."""
    REQUIRED_FIELDS = ["model_name", "algorithm", "task", "val_acc", "features", "test_size"]

    missing = [f for f in REQUIRED_FIELDS if f not in state["raw_experiment"]]
    if missing:
        raise ValueError(f"Experiment card is missing required fields: {missing}")

    parsed_card = state["raw_experiment"]

    audit_entry = {
        "node":      "ingest",
        "timestamp": datetime.utcnow().isoformat(),
        "summary":   f"Validated experiment card for model '{parsed_card['model_name']}'. All required fields present."
    }

    return {
        "parsed_card": parsed_card,
        "audit_trail": state.get("audit_trail", []) + [audit_entry]
    }

# ── NODE 2: Retrieve ──────────────────────────────────────────────────────────
def node_retrieve(state: ExperimentState) -> dict:
    """Find similar past experiments for context."""
    card = state["parsed_card"]

    query = (f"{card['algorithm']} {card['task']} "
             f"features={' '.join(card['features'])} "
             f"val_acc={card.get('val_acc')}")

    similar = hybrid_retrieve_experiments(
        query,
        top_k=3,
        exclude_id=card.get("id")
    )

    audit_entry = {
        "node":      "retrieve",
        "timestamp": datetime.utcnow().isoformat(),
        "summary":   f"Retrieved {len(similar)} similar experiments: "
                     f"{[r['experiment']['id'] for r in similar]}"
    }

    return {
        "similar_experiments": similar,
        "audit_trail": state.get("audit_trail", []) + [audit_entry]
    }

# ── NODE 3: Analyse ───────────────────────────────────────────────────────────
def node_analyse(state: ExperimentState) -> dict:
    """Run all 3 diagnostic tools in sequence."""
    card = state["parsed_card"]

    metric_result  = metric_analyser(card)
    leakage_result = leakage_detector(card)
    bias_result    = bias_checker(card)

    audit_entry = {
        "node":      "analyse",
        "timestamp": datetime.utcnow().isoformat(),
        "summary":   (f"metric={metric_result['risk_level']} | "
                      f"leakage={leakage_result['risk_level']} (blocks={leakage_result['blocks_deployment']}) | "
                      f"bias={bias_result['risk_level']} (blocks={bias_result['blocks_deployment']})")
    }

    return {
        "metric_result":  metric_result,
        "leakage_result": leakage_result,
        "bias_result":    bias_result,
        "audit_trail":    state.get("audit_trail", []) + [audit_entry]
    }

# ── ROUTING: fast-path rejection ──────────────────────────────────────────────
def route_after_analyse(state: ExperimentState) -> str:
    """Skip the review node and go straight to HITL for critical leakage/bias."""
    if state["leakage_result"]["blocks_deployment"] or state["bias_result"]["blocks_deployment"]:
        return "hitl_gate"
    return "review"

# ── NODE 4: Review ────────────────────────────────────────────────────────────
def node_review(state: ExperimentState) -> dict:
    """Generate structured review report using LLM."""
    prompt = REVIEW_TMPL.render(
        experiment         = state["parsed_card"],
        metric_result      = state["metric_result"],
        leakage_result     = state["leakage_result"],
        bias_result        = state["bias_result"],
        similar_experiments= state.get("similar_experiments", []),
        human_feedback     = state.get("human_feedback", "")
    )

    try:
        raw = get_llm_response(prompt)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        report = ReviewReport.model_validate_json(raw)
    except Exception as e:
        print(f"[DEBUG] node_review parse error: {type(e).__name__}: {e}")
        report = ReviewReport(
            experiment_id       = state["parsed_card"].get("id", "unknown"),
            model_name          = state["parsed_card"].get("model_name", "unknown"),
            overall_verdict     = "escalate",
            verdict_reasoning   = f"LLM review failed — escalating for manual review. Error: {e}",
            risks               = [],
            similar_experiments = [r["experiment"]["id"] for r in state.get("similar_experiments", [])],
            deployment_conditions = [],
            estimated_fix_effort  = "weeks"
        )

    audit_entry = {
        "node":      "review",
        "timestamp": datetime.utcnow().isoformat(),
        "summary":   f"LLM verdict: {report.overall_verdict} | effort: {report.estimated_fix_effort} | risks: {len(report.risks)}"
    }

    return {
        "review_report": report.model_dump(),
        "audit_trail":   state.get("audit_trail", []) + [audit_entry]
    }

# ── NODE 5: HITL Gate ─────────────────────────────────────────────────────────
def node_hitl(state: ExperimentState) -> dict:
    """Display review report to human approver. Waits for update_state() injection."""
    print("\n" + "═" * 60)
    print("  HUMAN-IN-THE-LOOP REVIEW REQUIRED")
    print("═" * 60)

    if state.get("review_report"):
        # Normal path — LLM has produced a full review report
        report = state["review_report"]
        print(f"  Model    : {report.get('model_name')}")
        print(f"  Verdict  : {report.get('overall_verdict').upper()}")
        print(f"  Reasoning: {report.get('verdict_reasoning')}")
        print(f"\n  Risks ({len(report.get('risks', []))}):")
        for risk in report.get("risks", []):
            print(f"    [{risk['severity'].upper()}] {risk['risk_type']} — {risk['evidence']}")
            print(f"           → {risk['recommendation']}")
        conditions = report.get("deployment_conditions", [])
        if conditions:
            print(f"\n  Deployment conditions:")
            for c in conditions:
                print(f"    • {c}")
    else:
        # Fast path — hard block from leakage or bias, no LLM review was run
        print(f"  Model    : {state['parsed_card'].get('model_name')}")
        print("  Verdict  : AUTO-REJECTED (hard block — no LLM review run)")
        if state["leakage_result"]["blocks_deployment"]:
            print(f"  Leakage  : {state['leakage_result']['findings']}")
        if state["bias_result"]["blocks_deployment"]:
            print(f"  Bias     : {state['bias_result']['findings']}")

    print("\n  Awaiting decision via graph.update_state()")
    print("  Set hitl_decision = 'approved' | 'rejected' | 'revision_requested'")
    print("═" * 60 + "\n")

    audit_entry = {
        "node":      "hitl_gate",
        "timestamp": datetime.utcnow().isoformat(),
        "summary":   "Presented to human reviewer. Awaiting hitl_decision."
    }

    return {
        "audit_trail": state.get("audit_trail", []) + [audit_entry]
    }

# ── NODE 6: Deploy ────────────────────────────────────────────────────────────
def node_deploy(state: ExperimentState) -> dict:
    """Register approved model to model registry with lineage metadata."""
    card   = state["parsed_card"]
    report = state.get("review_report", {})
    now    = datetime.utcnow().isoformat()

    deployment_record = {
        "status":             "deployed",
        "model_name":         card.get("model_name"),
        "algorithm":          card.get("algorithm"),
        "task":               card.get("task"),
        "verdict":            report.get("overall_verdict", "approved"),
        "deployment_conditions": report.get("deployment_conditions", []),
        "similar_experiments":   [r["experiment"]["id"] for r in state.get("similar_experiments", [])],
        "metrics": {
            "val_acc":  card.get("val_acc"),
            "test_acc": card.get("test_acc"),
            "roc_auc":  card.get("roc_auc"),
        },
        "approved_by":        state.get("hitl_decision", "approved"),
        "deployed_at":        now,
        "audit_trail_length": len(state.get("audit_trail", [])) + 1
    }

    print(f"\n✅ Model '{card.get('model_name')}' registered to model registry.")
    if deployment_record["deployment_conditions"]:
        print("   Conditions:")
        for c in deployment_record["deployment_conditions"]:
            print(f"     • {c}")

    audit_entry = {
        "node":      "deploy",
        "timestamp": now,
        "summary":   f"Model '{card.get('model_name')}' deployed. Verdict: {deployment_record['verdict']}. Conditions: {len(deployment_record['deployment_conditions'])}."
    }

    return {
        "deployment_record": deployment_record,
        "audit_trail":       state.get("audit_trail", []) + [audit_entry]
    }

# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_hitl(state: ExperimentState) -> str:
    """approved -> deploy; rejected with retries left -> review; otherwise -> END (escalate).
    Fast-path (no review_report = leakage/bias hard block) never gets retries — goes straight to END.
    """
    decision     = state.get("hitl_decision", "")
    attempts     = state.get("approval_attempts", 0)
    is_fast_path = not bool(state.get("review_report"))  # LLM review was skipped

    if decision == "approved":
        return "deploy"

    # Hard block (leakage / bias) — no retry loop, reject immediately
    if is_fast_path:
        return END

    if decision == "revision_requested" and attempts < 2:
        return "review"

    return END

# ── Build graph ───────────────────────────────────────────────────────────────
builder = StateGraph(ExperimentState)
builder.add_node("ingest",   node_ingest)
builder.add_node("retrieve", node_retrieve)
builder.add_node("analyse",  node_analyse)
builder.add_node("review",   node_review)
builder.add_node("hitl_gate",node_hitl)
builder.add_node("deploy",   node_deploy)

builder.set_entry_point("ingest")
builder.add_edge("ingest",   "retrieve")
builder.add_edge("retrieve", "analyse")
builder.add_conditional_edges("analyse", route_after_analyse,
    {"review":"review","hitl_gate":"hitl_gate"})
builder.add_edge("review",   "hitl_gate")
builder.add_conditional_edges("hitl_gate", route_after_hitl,
    {"deploy":"deploy","review":"review",END:END})
builder.add_edge("deploy",   END)

memory = MemorySaver()
graph  = builder.compile(checkpointer=memory, interrupt_before=["hitl_gate"])
print("✓ Experiment review graph compiled")