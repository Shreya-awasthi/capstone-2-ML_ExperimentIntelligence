from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json, time, os, sys, uuid

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agent'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag'))
from graph import graph

app = FastAPI(title="ML Experiment Review API", version="1.0")
START_TIME = time.time()
REVIEWS_PROCESSED = 0
# In-memory store so any client (Streamlit, GitHub Actions, MCP) can discover reviews
REVIEW_STORE: dict = {}   # review_id → {experiment_id, model_name, submitted_at, source}

class ExperimentRequest(BaseModel):
    review_id:    Optional[str]   = None   # pre-specified by CI/CD so Streamlit can track it
    experiment_id: Optional[str]  = None
    model_name: str
    algorithm:  str
    task:       str
    train_acc:  Optional[float] = None
    val_acc:    Optional[float] = None
    test_acc:   Optional[float] = None
    roc_auc:    Optional[float] = None
    features:   List[str]
    train_size: int
    test_size:  int
    leakage_risk: str = "none"
    bias_flags:   List[str] = []

class DecisionRequest(BaseModel):
    decision: str   # "approved" | "rejected"
    feedback: str = ""

@app.post("/experiment")
async def submit_experiment(req: ExperimentRequest,
                             x_pipeline_key: Optional[str] = Header(None)):
    global REVIEWS_PROCESSED
    # Use pre-specified review_id (from CI/CD) or generate a new one
    review_id = req.review_id or f"rev-{uuid.uuid4().hex[:8]}"
    source    = "cicd" if req.review_id else "manual"
    config    = {"configurable": {"thread_id": review_id}}
    exp_id    = req.experiment_id or f"EXP-{uuid.uuid4().hex[:6].upper()}"

    initial_state = {
        "raw_experiment":    {**req.model_dump(exclude={"review_id"}), "id": exp_id},
        "audit_trail":       [],
        "approval_attempts": 0,
    }

    # Run graph — pauses automatically at interrupt_before=["hitl_gate"]
    graph.invoke(initial_state, config)
    REVIEWS_PROCESSED += 1

    # Track in store so any client can discover this review
    REVIEW_STORE[review_id] = {
        "experiment_id": exp_id,
        "model_name":    req.model_name,
        "submitted_at":  time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "source":        source,
    }

    state  = graph.get_state(config).values
    report = state.get("review_report", {})

    return {
        "review_id":           review_id,
        "status":              "pending_approval",
        "model_name":          req.model_name,
        "preliminary_verdict": report.get("overall_verdict", "pending"),
        "message":             f"Review pipeline complete. Awaiting human decision at /review/{review_id}/decision"
    }

@app.post("/review/{review_id}/decision")
async def submit_decision(review_id: str, req: DecisionRequest,
                           x_reviewer_key: Optional[str] = Header(None)):
    config = {"configurable": {"thread_id": review_id}}

    # Validate review exists
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    current_attempts = snapshot.values.get("approval_attempts", 0)

    graph.update_state(config, {
        "hitl_decision":     req.decision,
        "human_feedback":    req.feedback,
        "approval_attempts": current_attempts + 1
    })

    # Resume graph from hitl_gate
    graph.invoke(None, config)

    new_snapshot = graph.get_state(config)
    new_state    = new_snapshot.values
    record       = new_state.get("deployment_record", {})

    # Determine what happened after the graph resumed
    if new_snapshot.next:                          # paused again → revision cycle
        final_status = "pending_approval"
    elif record.get("status") == "deployed":       # deploy node ran
        final_status = "deployed"
    else:                                          # routed to END → escalated
        final_status = "escalated"

    return {
        "review_id":    review_id,
        "decision":     req.decision,
        "final_status": final_status,
        "can_deploy":   final_status == "deployed",
        "model_name":   record.get("model_name"),
    }

@app.get("/review/{review_id}")
async def get_review(review_id: str):
    config   = {"configurable": {"thread_id": review_id}}
    snapshot = graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    state  = snapshot.values
    report = state.get("review_report", {})
    record = state.get("deployment_record", {})

    # Derive status from graph execution state rather than deployment_record alone
    is_complete = not bool(snapshot.next)   # graph has no remaining steps
    is_deployed = record.get("status") == "deployed"
    if is_deployed:
        computed_status = "deployed"
    elif is_complete:
        computed_status = "escalated"       # graph ended without deploying → rejected/escalated
    else:
        computed_status = "pending_approval"

    return {
        "review_id":             review_id,
        "experiment_id":         state.get("parsed_card", {}).get("id"),
        "model_name":            state.get("parsed_card", {}).get("model_name"),
        "status":                computed_status,
        "verdict":               report.get("overall_verdict") or record.get("verdict"),
        "can_deploy":            record.get("status") == "deployed",
        # Full diagnostic state for the reviewer UI
        "parsed_card":           state.get("parsed_card", {}),
        "metric_result":         state.get("metric_result", {}),
        "leakage_result":        state.get("leakage_result", {}),
        "bias_result":           state.get("bias_result", {}),
        "similar_experiments":   state.get("similar_experiments", []),
        "review_report":         report,
        "deployment_record":     record,
        "audit_trail":           state.get("audit_trail", []),
        "approval_attempts":     state.get("approval_attempts", 0),
        "risks":                 report.get("risks", []),
        "deployment_conditions": report.get("deployment_conditions", []),
    }

@app.get("/experiments/leaderboard")
async def leaderboard():
    try:
        data = json.load(open(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'experiments.json')
        ))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="experiments.json not found")

    approved = [
        e for e in data
        if e.get("outcome") in ("approved", "approved_with_monitoring")
    ]
    approved.sort(key=lambda e: e.get("val_acc") or 0, reverse=True)

    return {
        "total_approved": len(approved),
        "leaderboard": [
            {
                "rank":       i + 1,
                "id":         e["id"],
                "model":      e["model"],
                "algorithm":  e["algorithm"],
                "task":       e["task"],
                "val_acc":    e.get("val_acc"),
                "roc_auc":    e.get("roc_auc"),
                "outcome":    e.get("outcome"),
            }
            for i, e in enumerate(approved)
        ]
    }

@app.get("/reviews")
async def list_reviews():
    """Return all review IDs known to this server — used by Streamlit to auto-discover CI/CD submissions."""
    return {"review_ids": list(REVIEW_STORE.keys()), "reviews": REVIEW_STORE, "total": len(REVIEW_STORE)}

@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "uptime_s":           round(time.time() - START_TIME, 1),
        "reviews_processed":  REVIEWS_PROCESSED,
        "model":              os.getenv("deployment_name", "unknown")
    }
