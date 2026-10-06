from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Literal, Optional
import json, os, requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PIPELINE_KEY = os.getenv("PIPELINE_KEY", "dev-pipeline-key")
REVIEWER_KEY = os.getenv("REVIEWER_KEY", "dev-reviewer-key")

mcp = FastMCP("ml-experiment-review", host="0.0.0.0", port=8001)

class ExperimentSubmission(BaseModel):
    model_name:   str
    algorithm:    str
    task:         str
    val_acc:      Optional[float]
    test_acc:     Optional[float]
    features:     list
    train_size:   int
    test_size:    int
    leakage_risk: str = "none"
    bias_flags:   list = []

class ReviewDecision(BaseModel):
    review_id:  str
    decision:   Literal["approved", "rejected"]
    feedback:   str = ""

@mcp.tool()
def submit_experiment(experiment: ExperimentSubmission) -> dict:
    """Submit an ML experiment for automated review.
    Called by training CI/CD at end of run. Returns review_id for polling.
    """
    resp = requests.post(
        f"{API_BASE_URL}/experiment",
        json=experiment.model_dump(),
        headers={"X-Pipeline-Key": PIPELINE_KEY},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()

@mcp.tool()
def submit_review_decision(decision: ReviewDecision) -> dict:
    """Submit a human approval or rejection. Called by reviewer from Slack or UI."""
    resp = requests.post(
        f"{API_BASE_URL}/review/{decision.review_id}/decision",
        json={"decision": decision.decision, "feedback": decision.feedback},
        headers={"X-Reviewer-Key": REVIEWER_KEY},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()

@mcp.tool()
def get_review_status(review_id: str) -> dict:
    """Poll review status. CI/CD uses this to decide whether to deploy.
    Returns: pending_approval | deployed | escalated
    """
    resp = requests.get(f"{API_BASE_URL}/review/{review_id}", timeout=10)
    if resp.status_code == 404:
        return {"review_id": review_id, "status": "not_found", "can_deploy": False}
    resp.raise_for_status()
    data = resp.json()
    return {
        "review_id":  review_id,
        "status":     data.get("status", "pending_approval"),
        "verdict":    data.get("verdict"),
        "can_deploy": data.get("can_deploy", False),
    }

@mcp.resource("experiments://registry")
def experiment_registry() -> str:
    """Read-only registry of all reviewed experiments and their outcomes."""
    try:
        data = json.load(open("data/experiments.json"))
        return json.dumps({"experiments": data, "total": len(data)})
    except Exception:
        return json.dumps({"experiments": [], "total": 0})

if __name__ == "__main__":
    mcp.run(transport="sse")
