#!/usr/bin/env python3
"""
ML Experiment CI/CD submission script.

Called by GitHub Actions at the end of a training run.
Submits the experiment card to the review pipeline (via FastAPI, representing
what the MCP submit_experiment tool does) and polls until the human reviewer
approves or rejects — exactly what a real CI/CD job would do.

Exit codes:
    0 — experiment approved, can_deploy = true
    1 — rejected / escalated / timed out, can_deploy = false

Usage:
    python scripts/cicd_submit.py \
        --experiment-id  EXP-021 \
        --review-id      rev-abc123 \
        --model-name     churn_xgb_v4 \
        --algorithm      XGBoost \
        --task           binary_classification \
        --features       "recency, frequency, monetary" \
        --metrics        '{"train_acc":0.85,"val_acc":0.88,"test_acc":0.87,"roc_auc":0.92}' \
        --dataset-sizes  '{"train_size":120000,"test_size":15000}'
"""

import sys, os, time, argparse, json, requests

API_BASE_URL      = os.getenv("API_BASE_URL",           "http://localhost:8000")
PIPELINE_KEY      = os.getenv("PIPELINE_KEY",           "dev-pipeline-key")
POLL_INTERVAL     = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
MAX_POLLS         = int(os.getenv("MAX_POLLS",              "80"))   # 80 × 15s = 20 min


def log(msg: str) -> None:
    print(f"[CI/CD] {msg}", flush=True)


def review_already_exists(review_id: str) -> bool:
    """Return True if the review was already submitted directly from Streamlit."""
    if not review_id:
        return False
    try:
        resp = requests.get(f"{API_BASE_URL}/review/{review_id}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def submit_experiment(payload: dict) -> str:
    """Submit experiment to review pipeline. Returns review_id.

    If the review was already created (e.g. submitted directly from Streamlit),
    skip the POST and go straight to polling.
    """
    exp_id    = payload.get("experiment_id", "?")
    review_id = payload.get("review_id")

    if review_id and review_already_exists(review_id):
        log(f"Review '{review_id}' already exists in the backend — skipping POST, going to poll…")
        log("Waiting for human approval in Streamlit Review Queue…")
        log(f"Open Streamlit at http://localhost:8501 → 📋 Review Queue")
        return review_id

    log(f"Submitting experiment '{exp_id}' for review…")
    if review_id:
        log(f"Using pre-specified review_id={review_id} (Streamlit is already tracking this)")

    resp = requests.post(
        f"{API_BASE_URL}/experiment",
        json=payload,
        headers={"X-Pipeline-Key": PIPELINE_KEY},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()

    rid = data["review_id"]
    log(f"Review started  →  review_id={rid}")
    log(f"Preliminary verdict : {data.get('preliminary_verdict', 'pending')}")
    log(f"Pipeline status     : {data.get('status')}")
    log("")
    log("Waiting for human approval in Streamlit Review Queue…")
    log(f"Open Streamlit at http://localhost:8501 → 📋 Review Queue")
    return rid


def poll_until_decided(review_id: str) -> str:
    """Poll GET /review/{id} until a final decision. Returns final status string."""
    for attempt in range(1, MAX_POLLS + 1):
        try:
            resp = requests.get(f"{API_BASE_URL}/review/{review_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()

            status     = data.get("status",     "pending_approval")
            verdict    = data.get("verdict",    "—")
            can_deploy = data.get("can_deploy", False)

            log(f"Poll {attempt:>2}/{MAX_POLLS}  status={status:<20} can_deploy={can_deploy}  verdict={verdict}")

            if status == "deployed":
                log("")
                log("✅  Approved!  can_deploy=true")
                log(f"   Verdict  : {verdict}")
                log("   Model registered to registry. Deployment unblocked.")
                return "deployed"

            if status in ("escalated", "rejected"):
                log("")
                log(f"❌  {status.capitalize()}.  can_deploy=false")
                return status

            log(f"   Still pending — next poll in {POLL_INTERVAL}s…")
            time.sleep(POLL_INTERVAL)

        except requests.HTTPError as e:
            log(f"HTTP error on poll {attempt}: {e} — retrying…")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            log(f"Unexpected error on poll {attempt}: {e} — retrying…")
            time.sleep(POLL_INTERVAL)

    log(f"⏰  Timed out after {MAX_POLLS * POLL_INTERVAL}s without a decision.  can_deploy=false")
    return "timeout"


def main() -> None:
    p = argparse.ArgumentParser(description="ML Experiment CI/CD submission + polling script")
    p.add_argument("--experiment-id",  required=True,  help="Experiment ID (e.g. EXP-021)")
    p.add_argument("--review-id",      default=None,   help="Pre-specified review ID from Streamlit")
    p.add_argument("--model-name",     required=True)
    p.add_argument("--algorithm",      required=True)
    p.add_argument("--task",           required=True)
    p.add_argument("--features",       required=True,  help="Comma-separated feature names")
    p.add_argument("--metrics",        required=True,
                   help='JSON object with train_acc, val_acc, test_acc, roc_auc — '
                        'e.g. \'{"train_acc":0.85,"val_acc":0.88,"test_acc":0.87,"roc_auc":0.92}\'')
    p.add_argument("--dataset-sizes",  required=True,
                   help='JSON object with train_size, test_size — '
                        'e.g. \'{"train_size":10000,"test_size":2000}\'')
    p.add_argument("--leakage-risk",   default="none",
                   choices=["none","low","medium","high"])
    p.add_argument("--bias-flags",     default="",
                   help="Comma-separated bias flag strings")
    args = p.parse_args()

    # Parse JSON inputs
    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError as exc:
        log(f"ERROR: --metrics is not valid JSON: {exc}")
        sys.exit(1)

    try:
        sizes = json.loads(args.dataset_sizes)
    except json.JSONDecodeError as exc:
        log(f"ERROR: --dataset-sizes is not valid JSON: {exc}")
        sys.exit(1)

    val_acc    = float(metrics.get("val_acc",   0.0))
    test_acc   = float(metrics.get("test_acc",  0.0))
    train_acc  = float(metrics.get("train_acc", 0.0)) or None
    roc_auc_v  = float(metrics.get("roc_auc",   0.0))
    roc_auc    = roc_auc_v if roc_auc_v > 0 else None
    train_size = int(sizes.get("train_size", 1000))
    test_size  = int(sizes.get("test_size",  200))

    payload = {
        "review_id":     args.review_id,
        "experiment_id": args.experiment_id,
        "model_name":    args.model_name,
        "algorithm":     args.algorithm,
        "task":          args.task,
        "features":      [f.strip() for f in args.features.split(",") if f.strip()],
        "train_acc":     train_acc,
        "val_acc":       val_acc,
        "test_acc":      test_acc,
        "roc_auc":       roc_auc,
        "train_size":    train_size,
        "test_size":     test_size,
        "leakage_risk":  args.leakage_risk,
        "bias_flags":    [f.strip() for f in args.bias_flags.split(",") if f.strip()],
    }

    log("=" * 60)
    log("ML Experiment Intelligence — CI/CD Submission")
    log("=" * 60)
    log(f"API endpoint  : {API_BASE_URL}")
    log(f"Experiment    : {args.experiment_id}")
    log(f"Model         : {args.model_name}  ({args.algorithm})")
    log(f"Task          : {args.task}")
    log(f"Metrics       : {args.metrics}")
    log(f"Dataset sizes : {args.dataset_sizes}")
    log("=" * 60)

    try:
        review_id    = submit_experiment(payload)
        final_status = poll_until_decided(review_id)

        log("=" * 60)
        log(f"Final status : {final_status}")
        log(f"can_deploy   : {final_status == 'deployed'}")
        log("=" * 60)

        sys.exit(0 if final_status == "deployed" else 1)

    except requests.HTTPError as e:
        log(f"HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
