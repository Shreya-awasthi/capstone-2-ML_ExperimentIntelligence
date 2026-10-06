"""
ML Experiment Intelligence Suite — Streamlit UI
Run from project root:  streamlit run app.py
Requires FastAPI running: uvicorn api.main:app --port 8000
"""

import streamlit as st
import os, json, uuid, requests
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL  = os.getenv("API_BASE_URL",  "http://localhost:8000")
REVIEWER_KEY  = os.getenv("REVIEWER_KEY",  "dev-reviewer-key")
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN",  "")
GITHUB_REPO   = os.getenv("GITHUB_REPO",   "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Experiment Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Seed data (read-only, for registry display) ───────────────────────────────
@st.cache_data
def load_registry_data():
    with open(os.path.join(PROJECT_ROOT, "data", "model_registry.json")) as f:
        registry = json.load(f)
    with open(os.path.join(PROJECT_ROOT, "data", "experiments.json")) as f:
        experiments = json.load(f)
    return registry, {e["id"]: e for e in experiments}

seed_registry, exp_lookup = load_registry_data()

# ── Session state ─────────────────────────────────────────────────────────────
if "reviews" not in st.session_state:
    st.session_state.reviews: dict = {}   # review_id → {experiment_id, model_name, submitted_at, status}
if "last_review_id" not in st.session_state:
    st.session_state.last_review_id = None
if "last_submit_outcome" not in st.session_state:
    st.session_state.last_submit_outcome = None  # persists submission result across reruns

# ── Constants ─────────────────────────────────────────────────────────────────
TASK_OPTIONS    = ["binary_classification","regression","multi_class_classification",
                   "ranking","time_series","anomaly_detection","survival_analysis",
                   "uplift_modelling","reinforcement_learning","embedding","causal_inference"]
LEAKAGE_OPTIONS = ["none","low","medium","high"]

# ── API helpers ───────────────────────────────────────────────────────────────
def api_health() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False

def api_submit(data: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/experiment", json=data,
                      headers={"X-Pipeline-Key": "dev"}, timeout=180)
    r.raise_for_status()
    return r.json()

def api_get_review(review_id: str) -> dict:
    r = requests.get(f"{API_BASE_URL}/review/{review_id}", timeout=10)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()

def api_decide(review_id: str, decision: str, feedback: str = "") -> dict:
    r = requests.post(
        f"{API_BASE_URL}/review/{review_id}/decision",
        json={"decision": decision, "feedback": feedback},
        headers={"X-Reviewer-Key": REVIEWER_KEY},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()

def api_list_reviews() -> dict:
    """Fetch all reviews known to FastAPI (including those submitted by GitHub Actions)."""
    try:
        r = requests.get(f"{API_BASE_URL}/reviews", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def api_leaderboard() -> dict:
    """Fetch approved experiments sorted by val_acc from the leaderboard endpoint."""
    try:
        r = requests.get(f"{API_BASE_URL}/experiments/leaderboard", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "leaderboard": [], "total_approved": 0}

def trigger_github_actions(review_id: str, payload: dict) -> tuple:
    """
    Dispatch the ml-review.yml workflow via GitHub API.
    Returns (ok: bool, error_message: str).
    Metrics are passed as a single JSON string; dataset sizes as another JSON string.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False, "GITHUB_TOKEN or GITHUB_REPO not configured"

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/ml-review.yml/dispatches"

    # Pack metrics and sizes into compact JSON strings
    metrics_json = json.dumps({
        "train_acc": payload.get("train_acc") or 0.0,
        "val_acc":   payload.get("val_acc",  0.0),
        "test_acc":  payload.get("test_acc", 0.0),
        "roc_auc":   payload.get("roc_auc")  or 0.0,
    })
    sizes_json = json.dumps({
        "train_size": payload.get("train_size", 1000),
        "test_size":  payload.get("test_size",  200),
    })

    body = {
        "ref": "master",
        "inputs": {
            "review_id":     review_id,
            "experiment_id": payload.get("experiment_id", ""),
            "model_name":    payload.get("model_name", ""),
            "algorithm":     payload.get("algorithm", ""),
            "task":          payload.get("task", "binary_classification"),
            "features":      ", ".join(payload.get("features", [])),
            "metrics":       metrics_json,
            "dataset_sizes": sizes_json,
            "leakage_risk":  payload.get("leakage_risk", "none"),
            "bias_flags":    ", ".join(payload.get("bias_flags", [])),
        },
    }
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        if resp.status_code == 204:
            return True, ""
        # Surface the GitHub API error message so the user can act on it
        try:
            gh_msg = resp.json().get("message", resp.text)
        except Exception:
            gh_msg = resp.text
        return False, f"GitHub API returned HTTP {resp.status_code}: {gh_msg}"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error reaching GitHub API: {e}"
    except requests.exceptions.Timeout:
        return False, "Request to GitHub API timed out (15 s)"
    except Exception as e:
        return False, f"Unexpected error: {e}"

# ── Display helpers ───────────────────────────────────────────────────────────
def verdict_color(v: str) -> str:
    return {"approve":"green","approve_with_monitoring":"orange",
            "reject":"red","escalate":"red"}.get(v or "", "gray")

def risk_emoji(level: str) -> str:
    return {"low":"🟢","medium":"🟡","high":"🔴","critical":"🔴"}.get(level or "", "⚪")

def status_badge(s: str) -> str:
    return {"pending_approval":"🟡 Pending Review","deployed":"✅ Deployed",
            "rejected":"❌ Rejected","escalated":"🚨 Escalated"}.get(s, f"⚪ {s}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧪 ML Experiment Intelligence Suite")
st.caption("Meridian Analytics — Automated experiment review with human-in-the-loop approval")

# FastAPI health banner
if not api_health():
    st.error(
        f"⚠️ **FastAPI backend is not reachable** at `{API_BASE_URL}`.  \n"
        "If running locally: `uvicorn api.main:app --port 8000 --reload`  \n"
        "If running via Docker: `docker compose up --build`"
    )
    st.stop()

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
n_pending = sum(1 for r in st.session_state.reviews.values() if r["status"] == "pending_approval")

n_rejected = sum(1 for r in st.session_state.reviews.values() if r["status"] == "escalated")
tab_submit, tab_queue, tab_rejected, tab_registry, tab_leaderboard = st.tabs([
    "📤  Submit Experiment",
    f"📋  Review Queue   ({n_pending} pending)",
    f"❌  Rejected   ({n_rejected})",
    "🗂️  Model Registry",
    "🏆  Leaderboard",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Submit Experiment + CI/CD Status Panel
# ═══════════════════════════════════════════════════════════════════════════════
with tab_submit:
    st.subheader("Submit a new experiment for automated review")
    st.caption("This simulates what a training CI/CD job does at the end of a run — submits the experiment card and polls for a verdict.")

    # ── Persistent outcome banner (survives reruns) ───────────────────────────
    outcome = st.session_state.get("last_submit_outcome")
    if outcome:
        rid = outcome["review_id"]
        if outcome["pipeline_ok"]:
            st.success(
                f"✅ **Pipeline complete** for `{outcome['experiment_id']}` / **{outcome['model_name']}**  \n"
                f"Review ID: `{rid}`  │  "
                f"Preliminary verdict: `{outcome['pipeline_verdict']}`  \n"
                f"👉 Go to **📋 Review Queue** tab to approve or reject."
            )
        else:
            st.error(
                f"❌ **Pipeline failed** for `{outcome['experiment_id']}`:  \n"
                f"{outcome['pipeline_error']}"
            )

        if outcome["gh_dispatched"] is True:
            st.info(
                f"🔗 **GitHub Actions workflow dispatched** — runner is polling for the verdict.  \n"
                f"Track it at **GitHub → Actions → ML Experiment Review**."
            )
        elif outcome["gh_dispatched"] is False:
            st.warning(
                f"⚠️ **GitHub Actions dispatch failed** (pipeline already ran above):  \n"
                f"{outcome['gh_error']}  \n\n"
                f"Common fixes: token needs `workflow` scope · `GITHUB_REPO` must be `owner/repo` · "
                f"`ml-review.yml` must exist on the `main` branch."
            )
        elif outcome["pipeline_ok"] and not gh_configured:
            st.warning(
                "⚠️ **GitHub Actions not configured** — pipeline ran directly via FastAPI.  \n"
                "To also trigger CI/CD tracking, set `GITHUB_TOKEN` and `GITHUB_REPO` before starting Streamlit:  \n"
                "```\nexport GITHUB_TOKEN=ghp_your_token\n"
                "export GITHUB_REPO=owner/repo-name\nstreamlit run app.py\n```"
            )

        if st.button("🗑️ Dismiss", key="dismiss_outcome"):
            st.session_state.last_submit_outcome = None
            st.rerun()
        st.divider()

    with st.form("experiment_form", clear_on_submit=False):
        left, right = st.columns(2)

        with left:
            st.markdown("##### Experiment")
            experiment_id = st.text_input("Experiment ID *", placeholder="e.g. EXP-021",
                help="Unique ID for this run — from MLflow, W&B, or your own naming convention")
            st.markdown("##### Model")
            model_name   = st.text_input("Model Name *",  placeholder="e.g. churn_xgb_v4")
            algorithm    = st.text_input("Algorithm *",   placeholder="e.g. XGBoost")
            task         = st.selectbox("Task *", TASK_OPTIONS)
            features_raw = st.text_area("Features * (comma-separated)",
                placeholder="e.g. recency, frequency, monetary, days_since_last_purchase", height=80)
            st.markdown("##### Risk flags")
            leakage_risk = st.selectbox("Self-reported Leakage Risk", LEAKAGE_OPTIONS)
            bias_raw     = st.text_area("Bias Flags (one per line, leave blank if none)",
                placeholder="e.g. demographic_parity: approval rate gap detected", height=80)

        with right:
            st.markdown("##### Metrics (JSON)")
            metrics_raw = st.text_area(
                "Metrics * (JSON)",
                value='{"train_acc": 0.0, "val_acc": 0.0, "test_acc": 0.0, "roc_auc": 0.0}',
                height=120,
                help=(
                    "Provide all accuracy/AUC metrics as a JSON object.  \n"
                    "Keys: `train_acc` (0 if N/A), `val_acc`, `test_acc`, `roc_auc` (0 if N/A).  \n"
                    "Example: `{\"train_acc\": 0.85, \"val_acc\": 0.88, \"test_acc\": 0.87, \"roc_auc\": 0.92}`"
                ),
            )
            st.markdown("##### Dataset Sizes (JSON)")
            sizes_raw = st.text_area(
                "Dataset Sizes * (JSON)",
                value='{"train_size": 10000, "test_size": 2000}',
                height=80,
                help=(
                    "Provide row counts as a JSON object.  \n"
                    "Keys: `train_size`, `test_size`.  \n"
                    "Example: `{\"train_size\": 120000, \"test_size\": 15000}`"
                ),
            )

        submitted = st.form_submit_button("⚙️  Submit Review", type="primary", use_container_width=True, disabled=False)

    # ── GitHub Actions config note ──────────────────────────────────────────
    gh_configured = bool(GITHUB_TOKEN and GITHUB_REPO)
    if not gh_configured:
        st.warning(
            "⚠️ **GitHub Actions not configured.**  \n"
            "Set these env vars to enable CI/CD dispatch:\n\n"
            "```\nexport GITHUB_TOKEN=ghp_your_token\n"
            "export GITHUB_REPO=owner/repo-name\n"
            "streamlit run app.py\n```"
        )
    else:
        st.caption(f"🔗 GitHub Actions enabled  │  repo: `{GITHUB_REPO}`  │  self-hosted runner required")

    # ── On submit ──
    action = "github" if submitted else None
    if action:
        features   = [f.strip() for f in features_raw.split(",")  if f.strip()]
        bias_flags = [f.strip() for f in bias_raw.split("\n")     if f.strip()]

        # Parse JSON fields
        metrics_err = sizes_err = None
        try:
            metrics_parsed = json.loads(metrics_raw)
        except Exception:
            metrics_err = f"Metrics JSON is invalid — check for missing quotes or commas."

        try:
            sizes_parsed = json.loads(sizes_raw)
        except Exception:
            sizes_err = "Dataset Sizes JSON is invalid — check for missing quotes or commas."

        if not experiment_id.strip():
            st.error("Experiment ID is required.")
        elif not model_name or not algorithm or not features:
            st.error("Model Name, Algorithm, and at least one Feature are required.")
        elif metrics_err:
            st.error(f"❌ {metrics_err}")
        elif sizes_err:
            st.error(f"❌ {sizes_err}")
        else:
            train_acc  = float(metrics_parsed.get("train_acc", 0.0))
            val_acc    = float(metrics_parsed.get("val_acc",   0.0))
            test_acc   = float(metrics_parsed.get("test_acc",  0.0))
            roc_auc_v  = float(metrics_parsed.get("roc_auc",   0.0))
            roc_auc    = roc_auc_v if roc_auc_v > 0 else None
            train_size = int(sizes_parsed.get("train_size", 10000))
            test_size  = int(sizes_parsed.get("test_size",  2000))

            # Pre-generate review_id so both Streamlit and GitHub Actions track the same id
            pre_review_id = f"rev-{uuid.uuid4().hex[:8]}"
            payload = {
                "review_id":     pre_review_id,
                "experiment_id": experiment_id.strip().upper(),
                "model_name":    model_name,
                "algorithm":     algorithm,
                "task":          task,
                "train_acc":     train_acc if train_acc > 0 else None,
                "val_acc":       val_acc,
                "test_acc":      test_acc,
                "roc_auc":       roc_auc,
                "features":      features,
                "train_size":    train_size,
                "test_size":     test_size,
                "leakage_risk":  leakage_risk,
                "bias_flags":    bias_flags,
            }

            # Pre-register locally so CI/CD panel appears immediately
            st.session_state.reviews[pre_review_id] = {
                "experiment_id": experiment_id.strip().upper(),
                "model_name":    model_name,
                "submitted_at":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "status":        "pending_approval",
                "source":        "github_actions",
                "pipeline_ready": False,   # runner hasn't run yet
            }
            st.session_state.last_review_id = pre_review_id

            # ── Step 1: run the pipeline directly via FastAPI ────────────────
            outcome = {
                "review_id":        pre_review_id,
                "model_name":       model_name,
                "experiment_id":    experiment_id.strip().upper(),
                "pipeline_ok":      False,
                "pipeline_verdict": None,
                "pipeline_error":   None,
                "gh_dispatched":    None,   # True / False / None (not attempted)
                "gh_error":         None,
            }

            with st.spinner("🔬 Running ML review pipeline… (this may take 30–60 s)"):
                try:
                    result = api_submit(payload)
                    outcome["pipeline_ok"]      = True
                    outcome["pipeline_verdict"] = result.get("preliminary_verdict", "pending")
                except Exception as e:
                    outcome["pipeline_error"] = str(e)

            # ── Step 2: dispatch GitHub Actions for CI/CD tracking ───────────
            # The runner will check whether the review already exists and just poll.
            if outcome["pipeline_ok"] and gh_configured:
                ok, err_msg = trigger_github_actions(pre_review_id, payload)
                outcome["gh_dispatched"] = ok
                outcome["gh_error"]      = err_msg if not ok else None

            # Store outcome so it survives st.rerun()
            st.session_state.last_submit_outcome = outcome
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # CI/CD Status Panel — shows after any submit, persists across tab switches
    # ══════════════════════════════════════════════════════════════════════════

    # Auto-sync: pull any reviews submitted by GitHub Actions that this browser
    # session doesn't know about yet
    try:
        server_reviews = api_list_reviews().get("reviews", {})
        for rid, rmeta in server_reviews.items():
            if rid not in st.session_state.reviews:
                st.session_state.reviews[rid] = {
                    "experiment_id": rmeta.get("experiment_id", "—"),
                    "model_name":    rmeta.get("model_name", "—"),
                    "submitted_at":  rmeta.get("submitted_at", "—"),
                    "status":        "pending_approval",
                    "source":        rmeta.get("source", "cicd"),
                }
    except Exception:
        pass

    last_id = st.session_state.last_review_id
    if last_id and last_id in st.session_state.reviews:
        st.divider()
        meta = st.session_state.reviews[last_id]

        col_title, col_refresh = st.columns([5, 1])
        col_title.markdown("### 📡 CI/CD Pipeline Status  (via GitHub Actions)")
        if col_refresh.button("🔄 Refresh", key="refresh_cicd"):
            st.rerun()

        data = api_get_review(last_id)
        if not data:
            # Runner hasn't picked up the job yet — pipeline not started
            source = meta.get("source", "")
            if source == "github_actions":
                st.info(
                    f"⏳ **Waiting for self-hosted runner to pick up the job…**  \n"
                    f"Review ID: `{last_id}`  \n"
                    f"The GitHub Actions runner will start the pipeline shortly. "
                    f"Click **🔄 Refresh** every few seconds, or check "
                    f"**GitHub → Actions → ML Experiment Review** for job status."
                )
                st.caption(
                    f"Experiment: `{meta['experiment_id']}`  │  "
                    f"Model: **{meta['model_name']}**  │  "
                    f"Dispatched: {meta['submitted_at']}"
                )
            else:
                st.warning(f"Could not fetch state for `{last_id}` from API.")
        else:
            # Sync local status from API
            api_status = data.get("status", "pending_approval")
            if st.session_state.reviews[last_id]["status"] != api_status and api_status in ("deployed","escalated"):
                st.session_state.reviews[last_id]["status"] = api_status

            st.caption(
                f"Review ID: `{last_id}`  │  "
                f"Experiment: `{meta['experiment_id']}`  │  "
                f"Model: **{meta['model_name']}**  │  "
                f"Submitted: {meta['submitted_at']}"
            )

            # ── Pipeline steps from audit trail ──
            st.markdown("**Pipeline execution log:**")
            audit = data.get("audit_trail", [])
            step_icons = {"ingest":"📥","retrieve":"🔍","analyse":"🔬",
                          "review":"📝","hitl_gate":"✋","deploy":"🚀"}

            for entry in audit:
                node = entry.get("node","?")
                icon = step_icons.get(node, "▶️")
                ts   = entry.get("timestamp","")[:19]
                summ = entry.get("summary","")
                st.success(f"{icon} **[{node}]** {summ}  `{ts}`")

            lr = data.get("leakage_result", {})
            br = data.get("bias_result", {})
            if lr.get("blocks_deployment") or br.get("blocks_deployment"):
                st.warning("⚡ **Fast path triggered** — blocking finding detected. `review` node was skipped.")

            # ── Current status banner ──
            st.markdown("---")
            status = data.get("status", "pending_approval")
            verdict = data.get("verdict")

            if status == "deployed":
                st.success(
                    f"✅ **CI/CD unblocked** — `can_deploy: true`  \n"
                    f"Model **{meta['model_name']}** registered to registry.  "
                    f"Verdict: `{verdict}`"
                )
            elif status == "escalated":
                st.error("🚨 **Escalated** — retry limit reached. Manual investigation required. `can_deploy: false`")
            else:
                col_s, col_v = st.columns(2)
                col_s.metric("Status",     "🟡 pending_approval")
                col_v.metric("can_deploy", "false")
                if verdict:
                    pv_color = verdict_color(verdict)
                    st.markdown(f"Preliminary verdict: :{pv_color}[**{verdict.upper()}**]")
                st.info("⏳ **Waiting for human review** — go to **📋 Review Queue** to approve or reject this experiment.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Review Queue
# ═══════════════════════════════════════════════════════════════════════════════
with tab_queue:
    # Auto-sync CI/CD reviews from server before rendering queue
    try:
        server_reviews = api_list_reviews().get("reviews", {})
        new_cicd = [rid for rid in server_reviews if rid not in st.session_state.reviews]
        for rid in new_cicd:
            rmeta = server_reviews[rid]
            st.session_state.reviews[rid] = {
                "experiment_id": rmeta.get("experiment_id", "—"),
                "model_name":    rmeta.get("model_name", "—"),
                "submitted_at":  rmeta.get("submitted_at", "—"),
                "status":        "pending_approval",
                "source":        rmeta.get("source", "cicd"),
            }
        if new_cicd:
            st.info(f"🔄 **{len(new_cicd)} new CI/CD review(s) discovered** — submitted by GitHub Actions / MCP server.")
    except Exception:
        pass

    if not st.session_state.reviews:
        st.info("No experiments submitted yet. Go to **📤 Submit Experiment** to get started.")
    else:
        sorted_reviews = sorted(
            st.session_state.reviews.items(),
            key=lambda x: (x[1]["status"] != "pending_approval", x[1]["submitted_at"]),
        )

        for review_id, meta in sorted_reviews:
            data           = api_get_review(review_id)
            report         = data.get("review_report", {})
            metric_result  = data.get("metric_result", {})
            leakage_result = data.get("leakage_result", {})
            bias_result    = data.get("bias_result", {})
            similar_exps   = data.get("similar_experiments", [])
            parsed_card    = data.get("parsed_card", {})
            deploy_record  = data.get("deployment_record", {})
            verdict        = data.get("verdict", "—")
            is_fast_path   = not bool(report) and bool(metric_result)
            is_pending     = data.get("status", "pending_approval") == "pending_approval"

            # Sync session state status from API
            api_status = data.get("status", "pending_approval")
            if meta["status"] != api_status:
                st.session_state.reviews[review_id]["status"] = api_status

            exp_label = (
                f"{status_badge(meta['status'])}  │  "
                f"`{meta.get('experiment_id','—')}`  │  "
                f"**{meta['model_name']}**  │  `{review_id}`  │  {meta['submitted_at']}"
            )

            with st.expander(exp_label, expanded=is_pending):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Experiment ID", meta.get("experiment_id","—"))
                c2.metric("Algorithm",     parsed_card.get("algorithm","—"))
                c3.metric("Task",          parsed_card.get("task","—"))
                c4.metric("Val Acc",       str(parsed_card.get("val_acc","—")))
                c5.metric("Test Size",     f"{parsed_card.get('test_size',0):,}")

                # ── Diagnostic findings ──
                st.markdown("#### 🔍 Diagnostic Findings")
                d1, d2, d3 = st.columns(3)

                with d1:
                    if metric_result:
                        lvl = metric_result.get("risk_level","?")
                        st.markdown(f"**Metric Analyser** {risk_emoji(lvl)} `{lvl}`")
                        og = metric_result.get("overfit_gap")
                        vt = metric_result.get("val_test_delta")
                        if og is not None: st.caption(f"Overfit gap: `{og}`")
                        if vt is not None: st.caption(f"Val–test delta: `{vt}`")
                        for f in metric_result.get("findings",[]): st.caption(f"• {f}")
                    else: st.caption("—")

                with d2:
                    if leakage_result:
                        lvl    = leakage_result.get("risk_level","?")
                        blocks = leakage_result.get("blocks_deployment", False)
                        st.markdown(f"**Leakage Detector** {risk_emoji(lvl)} `{lvl}`" + (" 🚫" if blocks else ""))
                        st.caption(f"Type: `{leakage_result.get('leakage_type','none')}`")
                        for f in leakage_result.get("findings",[]): st.caption(f"• {f}")
                    else: st.caption("—")

                with d3:
                    if bias_result:
                        lvl    = bias_result.get("risk_level","?")
                        blocks = bias_result.get("blocks_deployment", False)
                        mon    = bias_result.get("requires_monitoring", False)
                        st.markdown(f"**Bias Checker** {risk_emoji(lvl)} `{lvl}`"
                                    + (" 🚫" if blocks else "") + (" 👁" if mon else ""))
                        for f in bias_result.get("findings",[]): st.caption(f"• {f}")
                    else: st.caption("—")

                if is_fast_path:
                    st.error("⚡ **Fast path** — blocking finding detected. LLM review was skipped. Raw evidence above.")

                # ── LLM Review Report ──
                if report:
                    st.markdown("#### 📝 LLM Review Report")
                    vc, ec = st.columns([3,1])
                    color = verdict_color(verdict)
                    vc.markdown(f"**Verdict:** :{color}[**{verdict.upper()}**]")
                    ec.markdown(f"**Fix effort:** `{report.get('estimated_fix_effort','—')}`")
                    st.markdown("**Reasoning (CoT trace):**")
                    st.info(report.get("verdict_reasoning","—"))
                    for risk in report.get("risks",[]):
                        sev   = risk.get("severity","low")
                        block = " 🚫 blocks deployment" if risk.get("blocks_deployment") else ""
                        st.markdown(f"{risk_emoji(sev)} **{risk.get('risk_type','').upper()}** `{sev}`{block}")
                        st.caption(f"Evidence: {risk.get('evidence','—')}")
                        st.caption(f"→ Fix: {risk.get('recommendation','—')}")
                    for c in report.get("deployment_conditions",[]):
                        st.caption(f"• {c}")

                elif deploy_record:
                    st.success(f"✅ Deployed — verdict `{deploy_record.get('verdict')}`")

                # ── Similar experiments ──
                if similar_exps:
                    st.markdown("#### 🔗 Similar Past Experiments")
                    for r in similar_exps:
                        e       = r["experiment"]
                        o_color = "green" if "approved" in e["outcome"] else "red"
                        st.caption(
                            f"`{e['id']}` — **{e['model']}** ({e['algorithm']})  "
                            f":{o_color}[{e['outcome']}]  leakage=`{e['leakage_risk']}`  rrf=`{r['rrf_score']}`"
                        )

                # ── Audit trail ──
                audit = data.get("audit_trail",[])
                if audit:
                    with st.expander(f"📋 Audit Trail ({len(audit)} entries)", expanded=False):
                        for entry in audit:
                            st.caption(
                                f"`{entry.get('timestamp','')[:19]}`  "
                                f"**[{entry.get('node','?')}]**  {entry.get('summary','')}"
                            )

                # ── Decision panel ──
                if is_pending:
                    st.divider()
                    st.markdown("#### ✋ Your Decision")

                    attempts     = data.get("approval_attempts", 0)
                    retries_left = 2 - attempts

                    if is_fast_path:
                        # Hard block — leakage or bias blocks_deployment=True. No retry loop.
                        st.error("🚫 **Hard block detected.** Rejection is final — no retries are given for leakage or regulatory bias blocks.")
                        feedback = st.text_area(
                            "Rejection reason (recorded in audit trail):",
                            placeholder="e.g. 'Target leakage via future_purchase_flag. Remove feature and retrain.'",
                            key=f"fb_{review_id}",
                        )
                        a_col, r_col = st.columns(2)
                        if a_col.button("✅  Approve anyway", key=f"approve_{review_id}",
                                        use_container_width=True):
                            with st.spinner("Deploying model to registry…"):
                                result = api_decide(review_id, "approved", feedback)
                            st.session_state.reviews[review_id]["status"] = result.get("final_status", "deployed")
                            st.success(f"✅ **{meta['model_name']}** approved. CI/CD unblocked.")
                            st.rerun()
                        if r_col.button("🚫  Reject (Final)", key=f"reject_{review_id}",
                                        type="primary", use_container_width=True):
                            with st.spinner("Processing final rejection…"):
                                result = api_decide(review_id, "rejected", feedback)
                            st.session_state.reviews[review_id]["status"] = "escalated"
                            st.error("🚨 **Rejected.** CI/CD will see `can_deploy: false`. Experiment moved to Rejected tab.")
                            st.rerun()
                    else:
                        # Normal path — LLM review ran, retries allowed
                        if retries_left <= 0:
                            st.warning("⚠️ Retry limit reached. A rejection now will **escalate** this experiment.")
                        else:
                            st.caption(f"Retries remaining after rejection: **{max(retries_left-1,0)}**  *(feedback is sent to LLM for re-review)*")

                        feedback = st.text_area(
                            "Revision request for LLM re-review (required on rejection):",
                            placeholder="e.g. 'Overfit gap of 0.09 is borderline — verify holdout split. Re-evaluate risk level.'",
                            help="This feedback is injected into the LLM prompt. The review agent will re-run its analysis addressing your specific concerns.",
                            key=f"fb_{review_id}",
                        )

                        a_col, r_col = st.columns(2)

                        if a_col.button("✅  Approve", key=f"approve_{review_id}",
                                        type="primary", use_container_width=True):
                            with st.spinner("Deploying model to registry…"):
                                result = api_decide(review_id, "approved", feedback)
                            st.session_state.reviews[review_id]["status"] = result.get("final_status","deployed")
                            st.success(f"✅ **{meta['model_name']}** approved.  CI/CD unblocked — `can_deploy: true`")
                            st.rerun()

                        if r_col.button("❌  Reject / Request Revision", key=f"reject_{review_id}",
                                        use_container_width=True):
                            if not feedback.strip():
                                st.warning("Please provide feedback before rejecting.")
                            else:
                                spinner_msg = "Re-running review with your feedback…" if retries_left > 1 else "Processing final rejection…"
                                with st.spinner(spinner_msg):
                                    result = api_decide(review_id, "revision_requested", feedback)
                                final_status = result.get("final_status","escalated")
                                st.session_state.reviews[review_id]["status"] = final_status
                                if final_status == "pending_approval":
                                    st.info("🔄 LLM re-reviewed with your feedback. Scroll up to see the revised report.")
                                else:
                                    st.error("🚨 Retry limit reached. Experiment escalated. CI/CD will see `can_deploy: false`.")
                                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Rejected Experiments
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rejected:
    st.subheader("❌ Rejected Experiments")
    st.caption("Experiments that were rejected or escalated — either via fast-path hard block or after exhausting retries.")

    # Sync statuses from API before rendering
    for rid in list(st.session_state.reviews.keys()):
        if st.session_state.reviews[rid]["status"] not in ("deployed", "escalated"):
            fresh = api_get_review(rid)
            if fresh.get("status") in ("escalated",):
                st.session_state.reviews[rid]["status"] = "escalated"

    rejected_reviews = [(rid, meta) for rid, meta in st.session_state.reviews.items()
                        if meta["status"] == "escalated"]
    rejected_reviews.sort(key=lambda x: x[1]["submitted_at"], reverse=True)

    if not rejected_reviews:
        st.info("No rejected experiments yet. Rejections will appear here once you reject a review in the **📋 Review Queue**.")
    else:
        st.caption(f"**{len(rejected_reviews)} rejected experiment(s)** this session")
        st.divider()
        for review_id, meta in rejected_reviews:
            data           = api_get_review(review_id)
            report         = data.get("review_report", {})
            leakage_result = data.get("leakage_result", {})
            bias_result    = data.get("bias_result", {})
            metric_result  = data.get("metric_result", {})
            parsed_card    = data.get("parsed_card", {})
            audit          = data.get("audit_trail", [])
            attempts       = data.get("approval_attempts", 0)
            is_fp          = not bool(report) and bool(metric_result)

            exp_label = (
                f"🚫  `{meta.get('experiment_id','—')}`  │  "
                f"**{meta['model_name']}**  │  `{review_id}`  │  {meta['submitted_at']}"
                + ("  ⚡ fast-path" if is_fp else f"  ({attempts} attempt(s))")
            )

            with st.expander(exp_label, expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Experiment ID", meta.get("experiment_id","—"))
                c2.metric("Algorithm",     parsed_card.get("algorithm","—"))
                c3.metric("Task",          parsed_card.get("task","—"))

                # ── Rejection reason ──
                st.markdown("#### 🔍 Why it was rejected")
                if is_fp:
                    st.error("⚡ **Fast-path hard block** — LLM review was skipped entirely.")
                    if leakage_result.get("blocks_deployment"):
                        st.markdown(f"**Leakage:** `{leakage_result.get('leakage_type','—')}`")
                        for f in leakage_result.get("findings",[]): st.caption(f"• {f}")
                    if bias_result.get("blocks_deployment"):
                        st.markdown("**Bias/Fairness block:**")
                        for f in bias_result.get("findings",[]): st.caption(f"• {f}")
                else:
                    verdict = report.get("overall_verdict","—")
                    st.markdown(f"**LLM Verdict:** `{verdict}`")
                    st.info(report.get("verdict_reasoning","—"))
                    for risk in report.get("risks",[]):
                        sev = risk.get("severity","low")
                        st.markdown(f"{risk_emoji(sev)} **{risk.get('risk_type','').upper()}** `{sev}`")
                        st.caption(f"Evidence: {risk.get('evidence','—')}")
                        st.caption(f"→ Fix: {risk.get('recommendation','—')}")

                # ── Human feedback given ──
                last_feedback = next(
                    (e.get("summary","") for e in reversed(audit) if "feedback" in e.get("summary","").lower()),
                    None
                )
                if last_feedback:
                    st.markdown("**Your feedback:**")
                    st.caption(last_feedback)

                # ── Audit trail ──
                if audit:
                    with st.expander(f"📋 Audit Trail ({len(audit)} entries)", expanded=False):
                        for entry in audit:
                            st.caption(
                                f"`{entry.get('timestamp','')[:19]}`  "
                                f"**[{entry.get('node','?')}]**  {entry.get('summary','')}"
                            )
        st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Registry
# ═══════════════════════════════════════════════════════════════════════════════
with tab_registry:
    st.subheader("🗂️ Model Registry")

    session_deployed = [(rid, meta) for rid, meta in st.session_state.reviews.items()
                        if meta["status"] == "deployed"]
    session_deployed.sort(key=lambda x: x[1]["submitted_at"], reverse=True)

    total = len(seed_registry) + len(session_deployed)
    st.caption(f"**{total} model(s)** registered  —  {len(seed_registry)} existing  +  {len(session_deployed)} approved this session")
    st.divider()

    # ── Section A: approved this session ──
    if session_deployed:
        st.markdown("### ✨ Approved This Session")
        for review_id, meta in session_deployed:
            data          = api_get_review(review_id)
            deploy_record = data.get("deployment_record", {})
            report        = data.get("review_report", {})
            similar_exps  = data.get("similar_experiments", [])
            verdict       = deploy_record.get("verdict") or report.get("overall_verdict","approved")
            verdict_icon  = {"approve":"✅","approve_with_monitoring":"⚠️"}.get(verdict,"✅")
            metrics       = deploy_record.get("metrics",{})

            with st.container():
                h1, h2, h3 = st.columns([1.2, 3, 1.8])
                h1.markdown(f"#### {verdict_icon} `{meta.get('experiment_id','—')}`")
                h2.markdown(f"**{deploy_record.get('model_name', meta['model_name'])}**  \n"
                            f"`{deploy_record.get('algorithm','—')}` — {deploy_record.get('task','—')}")
                h3.markdown(f"Approved: `{meta['submitted_at']}`")

                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Val Acc",   metrics.get("val_acc")  or "—")
                m2.metric("Test Acc",  metrics.get("test_acc") or "—")
                m3.metric("ROC-AUC",   metrics.get("roc_auc")  or "—")
                m4.metric("Review ID", review_id)
                m5.metric("Verdict",   verdict)

                for c in deploy_record.get("deployment_conditions",[]):
                    st.warning(f"⚠️ Condition: {c}")
                if similar_exps:
                    st.caption("📎 Precedent: " + ", ".join(f"`{r['experiment']['id']}`" for r in similar_exps))

                audit = data.get("audit_trail",[])
                with st.expander("📋 Audit Trail", expanded=False):
                    for entry in audit:
                        st.caption(f"`{entry.get('timestamp','')[:19]}`  **[{entry.get('node','?')}]**  {entry.get('summary','')}")
            st.divider()

    # ── Section B: existing seed registry ──
    st.markdown("### 📦 Existing Registry")
    st.caption("Loaded from `data/model_registry.json` — pre-existing production models")
    status_icon = {"production":"🟢","limited":"🔵","deprecated":"🔴"}

    for mdl in seed_registry:
        exp = exp_lookup.get(mdl["experiment_id"],{})
        with st.container():
            h1, h2, h3 = st.columns([1.2, 3, 1.8])
            h1.markdown(f"#### {status_icon.get(mdl['status'],'⚪')} `{mdl['experiment_id']}`")
            h2.markdown(f"**{mdl['name']}**  v{mdl['version']}  \n"
                        f"`{exp.get('algorithm','—')}` — {exp.get('task','—')}")
            h3.markdown(f"Status: `{mdl['status']}`  \nBy: `{mdl.get('deployed_by','—')}`")

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Val Acc",     exp.get("val_acc")  or "—")
            m2.metric("ROC-AUC",     exp.get("roc_auc")  or "—")
            m3.metric("SLA",         f"{mdl.get('sla_latency_ms','—')} ms")
            m4.metric("Registry ID", mdl["id"])
            m5.metric("Outcome",     exp.get("outcome","—"))

            for flag in exp.get("bias_flags",[]):
                st.warning(f"⚠️ {flag}")
        st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_leaderboard:
    st.subheader("🏆 Experiment Leaderboard")
    st.caption("Approved experiments ranked by validation accuracy — the artifact the team checks day to day.")

    col_refresh, col_filter = st.columns([1, 3])
    if col_refresh.button("🔄 Refresh", key="refresh_leaderboard"):
        st.rerun()

    lb_data = api_leaderboard()

    if "error" in lb_data:
        st.error(f"❌ Could not fetch leaderboard: {lb_data['error']}")
    else:
        entries = lb_data.get("leaderboard", [])
        total   = lb_data.get("total_approved", 0)

        # ── Filter controls ──────────────────────────────────────────────────
        all_tasks       = sorted({e["task"]      for e in entries if e.get("task")})
        all_algorithms  = sorted({e["algorithm"] for e in entries if e.get("algorithm")})

        with col_filter:
            task_filter = st.multiselect(
                "Filter by Task", all_tasks,
                placeholder="All tasks", label_visibility="collapsed"
            )

        alg_col, outcome_col = st.columns(2)
        with alg_col:
            alg_filter = st.multiselect(
                "Filter by Algorithm", all_algorithms,
                placeholder="All algorithms", label_visibility="visible",
                key="alg_filter"
            )
        with outcome_col:
            outcome_filter = st.multiselect(
                "Filter by Outcome",
                ["approved", "approved_with_monitoring"],
                placeholder="All outcomes", label_visibility="visible",
                key="outcome_filter"
            )

        # Apply filters
        filtered = entries
        if task_filter:
            filtered = [e for e in filtered if e.get("task") in task_filter]
        if alg_filter:
            filtered = [e for e in filtered if e.get("algorithm") in alg_filter]
        if outcome_filter:
            filtered = [e for e in filtered if e.get("outcome") in outcome_filter]

        # Re-rank after filtering
        for i, e in enumerate(filtered):
            e["rank"] = i + 1

        st.caption(
            f"Showing **{len(filtered)}** of **{total}** approved experiment(s)  "
            + ("— filters active" if (task_filter or alg_filter or outcome_filter) else "")
        )
        st.divider()

        if not filtered:
            st.info("No experiments match the selected filters.")
        else:
            # ── Medal rows ───────────────────────────────────────────────────
            MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

            for entry in filtered:
                rank    = entry["rank"]
                medal   = MEDALS.get(rank, f"**#{rank}**")
                outcome = entry.get("outcome", "—")
                o_color = "green" if outcome == "approved" else "orange"
                val_acc = entry.get("val_acc")
                roc_auc = entry.get("roc_auc")

                with st.container():
                    r1, r2, r3, r4, r5, r6 = st.columns([0.6, 2.4, 1.6, 1.2, 1.2, 1.4])
                    r1.markdown(f"### {medal}")
                    r2.markdown(
                        f"**{entry['model']}**  \n"
                        f"`{entry['algorithm']}` — {entry['task']}"
                    )
                    r3.metric("Val Acc",  f"{val_acc:.4f}"  if val_acc  is not None else "—")
                    r4.metric("ROC-AUC",  f"{roc_auc:.4f}"  if roc_auc  is not None else "—")
                    r5.markdown(f"**Outcome**  \n:{o_color}[{outcome}]")
                    r6.metric("Experiment", entry["id"])

                st.divider()

            # ── Summary stats ────────────────────────────────────────────────
            st.markdown("#### 📊 Summary")
            s1, s2, s3, s4 = st.columns(4)
            val_accs = [e["val_acc"] for e in filtered if e.get("val_acc") is not None]
            roc_aucs = [e["roc_auc"] for e in filtered if e.get("roc_auc") is not None]

            s1.metric("Total Shown",    len(filtered))
            s2.metric("Best Val Acc",   f"{max(val_accs):.4f}"  if val_accs else "—")
            s3.metric("Avg Val Acc",    f"{sum(val_accs)/len(val_accs):.4f}" if val_accs else "—")
            s4.metric("Avg ROC-AUC",    f"{sum(roc_aucs)/len(roc_aucs):.4f}" if roc_aucs else "—")

            # ── With-monitoring badge ────────────────────────────────────────
            monitored = [e for e in filtered if e.get("outcome") == "approved_with_monitoring"]
            if monitored:
                st.warning(
                    f"⚠️ **{len(monitored)} model(s) approved with monitoring conditions** — "
                    + ", ".join(f"`{e['model']}`" for e in monitored)
                )
