# ML Experiment Intelligence Suite

An automated review pipeline for ML experiments at Meridian Analytics. It replaces ad hoc manual notebook review with a consistent, evidence-based pipeline: every submitted experiment is validated, contextualised against prior runs, diagnosed by three independent checks, and routed to a senior human reviewer with an auditable trail — never silently approved when leakage or bias is present.

A **Streamlit UI** provides a full reviewer dashboard (submit, queue, registry, leaderboard), and a **GitHub Actions workflow** allows CI/CD pipelines running on a self-hosted runner to track the review end-to-end.

---

## Architecture

```
                             EXPERIMENT CARD
                                   │
                                   ▼
                         ┌─────────────────┐
                         │  INGEST         │  Validates submission against fixed schema
                         └───────┬─────────┘
                                 ▼
                         ┌─────────────────┐
                ┌───────▶│  RETRIEVE       │  Hybrid BM25 + vector search over past
                │        └───────┬─────────┘  experiments (index updated by DEPLOY /
                │                ▼             terminal outcomes)
                │        ┌─────────────────┐
                │        │  ANALYSE        │  metric_analyser, leakage_detector,
                │        │                 │  bias_checker → structured, evidence-
                │        └───────┬─────────┘  cited findings
                │                │
                │      ┌─────────┴──────────┐
                │  blocking              no blocking
                │  finding                finding
                │      │                    ▼
                │      │          ┌─────────────────┐
                │      │          │  REVIEW         │  Narrates findings + retrieved
                │      │          └────────┬────────┘  context into a verdict
                │      └─────────┬─────────┘
                │                ▼
                │        ┌─────────────────┐
                │        │  APPROVAL GATE  │  Senior reviewer approves / rejects
                │        │  (human-in-loop)│  with optional feedback
                │        └────────┬────────┘
                │      approved   │   rejected
                │                 ▼
                │        ┌─────────────────┐
                └────────┤  RETRY GATE     │  Reads rejection's reason_type;
                 bound     │                 │  routes back to ANALYSE or REVIEW
                 exceeded  └────────┬────────┘  (max 2 resubmissions)
                     ▼             │
              TERMINAL REJECT      ▼
                         ┌─────────────────┐
                         │  DEPLOY         │  Registers to model registry with lineage
                         └───────┬─────────┘
                                 ▼
                         ┌─────────────────┐
                         │  MCP SERVER     │  CI/CD submission + polling interface
                         └─────────────────┘  (never bypasses the approval gate)
```

**Fast path:** if `analyse` returns a blocking leakage or bias finding, the graph routes straight to the approval gate — skipping `review`. A confirmed leakage or bias risk doesn't need a generated narrative to justify escalation; it needs to reach a human immediately, with the raw evidence and retrieved precedents attached.

---

## Project Structure

```
.
├── agent/
│   ├── graph.py             # LangGraph: ExperimentState, all 6 nodes, routing functions,
│   │                        #   compiled graph with interrupt_before=["hitl_gate"]
│   ├── tools.py             # metric_analyser, leakage_detector, bias_checker + LLM client
│   └── prompts.py           # ExperimentCard, ExperimentRisk, ReviewReport schemas;
│                            #   REVIEW_TMPL (risk synthesiser), REMEDIATION_TMPL (few-shot)
├── rag/
│   ├── ingest.py            # BM25 index + ChromaDB vector store construction
│   └── retriever.py         # hybrid_retrieve_experiments() — RRF fusion, self-exclusion
├── api/
│   └── main.py              # FastAPI: POST /experiment, POST /review/{id}/decision,
│                            #   GET /review/{id}, GET /experiments/leaderboard,
│                            #   GET /reviews, GET /health
├── mcp_server/
│   └── server.py            # FastMCP: submit_experiment, submit_review_decision,
│                            #   get_review_status, experiments://registry resource
├── scripts/
│   └── cicd_submit.py       # Standalone script for CI/CD runners — submits experiment
│                            #   and polls for verdict; used by the GitHub Actions workflow
├── .github/
│   └── workflows/
│       └── ml-review.yml    # GitHub Actions: submit → pipeline → poll for verdict
│                            #   (self-hosted runner; triggered from Streamlit or manually)
├── data/
│   ├── experiments.json     # Seed corpus — 20 past experiments with outcomes
│   └── model_registry.json  # Production model records, linked by experiment_id
├── app.py                   # Streamlit UI — 5-tab reviewer dashboard (see below)
├── Capstone2_MLExperimentIntelligence.ipynb  # Development notebook (see Testing below)
├── Dockerfile               # Multi-stage build, non-root appuser, /health HEALTHCHECK
├── docker-compose.yml       # api (port 8000) + mcp_server (port 8001) services
├── requirements.txt
└── .env                     # endpoint, deployment_name, OPENAI_API_KEY
```

---

## Streamlit UI

Run alongside the FastAPI backend to get a full reviewer dashboard:

```bash
streamlit run app.py
```

The UI is available at `http://localhost:8501` and provides five tabs:

| Tab | Description |
|---|---|
| **📤 Submit Experiment** | Form-based experiment card submission; runs the review pipeline directly via FastAPI, then optionally dispatches a GitHub Actions workflow for CI/CD tracking. Shows a persistent CI/CD status panel with an audit trail log after submission. |
| **📋 Review Queue** | Lists all pending and past reviews (including those discovered from GitHub Actions/MCP via `/reviews` auto-sync). Shows diagnostic findings, the LLM report with CoT trace, similar past experiments (RRF-ranked), and the Approve / Reject / Request Revision decision panel with retry counter. |
| **❌ Rejected** | All escalated experiments with rejection reasoning, human feedback given, and full audit trail. |
| **🗂️ Model Registry** | Session-deployed models + the full seed registry loaded from `data/model_registry.json`. Shows metrics, verdict, deployment conditions, and precedent links. |
| **🏆 Leaderboard** | Approved experiments ranked by `val_acc` descending, with task/algorithm/outcome filters and summary stats. |

### GitHub Actions integration (Streamlit → CI/CD)

When `GITHUB_TOKEN` and `GITHUB_REPO` are set, the **Submit Experiment** tab also dispatches `ml-review.yml` so a self-hosted runner can independently poll for the verdict:

```bash
export GITHUB_TOKEN=ghp_your_token
export GITHUB_REPO=owner/repo-name
streamlit run app.py
```

The Streamlit UI and the GitHub Actions runner share the same `review_id` (pre-generated before submission), so both views stay in sync via the FastAPI `/reviews` endpoint.

---

## Design Memo

### 1. Why run leakage detection as a discrete tool call rather than folding it into the review prompt?

Leakage detection has to produce a specific, checkable claim — "feature X is computed after the label is known" — that gates a routing decision (`blocks_deployment`), not a stylistic judgment folded into a longer narrative. Keeping it as a separate tool call gives it an independent, testable output that other nodes can consume directly: `route_after_analyse` reads `leakage_result['blocks_deployment']` without parsing it back out of a free-form paragraph. It also means leakage detection runs and produces a verdict even when the fast path skips `review` entirely — if it were embedded inside the review prompt, a skipped review would mean no leakage check ran at all. Finally, independence lets a reviewer trust each finding on its own: if `leakage_detector` and `review` were the same call, a good narrative could mask a shaky underlying leakage judgment, and there would be no way to separately audit or re-run just the leakage logic when reviewing the tool's own accuracy over time.

### 2. Should a bias finding ever block deployment outright, or should it always route to a human first?

In this design, a *high-confidence* bias finding — an explicit `bias_flags` entry describing a clear disparity, or a test set too small to trust — sets `blocks_deployment` and fast-paths to the human gate. But "blocks deployment" means "forces immediate human review," not "auto-rejects without a human." The human reviewer is still the one who makes the final call; the flag only removes the option of a model with a stated bias problem sliding through on an `approve_with_monitoring` verdict without that specific evidence being surfaced first. A *softer* signal — like EXP-003's 3% APAC representation — doesn't block; it goes through the normal `review` path so the LLM can contextualise severity before the human sees it. The line is drawn at confidence and stated evidence, not at the mere presence of any bias-adjacent note, because treating every domain-coverage gap as a hard block would make the fast path noisy and erode the reviewer's trust in it.

### 3. What do you embed for retrieval — raw metrics, or derived descriptions?

Derived descriptions — combining algorithm, task, outcome, tags, and notes — not raw metric values. Two experiments can have superficially similar numbers (e.g., both around 0.85 val accuracy) while having nothing in common in terms of *why* they succeeded or failed, and two experiments can share a real failure pattern (temporal leakage from lag features) while having very different numbers. Embedding a semantic description captures "experiment type and failure pattern," which is what a reviewer actually wants when asking "has something like this happened before." Raw metrics are still useful — they're kept as ChromaDB metadata for filtering (e.g., search only non-approved outcomes) and fed directly to `metric_analyser`, which needs exact numbers, not semantic similarity.

---

## Improvements Over the Provided Hints

| Area | Hint as given | What was built instead | Rationale |
|---|---|---|---|
| Fast-path auditability | Fast path skips `review`, implying the narrative is what's skipped | `analyse` always emits a structured, evidence-cited findings object (feature name, metric, prior experiment ID); `review` only adds narration on top | The constraint "every verdict must cite specific evidence" applies to every path, including the fast path. If structuring only happened inside `review`, skipping `review` would silently also skip the audit trail. |
| Retrieval context on fast path | Not specified | Retrieved similar experiments are attached to the findings object itself, not just passed into `review`'s input | Without this, a fast-tracked leakage finding reaches the human with no historical precedent attached, even though `retrieve` already ran and found relevant context (e.g., "this resembles EXP-002"). |
| Retry loop routing | Diagram shows `rejected → review` unconditionally | `RETRY GATE` reads the rejection's `reason_type` (`analysis_dispute` vs `reasoning_dispute`) and routes back to `analyse` or `review` accordingly | If a reviewer rejects because a metric or finding itself was wrong, re-running only `review` regenerates a narrative over stale, disputed diagnostics. Disputes about the underlying analysis must route back to `analyse`. |
| Corpus growth | Not specified | Feedback edge from `deploy` and from terminal rejections back into the `retrieve` index | Without this, the retrieval corpus is static at 20 seed experiments forever. New experiments — approved or rejected — should become future precedent, which is the entire point of the retrieval layer. |
| Retry bound enforcement | "Bounded revise-and-resubmit loop" mentioned, no explicit mechanism | Explicit `approval_attempts` counter in `ExperimentState`, checked before each loop-back, forcing a terminal `escalate` state once exhausted | Without an explicit counter and check, "bounded" has no enforcement mechanism and the loop could run indefinitely in practice. |
| MCP status semantics | "Poll for a verdict without a human in that loop" | `get_review_status` returns pipeline *state* (`pending_human_review`, `approved`, `rejected`, `revise_requested`) — never a value implying the human step was bypassed | `X-Pipeline-Key` (CI/CD) cannot call `submit_review_decision`. The status contract makes that boundary explicit. |

---

## Assumptions

- **Per-class test-set threshold for `bias_checker`.** Only total `test_size` is available — no per-class breakdown. For binary classification tasks the threshold is applied against `test_size / 2` (i.e., ~2,000 total minimum). For non-classification tasks (regression, forecasting, causal inference) the threshold is applied against total `test_size` with a lower absolute floor, reported as "insufficient data to evaluate statistical power" rather than a hard fail when per-class division is meaningless.

- **Outcome values that count as "approved" on the leaderboard.** The seed data includes `approved`, `approved_with_monitoring`, and `approved_limited`. All three are included in `GET /experiments/leaderboard` since all three represent live, deployed models. Outcomes `rejected`, `rejected_bias`, `rejected_performance`, and `rejected_stats` are excluded.

- **Experiments with `val_acc: None` on the leaderboard.** Excluded from the sorted result entirely — not coerced to 0 — since unsupervised and survival-analysis experiments (e.g., `IsolationForest`, `CoxPH`) have no comparable `val_acc` and including them in a numeric sort would be misleading.

- **`approval_attempts` bound.** Not specified numerically in the brief. Set to **2 resubmissions (3 total review attempts)** before forced escalation — enough to give a data scientist a real chance to fix a genuine issue, not enough to let a marginal experiment loop indefinitely.

- **Reviewer identity.** The brief describes "a senior reviewer" without specifying per-reviewer authentication. A single shared `X-Reviewer-Key` is sufficient for this capstone's scope; per-reviewer identity and audit logging are noted as natural extensions.

- **LLM reasoning in `leakage_detector`.** The LLM reasoning pass runs *in addition to* the rule-based `leakage_risk` field check — not only when that field is missing — since the field itself may under-report leakage that only becomes apparent from reasoning about feature names against task type (e.g., a feature named `lag_7` in a regression where the split wasn't time-ordered).

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env` and fill in your credentials:

```bash
cp .env.example .env            # or create .env manually
```

Required environment variables:

```
endpoint=<your OpenAI-compatible base URL>
deployment_name=<model deployment name>
OPENAI_API_KEY=<your API key>
```

Optional (for Streamlit → GitHub Actions CI/CD dispatch):

```
GITHUB_TOKEN=<personal access token with workflow scope>
GITHUB_REPO=<owner/repo-name>
REVIEWER_KEY=<reviewer key sent in X-Reviewer-Key header; defaults to dev-reviewer-key>
API_BASE_URL=<FastAPI base URL; defaults to http://localhost:8000>
```

---

## Running

### Option A — FastAPI + MCP server + Streamlit UI directly

```bash
# Terminal 1 — REST API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — MCP server (CI/CD interface)
python mcp_server/server.py

# Terminal 3 — Streamlit reviewer dashboard
streamlit run app.py
```

- API: `http://localhost:8000`
- MCP server: `http://localhost:8001`
- Streamlit UI: `http://localhost:8501`
- Health check: `http://localhost:8000/health`

### Option B — Docker Compose (API + MCP server together)

```bash
docker-compose up --build
```

Then run the Streamlit UI separately (it calls the API at `localhost:8000`):

```bash
streamlit run app.py
```

---

## Testing the Three Required Paths


| Path | Experiment to use | What to observe |
|---|---|---|
| **Clean approval** | `EXP-001` or `EXP-004` | Full path: `ingest → retrieve → analyse → review → pause → approve → deploy` |
| **Leakage fast rejection** | `EXP-002` or `EXP-007` | `review` is skipped; raw leakage finding + retrieved precedents reach the gate directly |
| **Rejection-with-feedback loop** | `EXP-010` | Reject with feedback; reviewer's note flows into the resubmitted review; retry counter increments; loop terminates on second rejection |

You can also trigger all three paths interactively through the **Streamlit UI** without touching the notebook.

### API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/experiment` | `X-Pipeline-Key` | Submit an experiment card for review |
| `POST` | `/review/{id}/decision` | `X-Reviewer-Key` | Approve or reject with optional feedback |
| `GET` | `/review/{id}` | — | Fetch current pipeline state for a review |
| `GET` | `/reviews` | — | List all review IDs known to the server (used by Streamlit auto-sync) |
| `GET` | `/experiments/leaderboard` | — | Approved experiments sorted by `val_acc` descending |
| `GET` | `/health` | — | Uptime, model name, reviews processed |

### MCP tools (CI/CD interface)

| Tool / Resource | Caller | Description |
|---|---|---|
| `submit_experiment` | Training job (`X-Pipeline-Key`) | Starts a review, returns `review_id` |
| `get_review_status` | CI/CD polling | Returns pipeline state; `can_deploy` is only `true` after human approval |
| `submit_review_decision` | Senior reviewer (`X-Reviewer-Key`) | Injects approve/reject + feedback into the running graph |
| `experiments://registry` | Read-only | Full reviewed experiment corpus |

> **Auth boundary:** `X-Pipeline-Key` can call `submit_experiment` only. It cannot call `submit_review_decision`. A training job can never approve its own experiment.

### GitHub Actions workflow (CI/CD tracking)

The workflow at `.github/workflows/ml-review.yml` runs on a **self-hosted runner** so it can reach `localhost:8000`. It is triggered automatically when you submit from the Streamlit UI (if `GITHUB_TOKEN`/`GITHUB_REPO` are configured) or manually from **GitHub → Actions → ML Experiment Review → Run workflow**.

**One-time setup:**

1. Register your machine as a self-hosted runner:
   `GitHub repo → Settings → Actions → Runners → New self-hosted runner`
2. Ensure FastAPI is running before triggering: `uvicorn api.main:app --port 8000 --reload`
3. Set `API_BASE_URL = http://localhost:8000` in **GitHub → Settings → Variables**
4. Optionally set `PIPELINE_KEY` in **GitHub → Settings → Secrets**

The workflow calls `scripts/cicd_submit.py`, which submits the experiment card to FastAPI and polls `GET /review/{id}` until `can_deploy` is `true` or the poll limit is reached (`MAX_POLLS=80`, `POLL_INTERVAL_SECONDS=15`).

---

## Tech Stack

| Component | Technology |
|---|---|
| Agentic pipeline | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph, MemorySaver, interrupt_before) |
| LLM | Azure OpenAI / OpenAI-compatible endpoint (via `openai` SDK) |
| Retrieval | BM25 (`rank-bm25`) + ChromaDB vector store, RRF fusion |
| REST API | FastAPI + Uvicorn |
| Reviewer UI | Streamlit |
| CI/CD interface | FastMCP (`mcp[cli]`) |
| CI/CD workflow | GitHub Actions (self-hosted runner, `scripts/cicd_submit.py`) |
| Containerisation | Docker + Docker Compose |
| Schema validation | Pydantic v2, Jinja2 |

---

## Demo

A short end-to-end recording accompanies this submission, showing:

1. **Clean approval** — a low-risk churn or LTV experiment proceeding through the full pipeline to deployment.
2. **Leakage fast rejection** — `review` being skipped and the raw leakage finding reaching the approval gate directly, with retrieved precedents attached.
3. **Rejection-with-feedback cycle** — the reviewer's feedback flowing into a resubmitted review that references the prior rejection reason, and the retry loop terminating correctly on exhaustion.
