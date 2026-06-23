# NIDS AI Project — Claude Code Context

## Who I Am
- Student: Chaima (ML Engineer role)
- Project: Final Year Project (Projet de Fin d'Etudes 2025-2026)
- Role: Build the ML pipeline, train models, evaluate and export them
- Teammate: Building the FastAPI backend, Docker, PostgreSQL, Redis, React dashboard

---

## Project Overview
AI-powered Network Intrusion Detection System (NIDS) that detects and classifies network attacks in real time using machine learning trained on the CICIDS2017 dataset.

**Full stack:**
- ML model (my work) → FastAPI backend → PostgreSQL + Redis → React dashboard
- Containerized with Docker Compose
- Target: precision ≥97%, recall ≥95%, F1 ≥0.96 on CICIDS2017

---

## Dataset — CICIDS2017
- Source: University of New Brunswick, Canada
- 8 daily CSV files (Monday–Friday), ~2.8GB raw
- Total rows after cleaning: **2,059,411**
- Original columns: 79 (78 features + 1 Label)
- Platform: Kaggle (all code runs on Kaggle notebooks)
- Data path on Kaggle: `/kaggle/input/datasets/chaimaabid233/cicids2017/MachineLearningCVE/`

### Attack Classes (15 total)
| Class | Count | % |
|-------|-------|---|
| Benign | 1,718,709 | 83.46% |
| Dos Hulk | 138,279 | 6.71% |
| DDoS | 102,413 | 4.97% |
| PortScan | 72,655 | 3.53% |
| Dos GoldenEye | 8,229 | 0.40% |
| FTP-Patator | 4,746 | 0.23% |
| Dos Slowloris | 4,308 | 0.21% |
| Dos Slowhttptest | 4,182 | 0.20% |
| SSH-Patator | 2,575 | 0.13% |
| Bot | 1,562 | 0.08% |
| Web Attack - Brute Force | 1,176 | 0.06% |
| Web Attack - XSS | 522 | 0.03% |
| Infiltration | 29 | 0.001% |
| Web Attack - SQL Injection | 17 | 0.001% |
| Heartbleed | 9 | 0.000% |

**Imbalance ratio: 190,967x (Benign vs Heartbleed)**

---

## Week 1 — COMPLETED ✅
All preprocessing done. Files saved to `/kaggle/working/`.

### Preprocessing Pipeline (12 steps)
1. Strip column name whitespace
2. Remove 8 zero-variance features
3. Replace negative impossible values with NaN
4. Replace infinite values with NaN
5. Train/test split 80/20 stratified
6. Compute medians from X_train only
7. Fill NaN with training medians
8. Apply same medians to X_test
9. Remove 23 correlated features (>0.95 Pearson)
10. StandardScaler fit on X_train only
11. Apply scaler to both sets
12. Encode labels with LabelEncoder

### Feature Counts
- Raw features: 78
- After zero-variance removal: 71 (removed 8)
- After correlation removal: **47** (removed 23 from 70)
- **Final feature count: 47**

### Zero-variance features removed (8)
`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`

### Correlated features removed (23)
`Avg Bwd Segment Size`, `SYN Flag Count`, `Subflow Fwd Packets`, `CWE Flag Count`, `Avg Fwd Segment Size`, `Subflow Bwd Packets`, `Fwd Header Length.1`, `Subflow Bwd Bytes`, `Subflow Fwd Bytes`, `Total Backward Packets`, `Fwd IAT Total`, `ECE Flag Count`, `Fwd IAT Max`, `Average Packet Size`, `Total Length of Bwd Packets`, `Idle Max`, `Idle Min`, `Packet Length Std`, `Bwd Packet Length Std`, `Fwd Packets/s`, `Idle Mean`, `Fwd Packet Length Std`, `Bwd Packet Length Mean`

### Imbalance Strategy: Class Weights (chosen over SMOTE and undersampling)
- Heartbleed weight: 15,254.9x
- Benign weight: 0.08x
- Reason: real data, zero memory overhead, mathematically equivalent to balanced sampling

### Saved pkl Files (all in /kaggle/working/)
| File | Shape | Description |
|------|-------|-------------|
| X_train_final.pkl | (2,059,411, 47) | Training features scaled |
| X_test_final.pkl | (514,853, 47) | Test features scaled |
| y_train_final.pkl | (2,059,411,) | Multiclass labels (0-14) |
| y_test_final.pkl | (514,853,) | Multiclass test labels |
| y_train_binary_final.pkl | (2,059,411,) | Binary labels (0=benign, 1=attack) |
| y_test_binary_final.pkl | (514,853,) | Binary test labels |
| preprocessing_pipeline.pkl | — | Scaler + imputer chained |
| class_weights.pkl | — | Per-class penalty weights |
| label_encoder.pkl | — | Integer ↔ label name mapping |
| feature_names.pkl | — | List of 47 final feature names |
| label_classes.pkl | — | Array of 15 class names |
| removed_zero_variance.pkl | — | 8 removed constant features |
| removed_correlated_features.pkl | — | 23 removed correlated features |

---

## Week 2 — IN PROGRESS 🔄
**Goal: Train Random Forest + XGBoost, evaluate properly, export best model**

### My Tasks
- Train Random Forest (binary + multiclass)
- Train XGBoost (binary + multiclass)
- Hyperparameter tuning with RandomizedSearchCV
- Evaluate with classification_report per class
- Target: precision ≥97%, recall ≥95% on attacks
- Export best model as pkl file for teammate

### How to Run Code on Kaggle
All prompts are run using:
```python
code = open('/kaggle/input/datasets/chaimaabid233/DATASET_NAME/script.py', encoding='utf-8-sig').read()
import re
code = re.sub(
    r'try:.*?except ImportError:.*?print\("Environment: Kaggle.*?SAVE_DIR\)',
    'SAVE_DIR = \'/kaggle/working/\'\nENV = \'kaggle\'\nprint("Environment: Kaggle saving to", SAVE_DIR)',
    code, flags=re.DOTALL
)
exec(code, globals())
```

---

## Teammate's Backend — Week 1 COMPLETED ✅
Repo: (ask teammate for GitHub link)

### Stack
- FastAPI (Python) — REST API
- PostgreSQL — permanent alert storage
- Redis — real-time push
- Docker Compose — one command startup: `docker compose up --build`
- Tested locally on Ubuntu 26.04 in VirtualBox

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Health check → `{"status": "NIDS API is running"}` |
| /api/predict | POST | Send 47 features → get prediction |
| /api/alerts | GET | Get past alerts from PostgreSQL |
| /ws/alerts | WS | Real-time alert push via WebSocket |

### Current predict.py (placeholder — Week 2 integration)
```python
@router.post("/predict")
def predict(features: dict):
    # Placeholder — real model integrated in Week 2
    return {
        "prediction": "BENIGN",
        "confidence": 0.99,
        "message": "Model not yet integrated"
    }
```

### What teammate needs from me (end of Week 2)
- `model_best.pkl` — trained and evaluated model
- `preprocessing_pipeline.pkl` — already done in Week 1
- `feature_names.pkl` — already done in Week 1
- `label_classes.pkl` — already done in Week 1

### Docker services
- `backend` → port 8000 (FastAPI)
- `db` → port 5432 (PostgreSQL, db=nidsdb, user=nids, pass=nids123)
- `redis` → port 6379

---

## Project Plan — 4 Weeks Total (1 Month)

### Week 1 ✅ COMPLETE
**Me (Days 1-3):** Load CICIDS2017, merge, EDA, class distribution, feature correlation analysis
**Me (Days 4-5):** Class imbalance — SMOTE vs undersampling vs class weights. Chose class weights.
**Friend (Days 1-3):** GitHub repo, FastAPI skeleton, PostgreSQL, Redis, Docker Compose
**Friend (Days 4-5):** FastAPI predict endpoint with dummy response
**Weekend 1:** Teach each other — I explain CICIDS2017 + imbalance, he explains FastAPI + Docker

### Week 2 🔄 IN PROGRESS
**Me (Days 1-2):** Random Forest — bootstrap sampling, feature importance, bias-variance tradeoff
**Me (Days 3-4):** XGBoost — gradient boosting, sequential trees, compare vs RF
**Me (Day 5):** Hyperparameter tuning (RandomizedSearchCV), export best model with joblib
**Friend (Weekend 2):** Integrate my saved model into FastAPI — first real end-to-end demo

### Week 3
**Me (Days 1-2):** Neural network PyTorch/Keras — Input(47) after preprocessing → compare RF vs XGBoost vs NN
  Note: Plan mentions 78 features but our preprocessing reduces to 47 — neural net uses 47 scaled features
**Me (Days 3-5):** Feature extraction spec — map 47 features to Scapy measurements
**Friend (Days 3-5):** Alert system, PostgreSQL storage, WebSocket push, Scapy capture module

### Week 4
**Me (Days 1-2):** Full evaluation report (model comparison, per-class metrics, benchmarks)
**Me (Days 3-4):** Docker ML container, end-to-end testing, integration debugging
**Me (Day 5):** Code cleanup, docstrings, README, final GitHub push
**Friend (Days 1-2):** React dashboard (real-time alerts, traffic charts, model metrics)
**Friend (Days 3-4):** Full Docker Compose, one-command deployment
**Both (Day 5):** End-to-end test: packet → features → prediction → alert → dashboard

---

## Evaluation Rules — CRITICAL
**NEVER use accuracy as primary metric — it is misleading on imbalanced data.**
A dummy classifier scores 83.5% accuracy by always predicting BENIGN while detecting ZERO attacks.

**Always use:**
- `classification_report` with per-class precision, recall, F1
- Macro average F1 (treats all classes equally)
- Recall per attack class (missing an attack = dangerous)
- Confusion matrix

**Targets from CDC:**
- Precision ≥ 97%
- Recall ≥ 95% for critical attacks
- F1 ≥ 0.96 on test set
- False positive rate < 3% on BENIGN

---

## CDC Requirements (Cahier des Charges)
- F-06: Binary classification (BENIGN vs ATTACK) — HIGH priority
- F-07: Multiclass (15 attack types) — HIGH priority
- F-08: Confidence score per prediction — MEDIUM priority
- F-09: Ensemble model RF + XGBoost — HIGH priority
- F-10: Scheduled retraining without service interruption — LOW priority

**Security requirements:**
- JWT authentication with expiration
- TLS 1.3 for all API communications
- bcrypt passwords (cost factor ≥12)
- RBAC: Admin / Analyst / Reader roles
- Rate limiting on API endpoints
- SQLAlchemy ORM (prevents SQL injection)

---

## Code Style Preferences
- All scripts follow the same structure as prompt1-4 code
- STEP-by-STEP print statements with `="*60` separators
- Each step has timing with `time.time()`
- FINDING: print at end of each step explaining what was found
- Save all artifacts with joblib to `/kaggle/working/`
- Verify every save immediately after writing
- Environment detection: Kaggle vs Colab auto-detect

---

## Key Numbers to Remember
- Total rows: 2,059,411
- Train: 1,647,529 (80%) | Test: 514,853 (20%)
- Final features: 47
- Classes: 15
- Imbalance ratio: 190,967x
- Rarest class: Heartbleed (9 rows)
- Heartbleed class weight: 15,254.9x
- Literature agreement (top features): 38% overlap with Sharafaldin et al. 2017

---

## Literature Reference
Sharafaldin et al. (2017) top features: `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `Fwd Packet Length Max`, `Bwd Packet Length Max`, `Flow IAT Mean`, `SYN Flag Count`

Our top features confirmed YES match: `Bwd Packet Length Max`, `Fwd Packet Length Max`, `Total Length of Fwd Packets`

---

## File Locations
| Location | What's There |
|----------|-------------|
| `/kaggle/working/` | All pkl files, model outputs, plots |
| `/kaggle/input/datasets/chaimaabid233/cicids2017/` | Raw CICIDS2017 CSV files |
| `/kaggle/input/datasets/chaimaabid233/nids-week1-prompt1/` | prompt1_code.py |
| `/kaggle/input/datasets/chaimaabid233/nids-week1-prompt21/` | prompt2_code (1).py (fixed version) |
| `/kaggle/input/datasets/chaimaabid233/nids-week1-prompt3/` | prompt3_code.py |
| GitHub: nids-ml repo | prompt1-4 code, notebook, this CLAUDE.md |

---

## NIDS Learning Log
The project plan requires daily updates to `NIDS-learning-log.md` (second brain file).
This file should be updated every evening with:
- What I built today
- What I understood deeply
- What I used without fully understanding
- Questions still open

---

## Agent skills

### Issue tracker

Issues tracked as GitHub Issues (github.com/chaimaabid-cloud/NIDS-ML), via gh CLI. PRs not used as a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one CONTEXT.md + docs/adr/ at repo root. See `docs/agents/domain.md`.

---

## Update Log
- 2026-06-13: Week 1 complete, all pkl files saved, teammate backend running
- 2026-06-22: CLAUDE.md created, GitHub repo set up, Ubuntu VM configured
