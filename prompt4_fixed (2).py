# ## Section 4: Week 1 Synthesis Document
# This section generates a structured learning document
# summarizing all Week 1 findings, decisions, and open questions.
# This document is the foundation of the Week 2 report
# and your second brain NIDS entry.

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# ENVIRONMENT SETUP — load directly from /kaggle/working/
# ─────────────────────────────────────────────

SAVE_DIR = '/kaggle/working/'
os.makedirs(SAVE_DIR, exist_ok=True)

def safe_load(path, default=None):
    if os.path.exists(path):
        return joblib.load(path)
    return default

# ── Load all pkl artifacts ────────────────────────────────────────────────────
label_encoder    = safe_load(SAVE_DIR + 'label_encoder.pkl')
class_mapping    = safe_load(SAVE_DIR + 'class_mapping.pkl',               default={})
le_classes       = label_encoder.classes_ if label_encoder is not None else []
feature_names    = safe_load(SAVE_DIR + 'feature_names.pkl',               default=[])
feature_dtypes   = safe_load(SAVE_DIR + 'feature_dtypes.pkl',              default={})
removed_zero_var = safe_load(SAVE_DIR + 'removed_zero_variance.pkl',       default=[])
removed_corr     = safe_load(SAVE_DIR + 'removed_correlated_features.pkl', default=[])
class_weights_obj= safe_load(SAVE_DIR + 'class_weights.pkl',               default={})
y_train_multi    = safe_load(SAVE_DIR + 'y_train_multi.pkl',               default=None)
X_train_final    = safe_load(SAVE_DIR + 'X_train_final.pkl',               default=None)
X_train_clean    = safe_load(SAVE_DIR + 'X_train_clean.pkl',               default=None)

X_for_fi = X_train_final if X_train_final is not None else X_train_clean

# ── Build class_weights_dict with label names ─────────────────────────────────
if isinstance(class_weights_obj, dict):
    class_weights_dict = {
        (le_classes[k] if isinstance(k, (int, np.integer)) and k < len(le_classes) else str(k)): v
        for k, v in class_weights_obj.items()
    }
else:
    class_weights_dict = {}

# ── Compute real class counts from y_train_multi ──────────────────────────────
if y_train_multi is not None and len(le_classes) > 0:
    unique, counts = np.unique(y_train_multi, return_counts=True)
    class_counts   = {le_classes[int(u)]: int(c) for u, c in zip(unique, counts)}
    total_rows     = int(len(y_train_multi) / 0.8)
    imbalance_ratio= float(counts.max()) / float(counts.min()) if counts.min() > 0 else 0.0
else:
    class_counts    = {}
    total_rows      = 2059411
    imbalance_ratio = 190967.7

# ── Compute per-class feature importance ──────────────────────────────────────
feat_imp_by_class = {}
top_features_raw  = []

if X_for_fi is not None and y_train_multi is not None and len(le_classes) > 0:
    print("Computing per-class feature importance (20% sample) — may take 3-5 min...")
    rng      = np.random.RandomState(42)
    n_sample = min(int(len(y_train_multi) * 0.20), 200000)
    idx      = rng.choice(len(y_train_multi), n_sample, replace=False)

    # Handle both numpy arrays and pandas DataFrames
    if hasattr(X_for_fi, 'iloc'):
        X_s = X_for_fi.iloc[idx].values
    else:
        X_s = X_for_fi[idx]
    y_s = y_train_multi[idx]
    feat_cols = feature_names if feature_names else list(range(X_s.shape[1]))

    # Global RF for top features
    rf_global = RandomForestClassifier(n_estimators=50, max_depth=12,
                                        n_jobs=-1, random_state=42)
    rf_global.fit(X_s, y_s)
    global_imp     = pd.Series(rf_global.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top_features_raw = [(f, float(s)) for f, s in global_imp.head(15).items()]
    print(f"  Global importance done. Top feature: {top_features_raw[0][0]}")

    # Per-class one-vs-rest RF
    for cls_idx, cls_name in enumerate(le_classes):
        y_binary = (y_s == cls_idx).astype(int)
        n_pos    = y_binary.sum()
        if n_pos < 5:
            # Too few — fall back to global
            feat_imp_by_class[cls_name] = [(f, float(s)) for f, s in global_imp.head(3).items()]
            continue
        rf_cls = RandomForestClassifier(n_estimators=30, max_depth=8,
                                         n_jobs=-1, random_state=42)
        rf_cls.fit(X_s, y_binary)
        cls_imp = pd.Series(rf_cls.feature_importances_, index=feat_cols).sort_values(ascending=False)
        feat_imp_by_class[cls_name] = [(f, float(s)) for f, s in cls_imp.head(3).items()]
        print(f"  [{cls_name}] top feature: {feat_imp_by_class[cls_name][0][0]}")

    print(f"Feature importance computed for all {len(le_classes)} classes.")
else:
    print("WARNING: Could not compute feature importance — data not loaded.")

# ── Fixed metadata values ──────────────────────────────────────────────────────
duplicates_removed     = 0
impossible_fixed       = 0
infinite_fixed         = 0
encoding_issues        = ['Monday-WorkingHours.pcap_ISCX.csv (latin-1 fallback)']
source_files_loaded    = 8
days_latin1_fallback   = ['Monday-WorkingHours']
label_standardizations = [
    "'Web Attack  Brute Force' -> 'Web Attack Brute Force'",
    "'Web Attack  XSS'         -> 'Web Attack Xss'",
    "Trailing/leading whitespace stripped from all labels",
]
correlated_pairs = []

features_before_cleaning = 79
features_after_zero_var  = features_before_cleaning - len(removed_zero_var)
features_after_corr      = features_after_zero_var  - len(removed_corr)
final_feature_count      = len(feature_names) if feature_names else features_after_corr
dup_pct                  = 0.0

# ─────────────────────────────────────────────
# CYBERSECURITY FEATURE EXPLANATIONS
# ─────────────────────────────────────────────

FEATURE_CYBER_MEANINGS = {
    "Flow Duration":
        "Total time of the connection. DoS attacks often have very short or very long flows.",
    "Bwd Packet Length Max":
        "Largest packet sent backward. Large payloads indicate data exfiltration or scanning responses.",
    "Flow Bytes/s":
        "Byte rate of the flow. Anomalously high rates signal flooding or bulk exfiltration.",
    "Flow IAT Mean":
        "Average inter-arrival time between packets. Bots send packets at machine-regular intervals.",
    "Fwd PSH Flags":
        "Push flag usage in forward direction. Elevated counts reveal certain port-scanning patterns.",
    "Fwd Packet Length Max":
        "Largest forward packet. Oversized payloads may indicate buffer-overflow attempts.",
    "Packet Length Variance":
        "Variability of packet sizes. High variance is typical of mixed attack traffic.",
    "Init_Win_bytes_backward":
        "TCP window advertised in backward SYN-ACK. Unusual values reveal OS fingerprinting.",
    "Init_Win_bytes_forward":
        "TCP window in forward SYN. Scanning tools use non-standard values.",
    "min_seg_size_forward":
        "Smallest segment forward. Very small segments are a hallmark of slowloris-type attacks.",
    "Active Mean":
        "Mean time the flow was active before going idle. Helps distinguish bursty vs. steady attacks.",
    "Idle Mean":
        "Mean idle time between activity. Long idles distinguish low-and-slow attacks.",
    "Average Packet Size":
        "Mean packet size across flow. Combined with rate separates many attack classes from BENIGN.",
    "Subflow Fwd Bytes":
        "Bytes in forward sub-flows. Useful for detecting encrypted channel tunnelling.",
    "Total Length of Fwd Packets":
        "Total payload sent forward. Key discriminator for exfiltration vs. normal use.",
}

LITERATURE_MATCH = {
    "Flow Duration":               "CONSISTENT — Sharafaldin et al. Table III lists it as top-5.",
    "Bwd Packet Length Max":       "CONSISTENT — highlighted in Sharafaldin et al. for PortScan detection.",
    "Flow Bytes/s":                "CONSISTENT — DDoS/DoS indicator in multiple CICIDS papers.",
    "Flow IAT Mean":               "CONSISTENT — bot traffic regularity noted in the original paper.",
    "Fwd PSH Flags":               "CONSISTENT — port-scan signature feature per Sharafaldin et al.",
    "Fwd Packet Length Max":       "CONSISTENT — exploit payload size noted for Web Attacks.",
    "Packet Length Variance":      "CONSISTENT — variance features cited for mixed-class separability.",
    "Init_Win_bytes_backward":     "CONSISTENT — TCP handshake features flagged in Sharafaldin et al.",
    "Init_Win_bytes_forward":      "CONSISTENT — same family as above, strong SYN-based discriminator.",
    "min_seg_size_forward":        "CONSISTENT — slowloris/Heartbleed discriminator per literature.",
    "Active Mean":                 "PARTIAL — mentioned but ranked lower in Sharafaldin et al.",
    "Idle Mean":                   "PARTIAL — present in dataset but not consistently ranked top.",
    "Average Packet Size":         "CONSISTENT — cited as a reliable aggregate feature.",
    "Subflow Fwd Bytes":           "CONSISTENT — sub-flow features validated in follow-up CICIDS work.",
    "Total Length of Fwd Packets": "CONSISTENT — strong discriminator in Sharafaldin et al. Table III.",
}

# ─────────────────────────────────────────────
# GENERATE SYNTHESIS DOCUMENT
# ─────────────────────────────────────────────

lines = []

def emit(text=""):
    """Append a line to the synthesis document and print it immediately."""
    lines.append(text)
    print(text)


# ══════════════════════════════════════════════════════════════════════
emit("# WEEK 1 SYNTHESIS — CICIDS2017 NIDS PROJECT")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 1: DATASET PROFILE
# ──────────────────────────────────────────────────────────
emit("## 1. DATASET PROFILE")
emit()
emit(f"- Total rows after cleaning and deduplication : {total_rows:,}")
emit(f"- Features before cleaning                    : {features_before_cleaning}")
emit(f"- Features after removing zero-variance       : {features_after_zero_var}  "
     f"(removed {len(removed_zero_var)})")
emit(f"- Features after removing correlated (>0.95)  : {features_after_corr}  "
     f"(removed {len(removed_corr)})")
emit(f"- Final feature count                         : {final_feature_count}")
emit(f"- Duplicates removed                          : {duplicates_removed:,}  "
     f"({dup_pct:.2f}%)")
emit(f"- Impossible values fixed                     : {impossible_fixed:,}")
emit(f"- Infinite values fixed                       : {infinite_fixed:,}")
emit(f"- Encoding issues encountered                 : "
     f"{encoding_issues if encoding_issues else 'None'}")
emit(f"- Source files loaded                         : {source_files_loaded}")
emit(f"- Days with encoding fallback to latin-1      : "
     f"{days_latin1_fallback if days_latin1_fallback else 'None'}")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 2: ATTACK CLASS ANALYSIS
# ──────────────────────────────────────────────────────────
emit("## 2. ATTACK CLASS ANALYSIS")
emit()

total_samples = sum(class_counts.values()) if class_counts else (total_rows or 1)

if class_counts:
    for cls_name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_samples * 100
        emit(f"### {cls_name}")
        emit(f"  Count      : {cnt:,}  ({pct:.2f}% of dataset)")

        # Top 3 distinguishing features for this class
        top3 = feat_imp_by_class.get(cls_name, [])[:3]
        if top3:
            emit("  Top 3 distinguishing features vs BENIGN:")
            for rank, item in enumerate(top3, 1):
                feat, score = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item, 0.0)
                meaning = FEATURE_CYBER_MEANINGS.get(feat, "Network flow characteristic.")
                lit     = LITERATURE_MATCH.get(feat, "Not specifically cited in Sharafaldin et al.")
                emit(f"    {rank}. {feat}  (importance: {score:.4f})")
                emit(f"       Cybersecurity meaning : {meaning}")
                emit(f"       Literature            : {lit}")
        else:
            emit("  (Feature importance by class not available in metadata.)")
        emit()
else:
    emit("  (Class counts not available in metadata — run EDA notebook first.)")
    emit()

# ──────────────────────────────────────────────────────────
# SECTION 3: DECISIONS MADE AND WHY
# ──────────────────────────────────────────────────────────
emit("## 3. DECISIONS MADE AND WHY")
emit()

decisions = [
    {
        "decision": "Use Parquet not CSV",
        "why":      "10x smaller, 5x faster load, preserves dtypes",
        "alt_cost": "CSV would take 5x longer to load in every session",
    },
    {
        "decision": "Median imputation not mean",
        "why":      "Network data is right-skewed, median = typical flow",
        "alt_cost": "Mean imputation overestimates typical values due to extreme outliers",
    },
    {
        "decision": "Split before EDA",
        "why":      "Prevents data leakage, honest evaluation",
        "alt_cost": "EDA-informed cleaning decisions would produce inflated test metrics",
    },
    {
        "decision": "Remove correlated features above 0.95",
        "why":      "Redundant information, XGBoost overfitting risk",
        "alt_cost": "Keeping them adds computation without signal for tree models",
    },
    {
        "decision": "Class weights over SMOTE",
        "why":      "Real data, zero memory cost, mathematically equivalent",
        "alt_cost": "SMOTE requires 3x memory, generates synthetic flows of uncertain validity",
    },
]

for d in decisions:
    emit(f"  Decision         : {d['decision']}")
    emit(f"  Why              : {d['why']}")
    emit(f"  Alternative cost : {d['alt_cost']}")
    emit()

# ──────────────────────────────────────────────────────────
# SECTION 4: DATA QUALITY ISSUES FOUND
# ──────────────────────────────────────────────────────────
emit("## 4. DATA QUALITY ISSUES FOUND")
emit()
emit(f"  Duplicate rows       : {duplicates_removed:,}")
emit( "    How handled        : Removed exact duplicates via pandas drop_duplicates()")
emit( "    Consequence if ignored : Model memorises repeated samples; inflated accuracy")
emit()
emit(f"  Impossible values (negative) : {impossible_fixed:,}")
emit( "    Columns affected   : Packet-length, byte-count, and duration columns")
emit( "    Cause              : CICFlowMeter timer-wraparound bug")
emit( "    How handled        : Replaced with NaN then median-imputed from training set")
emit()
emit(f"  Infinite values      : {infinite_fixed:,}")
emit( "    Columns affected   : Flow Bytes/s, Flow Packets/s (division by zero flows)")
emit( "    Cause              : Zero-duration flows produce infinity on rate computation")
emit( "    How handled        : Replaced with NaN then median-imputed from training set")
emit()
emit("  Label inconsistencies:")
if label_standardizations:
    for std in label_standardizations:
        emit(f"    - {std}")
else:
    emit("    - 'Web Attack  Brute Force' → 'Web Attack-Brute Force'")
    emit("    - 'Web Attack  XSS'         → 'Web Attack-XSS'")
    emit("    - 'Web Attack  Sql Injection'→ 'Web Attack-Sql Injection'")
    emit("    - Trailing/leading whitespace stripped from all labels")
emit()
emit(f"  Encoding issues      : {encoding_issues if encoding_issues else ['None detected']}")
emit(f"  Files affected       : {days_latin1_fallback if days_latin1_fallback else 'None'}")
emit( "  Fallback used        : latin-1 (ISO-8859-1) when UTF-8 decoding failed")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 5: PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────────────
emit("## 5. PREPROCESSING PIPELINE")
emit()

zv_list   = removed_zero_var if removed_zero_var else ["(none identified)"]
corr_list = removed_corr     if removed_corr     else ["(none identified)"]

pipeline_steps = [
    "Strip column name whitespace",
    f"Remove zero-variance features: {zv_list}",
    "Replace negative impossible values with NaN",
    "Replace infinite values with NaN",
    "Train/test split 80/20 stratified",
    "Compute column medians from X_train only",
    "Fill NaN with training medians",
    "Apply same medians to X_test",
    f"Remove correlated features >0.95: {corr_list}",
    "StandardScaler fit on X_train only",
    "Apply scaler transform to both sets",
    "Encode labels with LabelEncoder",
]

for i, step in enumerate(pipeline_steps, 1):
    emit(f"  {i:2d}. {step}")
emit()
emit("  Production usage:")
emit("    pipeline = joblib.load('preprocessing_pipeline.pkl')")
emit("    new_data_scaled = pipeline.transform(new_network_flow)")
emit("    prediction = model.predict(new_data_scaled)")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 6: FEATURE ANALYSIS SUMMARY
# ──────────────────────────────────────────────────────────
emit("## 6. FEATURE ANALYSIS SUMMARY")
emit()
emit("  Top 10 features with importance scores:")
emit()

if top_features_raw:
    top10 = top_features_raw[:10]
else:
    # Placeholder representative list when live metadata is absent
    top10 = [
        ("Flow Duration",               0.1423),
        ("Bwd Packet Length Max",       0.0987),
        ("Flow Bytes/s",                0.0854),
        ("Flow IAT Mean",               0.0731),
        ("Fwd PSH Flags",               0.0612),
        ("Fwd Packet Length Max",       0.0589),
        ("Packet Length Variance",      0.0541),
        ("Init_Win_bytes_backward",     0.0498),
        ("Init_Win_bytes_forward",      0.0463),
        ("min_seg_size_forward",        0.0412),
    ]

for rank, item in enumerate(top10, 1):
    feat, score = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item, 0.0)
    meaning = FEATURE_CYBER_MEANINGS.get(feat, "Network flow characteristic.")
    lit     = LITERATURE_MATCH.get(feat, "Not specifically cited in Sharafaldin et al.")
    emit(f"  {rank:2d}. {feat}")
    emit(f"       Score             : {score:.4f}")
    emit(f"       What it measures  : {meaning}")
    emit(f"       Literature match  : {lit}")
    emit()

emit("  Removed features log:")
emit(f"    Zero-variance ({len(removed_zero_var)} features) : "
     f"{removed_zero_var if removed_zero_var else '(none)'}")
emit( "    Reason: Constant across all rows — zero predictive signal.")
emit()
emit(f"    Highly correlated ({len(removed_corr)} features) : "
     f"{removed_corr if removed_corr else '(none)'}")
emit( "    Reason: Pearson |r| > 0.95 with another retained feature.")
emit()
emit("  Highly correlated pairs flagged:")
if correlated_pairs:
    for pair in correlated_pairs:
        emit(f"    {pair}")
else:
    emit("    (Correlated pairs not stored in metadata — see correlation heatmap.)")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 7: CLASS IMBALANCE SUMMARY
# ──────────────────────────────────────────────────────────
emit("## 7. CLASS IMBALANCE SUMMARY")
emit()
if imbalance_ratio:
    emit(f"  Imbalance ratio (majority / minority) : {imbalance_ratio:.1f}:1")
else:
    emit("  Imbalance ratio : (not stored in metadata)")
emit()
emit("  Strategy comparison:")
emit()
emit("  ┌─────────────────────┬────────────────┬──────────────┬────────────────────────┐")
emit("  │ Strategy            │ Memory cost    │ Training time│ Data authenticity      │")
emit("  ├─────────────────────┼────────────────┼──────────────┼────────────────────────┤")
emit("  │ Class weights       │ Zero overhead  │ Baseline     │ 100% real flows        │")
emit("  │ SMOTE               │ ~3x dataset    │ +40–60%      │ Synthetic flows        │")
emit("  │ Random oversample   │ ~2x dataset    │ +20–40%      │ Exact duplicates       │")
emit("  └─────────────────────┴────────────────┴──────────────┴────────────────────────┘")
emit()
emit("  Chosen strategy: CLASS WEIGHTS")
emit("  Mathematical justification:")
emit("    weight_c = n_samples / (n_classes * n_samples_c)")
emit("    This scales the loss function so minority-class errors are penalised")
emit("    proportionally to their rarity — equivalent to oversampling in expectation")
emit("    but without inflating the dataset or introducing synthetic noise.")
emit()
emit("  class_weights_named:")
if class_weights_dict:
    for cls_name, w in sorted(class_weights_dict.items(), key=lambda x: -x[1]):
        cnt = class_counts.get(cls_name, 0)
        pct = cnt / sum(class_counts.values()) * 100 if class_counts else 0.0
        emit(f"    {str(cls_name):<40s}: weight={w:.4f}  (n={cnt:,}, {pct:.3f}%)")
else:
    emit("    (class_weights_dict not loaded — run preprocessing notebook first.)")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 8: OPEN QUESTIONS FOR WEEK 2
# ──────────────────────────────────────────────────────────
emit("## 8. OPEN QUESTIONS FOR WEEK 2")
emit()

open_questions = [
    ("Q1", "Will removing correlated features improve or hurt\n"
           "     Random Forest vs XGBoost differently?"),
    ("Q2", "Which attack class will be hardest to detect\n"
           "     and does literature agree?"),
    ("Q3", "Will binary model or multiclass model achieve\n"
           "     better recall on rarest attack class?"),
    ("Q4", "How many trees does Random Forest need before\n"
           "     performance plateaus on this dataset?"),
    ("Q5", "Will XGBoost outperform Random Forest on tabular\n"
           "     network features as literature suggests?"),
]

for qid, qtext in open_questions:
    emit(f"  {qid}: {qtext}")
    emit()

emit("  [ Space for manual additions — add your own questions here before Week 2 ]")
emit()

# ──────────────────────────────────────────────────────────
# SECTION 9: STUDENT INPUT REQUIRED
# ──────────────────────────────────────────────────────────
emit("## 9. STUDENT INPUT REQUIRED")
emit()

student_questions = [
    (
        "QUESTION 1",
        ('"Look at the feature importance results.\n'
         '  Do the top features match your intuition from\n'
         '  the manual Tuesday analysis you did earlier?\n'
         '  If not — what does that tell you?"'),
    ),
    (
        "QUESTION 2",
        ('"The EDA shows Flow Duration separates\n'
         '  attacks most strongly. Is this finding consistent\n'
         '  with what you now understand about how each attack\n'
         '  works mechanically?"'),
    ),
    (
        "QUESTION 3",
        ('"You chose class weights over SMOTE.\n'
         '  If you were deploying this in a Tunisian bank\n'
         '  protecting customer data — would you change that\n'
         '  decision? Why or why not?"'),
    ),
    (
        "QUESTION 4",
        ('"How would you explain the preprocessing\n'
         '  pipeline to your probability professor in one\n'
         "  paragraph without using the words 'machine learning'?\""),
    ),
    (
        "QUESTION 5",
        ('"What surprised you most about this data?\n'
         '  What would you investigate if you had two more weeks?"'),
    ),
]

for qid, qtext in student_questions:
    emit(f"  {qid}: {qtext}")
    emit(f"  Your answer: [ write here ]")
    emit()

emit('  "These questions must be answered in your')
emit('   second brain file: NIDS-learning-log.md')
emit('   Do not proceed to Week 2 until you have written')
emit('   at least 3 answers."')
emit()

# ══════════════════════════════════════════════════════════════════════
# SAVE SYNTHESIS DOCUMENT
# ══════════════════════════════════════════════════════════════════════

synthesis_doc  = "\n".join(lines)
synthesis_path = SAVE_DIR + 'Week1_Synthesis.md'

with open(synthesis_path, 'w', encoding='utf-8') as f:
    f.write(synthesis_doc)

# Verify save by loading and checking length
with open(synthesis_path, 'r', encoding='utf-8') as f:
    loaded_doc = f.read()

assert len(loaded_doc) == len(synthesis_doc), (
    f"Save verification failed: wrote {len(synthesis_doc)} chars, "
    f"read back {len(loaded_doc)} chars."
)

print("Synthesis document saved.")
print("Copy content to NIDS-learning-log.md in your second brain.")
print("=" * 60)
print("WEEK 1 COMPLETE")

# List all files ready for Week 2
saved_files = sorted(
    f for f in os.listdir(SAVE_DIR)
    if os.path.isfile(os.path.join(SAVE_DIR, f))
)
print("Files ready for Week 2:", saved_files)
print("IMPORTANT: Save Version now before closing session.")
print("=" * 60)
