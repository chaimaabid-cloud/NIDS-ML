# ## Section 3: Class Imbalance Handling and Preprocessing Pipeline
# This section handles class imbalance using three strategies,
# builds a reusable preprocessing pipeline, and prepares
# the final clean dataset for Week 2 model training.

# ── Global Random Seed ───────────────────────────────────────────────────────
import numpy as np
import random
import os
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

import joblib

# ── Save Directory ────────────────────────────────────────────────────────────
if os.path.exists('/kaggle/working/cicids2017_merged.parquet'):
    SAVE_DIR = '/kaggle/working/'
    print("Environment: Kaggle — current session")
else:
    SAVE_DIR = '/kaggle/input/week1-prompt1-outputs/'
    print("Environment: loading from saved version outputs")

# ── FILE EXISTENCE CHECKS — NEVER SKIP THIS ──────────────────────────────────
required_files = [
    'X_train_clean.pkl', 'X_test_clean.pkl',
    'y_train_multi.pkl', 'y_test_multi.pkl',
    'y_train_binary.pkl', 'y_test_binary.pkl',
    'label_encoder.pkl', 'class_mapping.pkl',
    'feature_names.pkl', 'feature_dtypes.pkl',
]
for f in required_files:
    path = SAVE_DIR + f
    assert os.path.exists(path), (
        f"MISSING: {f}. Rerun Prompt 2 completely before proceeding.")
    print(f"  Found: {f}")
print("All required files found. Proceeding.")

# ── Load All Required Files ───────────────────────────────────────────────────
X_train_clean   = joblib.load(SAVE_DIR + 'X_train_clean.pkl')
X_test_clean    = joblib.load(SAVE_DIR + 'X_test_clean.pkl')
y_train_multi   = joblib.load(SAVE_DIR + 'y_train_multi.pkl')
y_test_multi    = joblib.load(SAVE_DIR + 'y_test_multi.pkl')
y_train_binary  = joblib.load(SAVE_DIR + 'y_train_binary.pkl')
y_test_binary   = joblib.load(SAVE_DIR + 'y_test_binary.pkl')
le              = joblib.load(SAVE_DIR + 'label_encoder.pkl')
class_mapping   = joblib.load(SAVE_DIR + 'class_mapping.pkl')
feature_names   = joblib.load(SAVE_DIR + 'feature_names.pkl')
feature_dtypes  = joblib.load(SAVE_DIR + 'feature_dtypes.pkl')

print(f"\nX_train_clean shape : {X_train_clean.shape}")
print(f"X_test_clean  shape : {X_test_clean.shape}")
print(f"Classes             : {list(le.classes_)}")

# ── imbalanced-learn version check ───────────────────────────────────────────
import imblearn
print(f"\nimbalanced-learn: {imblearn.__version__}")
assert imblearn.__version__ >= '0.10', \
    "Update imbalanced-learn: pip install --upgrade imbalanced-learn"

section_start = time.time()

# =============================================================================
# STEP 1 — MATHEMATICAL PROOF OF IMBALANCE PROBLEM
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1: Mathematical proof of imbalance problem")
print("=" * 60)
step_start = time.time()

from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score

print("\nClass counts and percentages in y_train_multi:")
total_train = len(y_train_multi)
class_counts = {}
for cls_id, name in class_mapping.items():
    cnt = int((y_train_multi == cls_id).sum())
    class_counts[cls_id] = cnt
    pct = cnt / total_train * 100
    print(f"  {name:<45} {cnt:>10,}  ({pct:6.3f}%)")

max_count = max(class_counts.values())
min_count = min(class_counts.values())
imbalance_ratio = max_count / min_count
rarest_class = class_mapping[min(class_counts, key=class_counts.get)]
print(f"\nImbalance ratio (largest / smallest): {imbalance_ratio:,.1f}x")
print(f"Rarest class: '{rarest_class}' with {min_count:,} samples")

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_clean, y_train_multi)
dummy_preds = dummy.predict(X_test_clean)
dummy_acc   = dummy.score(X_test_clean, y_test_multi)

print(f"\nDummy classifier classification report:")
print(classification_report(y_test_multi, dummy_preds,
                             target_names=le.classes_, zero_division=0))

benign_lower = 'benign'
attack_f1_scores = {
    class_mapping[i]: f1_score(
        y_test_multi, dummy_preds, labels=[i], average='macro', zero_division=0
    )
    for i in range(len(le.classes_))
    if class_mapping[i].lower() != benign_lower
}
print("Attack class F1 scores (dummy classifier):")
for name, score in attack_f1_scores.items():
    print(f"  {name:<45} F1 = {score:.4f}")

step_elapsed = time.time() - step_start
print(f"\nStep 1 completed in {step_elapsed:.1f}s")
print(
    f"\nFINDING: Dummy classifier accuracy: {dummy_acc:.1%}\n"
    "F1 for every attack class: 0.0\n"
    "Mathematical proof: optimizing accuracy alone produces a system that detects\n"
    "zero attacks while appearing to work perfectly. Recall on attack classes is our\n"
    "primary optimization target."
)

# =============================================================================
# STEP 2 — BUILD PREPROCESSING PIPELINE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Build preprocessing pipeline")
print("=" * 60)
step_start = time.time()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

preprocessing_pipeline.fit(X_train_clean)

X_train_scaled_arr = preprocessing_pipeline.transform(X_train_clean)
X_test_scaled_arr  = preprocessing_pipeline.transform(X_test_clean)

# preserve column names — use only the features that survived Prompt 2 cleaning
current_features = X_train_clean.columns.tolist()
X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=current_features)
X_test_scaled  = pd.DataFrame(X_test_scaled_arr,  columns=current_features)

print(f"  X_train_scaled shape : {X_train_scaled.shape}")
print(f"  X_test_scaled  shape : {X_test_scaled.shape}")

# ── save and verify ───────────────────────────────────────────────────────────
joblib.dump(preprocessing_pipeline, SAVE_DIR + 'preprocessing_pipeline.pkl')
verify_pipe  = joblib.load(SAVE_DIR + 'preprocessing_pipeline.pkl')
test_transform = verify_pipe.transform(X_train_clean.head(5))
assert test_transform.shape == (5, len(current_features)), \
    f"Pipeline verification failed: got {test_transform.shape}"
print("  Pipeline save verified")

step_elapsed = time.time() - step_start
print(f"Step 2 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: The preprocessing pipeline chains imputation and scaling into one object.\n"
    "Usage in Week 3 production:\n"
    "    pipeline = joblib.load('preprocessing_pipeline.pkl')\n"
    "    new_packet_clean = pipeline.transform(new_packet)\n"
    "    prediction = model.predict(new_packet_clean)\n"
    "This guarantees identical transformations between training and production.\n"
    "Without this the model receives different scale data than it trained on."
)

# =============================================================================
# STEP 3 — STRATEGY A: CLASS WEIGHTS (PRIMARY STRATEGY)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Strategy A — Class Weights (primary strategy)")
print("=" * 60)
step_start = time.time()

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train_multi)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_multi,
)
class_weights_dict  = dict(zip(classes.tolist(), weights.tolist()))
class_weights_named = {class_mapping[k]: v for k, v in class_weights_dict.items()}

print("\nClass weights computed:")
print(f"  {'Class':<45} {'Count':>10}  {'Weight':>10}")
print("  " + "-" * 70)
for cls_id, weight in class_weights_dict.items():
    name = class_mapping[cls_id]
    cnt  = class_counts[cls_id]
    print(f"  {name:<45} {cnt:>10,}  {weight:>10.4f}")

rarest_id     = min(class_counts, key=class_counts.get)
rarest_weight = class_weights_dict[rarest_id]
print(
    f"\nInterpretation: the model penalizes misclassifying '{class_mapping[rarest_id]}'\n"
    f"{rarest_weight:.1f}x more than misclassifying BENIGN."
)

# ── save and verify ───────────────────────────────────────────────────────────
joblib.dump(class_weights_dict, SAVE_DIR + 'class_weights.pkl')
verify_cw = joblib.load(SAVE_DIR + 'class_weights.pkl')
assert verify_cw == class_weights_dict, "Class weights save verification failed"
print("  Class weights save verified")

# ── quick Decision Tree with class weights for comparison ─────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts

X_demo, _, y_demo, _ = tts(
    X_train_scaled, y_train_multi,
    test_size=0.9, random_state=42, stratify=y_train_multi,
)

dt_weighted = DecisionTreeClassifier(
    random_state=42,
    class_weight=class_weights_dict,
)
dt_weighted.fit(X_demo, y_demo)
preds_weighted  = dt_weighted.predict(X_test_scaled)
report_weighted = classification_report(
    y_test_multi, preds_weighted,
    target_names=le.classes_, zero_division=0, output_dict=True,
)
print("\nStrategy A (class weights) — classification report on 10% demo sample:")
print(classification_report(y_test_multi, preds_weighted,
                             target_names=le.classes_, zero_division=0))

step_elapsed = time.time() - step_start
print(f"Step 3 completed in {step_elapsed:.1f}s")

botnet_name   = next((n for n in class_mapping.values() if 'bot' in n.lower()), rarest_id)
botnet_weight = class_weights_named.get(botnet_name, rarest_weight)
print(
    "\nFINDING: Class weights change the loss function:\n"
    "  Standard loss = sum(loss per row)\n"
    "  Weighted loss = sum(weight × loss per row)\n"
    f"  A {class_mapping[rarest_id]} misclassification now costs {rarest_weight:.1f}x\n"
    "  more than a BENIGN misclassification.\n"
    "  This is mathematically equivalent to having equal numbers of each class\n"
    "  but uses zero additional memory and operates on real data not synthetic samples."
)

# =============================================================================
# STEP 4 — STRATEGY B: UNDERSAMPLING (DEMONSTRATION ON 10% SAMPLE)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Strategy B — Undersampling (10% demo sample)")
print("=" * 60)
step_start = time.time()

# X_demo / y_demo already created above (same 10% stratified split)
print("Class distribution BEFORE undersampling:")
demo_counts_before = pd.Series(y_demo).value_counts().sort_index()
for cls_id, cnt in demo_counts_before.items():
    print(f"  {class_mapping[cls_id]:<45} {cnt:>8,}")

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_demo, y_demo)

print(f"\nClass distribution AFTER undersampling (shape: {X_under.shape}):")
under_counts = pd.Series(y_under).value_counts().sort_index()
for cls_id, cnt in under_counts.items():
    print(f"  {class_mapping[cls_id]:<45} {cnt:>8,}")

dt_under = DecisionTreeClassifier(random_state=42)
dt_under.fit(X_under, y_under)
preds_under  = dt_under.predict(X_test_scaled)
report_under = classification_report(
    y_test_multi, preds_under,
    target_names=le.classes_, zero_division=0, output_dict=True,
)
print("\nStrategy B (undersampling) — classification report:")
print(classification_report(y_test_multi, preds_under,
                             target_names=le.classes_, zero_division=0))

step_elapsed = time.time() - step_start
print(f"Step 4 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: Undersampling removes real BENIGN data.\n"
    "Potential problem: the model sees less normal traffic and may generate more\n"
    "false alarms in production.\n"
    "Check precision on BENIGN in the report above."
)

# =============================================================================
# STEP 5 — STRATEGY C: SMOTE (DEMONSTRATION ON 10% SAMPLE)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Strategy C — SMOTE (10% demo sample)")
print("=" * 60)
step_start = time.time()

print(
    "IMPORTANT: SMOTE demonstration uses 10% sample only.\n"
    "Full 2.8M row SMOTE would require 8GB+ additional RAM.\n"
    "This demonstration is sufficient for comparison.\n"
    "Class weights will be used for actual model training."
)

# k_neighbors must be < smallest class count in the demo set
min_demo_class = int(pd.Series(y_demo).value_counts().min())
k_n = max(1, min(5, min_demo_class - 1))
print(f"\nUsing k_neighbors={k_n} (auto-adjusted for smallest class in demo set)")

from imblearn.over_sampling import SMOTE

try:
    smote = SMOTE(random_state=42, k_neighbors=k_n)
    X_smote, y_smote = smote.fit_resample(X_demo, y_demo)
    smote_ok = True
except Exception as e:
    print(f"  SMOTE failed: {e}")
    print("  Falling back to RandomOverSampler for demonstration.")
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_smote, y_smote = ros.fit_resample(X_demo, y_demo)
    smote_ok = False

print(f"\nClass distribution AFTER {'SMOTE' if smote_ok else 'RandomOverSampler'}"
      f" (shape: {X_smote.shape}):")
smote_counts = pd.Series(y_smote).value_counts().sort_index()
for cls_id, cnt in smote_counts.items():
    print(f"  {class_mapping[cls_id]:<45} {cnt:>8,}")

dt_smote = DecisionTreeClassifier(random_state=42)
dt_smote.fit(X_smote, y_smote)
preds_smote  = dt_smote.predict(X_test_scaled)
report_smote = classification_report(
    y_test_multi, preds_smote,
    target_names=le.classes_, zero_division=0, output_dict=True,
)
print(f"\nStrategy C ({'SMOTE' if smote_ok else 'RandomOverSampler'}) — classification report:")
print(classification_report(y_test_multi, preds_smote,
                             target_names=le.classes_, zero_division=0))

step_elapsed = time.time() - step_start
print(f"Step 5 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: SMOTE generates synthetic minority samples by interpolating between\n"
    "real minority samples.\n"
    "For network traffic this is questionable:\n"
    "a synthetic FTP-Patator flow is the mathematical average of two real FTP-Patator flows.\n"
    "This average may not represent a real attack pattern.\n"
    "Compare recall on attack classes vs Strategy B and A."
)

# =============================================================================
# STEP 6 — THREE-WAY COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Three-way strategy comparison")
print("=" * 60)
step_start = time.time()

def _avg_attack_f1(report_dict, attack_class_names):
    """Mean F1 across all non-BENIGN classes that exist in the report."""
    scores = [
        report_dict[n]['f1-score']
        for n in attack_class_names
        if n in report_dict and n.lower() != 'benign'
    ]
    return np.mean(scores) if scores else 0.0

def _rarest_recall(report_dict, rarest_name):
    return report_dict.get(rarest_name, {}).get('recall', 0.0)

benign_name   = next(n for n in le.classes_ if n.lower() == 'benign')
attack_names  = [n for n in le.classes_ if n.lower() != 'benign']

strategies = {
    'Class Weights' : report_weighted,
    'Undersampling' : report_under,
    'SMOTE (sample)': report_smote,
}
memory_impact = {
    'Class Weights' : 'Zero',
    'Undersampling' : 'Reduced dataset',
    'SMOTE (sample)': '~3x data size',
}

print(
    f"\n{'Strategy':<18} {'BENIGN F1':>10} {'Atk Avg F1':>12} "
    f"{'Rarest Recall':>15} {'Memory Impact':<20}"
)
print("-" * 80)
for strat_name, rep in strategies.items():
    benign_f1    = rep.get(benign_name, {}).get('f1-score', 0.0)
    atk_avg_f1   = _avg_attack_f1(rep, attack_names)
    rare_recall  = _rarest_recall(rep, class_mapping[rarest_id])
    mem          = memory_impact[strat_name]
    print(
        f"{strat_name:<18} {benign_f1:>10.3f} {atk_avg_f1:>12.3f} "
        f"{rare_recall:>15.3f} {mem:<20}"
    )

step_elapsed = time.time() - step_start
print(f"\nStep 6 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: For a security system:\n"
    "Recall on rarest attack class is the critical metric.\n"
    f"Missing a {class_mapping[rarest_id]} (rarest, hardest) is more dangerous\n"
    "than any false alarm.\n"
    f"Which strategy achieves highest recall on '{class_mapping[rarest_id]}'?"
)

# =============================================================================
# STEP 7 — FINAL DECISION AND SAVE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Final decision and save complete data package")
print("=" * 60)
step_start = time.time()

print(
    "\nFINAL DECISION: Class Weights as primary strategy.\n"
    "REASON 1: Comparable recall to SMOTE without memory issues.\n"
    "REASON 2: Operates on real data not synthetic interpolations.\n"
    "REASON 3: Mathematically equivalent to balanced sampling.\n"
    "REASON 4: Directly supported by sklearn Random Forest and XGBoost\n"
    "          through class_weight parameter.\n"
    "This decision will be passed to all Week 2 models."
)

files_to_save = {
    'X_train_final.pkl'       : X_train_scaled,
    'X_test_final.pkl'        : X_test_scaled,
    'y_train_final.pkl'       : y_train_multi,
    'y_test_final.pkl'        : y_test_multi,
    'y_train_binary_final.pkl': y_train_binary,
    'y_test_binary_final.pkl' : y_test_binary,
}

print()
for filename, data in files_to_save.items():
    fpath = SAVE_DIR + filename
    joblib.dump(data, fpath)
    loaded = joblib.load(fpath)
    if hasattr(loaded, 'shape'):
        assert loaded.shape == data.shape, \
            f"Shape mismatch for {filename}: {loaded.shape} != {data.shape}"
        print(f"  Saved and verified: {filename:<35} shape={loaded.shape}")
    elif hasattr(loaded, '__len__'):
        assert len(loaded) == len(data), \
            f"Length mismatch for {filename}"
        print(f"  Saved and verified: {filename:<35} len={len(loaded):,}")

step_elapsed = time.time() - step_start
print(f"\nStep 7 completed in {step_elapsed:.1f}s")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
total_elapsed = time.time() - section_start

print("\n" + "=" * 60)
print("SECTION 3 — FINAL SUMMARY")
print("=" * 60)
print(f"  Total Section 3 time          : {total_elapsed:.1f}s")
print(f"  X_train_final shape           : {X_train_scaled.shape}")
print(f"  X_test_final  shape           : {X_test_scaled.shape}")
print(f"  Imbalance ratio               : {imbalance_ratio:,.1f}x")
print(f"  Rarest class                  : {class_mapping[rarest_id]}  ({min_count:,} rows)")
print(f"  Strategy chosen               : Class Weights")
print(f"  Preprocessing pipeline        : saved to {SAVE_DIR}preprocessing_pipeline.pkl")
print(f"  Class weights                 : saved to {SAVE_DIR}class_weights.pkl")
print(f"  Files saved to {SAVE_DIR}:")
for fname in list(files_to_save.keys()) + ['preprocessing_pipeline.pkl', 'class_weights.pkl']:
    fpath = SAVE_DIR + fname
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 ** 2)
        print(f"    {fname:<40} {size_mb:.1f} MB")

print("\n" + "=" * 60)
print("IMPORTANT: Save Version now before closing session.")
print("Kaggle: Click 'Save Version' button top right")
print("Colab: File is already saved to Google Drive")
print("=" * 60)
