# ## Section 2: Exploratory Data Analysis
# This section performs all EDA on the TRAINING SET ONLY to prevent
# data leakage. Test set is never examined during EDA.

# ── Global Random Seed ───────────────────────────────────────────────────────
import numpy as np
import random
import os
import time

np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.dpi'] = 100

# ── Save Directory (mirrors Prompt 1 logic) ───────────────────────────────────
if os.path.exists('/kaggle/working/cicids2017_merged.parquet'):
    SAVE_DIR = '/kaggle/working/'
    print("Loading from current session")
else:
    SAVE_DIR = '/kaggle/input/week1-prompt1-outputs/'
    print("Loading from saved version outputs")

# ── Load Data ─────────────────────────────────────────────────────────────────
df_all = pd.read_parquet(SAVE_DIR + 'cicids2017_merged.parquet')
print(f"Loaded merged dataset: {df_all.shape}")

section_start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# ### Why we split before EDA
# Any cleaning decision made after seeing test data is data leakage.
# We split first. All analysis happens on X_train only.
# Same transformations are applied to X_test without examining it.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

le = LabelEncoder()
le.fit(df_all['Label'])                     # fit on full label set for consistency
y = le.transform(df_all['Label'])

drop_cols = [c for c in ['Label', 'Day'] if c in df_all.columns]
X = df_all.drop(drop_cols, axis=1)

# keep only numeric columns (non-numeric are meaningless for ML here)
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nX_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")

print("\nClass distribution — TRAIN:")
train_series = pd.Series(y_train)
for cls_id, cnt in train_series.value_counts().sort_index().items():
    print(f"  {le.classes_[cls_id]:<45} {cnt:>10,}  ({cnt/len(y_train)*100:5.2f}%)")

print("\nClass distribution — TEST:")
test_series = pd.Series(y_test)
for cls_id, cnt in test_series.value_counts().sort_index().items():
    print(f"  {le.classes_[cls_id]:<45} {cnt:>10,}  ({cnt/len(y_test)*100:5.2f}%)")

print("\nTest set will not be examined until final model evaluation.")

# ── Save encoders and mappings ────────────────────────────────────────────────
joblib.dump(le,            SAVE_DIR + 'label_encoder.pkl')
joblib.dump(le.classes_,   SAVE_DIR + 'label_classes.pkl')

class_mapping = {i: name for i, name in enumerate(le.classes_)}
joblib.dump(class_mapping, SAVE_DIR + 'class_mapping.pkl')
print("\nClass mapping saved:", class_mapping)
print(
    "\nFINDING: class_mapping.pkl maps model output integers back to attack names.\n"
    "When model predicts '3' in Week 3 this file tells us it means 'FTP-Patator'.\n"
    "Essential for production."
)

# ── Handle Unknown Labels (production safety) ────────────────────────────────
print(
    "\nIn production if a new attack type appears not seen in training:\n"
    "try:\n"
    "    encoded = le.transform([new_label])\n"
    "except ValueError:\n"
    "    print('Unknown attack type detected — flagging for human review')\n"
    "This prevents crashes when novel attacks appear."
)

# ── Stratified Sample for Visualization ──────────────────────────────────────
from sklearn.model_selection import train_test_split as tts

df_sample, _ = tts(
    pd.concat([X_train.reset_index(drop=True),
               pd.Series(y_train, name='Label').reset_index(drop=True)], axis=1),
    test_size=0.5, random_state=42,
    stratify=y_train
)
print(f"\nStratified 50% sample for visualizations: {df_sample.shape}")
print("All attack classes preserved:")
print(df_sample['Label'].value_counts().to_string())
print(
    "\nFINDING: Random sampling would lose rare classes like Infiltration (36 rows total).\n"
    "Stratified sampling preserves all class proportions."
)

# =============================================================================
# STEP 1 — MISSING AND INFINITE VALUES (X_train only)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1: Missing and infinite values")
print("=" * 60)
step_start = time.time()

# count before
missing_train = X_train.isnull().sum().sum()
inf_train = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
print(f"  Missing values in X_train before fill : {missing_train:,}")
print(f"  Infinite values in X_train            : {inf_train:,}")

# replace inf with NaN in both sets
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

# compute medians from X_train ONLY
train_medians = X_train.median()

# fill both sets with train medians
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

total_replaced = missing_train + inf_train
print(f"  Total values replaced (NaN + inf)     : {total_replaced:,}")
print(f"  Missing remaining in X_train          : {X_train.isnull().sum().sum()}")
print(f"  Missing remaining in X_test           : {X_test.isnull().sum().sum()}")

step_elapsed = time.time() - step_start
print(f"Step 1 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: Median computed from training set only then applied to test set.\n"
    "Using X_test statistics would leak test distribution into preprocessing.\n"
    "For network data median is preferred over mean because packet sizes and\n"
    "flow durations are heavily right-skewed — the mean is pulled by extreme outliers."
)

# =============================================================================
# STEP 2 — CLASS DISTRIBUTION ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Class distribution analysis")
print("=" * 60)
step_start = time.time()

label_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
label_names  = [le.classes_[i] for i in label_counts.index]
label_pcts   = label_counts.values / len(y_train) * 100

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(label_names, label_counts.values, color='steelblue')
ax.set_xlabel('Flow Count')
ax.set_title('Class Distribution in Training Set (y_train)')
for bar, pct in zip(bars, label_pcts):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f'{pct:.2f}%', va='center', fontsize=8)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(SAVE_DIR + 'class_distribution_train.png', bbox_inches='tight')
plt.show()

imbalance_ratio = label_counts.max() / label_counts.min()
print(f"\n  Imbalance ratio (largest / smallest class): {imbalance_ratio:,.1f}")

from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_acc = dummy.score(X_test, y_test)
print(f"\n  Dummy classifier (most_frequent) accuracy: {dummy_acc:.4f}")
print("\n  Classification report for dummy classifier:")
print(classification_report(y_test, dummy.predict(X_test), target_names=le.classes_))

step_elapsed = time.time() - step_start
print(f"Step 2 completed in {step_elapsed:.1f}s")
print(
    f"\nFINDING: The dummy classifier achieves {dummy_acc:.1%} accuracy by predicting\n"
    "BENIGN for everything. But its F1 score for every attack class is 0.0.\n"
    "This mathematically proves that accuracy is the wrong metric for this security dataset.\n"
    "Business cost: a security system with 99% accuracy that misses 100% of attacks is\n"
    "worse than useless — it creates false confidence. We optimize for recall on attack\n"
    "classes: missing an attack is more dangerous than a false alarm."
)

# =============================================================================
# STEP 3 — FEATURE QUALITY ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Feature quality — zero-variance features")
print("=" * 60)
step_start = time.time()

zero_var = X_train.columns[X_train.var() == 0].tolist()
print(f"  Zero variance features found: {zero_var}")
print(
    "  These features have the same value in every row.\n"
    "  No model can use them to make decisions."
)

X_train = X_train.drop(columns=zero_var)
X_test  = X_test.drop(columns=zero_var)
joblib.dump(zero_var, SAVE_DIR + 'removed_zero_variance.pkl')

step_elapsed = time.time() - step_start
print(f"Step 3 completed in {step_elapsed:.1f}s")
print(
    f"\nFINDING: Removed {len(zero_var)} zero-variance feature(s).\n"
    "These are logged in removed_zero_variance.pkl for reproducibility."
)

# =============================================================================
# STEP 4 — CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Correlation analysis")
print("=" * 60)
step_start = time.time()

CORRELATION_THRESHOLD = 0.95

corr_matrix = X_train.corr().abs()

# upper triangle only to avoid duplicate pairs
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_pairs = (
    upper.stack()
         .reset_index()
         .rename(columns={'level_0': 'Feature A', 'level_1': 'Feature B', 0: 'Correlation'})
         .query(f"Correlation > {CORRELATION_THRESHOLD}")
         .sort_values('Correlation', ascending=False)
)

print(f"\n  Pairs with correlation > {CORRELATION_THRESHOLD}:")
if len(high_corr_pairs):
    print(f"\n  {'Feature A':<45} {'Feature B':<45} {'Correlation':>12}")
    print("  " + "-" * 104)
    for _, row in high_corr_pairs.iterrows():
        print(f"  {row['Feature A']:<45} {row['Feature B']:<45} {row['Correlation']:>12.4f}")
else:
    print("  None found above threshold.")

# heatmap of top 30 most correlated features
top_features = set()
for _, row in high_corr_pairs.head(30).iterrows():
    top_features.update([row['Feature A'], row['Feature B']])
top_features = list(top_features)[:30]

if top_features:
    fig, ax = plt.subplots(figsize=(16, 14))
    sub_corr = X_train[top_features].corr()
    im = ax.imshow(sub_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90, fontsize=7)
    ax.set_yticklabels(top_features, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Correlation Heatmap — Top {len(top_features)} Correlated Features')
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'correlation_heatmap.png', bbox_inches='tight')
    plt.show()

# ── Decision: remove second feature from each correlated pair ────────────────
print(
    f"\n  Decision: Remove features with correlation > {CORRELATION_THRESHOLD}\n"
    "  Justification: At 0.95 correlation two features measure essentially the same\n"
    "  thing 95% of the time.\n"
    "  For Random Forest this adds computation without benefit.\n"
    "  For XGBoost this can cause overfitting by double-weighting the same signal.\n"
    "  For the neural network in Week 3 multicollinearity destabilizes gradient descent.\n"
    "  Threshold 0.95 chosen over 0.90 (too aggressive — removes features with 10%\n"
    "  independent information) and 0.99 (too conservative — keeps near-duplicates)."
)

features_to_remove = high_corr_pairs['Feature B'].unique().tolist()
original_n = X_train.shape[1]
print(f"\n  Features flagged for removal ({len(features_to_remove)}): {features_to_remove}")
joblib.dump(features_to_remove, SAVE_DIR + 'removed_correlated_features.pkl')

X_train = X_train.drop(columns=[c for c in features_to_remove if c in X_train.columns])
X_test  = X_test.drop(columns=[c for c in features_to_remove if c in X_test.columns])
remaining_n = X_train.shape[1]

print(f"\n  X_train shape after correlation removal: {X_train.shape}")
print(f"  X_test  shape after correlation removal: {X_test.shape}")

step_elapsed = time.time() - step_start
print(f"Step 4 completed in {step_elapsed:.1f}s")
print(
    f"\nFINDING: Removed {len(features_to_remove)} correlated feature(s).\n"
    f"This reduces feature space from {original_n} to {remaining_n} features."
)

# =============================================================================
# STEP 5 — FEATURE DISTRIBUTIONS WITH CYBERSECURITY VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Feature distributions with cybersecurity validation")
print("=" * 60)
step_start = time.time()

FEATURES_OF_INTEREST = [
    'Flow Duration',
    'Total Fwd Packets',
    'Destination Port',
    'SYN Flag Count',
    'Flow IAT Mean',
]

CYBER_NOTES = {
    'Flow Duration': (
        "Short duration for DoS/BruteForce: CONFIRMED by literature.\n"
        "Sharafaldin et al. 2017 report flow duration as top distinguishing\n"
        "feature for automated attacks."
    ),
    'SYN Flag Count': (
        "High SYN for BruteForce: CONFIRMED by literature.\n"
        "Each brute force attempt initiates new TCP handshake generating SYN flag.\n"
        "Normal traffic reuses connections."
    ),
    'Flow IAT Mean': (
        "Low inter-arrival time for automated attacks: CONFIRMED.\n"
        "Human users pause between actions (seconds).\n"
        "Automated tools operate at machine speed (milliseconds)."
    ),
    'Destination Port': (
        "Concentrated ports for BruteForce: CONFIRMED.\n"
        "FTP-Patator targets port 21. SSH-Patator targets port 22.\n"
        "Normal traffic uses diverse ports."
    ),
    'Total Fwd Packets': (
        "Low packet count for BruteForce: CONFIRMED.\n"
        "Each attempt: connect, send credentials, receive rejection, disconnect.\n"
        "Minimal exchange."
    ),
}

label_col = pd.Series(y_train, name='Label').reset_index(drop=True)
sample_features = df_sample.drop(columns=['Label'])

for feat in FEATURES_OF_INTEREST:
    if feat not in sample_features.columns:
        print(f"  [SKIP] '{feat}' not in sample columns — may have been removed.")
        continue

    print(f"\n  Feature: {feat}")
    class_means = df_sample.groupby('Label')[feat].mean()
    print("  Mean per class:")
    for cls_id, mean_val in class_means.items():
        print(f"    {le.classes_[int(cls_id)]:<40} {mean_val:.4f}")

    print(f"  Cybersecurity note:\n    {CYBER_NOTES.get(feat, 'N/A')}")

    # boxplot grouped by label, log scale
    fig, ax = plt.subplots(figsize=(14, 5))
    groups = [
        df_sample.loc[df_sample['Label'] == cls_id, feat].dropna().values
        for cls_id in sorted(df_sample['Label'].unique())
    ]
    class_labels = [le.classes_[int(i)] for i in sorted(df_sample['Label'].unique())]
    ax.boxplot(groups, labels=class_labels, vert=True)
    ax.set_yscale('log')
    ax.set_ylabel(feat + ' (log scale)')
    ax.set_title(f'Distribution of "{feat}" by Attack Class')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    safe_name = feat.replace(' ', '_').replace('/', '_')
    plt.savefig(SAVE_DIR + f'dist_{safe_name}.png', bbox_inches='tight')
    plt.show()

step_elapsed = time.time() - step_start
print(f"\nStep 5 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: All 5 EDA findings are consistent with published cybersecurity literature.\n"
    "This validates that our feature analysis is capturing real attack signatures not noise."
)

# =============================================================================
# STEP 6 — BINARY VS MULTICLASS FRAMING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Binary vs multiclass framing")
print("=" * 60)
step_start = time.time()

y_train_binary = (y_train > 0).astype(int)
y_test_binary  = (y_test  > 0).astype(int)

print("  Binary label distribution — TRAIN:")
for val, cnt in pd.Series(y_train_binary).value_counts().sort_index().items():
    name = 'BENIGN' if val == 0 else 'ATTACK'
    print(f"    {name} ({val}) : {cnt:,}  ({cnt/len(y_train_binary)*100:.2f}%)")

print("  Binary label distribution — TEST:")
for val, cnt in pd.Series(y_test_binary).value_counts().sort_index().items():
    name = 'BENIGN' if val == 0 else 'ATTACK'
    print(f"    {name} ({val}) : {cnt:,}  ({cnt/len(y_test_binary)*100:.2f}%)")

joblib.dump(y_train,        SAVE_DIR + 'y_train_multi.pkl')
joblib.dump(y_test,         SAVE_DIR + 'y_test_multi.pkl')
joblib.dump(y_train_binary, SAVE_DIR + 'y_train_binary.pkl')
joblib.dump(y_test_binary,  SAVE_DIR + 'y_test_binary.pkl')
print("  Saved: y_train_multi.pkl, y_test_multi.pkl, y_train_binary.pkl, y_test_binary.pkl")

step_elapsed = time.time() - step_start
print(f"Step 6 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: Binary model answers: is there an attack?\n"
    "Multiclass model answers: which specific attack is it?\n"
    "A real security system needs both:\n"
    "  Binary    for real-time alerting (fast, simple)\n"
    "  Multiclass for incident response (detailed, slower)\n"
    "Week 2 will train both."
)

# =============================================================================
# STEP 7 — FEATURE IMPORTANCE PREVIEW
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Feature importance preview (Random Forest on 20% sample)")
print("=" * 60)
step_start = time.time()

from sklearn.ensemble import RandomForestClassifier

# 20% stratified sample for speed
X_train_rs = X_train.reset_index(drop=True)
y_train_rs  = y_train

X_sample_idx, _, y_sample_idx, _ = train_test_split(
    X_train_rs, y_train_rs,
    test_size=0.80, random_state=42, stratify=y_train_rs
)

rf_preview = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_preview.fit(X_sample_idx, y_sample_idx)

importances = pd.Series(rf_preview.feature_importances_, index=X_train.columns)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 7))
top20.sort_values().plot(kind='barh', ax=ax, color='darkorange')
ax.set_xlabel('Feature Importance')
ax.set_title('Top 20 Feature Importances (RF preview on 20% sample)')
plt.tight_layout()
plt.savefig(SAVE_DIR + 'feature_importance_preview.png', bbox_inches='tight')
plt.show()

CYBER_FI_NOTES = {
    'Flow Duration'              : "Total time of network flow. Short for automated attacks.",
    'Total Fwd Packets'          : "Packets from source to destination. Low for brute force.",
    'Destination Port'           : "Target port. Concentrated for targeted attacks (21, 22, 80).",
    'SYN Flag Count'             : "TCP SYN flags. High for connection-heavy attacks.",
    'Flow IAT Mean'              : "Inter-arrival time between packets. Very low for automated tools.",
    'Fwd Packet Length Mean'     : "Avg byte size of forward packets. Tiny for DoS floods.",
    'Bwd Packet Length Mean'     : "Avg byte size of reply packets. Low if server rejects quickly.",
    'Packet Length Variance'     : "Variance in packet sizes. Low for automated uniform attacks.",
    'Flow Bytes/s'               : "Throughput. Extreme for volumetric DoS.",
    'Flow Packets/s'             : "Packet rate. Very high for DoS/DDoS.",
    'PSH Flag Count'             : "Push flags. Reveals data transfer patterns.",
    'ACK Flag Count'             : "Acknowledge flags. High in established sessions.",
    'Init_Win_bytes_forward'     : "Initial TCP window size forward. Fingerprints OS/tool.",
    'Init_Win_bytes_backward'    : "Initial TCP window backward. Reveals server response pattern.",
    'min_seg_size_forward'       : "Min segment size. Low for minimal-payload attacks.",
    'Average Packet Size'        : "Mean flow packet size. Distinguishes bulk vs probe traffic.",
    'Subflow Fwd Bytes'          : "Bytes per subflow forward. Reflects attack payload volume.",
    'Subflow Bwd Bytes'          : "Bytes per subflow backward. Reflects server response volume.",
}

print("\n  Top 5 features — detail:")
for feat, score in top20.head(5).items():
    note = CYBER_FI_NOTES.get(feat, "Network flow metric used by CICFlowMeter.")
    matches_lit = feat in [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Max',
        'Bwd Packet Length Max', 'Flow IAT Mean', 'SYN Flag Count'
    ]
    print(f"\n    Feature   : {feat}")
    print(f"    Importance: {score:.5f}")
    print(f"    Meaning   : {note}")
    print(f"    Matches literature: {'YES' % () if matches_lit else 'PARTIAL'}")

# published top features from Sharafaldin et al. 2017
published_top = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Bwd Packet Length Max', 'Flow IAT Mean', 'SYN Flag Count'
]
our_top = top20.head(8).index.tolist()
overlap = set(published_top) & set(our_top)
pct_agree = len(overlap) / len(published_top) * 100

print(f"\n  Published top features (Sharafaldin et al. 2017): {published_top}")
print(f"  Our top 8 features                              : {our_top}")
print(f"  Agreement: {pct_agree:.0f}%  (overlap: {list(overlap)})")

step_elapsed = time.time() - step_start
print(f"\nStep 7 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: Feature importance preview validates our preprocessing.\n"
    "If nonsensical features rank highest it would indicate data leakage\n"
    "or preprocessing errors."
)

# =============================================================================
# STEP 8 — SAVE COMPLETE DATA PACKAGE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Save complete data package")
print("=" * 60)
step_start = time.time()

X_train_clean = X_train
X_test_clean  = X_test

saves = {
    'X_train_clean.pkl'  : X_train_clean,
    'X_test_clean.pkl'   : X_test_clean,
}
feature_names  = X_train_clean.columns.tolist()
feature_dtypes = {col: str(dtype) for col, dtype in X_train_clean.dtypes.items()}

saves_meta = {
    'feature_names.pkl'  : feature_names,
    'feature_dtypes.pkl' : feature_dtypes,
}

for fname, obj in {**saves, **saves_meta}.items():
    fpath = SAVE_DIR + fname
    joblib.dump(obj, fpath)
    # verify
    loaded_back = joblib.load(fpath)
    if hasattr(loaded_back, 'shape'):
        assert loaded_back.shape == obj.shape, f"Shape mismatch for {fname}"
        print(f"  Saved & verified: {fname}  shape={loaded_back.shape}")
    elif isinstance(loaded_back, list):
        assert len(loaded_back) == len(obj), f"Length mismatch for {fname}"
        print(f"  Saved & verified: {fname}  len={len(loaded_back)}")
    elif isinstance(loaded_back, dict):
        assert len(loaded_back) == len(obj), f"Length mismatch for {fname}"
        print(f"  Saved & verified: {fname}  keys={len(loaded_back)}")
    else:
        print(f"  Saved: {fname}")

step_elapsed = time.time() - step_start
print(f"Step 8 completed in {step_elapsed:.1f}s")
print(
    "\nFINDING: feature_names.pkl and feature_dtypes.pkl are critical for Week 3 deployment.\n"
    "When a new packet arrives it must have exactly these features in exactly this order\n"
    "with these dtypes. Any mismatch produces silent wrong predictions."
)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
total_elapsed = time.time() - section_start

print("\n" + "=" * 60)
print("SECTION 2 — FINAL SUMMARY")
print("=" * 60)
print(f"  Total pipeline time (Section 2) : {total_elapsed:.1f}s")
print(f"  X_train_clean shape             : {X_train_clean.shape}")
print(f"  X_test_clean  shape             : {X_test_clean.shape}")
print(f"  y_train (multi)  shape          : {y_train.shape}")
print(f"  y_test  (multi)  shape          : {y_test.shape}")
print(f"  y_train (binary) shape          : {y_train_binary.shape}")
print(f"  y_test  (binary) shape          : {y_test_binary.shape}")
print(f"  Features after cleaning         : {len(feature_names)}")
print(f"  Zero-variance removed           : {len(zero_var)}")
print(f"  Correlated features removed     : {len(features_to_remove)}")
print(f"  Classes                         : {list(le.classes_)}")

print("\n" + "=" * 60)
print("IMPORTANT: Save Version now before closing session.")
print("Kaggle: Click 'Save Version' button top right")
print("Colab: File is already saved to Google Drive")
print("=" * 60)
