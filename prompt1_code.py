# NIDS Week 1 Pipeline
# Section 1: Data Loading and Merging
# This section loads all CICIDS2017 CSV files, handles encoding issues,
# standardizes labels, removes duplicates, validates data quality,
# and saves the merged dataset. Must complete before any other section.

# ── Global Random Seed ──────────────────────────────────────────────────────
import numpy as np
import random
import os
import time

np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# ── Library Versions ─────────────────────────────────────────────────────────
import pandas
import sklearn
print(f"pandas: {pandas.__version__}")
print(f"sklearn: {sklearn.__version__}")
print(f"numpy: {numpy.__version__}")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

# ── Environment Detection & Save Directory ───────────────────────────────────
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/NIDS/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    ENV = 'colab'
    print("Environment: Google Colab — saving to Google Drive at", SAVE_DIR)
except ImportError:
    SAVE_DIR = '/kaggle/working/'
    ENV = 'kaggle'
    print("Environment: Kaggle — saving to", SAVE_DIR)

DATA_PATH = '/kaggle/input/datasets/chaimaabid233/cicids2017/MachineLearningCVE/'

FILES = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
]

# ── STEP 1 — LOAD ALL FILES WITH ROBUST ERROR HANDLING ──────────────────────
print("\n" + "=" * 60)
print("STEP 1: Loading CSV files")
print("=" * 60)
step_start = time.time()

pipeline_start = time.time()

dataframes = []
load_log = []          # (filename, rows, encoding, error)
encoding_issues = []

for fname in FILES:
    fpath = os.path.join(DATA_PATH, fname)
    loaded = False

    # ── attempt UTF-8 ──
    try:
        df_tmp = pd.read_csv(fpath, encoding='utf-8', low_memory=False)
        encoding_used = 'utf-8'
        loaded = True
        print(f"  [OK utf-8]  {fname}  →  {df_tmp.shape[0]:,} rows, {df_tmp.shape[1]} cols")
    except UnicodeDecodeError:
        # ── fall back to latin-1 ──
        try:
            df_tmp = pd.read_csv(fpath, encoding='latin-1', low_memory=False)
            encoding_used = 'latin-1'
            loaded = True
            encoding_issues.append(fname)
            print(f"  [OK latin1] {fname}  →  {df_tmp.shape[0]:,} rows, {df_tmp.shape[1]} cols  (encoding fallback)")
        except Exception as err:
            print(f"  [FAIL]      {fname}  →  {err}")
            load_log.append((fname, 0, 'none', str(err)))
            continue
    except FileNotFoundError:
        print(f"  [MISS]      {fname}  →  file not found at {fpath}")
        load_log.append((fname, 0, 'none', 'FileNotFoundError'))
        continue
    except Exception as err:
        print(f"  [FAIL]      {fname}  →  {err}")
        load_log.append((fname, 0, 'none', str(err)))
        continue

    if loaded:
        dataframes.append((fname, df_tmp, encoding_used))
        load_log.append((fname, len(df_tmp), encoding_used, None))

step_elapsed = time.time() - step_start
print(f"\nStep 1 completed in {step_elapsed:.1f}s")
print(f"\nFINDING: {len(dataframes)} of {len(FILES)} files loaded successfully.")
if encoding_issues:
    print(f"  Encoding issues (fell back to latin-1): {encoding_issues}")
else:
    print("  No encoding issues encountered.")

# ── STEP 2 — CLEAN COLUMN NAMES ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Stripping whitespace from column names")
print("=" * 60)

if dataframes:
    fname0, df0, _ = dataframes[0]
    before_cols = list(df0.columns[:5])

for i, (fname, df, enc) in enumerate(dataframes):
    df.columns = df.columns.str.strip()
    dataframes[i] = (fname, df, enc)

if dataframes:
    after_cols = list(dataframes[0][1].columns[:5])
    print(f"  First file: {dataframes[0][0]}")
    print(f"  BEFORE (first 5): {before_cols}")
    print(f"  AFTER  (first 5): {after_cols}")

print("FINDING: Column names had leading/trailing spaces which would")
print("         cause KeyError when accessing columns by name.")

# ── STEP 3 — ADD SOURCE TRACKING ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Adding 'Day' source-tracking column")
print("=" * 60)

dfs_tagged = []
for fname, df, enc in dataframes:
    df = df.copy()
    df['Day'] = fname
    dfs_tagged.append(df)

if dfs_tagged:
    print("  Sample 'Day' values:")
    sample = pd.concat([d[['Day']].head(2) for d in dfs_tagged[:3]], ignore_index=True)
    print(sample.to_string(index=False))

print("\nFINDING: Day column preserved for future temporal analysis.")

# ── Plot attack class distribution per day ────────────────────────────────────
if dfs_tagged:
    combined_for_plot = pd.concat(dfs_tagged, ignore_index=True)
    # strip label whitespace for plotting
    combined_for_plot['Label'] = combined_for_plot['Label'].astype(str).str.strip()
    pivot = combined_for_plot.groupby(['Day', 'Label']).size().unstack(fill_value=0)

    # shorten day names for readability
    pivot.index = [idx.replace('.pcap_ISCX.csv', '').replace('WorkingHours', '')
                   .replace('MachineLearningCVE/', '') for idx in pivot.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_xlabel('Day')
    ax.set_ylabel('Flow Count')
    ax.set_title('Attack Class Distribution per Day (CICIDS2017)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=7)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'attack_distribution_per_day.png'),
                bbox_inches='tight')
    plt.show()
    print("  Bar chart saved to", SAVE_DIR + 'attack_distribution_per_day.png')

# ── STEP 4 — MERGE ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Concatenating all dataframes")
print("=" * 60)
step_start = time.time()

df_all = pd.concat(dfs_tagged, ignore_index=True)
print(f"  Final merged shape: {df_all.shape}")

step_elapsed = time.time() - step_start
print(f"Step 4 completed in {step_elapsed:.1f}s")

# ── STEP 5 — LABEL STANDARDIZATION ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Label standardization")
print("=" * 60)
step_start = time.time()

print("Labels BEFORE cleaning (sorted):")
labels_before = sorted(df_all['Label'].astype(str).unique())
for lbl in labels_before:
    print(f"  '{lbl}'")

# ── apply normalisation and record mapping ────────────────────────────────────
label_mapping = {}
original_labels = df_all['Label'].astype(str).copy()

def normalize_label(lbl):
    """Strip, fix separators, and title-case."""
    lbl = lbl.strip()
    lbl = lbl.replace('  ', ' ')   # collapse double spaces
    lbl = lbl.replace('-', ' ')    # unify hyphen/space separators
    lbl = lbl.title()              # consistent casing
    return lbl

df_all['Label'] = df_all['Label'].astype(str).apply(normalize_label)

# build explicit before→after mapping for display
unique_before = original_labels.unique()
for old in sorted(unique_before):
    new = normalize_label(old)
    if old != new:
        label_mapping[old] = new

if label_mapping:
    print("\nLabel mapping (changed labels only):")
    for old, new in label_mapping.items():
        print(f"  '{old}'  →  '{new}'")
else:
    print("\nNo label changes required (all labels already consistent).")

print("\nLabels AFTER cleaning (sorted):")
labels_after = sorted(df_all['Label'].unique())
for lbl in labels_after:
    print(f"  '{lbl}'")

print("\nLabel value counts with percentages:")
total_rows = len(df_all)
vc = df_all['Label'].value_counts()
for lbl, cnt in vc.items():
    pct = cnt / total_rows * 100
    print(f"  {lbl:<45} {cnt:>10,}  ({pct:5.2f}%)")

step_elapsed = time.time() - step_start
print(f"\nStep 5 completed in {step_elapsed:.1f}s")

inconsistencies = list(label_mapping.keys())
print(f"\nFINDING: Label inconsistencies found across files: {inconsistencies if inconsistencies else 'none'}")
print("  These would have created phantom classes in the model.")

# ── STEP 6 — DUPLICATE DETECTION AND REMOVAL ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Duplicate detection and removal")
print("=" * 60)
step_start = time.time()

n_before_dedup = len(df_all)
n_duplicates = df_all.duplicated().sum()
pct_dup = n_duplicates / n_before_dedup * 100
print(f"  Exact duplicate rows: {n_duplicates:,}  ({pct_dup:.3f}%)")

df_all = df_all.drop_duplicates().reset_index(drop=True)
print(f"  Shape after deduplication: {df_all.shape}")

step_elapsed = time.time() - step_start
print(f"Step 6 completed in {step_elapsed:.1f}s")
print(f"\nFINDING: Removed {n_duplicates:,} duplicate rows ({pct_dup:.3f}%). Without removal,")
print("  duplicates could appear in both train and test sets. The model would memorize")
print("  these exact rows and report inflated test metrics that don't reflect real")
print("  generalization performance.")

# ── STEP 7 — DOMAIN VALIDATION (impossible values) ───────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Domain validation — impossible values")
print("=" * 60)
step_start = time.time()

NON_NEGATIVE_COLS = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
]

total_negative = 0
total_infinite = 0

print("  Negative values per column:")
for col in NON_NEGATIVE_COLS:
    if col not in df_all.columns:
        print(f"    {col:<45} [COLUMN NOT FOUND — skipping]")
        continue

    # coerce to numeric first (some files may have mixed types)
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    n_neg = (df_all[col] < 0).sum()
    total_negative += n_neg
    if n_neg:
        print(f"    {col:<45} {n_neg:>8,} negative values")
        df_all.loc[df_all[col] < 0, col] = np.nan
    else:
        print(f"    {col:<45} {n_neg:>8,}")

print("\n  Infinite values (across all numeric columns):")
numeric_cols = df_all.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    n_inf = np.isinf(df_all[col]).sum()
    if n_inf > 0:
        total_infinite += n_inf
        print(f"    {col:<45} {n_inf:>8,} infinite values")
        df_all[col] = df_all[col].replace([np.inf, -np.inf], np.nan)

total_nan = df_all.isnull().sum().sum()
print(f"\n  Total NaN count after all replacements: {total_nan:,}")

step_elapsed = time.time() - step_start
print(f"Step 7 completed in {step_elapsed:.1f}s")
print(f"\nFINDING: CICFlowMeter artifacts found: {total_negative:,} negative values and")
print(f"  {total_infinite:,} infinite values replaced with NaN. These are measurement errors,")
print("  not outliers — they represent physically impossible network measurements caused")
print("  by integer overflow and division by zero bugs in the capture tool.")

# ── STEP 8 — SAVE AND VERIFY ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Save and verify")
print("=" * 60)
step_start = time.time()

save_path = os.path.join(SAVE_DIR, 'cicids2017_merged.parquet')
df_all.to_parquet(save_path, index=False)

# ── immediate verification ─────────────────────────────────────────────────
loaded = pd.read_parquet(save_path)
assert loaded.shape == df_all.shape, \
    f"SAVE VERIFICATION FAILED: saved {loaded.shape} but expected {df_all.shape}"
print(f"  Save verified successfully: {loaded.shape}")

parquet_mb = os.path.getsize(save_path) / (1024 ** 2)
# rough CSV estimate: parquet is typically 5-8× smaller than CSV for this data
csv_estimate_mb = parquet_mb * 6.5
print(f"  Parquet file size : {parquet_mb:.1f} MB")
print(f"  CSV equivalent    : ~{csv_estimate_mb:.0f} MB (estimated)")

step_elapsed = time.time() - step_start
print(f"Step 8 completed in {step_elapsed:.1f}s")

print("\n" + "=" * 60)
print("IMPORTANT: SAVE YOUR WORK NOW")
print("Kaggle: Click 'Save Version' button top right")
print("Colab: File is already saved to Google Drive")
print("=" * 60)

# ── FINAL SUMMARY ────────────────────────────────────────────────────────────
total_elapsed = time.time() - pipeline_start

print("\n" + "=" * 60)
print("FINAL PIPELINE SUMMARY")
print("=" * 60)
print(f"  Total pipeline time       : {total_elapsed:.1f}s")
print(f"  Total rows                : {len(df_all):,}")
print(f"  Total columns             : {df_all.shape[1]}")
print(f"  Labels found              : {sorted(df_all['Label'].unique())}")
print(f"  Duplicates removed        : {n_duplicates:,}  ({pct_dup:.3f}%)")
print(f"  Impossible values fixed   : {total_negative + total_infinite:,}"
      f"  ({total_negative:,} negative + {total_infinite:,} infinite)")
print(f"  Encoding issues           : {len(encoding_issues)} file(s)  {encoding_issues}")
print("=" * 60)
