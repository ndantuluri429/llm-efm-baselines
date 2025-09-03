
"""## CLEAN AND MERGE CSVs
### goes through all of the CSV files and manually calculates values
"""

import pandas as pd
import numpy as np
import re
import ast
import os
from scipy.signal import find_peaks, welch

# --- PARAMETERS ---
FOLDER_PATH = ''
SR = 0.25  # sampling rate

# --- LLM-STYLE METRIC FUNCTION ---

def classify_deceleration(fhr_vals, uc_vals, sr=0.25):
    dec_type = "None"
    dec_pres = "No"
    dec_assoc = "No"

    drops = []
    for i in range(len(fhr_vals) - 30):  # Look ahead up to 30 seconds (120 samples)
        for j in range(i + 4, min(i + int(2 * 60 * sr), len(fhr_vals))):
            amp = fhr_vals[i] - fhr_vals[j]
            if amp >= 15:
                duration_sec = (j - i) / sr
                drops.append((i, j, amp, duration_sec))
                break

    if drops:
        dec_pres = "Yes"
        for start, end, amp, duration_sec in drops:
            uc_window = uc_vals[max(0, start - 4):end + 5]
            dec_assoc = "Yes" if any(u > 1.0 for u in uc_window) else "No"

            if duration_sec >= 120:
                return "Prolonged", dec_pres, dec_assoc

            elif 15 <= amp < 120:
                slope = abs(fhr_vals[end] - fhr_vals[start]) / duration_sec
                if slope > 2:
                    return "Variable", dec_pres, dec_assoc
                else:
                    nadir_index = start + np.argmin(fhr_vals[start:end+1])
                    uc_peak_index = max(0, start - 10) + np.argmax(uc_vals[max(0, start - 10):end + 10])

                    if nadir_index < uc_peak_index:
                        return "Early", dec_pres, dec_assoc
                    elif nadir_index > uc_peak_index:
                        return "Late", dec_pres, dec_assoc
                    else:
                        return "Late", dec_pres, dec_assoc

        # none of the conditions matched = assume sinusoidal
        return "Sinusoidal", dec_pres, "No"

    # No drops at all
    return "None", dec_pres, dec_assoc

def calc_metrics(fhr_vals, uc_vals, sr=0.25):
    if not fhr_vals or not uc_vals:
        return {
            "baseline_fhr": None,
            "variability_std": None,
            "variability_category": None,
            "accelerations": None,
            "decelerations_present": None,
            "decelerations_type": None,
            "decelerations_assoc_with_uc": None,
            "uc_freq_per_10min": None,
            "uc_classification": None
        }

    # --- Baseline and variability ---
    baseline = int(round(np.mean(fhr_vals)))
    std = round(np.std(fhr_vals), 1)
    var_cat = (
        "Absent" if std == 0 else
        "Minimal" if std < 5 else
        "Moderate" if std <= 25 else
        "Marked"
    )

    # --- Acceleration ---
    acc = "Present" if any(
        fhr_vals[i + 4] - fhr_vals[i] >= 15
        for i in range(len(fhr_vals) - 4)
    ) else "Absent"

    # --- Decelerations ---
    dec_type, dec_pres, dec_assoc = classify_deceleration(fhr_vals, uc_vals, sr)

    # --- UC contraction frequency ---
    events = 0
    i = 0
    while i < len(uc_vals):
        if uc_vals[i] > 1.0:
            cnt = 0
            while i < len(uc_vals) and uc_vals[i] > 1.0:
                cnt += 1
                i += 1
            if cnt >= 3:
                events += 1
        else:
            i += 1

    duration_min = len(uc_vals) / (sr * 60)
    freq10 = round(events / (duration_min / 10), 1) if duration_min else None
    uc_class = "Normal" if freq10 is not None and freq10 <= 5 else "Tachysystole"

    return {
        "baseline_fhr": baseline,
        "variability_std": std,
        "variability_category": var_cat,
        "accelerations": acc,
        "decelerations_present": dec_pres,
        "decelerations_type": dec_type,
        "decelerations_assoc_with_uc": dec_assoc,
        "uc_freq_per_10min": freq10,
        "uc_classification": uc_class
    }


# --- LOOP THROUGH FILES AND COMPUTE METRICS ---
all_results = []

for filename in os.listdir(FOLDER_PATH):
    if not filename.endswith('.csv'):
        continue

    filepath = os.path.join(FOLDER_PATH, filename)
    df = pd.read_csv(filepath)

    # Split into 3 equal steps
    step_size = len(df) // 3
    segments = {
        "Step 1": df.iloc[:step_size],
        "Step 2": df.iloc[step_size:2*step_size],
        "Step 3": df.iloc[2*step_size:]
    }

    for step, seg in segments.items():
        fhr_vals = seg['FHR'].tolist()
        uc_vals  = seg['UC'].tolist()
        metrics = calc_metrics(fhr_vals, uc_vals)

        if metrics is not None:
            metrics["PID"] = filename.replace('_smoothed.csv', '')
            metrics["segment"] = step
            all_results.append(metrics)

# --- SAVE RESULTS ---
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        '',
        index=False
    )
else:
    print("No valid results were collected. Check for data issues.")


"""### Clean manual calcuations table"""

import pandas as pd

df = pd.read_csv('')

cols = df.columns.tolist()
new_order = ['PID', 'segment'] + [col for col in cols if col not in ['PID', 'segment']]
df = df[new_order]

df['segment'] = df['segment'].str.replace('Step ', '', regex=False)
df = df.sort_values(by='PID').reset_index(drop=True)

# --- Convert to int ---
df['PID'] = df['PID'].astype(int)
df['segment'] = df['segment'].astype(int)

# --- Convert to float ---
df['baseline_fhr'] = df['baseline_fhr'].astype(float)
df['variability_std'] = df['variability_std'].astype(float)
df['uc_freq_per_10min'] = df['uc_freq_per_10min'].astype(float)

# --- Convert to categorical variables ---
# variability_category
var_type_map = {'Absent': 0, 'Minimal': 1, 'Moderate': 2, 'Marked': 3}
df['variability_category'] = df['variability_category'].map(var_type_map).astype('category')

# accelerations (0 = absent; 1 = present)
df['accelerations'] = df['accelerations'].map({'Absent': 0, 'Present': 1}).astype('category')

# decelerations_present (0 = no; 1 = yes)
df['decelerations_present'] = df['decelerations_present'].map({'No': 0, 'Yes': 1}).astype('category')

# decelerations_type (0 = early; 1 = late; 2 = variable; 3 = prolonged; 4 = sinusoidal)
dec_type_map = {'Early': 0, 'Late': 1, 'Variable': 2, 'Prolonged': 3, 'Sinusoidal': 4}
df['decelerations_type'] = df['decelerations_type'].map(dec_type_map).astype('category')

# decelerations_assoc_with_uc (0 = no; 1 = yes)
df['decelerations_assoc_with_uc'] = df['decelerations_assoc_with_uc'].map({'No': 0, 'Yes': 1}).astype('category')

# uc_classification (0 = normal; 1 = tachysystole)
df['uc_classification'] = df['uc_classification'].map({'Normal': 0, 'Tachysystole': 1}).astype('category')

# --- Rename columns ---
df = df.rename(columns={
    'segment': 'step',
    'baseline_fhr': 'baseline_fhr_calc',
    'variability_std': 'var_std_calc',
    'variability_category': 'var_category_calc',
    'accelerations': 'accel_calc',
    'decelerations_present': 'decel_present_calc',
    'decelerations_type': 'decel_type_calc',
    'decelerations_assoc_with_uc': 'decel_assoc_uc_calc',
    'uc_freq_per_10min': 'uc_freq_per_10min_calc',
    'uc_classification': 'uc_class_calc'
})

df.to_csv('', index=False)

"""### Clean LLM results table"""

import pandas as pd

df = pd.read_csv('')

df = df.rename(columns={'uterine_contractions_frequency': 'uc_freq_per_10min'})
df['uc_freq_per_10min'] = df['uc_freq_per_10min'].str.replace(' per 10 min', '', regex=False).astype(int)
df['baseline_fhr'] = df['baseline_fhr'].str.replace(' bpm', '', regex=False).astype(float)

# --- Convert to int ---
df['PID'] = df['PID'].astype(int)
df['step'] = df['step'].astype(int)

# --- Convert to float ---
df['baseline_fhr'] = df['baseline_fhr'].astype(float)
df['baseline_variability_std_dev'] = df['baseline_variability_std_dev'].astype(float)
df['uc_freq_per_10min'] = df['uc_freq_per_10min'].astype(float)

# --- Convert to categorical variables ---
# baseline_variability_category
var_type_map = {'Absent': 0, 'Minimal': 1, 'Moderate': 2, 'Marked': 3}
df['baseline_variability_category'] = df['baseline_variability_category'].map(var_type_map).astype('category')

# accelerations (0 = absent; 1 = present)
df['accelerations'] = df['accelerations'].map({'Absent': 0, 'Present': 1}).astype('category')

# decelerations_present (0 = no; 1 = yes)
df['decelerations_present'] = df['decelerations_present'].map({'No': 0, 'Yes': 1}).astype('category')

# decelerations_type (0 = early; 1 = late; 2 = variable; 3 = prolonged; 4 = sinusoidal)
dec_type_map = {'Early': 0, 'Late': 1, 'Variable': 2, 'Prolonged': 3, 'Sinusoidal': 4}
df['decelerations_type'] = df['decelerations_type'].map(dec_type_map).astype('category')

# decelerations_assoc_with_uc (0 = no; 1 = yes)
df['decelerations_associated_with_contractions'] = df['decelerations_associated_with_contractions'].map({'No': 0, 'Yes': 1}).astype('category')

# uc_classification (0 = normal; 1 = tachysystole)
df['uterine_contractions_classification'] = df['uterine_contractions_classification'].map({'Normal': 0, 'Tachysystole': 1}).astype('category')


# --- Rename columns ---
df = df.rename(columns={
    'baseline_fhr': 'baseline_fhr_LLM',
    'baseline_variability_std_dev': 'var_std_LLM',
    'baseline_variability_category': 'var_category_LLM',
    'accelerations': 'accel_LLM',
    'decelerations_present': 'decel_present_LLM',
    'decelerations_type': 'decel_type_LLM',
    'decelerations_associated_with_contractions': 'decel_assoc_uc_LLM',
    'uc_freq_per_10min': 'uc_freq_per_10min_LLM',
    'uterine_contractions_classification': 'uc_class_LLM'
})


df.to_csv('', index=False)

"""### merge LLM + manual results table"""

import pandas as pd
manual_path = ''
llm_path = ''

manual_df = pd.read_csv(manual_path)
llm_df = pd.read_csv(llm_path)

# Merge on 'PID' and 'step'
merged_df = pd.merge(manual_df, llm_df, on=['PID', 'step'], how='inner')

merged_df.to_csv('', index=False)

"""## COMPARE MANUAL CALC. VS LLM RESULTS

#### NUMERIC Variables
"""

# --- Numeric Variables --- #

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

merged_df = pd.read_csv('')

# difference and absolute error
merged_df['fhr_diff'] = abs(merged_df['baseline_fhr_calc'] - merged_df['baseline_fhr_LLM'])
merged_df['var_std_diff'] = abs(merged_df['var_std_calc'] - merged_df['var_std_LLM'])
merged_df['uc_freq_diff'] = abs(merged_df['uc_freq_per_10min_calc'] - merged_df['uc_freq_per_10min_LLM'])

# correlation btwn predicted and calculated
merged_df[['baseline_fhr_calc', 'baseline_fhr_LLM']].corr()
merged_df[['var_std_calc', 'var_std_LLM']].corr()
merged_df[['uc_freq_per_10min_calc', 'uc_freq_per_10min_LLM']].corr()

# mean absolute error
mae_fhr = mean_absolute_error(merged_df['baseline_fhr_calc'], merged_df['baseline_fhr_LLM'])
mae_var_std = mean_absolute_error(merged_df['var_std_calc'], merged_df['var_std_LLM'])
mae_uc_freq = mean_absolute_error(merged_df['uc_freq_per_10min_calc'], merged_df['uc_freq_per_10min_LLM'])

# pearson correlation
corr_fhr = merged_df[['baseline_fhr_calc', 'baseline_fhr_LLM']].corr().iloc[0, 1]
corr_var_std = merged_df[['var_std_calc', 'var_std_LLM']].corr().iloc[0, 1]
corr_uc_freq = merged_df[['uc_freq_per_10min_calc', 'uc_freq_per_10min_LLM']].corr().iloc[0, 1]

## show results
summary_table = pd.DataFrame({
    'Metric': ['Baseline FHR', 'Variability Std Dev', 'UC Frequency per 10min'],
    'MAE': [mae_fhr, mae_var_std, mae_uc_freq],
    'Correlation': [corr_fhr, corr_var_std, corr_uc_freq]
})
print(summary_table.round(3))


## show correlations
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.regplot(x='baseline_fhr_calc', y='baseline_fhr_LLM', data=merged_df, scatter_kws={'alpha':0.4})
plt.title('Baseline FHR')
plt.xlabel('Manual (calc)')
plt.ylabel('GPT')

plt.subplot(1, 3, 2)
sns.regplot(x='var_std_calc', y='var_std_LLM', data=merged_df, scatter_kws={'alpha':0.4})
plt.title('Variability Std Dev')
plt.xlabel('Manual (calc)')
plt.ylabel('GPT')

plt.subplot(1, 3, 3)
sns.regplot(x='uc_freq_per_10min_calc', y='uc_freq_per_10min_LLM', data=merged_df, scatter_kws={'alpha':0.4})
plt.title('UC Frequency per 10min')
plt.xlabel('Manual (calc)')
plt.ylabel('GPT')
plt.tight_layout()
plt.show()

## show absolute diff histograms
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(merged_df['fhr_diff'], bins=20, edgecolor='black')
plt.title(f'FHR Absolute Difference\n(MAE = {mae_fhr:.2f})')
plt.xlabel('Absolute Difference')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(merged_df['var_std_diff'], bins=20, edgecolor='black')
plt.title(f'Variability Std Absolute Difference\n(MAE = {mae_var_std:.2f})')
plt.xlabel('Absolute Difference')

plt.subplot(1, 3, 3)
plt.hist(merged_df['uc_freq_diff'], bins=20, edgecolor='black')
plt.title(f'UC Freq Absolute Difference\n(MAE = {mae_uc_freq:.2f})')
plt.xlabel('Absolute Difference')

plt.tight_layout()
plt.show()

plt.savefig("GPT_scatters.pdf", bbox_inches='tight')
plt.savefig("",
             dpi=300, bbox_inches='tight')

"""#### CATEGORICAL Variables"""

# --- Categorical Variables --- #

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

merged_df = pd.read_csv('')

# proportion of agreement
(merged_df['accel_calc'] == merged_df['accel_LLM']).mean()
(merged_df['var_category_calc'] == merged_df['var_category_LLM']).mean()
(merged_df['decel_present_calc'] == merged_df['decel_present_LLM']).mean()
(merged_df['decel_type_calc'] == merged_df['decel_type_LLM']).mean()
(merged_df['decel_assoc_uc_calc'] == merged_df['decel_assoc_uc_LLM']).mean()
(merged_df['uc_class_calc'] == merged_df['uc_class_LLM']).mean()

# confusion matrix
confusion_matrix(merged_df['var_category_calc'], merged_df['var_category_LLM'])

# show comparison
comparison_results = {
    'accel_agreement': (merged_df['accel_calc'] == merged_df['accel_LLM']).mean(),
    'var_category_agreement': (merged_df['var_category_calc'] == merged_df['var_category_LLM']).mean(),
    'decel_present_agreement': (merged_df['decel_present_calc'] == merged_df['decel_present_LLM']).mean(),
    'decel_type_agreement': (merged_df['decel_type_calc'] == merged_df['decel_type_LLM']).mean(),
    'decel_assoc_uc_agreement': (merged_df['decel_assoc_uc_calc'] == merged_df['decel_assoc_uc_LLM']).mean(),
    'uc_class_agreement': (merged_df['uc_class_calc'] == merged_df['uc_class_LLM']).mean()
}

pd.Series(comparison_results).sort_values(ascending=False)

"""### per class agreement"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def per_class_agreement_table(y_true, y_pred, feature_name):
    """
    Returns:
      - per-class table with support (GT count), pred_count, recall (per_class_agreement),
        precision (pred_class_agreement)
      - confusion matrix (DataFrame)
      - overall accuracy
    """

    # 1) Make Series, replace NaN, and coerce to strings (prevents sklearn "unknown" target type)
    y_true_s = pd.Series(y_true).where(pd.notna(y_true), "missing").astype(str)
    y_pred_s = pd.Series(y_pred).where(pd.notna(y_pred), "missing").astype(str)

    # 2) Fix labels (union of both sides) and ensure plain 1D arrays for sklearn
    labels = sorted(set(y_true_s.unique()).union(set(y_pred_s.unique())), key=lambda x: x)

    # 3) Confusion matrix
    cm_np = confusion_matrix(y_true_s.values, y_pred_s.values, labels=labels)
    cm = pd.DataFrame(cm_np, index=pd.Index(labels, name="GT"), columns=pd.Index(labels, name="Pred"))

    # 4) Counts
    gt_counts = cm.sum(axis=1)          # support per GT class
    pred_counts = cm.sum(axis=0)        # how often each class was predicted
    correct = pd.Series(np.diag(cm_np), index=labels)

    # 5) Metrics
    per_class_agreement = (correct / gt_counts).replace([np.inf, np.nan], 0.0)   # recall
    pred_class_agreement = (correct / pred_counts).replace([np.inf, np.nan], 0.0) # precision
    overall = (y_true_s.values == y_pred_s.values).mean()

    out = pd.DataFrame({
        "feature": feature_name,
        "class": labels,
        "support_gt": gt_counts.astype(int).values,
        "pred_count": pred_counts.astype(int).values,
        "per_class_agreement": per_class_agreement.values,   # recall
        "pred_class_agreement": pred_class_agreement.values, # precision
    })

    try:
        out["_cls_num"] = pd.to_numeric(out["class"])
        out = out.sort_values(["feature", "_cls_num"]).drop(columns=["_cls_num"])
    except Exception:
        out = out.sort_values(["feature", "class"])

    return out, cm, overall

    features = [
    ("accel_calc", "accel_LLM"),
    ("var_category_calc", "var_category_LLM"),
    ("decel_present_calc", "decel_present_LLM"),
    ("decel_type_calc", "decel_type_LLM"),
    ("decel_assoc_uc_calc", "decel_assoc_uc_LLM"),
    ("uc_class_calc", "uc_class_LLM"),
]

all_per_class = []
conf_mats = {}
overall_acc = {}

for gt_col, pred_col in features:
    tbl, cm, overall = per_class_agreement_table(
        merged_df[gt_col], merged_df[pred_col], feature_name=gt_col.replace("_calc", "")
    )
    all_per_class.append(tbl)
    conf_mats[gt_col.replace("_calc", "")] = cm
    overall_acc[gt_col.replace("_calc", "")] = overall

per_class_results = pd.concat(all_per_class, ignore_index=True)

print("\nOverall accuracy by feature:")
print(pd.Series(overall_acc).round(3).sort_values(ascending=False))

print("\nPer-class agreement (recall) & predicted-class agreement (precision):")
print(per_class_results.round(3))

import pandas as pd

df = pd.read_csv('')

features = [
    "accel",
    "decel_present",
    "var_category",
    "decel_type",
    "uc_class",
]

rows = []
for feature in features:
    gt = f"{feature}_calc"
    llm = f"{feature}_LLM"

    # Replace NaN with a consistent label
    gt_series = df[gt].fillna("missing")
    llm_series = df[llm].fillna("missing")

    # Overall accuracy
    acc = (gt_series == llm_series).mean()

    # Counts
    gt_counts = gt_series.value_counts()
    llm_counts = llm_series.value_counts()

    # Align classes
    all_classes = sorted(set(gt_counts.index).union(set(llm_counts.index)), key=lambda x: str(x))

    for cls in all_classes:
        rows.append({
            "Feature": feature,
            "Class": cls,
            "Ground Truth Count": int(gt_counts.get(cls, 0)),
            "LLM Count": int(llm_counts.get(cls, 0)),
            "Overall Accuracy": round(acc, 3),
        })

results_table = pd.DataFrame(rows).sort_values(by=["Feature", "Class"]).reset_index(drop=True)
print(results_table)

results_table.to_csv("", index=False)
#print("\nSaved to per_feature_class_counts_with_accuracy.csv")

import pandas as pd

df = pd.read_csv('')

# Pick out the presence/absence style binary features
binary_features = ["accel_calc", "decel_present_calc"]

# Row-wise count of how many are "present" (==1)
df["n_events_present"] = df[binary_features].sum(axis=1)

# Keep categorical features too for context
per_record = df[["PID", "step", "accel_calc", "decel_present_calc",
                 "decel_type_calc", "var_category_calc", "uc_class_calc",
                 "n_events_present"]]

print(per_record.head(10))

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the merged CSV
merged_df = pd.read_csv('')

# Full column pairs to compare
column_pairs = [
    ('accel_calc', 'accel_LLM'),
    ('var_category_calc', 'var_category_LLM'),
    ('decel_present_calc', 'decel_present_LLM'),
    ('decel_type_calc', 'decel_type_LLM'),
    ('decel_assoc_uc_calc', 'decel_assoc_uc_LLM'),
    ('uc_class_calc', 'uc_class_LLM')
]

# Generate confusion matrices
for calc_col, llm_col in column_pairs:
    if calc_col in merged_df.columns and llm_col in merged_df.columns:
        y_true = merged_df[calc_col].astype(str)
        y_pred = merged_df[llm_col].astype(str)
        labels = sorted(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("LLM (Predicted)")
        plt.ylabel("CALC (Manual)")
        plt.title(f"Confusion Matrix: {calc_col.replace('_calc', '')}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipped: {calc_col} or {llm_col} not found.")

"""### LLM v. Real pH"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the merged CSV
df = pd.read_csv('')

# Group by PID and calculate the average pH prediction
ph_avg_df = df.groupby('PID')['umbilical_artery_ph_prediction'].mean().reset_index()

# rename column to avg_ph_prediction'
ph_avg_df = ph_avg_df.rename(columns={'umbilical_artery_ph_prediction': 'avg_ph_prediction'})

# save as csv
ph_avg_df.to_csv('', index=False)

# Load files to compare
pred_df = pd.read_csv('')
true_df = pd.read_excel('')
true_df = true_df.rename(columns={'File': 'PID', 'pH': 'true_ph'})

# merge on PID
merged_df = pd.merge(pred_df, true_df, on='PID', how='inner')

# calculate absolute error
merged_df['abs_error'] = (merged_df['avg_ph_prediction'] - merged_df['true_ph']).abs()

# save comparison
merged_df.to_csv('', index=False)

# print(pH_df.columns)
# true_df.info()

!pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# compare the values
merged_df = pd.read_csv('')

# merged_df.info()

# Mean Absolute Error (already exists)
mae = merged_df['abs_error'].mean()

# RMSE (manual calc to avoid version issues)
rmse = np.sqrt(((merged_df['true_ph'] - merged_df['avg_ph_prediction']) ** 2).mean())

# Mean Signed Error (bias)
bias = (merged_df['avg_ph_prediction'] - merged_df['true_ph']).mean()

# R² Score
r2 = r2_score(merged_df['true_ph'], merged_df['avg_ph_prediction'])

# Print metrics in a clean table format
print(f"{'Metric':<30} {'Value'}")
print("-" * 40)
print(f"{'Mean Absolute Error (MAE)':<30} {mae:.4f}")
print(f"{'Root Mean Squared Error (RMSE)':<30} {rmse:.4f}")
print(f"{'Mean Signed Error (Bias)':<30} {bias:.4f}")
print(f"{'R² Score':<30} {r2:.4f}")

# True vs Predicted
plt.scatter(merged_df['true_ph'], merged_df['avg_ph_prediction'], alpha=0.6)
plt.plot([6.8, 7.4], [6.8, 7.4], color='red', linestyle='--')  # perfect prediction line
plt.xlabel("True pH")
plt.ylabel("Predicted pH")
plt.title("True vs Predicted pH")
plt.grid(True)
plt.show()

# errors
merged_df['abs_error'].hist(bins=30)
plt.xlabel("Absolute Error")
plt.title("Distribution of Prediction Errors")
plt.show()

# distrbiution of LLM v true pH
merged_df['avg_ph_prediction'].hist(alpha=0.5, label='Predicted', bins=30)
merged_df['true_ph'].hist(alpha=0.5, label='True', bins=30)
plt.legend()
plt.title("Distribution of Predicted vs True pH")
plt.show()

# MA%E
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1) Load the data
df = pd.read_csv('')

# 2) Compute basic errors
df['abs_error'] = (df['avg_ph_prediction'] - df['true_ph']).abs()
df['squared_error'] = (df['avg_ph_prediction'] - df['true_ph'])**2

# 3) Standard metrics
mae  = df['abs_error'].mean()
rmse = np.sqrt(df['squared_error'].mean())
bias = (df['avg_ph_prediction'] - df['true_ph']).mean()
r2   = r2_score(df['true_ph'], df['avg_ph_prediction'])

# 4) Mean Absolute Percentage Error (MAPE)
#    avoid division by zero
mask = df['true_ph'] != 0
df.loc[mask, 'pct_error'] = (df.loc[mask, 'abs_error'] / df.loc[mask, 'true_ph']).abs() * 100
mape = df.loc[mask, 'pct_error'].mean()

# 5) Symmetric MAPE (sMAPE)
df['smape'] = 2 * df['abs_error'] / (df['avg_ph_prediction'].abs() + df['true_ph'].abs()) * 100
smape = df['smape'].mean()

# 6) Print metrics
print(f"{'Metric':<40} {'Value'}")
print("-" * 55)
print(f"{'Mean Absolute Error (MAE)':<40} {mae:.4f}")
print(f"{'Root Mean Squared Error (RMSE)':<40} {rmse:.4f}")
print(f"{'Mean Signed Error (Bias)':<40} {bias:.4f}")
print(f"{'R² Score':<40} {r2:.4f}")
print(f"{'Mean Absolute Percentage Error (MAPE)':<40} {mape:.2f}%")
print(f"{'Symmetric MAPE (sMAPE)':<40} {smape:.2f}%")

# 7) Plot % error distribution
plt.figure(figsize=(8,5))
plt.hist(df['pct_error'].dropna(), bins=30, edgecolor='black')
plt.title("Distribution of Percentage Errors")
plt.xlabel("Absolute % Error")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# classify of acidemia or non-acidemia
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# pH threshold
threshold = 7.2

# classification labels
merged_df['true_class'] = merged_df['true_ph'] <= threshold
merged_df['pred_class'] = merged_df['avg_ph_prediction'] <= threshold


# Confusion Matrix
cm = confusion_matrix(merged_df['true_class'], merged_df['pred_class'])
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(merged_df['true_class'], merged_df['pred_class'])
print(f"\nAccuracy: {accuracy:.4f}")

# Precision, Recall, F1-score
report = classification_report(merged_df['true_class'], merged_df['pred_class'], target_names=["Normal", "Acidemia"])
print("\nClassification Report:")
print(report)


sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Normal", "Pred Acidemia"], yticklabels=["True Normal", "True Acidemia"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt


# Hypoxia‐class
def assign_hypoxia_label(ph):
    if pd.isna(ph):
        return -1            # uninterpretable
    elif ph > 7.2:
        return 1             # No hypoxia
    elif 7.1 < ph <= 7.2:
        return 2             # Mild hypoxia
    else:
        return 3             # Severe hypoxia

merged_df['true_class'] = merged_df['true_ph'].apply(assign_hypoxia_label)
merged_df['pred_class'] = merged_df['pred_class'].astype(int)  # ensure int

# drop any uninterpretable if present
merged_df = merged_df[merged_df['true_class'] > 0]

y_true = merged_df['true_class']
y_pred = merged_df['pred_class']

# 5) Compute metrics
acc    = accuracy_score(y_true, y_pred)
bacc   = balanced_accuracy_score(y_true, y_pred)
cm     = confusion_matrix(y_true, y_pred, labels=[1,2,3])
report = classification_report(
    y_true, y_pred,
    labels=[1,2,3],
    target_names=['No hypoxia','Mild','Severe'],
    digits=3
)
# mean absolute class‐step error
mean_cls_err = np.mean(np.abs(y_pred - y_true))

print(f"Overall accuracy:             {acc:.3f}")
print(f"Balanced accuracy:            {bacc:.3f}")
print(f"Mean |class_pred–class_true|: {mean_cls_err:.3f}\n")
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification Report:\n", report)

# 6) Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['No','Mild','Sev'],
    yticklabels=['No','Mild','Sev']
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Expert Step 4 vs. True Hypoxia Class")
plt.show()

### BALANCED ACCURACY

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# pH threshold classification
threshold = 7.2
merged_df['true_class'] = merged_df['true_ph'] <= threshold   # True=acidemia
merged_df['pred_class'] = merged_df['avg_ph_prediction'] <= threshold

# 1) Confusion Matrix
cm = confusion_matrix(merged_df['true_class'], merged_df['pred_class'])
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:\n", cm)

# 2) Standard accuracy
acc = accuracy_score(merged_df['true_class'], merged_df['pred_class'])
print(f"\nOverall accuracy: {acc:.3f}")

# 3) Balanced accuracy
bal_acc = balanced_accuracy_score(
    merged_df['true_class'],
    merged_df['pred_class']
)
print(f"Balanced accuracy: {bal_acc:.3f}")

# 4) Sensitivity & specificity
sensitivity = tp / (tp + fn) if (tp + fn)>0 else float('nan')
specificity = tn / (tn + fp) if (tn + fp)>0 else float('nan')
print(f"Sensitivity (TPR): {sensitivity:.3f}")
print(f"Specificity (TNR): {specificity:.3f}")

# 5) Full classification report (macro‐averaged)
print("\nClassification Report:")
print(classification_report(
    merged_df['true_class'],
    merged_df['pred_class'],
    target_names=["Normal","Acidemia"],
    digits=3
))

# 6) Plotting the confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Norm→Norm","Norm→Aci"],
    yticklabels=["True Norm","True Aci"]
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


## SINGLE CUTOFF
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# ---- choose cutoff for analysis ----
cutoff = 7.20

# clean / select needed columns
gpt_df = merged_df.dropna(subset=['true_ph','avg_ph_prediction']).copy()

# Binary ground truth at this cutoff (or reuse gpt_df['true_class'])
y_true = (gpt_df['true_ph'] <= cutoff).astype(int).values

# Score: larger = more likely acidemia (invert pH so lower pH ⇒ higher risk)
gpt_score = (cutoff - gpt_df['avg_ph_prediction']).values

# AUROC
auc_val = roc_auc_score(y_true, gpt_score)
print(f"GPT-4o AUROC (pH ≤ {cutoff:.2f}): {auc_val:.3f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_true, gpt_score)
plt.figure(figsize=(4.8,4.2))
plt.plot(fpr, tpr, label=f"GPT-4o (AUC = {auc_val:.3f})")
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel("1 − Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title(f"ROC — pH ≤ {cutoff:.2f}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Precision–Recall curve (optional)
prec, rec, _ = precision_recall_curve(y_true, gpt_score)
ap = average_precision_score(y_true, gpt_score)
plt.figure(figsize=(4.8,4.2))
plt.plot(rec, prec, label=f"GPT-4o (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall — pH ≤ {cutoff:.2f}")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

## 4 DIFF THRESHOLD CUTOFF
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

cutoffs = [7.05, 7.10, 7.15, 7.20]

plt.figure(figsize=(5.6,4.4))
for c in cutoffs:
    y_true = (gpt_df['true_ph'] <= c).astype(int).values
    score  = (c - gpt_df['avg_ph_prediction']).values  # invert pH at this cutoff
    fpr, tpr, _ = roc_curve(y_true, score)
    auc_val = roc_auc_score(y_true, score)
    plt.plot(fpr, tpr, label=f"pH ≤ {c:.2f}  (AUC {auc_val:.3f})")

plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel("1 − Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title("GPT-4o ROC at Different pH Cutoffs")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

# do the same but do LLM v Real v OBs

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load both files
ph_df = pd.read_csv('')  # has PID, true_ph, avg_ph_prediction
expert_df = pd.read_csv('')  # has rec_id, step, consensus_annotation

# Keep only step 4 expert annotations
expert_df = expert_df[expert_df['step'] == 4].rename(
    columns={'rec_id': 'PID', 'consensus_annotation': 'expert_annotation_step4'}
)

# Merge expert labels into the ph_df
merged_df = ph_df.merge(expert_df[['PID', 'expert_annotation_step4']], on='PID', how='left')

# Define the unified ground truth label from true_ph
def assign_hypoxia_label(ph):
    if pd.isna(ph):
        return -1  # uninterpretable
    elif ph > 7.2:
        return 1  # No hypoxia
    elif 7.1 < ph <= 7.2:
        return 2  # Mild hypoxia
    else:
        return 3  # Severe hypoxia

# Assign ground truth based on true_ph
merged_df['eval_step4_true'] = merged_df['true_ph'].apply(assign_hypoxia_label)

# Compare expert and model against that ground truth
merged_df['expert_matches_truth'] = merged_df['expert_annotation_step4'] == merged_df['eval_step4_true']

# map predicted pH into categories using the same truth function
merged_df['eval_step4_pred'] = merged_df['avg_ph_prediction'].apply(assign_hypoxia_label)
merged_df['model_matches_truth'] = merged_df['eval_step4_pred'] == merged_df['eval_step4_true']

print(merged_df[['PID', 'true_ph', 'avg_ph_prediction', 'eval_step4_true',
                 'expert_annotation_step4', 'eval_step4_pred',
                 'expert_matches_truth', 'model_matches_truth']].head())


## visualize ##
# Bar plot: % correct by class
acc_by_class = merged_df.groupby('eval_step4_true')[['expert_matches_truth', 'model_matches_truth']].mean().reset_index()
acc_by_class = acc_by_class.rename(columns={
    'expert_matches_truth': 'Expert Accuracy',
    'model_matches_truth': 'Model Accuracy',
    'eval_step4_true': 'True Hypoxia Class'
})

acc_by_class = acc_by_class.melt(id_vars='True Hypoxia Class', var_name='Source', value_name='Accuracy')

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=acc_by_class, x='True Hypoxia Class', y='Accuracy', hue='Source')
plt.title("Accuracy by Hypoxia Class (Based on True pH)")
plt.ylabel("Accuracy (Proportion Correct)")
plt.xlabel("Ground Truth Hypoxia Category")
plt.ylim(0, 1)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# count of examples per hypoxia class
count_df = merged_df['eval_step4_true'].value_counts().sort_index().reset_index()
count_df.columns = ['Hypoxia Class', 'Count']

# accuracy per class
acc_by_class = merged_df.groupby('eval_step4_true')[['expert_matches_truth', 'model_matches_truth']].mean().reset_index()
acc_by_class.columns = ['Hypoxia Class', 'Expert Accuracy', 'Model Accuracy']
acc_melted = acc_by_class.melt(id_vars='Hypoxia Class', var_name='Source', value_name='Accuracy')

# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: count of cases per class
sns.barplot(data=count_df, x='Hypoxia Class', y='Count', ax=axes[0], color='gray')
axes[0].set_title("Number of Cases per Hypoxia Class")
axes[0].set_ylim(0, count_df['Count'].max() + 10)

# Right plot: model vs expert accuracy per class
sns.barplot(data=acc_melted, x='Hypoxia Class', y='Accuracy', hue='Source', ax=axes[1])
axes[1].set_title("Accuracy by Hypoxia Class")
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()

### Multiclass confusion matrix

# Generate confusion matrix
cm = confusion_matrix(merged_df['eval_step4_true'], merged_df['eval_step4_pred'], labels=[1, 2, 3])

# display with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Hypoxia", "Mild", "Severe"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix: Model vs True Hypoxia Class")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

"""### exploring use of a class-weighted model"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define features and target
X = df[["baseline_fhr", "var_std", "accel_present", "decel_type", ...]]
y = df["eval_step4_true"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train class-weighted model
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["No Hypoxia", "Mild", "Severe"]))

"""#### are some steps more accurate than others?

### futher eval into disagreement
"""

import matplotlib.pyplot as plt

## --- flag cases with disagreements --- #
merged_df['accel_mismatch'] = merged_df['accel_calc'] != merged_df['accel_LLM']
merged_df['decel_type_mismatch'] = merged_df['decel_type_calc'] != merged_df['decel_type_LLM']
merged_df['uc_class_mismatch'] = merged_df['uc_class_calc'] != merged_df['uc_class_LLM']
merged_df['var_category_mismatch'] = merged_df['var_category_calc'] != merged_df['var_category_LLM']

print("Acceleration mismatches:", merged_df['accel_mismatch'].sum())
print("Deceleration type mismatches:", merged_df['decel_type_mismatch'].sum())
print("UC class mismatches:", merged_df['uc_class_mismatch'].sum())
print("Variability category mismatches:", merged_df['var_category_mismatch'].sum())

accel_disagreements = merged_df[merged_df['accel_mismatch']][['PID', 'step', 'accel_calc', 'accel_LLM']]
decel_type_disagreements = merged_df[merged_df['decel_type_mismatch']][['PID', 'step', 'decel_type_calc', 'decel_type_LLM']]


# plot disagreements + agreements

# acceleration
merged_df['accel_match'] = merged_df['accel_mismatch'].map({True: 'Mismatch', False: 'Match'})
merged_df['accel_match'].value_counts().plot(kind='bar')
plt.title('Acceleration Match vs Mismatch')
plt.ylabel('Number of Cases')
plt.show()

# decel type
merged_df['decel_type_match'] = merged_df['decel_type_mismatch'].map({True: 'Mismatch', False: 'Match'})
merged_df['decel_type_match'].value_counts().plot(kind='bar')
plt.title('Deceleration Type Match vs Mismatch')
plt.ylabel('Number of Cases')
plt.show()


### LLAMA RESULTS

import pandas as pd

df = pd.read_csv('')

df = df.rename(columns={'uterine_contractions_frequency': 'uc_freq_per_10min'})
df['baseline_fhr'] = df['baseline_fhr'].str.replace(' bpm', '', regex=False).astype(float)

# --- Convert to float ---
df['baseline_fhr'] = df['baseline_fhr'].astype(float)
df['baseline_variability_std_dev'] = df['baseline_variability_std_dev'].astype(float)
# df['uc_freq_per_10min'] = df['uc_freq_per_10min'].astype(float)

# make sure it’s a string first
df['uc_freq_per_10min'] = (
    df['uc_freq_per_10min']
      .astype(str)
      .str.replace(r'\s*per\s*10\s*min', '', regex=True)
      .astype(float)
)
df['uc_freq_per_10min'] = (
    df['uc_freq_per_10min']
      .astype(str)
      .str.extract(r'(\d+\.?\d*)')    # grab integer or decimal
      .astype(float)
)

# --- Convert to categorical variables ---
# baseline_variability_category
var_type_map = {'Absent': 0, 'Minimal': 1, 'Moderate': 2, 'Marked': 3}
df['baseline_variability_category'] = df['baseline_variability_category'].map(var_type_map).astype('category')

# accelerations (0 = absent; 1 = present)
df['accelerations'] = df['accelerations'].map({'Absent': 0, 'Present': 1}).astype('category')

# decelerations_present (0 = no; 1 = yes)
df['decelerations_present'] = df['decelerations_present'].map({'No': 0, 'Yes': 1}).astype('category')

# decelerations_type (0 = early; 1 = late; 2 = variable; 3 = prolonged; 4 = sinusoidal)
dec_type_map = {'Early': 0, 'Late': 1, 'Variable': 2, 'Prolonged': 3, 'Sinusoidal': 4}
df['decelerations_type'] = df['decelerations_type'].map(dec_type_map).astype('category')

# decelerations_assoc_with_uc (0 = no; 1 = yes)
df['decelerations_associated_with_contractions'] = df['decelerations_associated_with_contractions'].map({'No': 0, 'Yes': 1}).astype('category')

# uc_classification (0 = normal; 1 = tachysystole)
df['uterine_contractions_classification'] = df['uterine_contractions_classification'].map({'Normal': 0, 'Tachysystole': 1}).astype('category')


# --- Rename columns ---
df = df.rename(columns={
    'baseline_fhr': 'baseline_fhr_y',
    'baseline_variability_std_dev': 'baseline_variability_std_dev_y',
    'baseline_variability_category': 'baseline_variability_category_y',
    'accelerations': 'accelerations_y',
    'decelerations_present': 'decelerations_present_y',
    'decelerations_type': 'decelerations_type_y',
    'decelerations_associated_with_contractions': 'decelerations_associated_with_contractions_y',
    'uc_freq_per_10min': 'uterine_contractions_frequency_y',
    'uterine_contractions_classification': 'uterine_contractions_classification_y',
    'reasoning': 'reasoning_y',
    'figo_classification': 'figo_classification_y',
    'umbilical_artery_ph_classification': 'umbilical_artery_ph_classification_y',
    'umbilical_artery_ph_prediction': 'umbilical_artery_ph_prediction_y'
})


df.to_csv('', index=False)

## merge all dfs

import pandas as pd

manual_df = pd.read_csv('')
llm_df    = pd.read_csv('')
LLAMA_path = ''

manual_df = pd.read_csv(manual_path)
llm_df    = pd.read_csv(llm_path)
LLAMA_df  = pd.read_csv(LLAMA_path)

# first merge manual & llm
temp = pd.merge(
    manual_df,
    llm_df,
    on=['PID', 'step'],
    how='inner'
)

# then merge the result with LLAMA
merged_df = pd.merge(
    temp,
    LLAMA_df,
    on=['PID', 'step'],
    how='inner'
)
# save
out_path = ''
merged_df.to_csv(out_path, index=False)
print(f"Saved merged CSV to {out_path!r}")


import pandas as pd

merged_df_LLAMA = pd.read_csv('')

def strip_and_float(col, unit_regex):
    return pd.to_numeric(
        col.astype(str).str.replace(unit_regex, '', regex=True),
        errors='coerce'
    )

merged_df_LLAMA['baseline_fhr_y'] = strip_and_float(
    merged_df_LLAMA['baseline_fhr_y'], r'\s*bpm$'
)
merged_df_LLAMA['baseline_variability_std_dev_y'] = pd.to_numeric(
    merged_df_LLAMA['baseline_variability_std_dev_y'], errors='coerce'
)
merged_df_LLAMA['uterine_contractions_frequency_y'] = strip_and_float(
    merged_df_LLAMA['uterine_contractions_frequency_y'], r'\s*per\s*10\s*min$'
)

merged_df_LLAMA['fhr_diff']     = (merged_df_LLAMA['baseline_fhr_calc']      - merged_df_LLAMA['baseline_fhr_y']).abs()
merged_df_LLAMA['var_std_diff'] = (merged_df_LLAMA['var_std_calc']            - merged_df_LLAMA['baseline_variability_std_dev_y']).abs()
merged_df_LLAMA['uc_freq_diff'] = (merged_df_LLAMA['uc_freq_per_10min_calc'] - merged_df_LLAMA['uterine_contractions_frequency_y']).abs()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# compute MAEs & correlations
mae_fhr     = mean_absolute_error(df['baseline_fhr_calc'], df['baseline_fhr_y'])
mae_var_std = mean_absolute_error(df['var_std_calc'], df['baseline_variability_std_dev_y'])
mae_uc      = mean_absolute_error(df['uc_freq_per_10min_calc'], df['uterine_contractions_frequency_y'])

corr_fhr     = df[['baseline_fhr_calc','baseline_fhr_y']].corr().iloc[0,1]
corr_var_std = df[['var_std_calc','baseline_variability_std_dev_y']].corr().iloc[0,1]
corr_uc      = df[['uc_freq_per_10min_calc','uterine_contractions_frequency_y']].corr().iloc[0,1]

# make a summary table
summary = pd.DataFrame({
    'Metric':      ['Baseline FHR','Variability Std Dev','UC Freq per 10 min'],
    'MAE':         [mae_fhr, mae_var_std, mae_uc],
    'Correlation': [corr_fhr, corr_var_std, corr_uc]
}).round(3)
print(summary)

# scatter + regression plots
fig, axes = plt.subplots(1,3, figsize=(18,5))
sns.regplot(
    data=df,
    x='baseline_fhr_calc',
    y='baseline_fhr_y',
    ax=axes[0],
    scatter_kws={'alpha': 0.4},
)
axes[0].set(title='Baseline FHR', xlabel='Manual (calc)', ylabel='LLAMA')

sns.regplot(
    data=df,
    x='var_std_calc',
    y='baseline_variability_std_dev_y',
    ax=axes[1],
    scatter_kws={'alpha': 0.4},
)
axes[1].set(title='Var Std Dev', xlabel='Manual (calc)', ylabel='LLAMA')

sns.regplot(
    data=df,
    x='uc_freq_per_10min_calc',
    y='uterine_contractions_frequency_y',
    ax=axes[2],
    scatter_kws={'alpha': 0.4},
)
axes[2].set(title='UC Freq per 10 min', xlabel='Manual (calc)', ylabel='LLAMA')
plt.tight_layout()
plt.show()

# histograms of absolute errors
fig, axes = plt.subplots(1,3, figsize=(18,5))
axes[0].hist(df['fhr_diff'], bins=20, edgecolor='black')
axes[0].set(title=f'FHR |Δ| (MAE={mae_fhr:.2f})', xlabel='|Difference|', ylabel='Count')
axes[1].hist(df['var_std_diff'], bins=20, edgecolor='black')
axes[1].set(title=f'Var Std |Δ| (MAE={mae_var_std:.2f})', xlabel='|Difference|')
axes[2].hist(df['uc_freq_diff'], bins=20, edgecolor='black')
axes[2].set(title=f'UC Freq |Δ| (MAE={mae_uc:.2f})', xlabel='|Difference|')
plt.tight_layout()
plt.show()

# --- Categorical Variables --- #

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

merged_df = pd.read_csv('')

# proportion of agreement
(merged_df['accel_calc'] == merged_df['accelerations_y']).mean()
(merged_df['var_category_calc'] == merged_df['baseline_variability_category_y']).mean()
(merged_df['decel_present_calc'] == merged_df['decelerations_present_y']).mean()
(merged_df['decel_type_calc'] == merged_df['decelerations_type_y']).mean()
(merged_df['decel_assoc_uc_calc'] == merged_df['decelerations_associated_with_contractions_y']).mean()
(merged_df['uc_class_calc'] == merged_df['uterine_contractions_classification_y']).mean()

# confusion matrix
# confusion_matrix(merged_df['var_category_calc'], merged_df['var_category_LLM'])

# show comparison
comparison_results = {
    'accel_agreement': (merged_df['accel_calc'] == merged_df['accelerations_y']).mean(),
    'var_category_agreement': (merged_df['var_category_calc'] == merged_df['baseline_variability_category_y']).mean(),
    'decel_present_agreement': (merged_df['decel_present_calc'] == merged_df['decelerations_present_y']).mean(),
    'decel_type_agreement': (merged_df['decel_type_calc'] == merged_df['decelerations_type_y']).mean(),
    'decel_assoc_uc_agreement': (merged_df['decel_assoc_uc_calc'] == merged_df['decelerations_associated_with_contractions_y']).mean(),
    'uc_class_agreement': (merged_df['uc_class_calc'] == merged_df['uterine_contractions_classification_y']).mean()
}

pd.Series(comparison_results).sort_values(ascending=False)

## calculate pHs
import pandas as pd
import matplotlib.pyplot as plt

# Load the merged CSV
df = pd.read_csv('')

# Group by PID and calculate the average pH prediction
ph_avg_df = df.groupby('PID')['umbilical_artery_ph_prediction_y'].mean().reset_index()

# rename column to avg_ph_prediction'
ph_avg_df = ph_avg_df.rename(columns={'umbilical_artery_ph_prediction_y': 'avg_ph_prediction_y'})

# save as csv
ph_avg_df.to_csv('', index=False)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# compare the values
merged_df = pd.read_csv('')

# Mean Absolute Error (
mae = merged_df['abs_error'].mean()

# RMSE
rmse = np.sqrt(((merged_df['true_ph'] - merged_df['avg_ph_prediction_y']) ** 2).mean())

# Mean Signed Error
bias = (merged_df['avg_ph_prediction_y'] - merged_df['true_ph']).mean()

# R² Score
r2 = r2_score(merged_df['true_ph'], merged_df['avg_ph_prediction_y'])

# Print metrics in a clean table format
print(f"{'Metric':<30} {'Value'}")
print("-" * 40)
print(f"{'Mean Absolute Error (MAE)':<30} {mae:.4f}")
print(f"{'Root Mean Squared Error (RMSE)':<30} {rmse:.4f}")
print(f"{'Mean Signed Error (Bias)':<30} {bias:.4f}")
print(f"{'R² Score':<30} {r2:.4f}")

# True vs Predicted
plt.scatter(merged_df['true_ph'], merged_df['avg_ph_prediction_y'], alpha=0.6)
plt.plot([6.8, 7.4], [6.8, 7.4], color='red', linestyle='--')
plt.xlabel("True pH")
plt.ylabel("Predicted pH")
plt.title("True vs Predicted pH")
plt.grid(True)
plt.show()

# errors
merged_df['abs_error'].hist(bins=30)
plt.xlabel("Absolute Error")
plt.title("Distribution of Prediction Errors")
plt.show()

# distrbiution of LLM v true pH
merged_df['avg_ph_prediction_y'].hist(alpha=0.5, label='Predicted', bins=30)
merged_df['true_ph'].hist(alpha=0.5, label='True', bins=30)
plt.legend()
plt.title("Distribution of Predicted vs True pH")
plt.show()

### OVERALL AND BALANCED ACCURACY OF LLAMA

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# pH threshold
threshold = 7.2

# classification labels
merged_df['true_class'] = merged_df['true_ph'] <= threshold
merged_df['pred_class'] = merged_df['avg_ph_prediction_y'] <= threshold

# Confusion Matrix
cm = confusion_matrix(merged_df['true_class'], merged_df['pred_class'])
print("Confusion Matrix:")
print(cm)

# Overall Accuracy
accuracy = accuracy_score(merged_df['true_class'], merged_df['pred_class'])
print(f"\nOverall Accuracy:  {accuracy:.4f}")

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(merged_df['true_class'], merged_df['pred_class'])
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# Precision, Recall, F1-score
report = classification_report(merged_df['true_class'], merged_df['pred_class'], target_names=["Normal", "Acidemia"])
print("\nClassification Report:")
print(report)

# Heatmap of Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Normal", "Pred Acidemia"],
            yticklabels=["True Normal", "True Acidemia"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# === OVERALL AND BALANCED ACCURACY OF LLAMA (3-class) ===
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

merged_df = pd.read_csv('')

# 1) Helper: map pH -> class {1: No hypoxia, 2: Mild, 3: Severe}
def ph_to_class(ph):
    if pd.isna(ph):
        return np.nan
    if ph > 7.2:
        return 1      # No hypoxia
    elif 7.1 <= ph <= 7.2:
        return 2      # Mild hypoxia
    else:
        return 3      # Severe hypoxia

# 2) Build class labels from ground truth and LLAMA prediction
# Expect columns: 'true_ph' and 'avg_ph_prediction_y'
merged_df['true_class'] = merged_df['true_ph'].apply(ph_to_class)
merged_df['pred_class'] = merged_df['avg_ph_prediction_y'].apply(ph_to_class)

# 3) Drop rows with missing classes (if any), cast to int
_eval = merged_df.dropna(subset=['true_class','pred_class']).copy()
y_true = _eval['true_class'].astype(int).values
y_pred = _eval['pred_class'].astype(int).values

# 4) Confusion Matrix (rows=true, cols=pred) with fixed label order
labels = [1, 2, 3]
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("Confusion Matrix (rows=true, cols=pred; 1=No, 2=Mild, 3=Severe):")
print(cm)

# 5) Overall & Balanced Accuracy
acc = accuracy_score(y_true, y_pred)
bacc = balanced_accuracy_score(y_true, y_pred)  # macro-averaged recall
print(f"\nOverall Accuracy:   {acc:.3f}")
print(f"Balanced Accuracy:  {bacc:.3f}")

# 6) Classification Report
target_names = ["No hypoxia (1)", "Mild (2)", "Severe (3)"]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=3))

# 7) Mean |Δclass| for ordinal error
mean_abs_class_err = np.mean(np.abs(y_pred - y_true))
print(f"Mean |class_pred – class_true|: {mean_abs_class_err:.3f}")

plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=target_names, yticklabels=target_names
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("LLAMA: Confusion Matrix (3-class pH thresholds)")
plt.tight_layout()
plt.show()

## Expert Annotation Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- Build merged_df and map expert labels to binary ---
# Map expert 3-class labels to binary: 1=No hypoxia -> False; 2/3=Hypoxia -> True
expert_to_binary = {1: False, 2: True, 3: True}

merged_df = (
    ph_df.merge(expert_df, on='PID', how='inner')
         .assign(pred_class=lambda d: d['pred_class'].map(expert_to_binary))
         .dropna(subset=['true_ph','pred_class'])
)

# pH threshold classification
threshold = 7.2
merged_df['true_class'] = merged_df['true_ph'] <= threshold   # True=acidemia
# pred_class already set from expert labels (boolean)

# 1) Confusion Matrix
cm = confusion_matrix(merged_df['true_class'], merged_df['pred_class'], labels=[False, True])
tn, fp = cm[0,0], cm[0,1]
fn, tp = cm[1,0], cm[1,1]
print("Confusion Matrix:\n", cm)

# 2) Standard accuracy
acc = accuracy_score(merged_df['true_class'], merged_df['pred_class'])
print(f"\nOverall accuracy: {acc:.3f}")

# 3) Balanced accuracy
bal_acc = balanced_accuracy_score(
    merged_df['true_class'],
    merged_df['pred_class']
)
print(f"Balanced accuracy: {bal_acc:.3f}")

# 4) Sensitivity & specificity
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')  # TPR, recall on Hypoxia
specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')  # TNR, recall on Normal
print(f"Sensitivity (TPR): {sensitivity:.3f}")
print(f"Specificity (TNR): {specificity:.3f}")

# 5) Full classification report (macro‐averaged)
print("\nClassification Report:")
print(classification_report(
    merged_df['true_class'],
    merged_df['pred_class'],
    labels=[False, True],
    target_names=["Normal","Acidemia"],
    digits=3
))

# 6) Plotting the confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Norm→Norm","Norm→Aci"],
    yticklabels=["True Norm","True Aci"]
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

# 3) Hypoxia‐class helper
def assign_hypoxia_label(ph):
    if pd.isna(ph):
        return -1            # uninterpretable
    elif ph > 7.2:
        return 1             # No hypoxia
    elif 7.1 < ph <= 7.2:
        return 2             # Mild hypoxia
    else:
        return 3             # Severe hypoxia

df = (
    ph_df
      .merge(expert_df, on='PID', how='inner')
)
df['true_class'] = df['true_ph'].apply(assign_hypoxia_label)
df['pred_class'] = df['pred_class'].astype(int)  # ensure int

# drop any uninterpretable if present
df = df[df['true_class'] > 0]

y_true = df['true_class']
y_pred = df['pred_class']

acc    = accuracy_score(y_true, y_pred)
bacc   = balanced_accuracy_score(y_true, y_pred)
cm     = confusion_matrix(y_true, y_pred, labels=[1,2,3])
report = classification_report(
    y_true, y_pred,
    labels=[1,2,3],
    target_names=['No hypoxia','Mild','Severe'],
    digits=3
)
mean_cls_err = np.mean(np.abs(y_pred - y_true))

print(f"Overall accuracy:             {acc:.3f}")
print(f"Balanced accuracy:            {bacc:.3f}")
print(f"Mean |class_pred–class_true|: {mean_cls_err:.3f}\n")
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification Report:\n", report)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['No','Mild','Sev'],
    yticklabels=['No','Mild','Sev']
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Expert Step 4 vs. True Hypoxia Class")
plt.show()

### AUROC (LLAMA)

from sklearn.metrics import roc_auc_score, roc_curve

threshold = 7.2

# Binary ground truth as 0/1
y_true = merged_df['true_class'].astype(int).values

# LLaMA continuous score: lower predicted pH -> higher risk
llama_score = (threshold - merged_df['avg_ph_prediction_y']).values
# (equivalently: -merged_df['avg_ph_prediction_y'].values)

# AUROC
auc_llama = roc_auc_score(y_true, llama_score)
print(f"LLaMA AUROC (pH ≤ {threshold}): {auc_llama:.3f}")

# (optional) ROC curve points if you want to plot
fpr, tpr, _ = roc_curve(y_true, llama_score)


### PLOTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

cutoff = 7.2

# ground truth (0/1) and LLaMA score (higher = more acidemia)
y_true = merged_df['true_class'].astype(int).values
llama_score = (cutoff - merged_df['avg_ph_prediction_y']).values  # invert pH

# ROC
fpr, tpr, _ = roc_curve(y_true, llama_score)
auc_val = roc_auc_score(y_true, llama_score)

plt.figure(figsize=(4.5,4))
plt.plot(fpr, tpr, label=f"LLaMA (AUC = {auc_val:.3f})")
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel("1 − Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title(f"ROC — pH ≤ {cutoff}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

### PLOTS w/ THRESHOLDS
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

cutoffs = [7.05, 7.10, 7.15, 7.20]

plt.figure(figsize=(5.2,4.5))
for c in cutoffs:
    y_true = (merged_df['true_ph'] <= c).astype(int).values
    score  = (c - merged_df['avg_ph_prediction_y']).values   # invert pH at this cutoff
    fpr, tpr, _ = roc_curve(y_true, score)
    auc_val = roc_auc_score(y_true, score)
    plt.plot(fpr, tpr, label=f"pH ≤ {c:.2f}  (AUC {auc_val:.3f})")

plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel("1 − Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title("LLaMA ROC at Different pH Cutoffs")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()






