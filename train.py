"""
Cardiac Mortality Prediction — Model Training Script
Run this first: python train.py
It will save model.pkl and feature_info.json used by the API.
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, brier_score_loss, average_precision_score
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("CARDIAC MORTALITY PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("dataset.csv", sep=",",          # change sep="," if your file is comma-separated
                 na_values=[' ', '', 'NULL', 'null', 'NA', 'N/A', 'NaN', '#N/A'])

# Convert ALL columns to numeric where possible — blank strings become NaN
# errors='coerce' turns non-numeric values into NaN (errors='ignore' removed in pandas 2.x)
for col in df.columns:
    converted = pd.to_numeric(df[col], errors='coerce')
    # Only replace if conversion was mostly successful (preserves date/text columns)
    if converted.notna().sum() >= df[col].notna().sum() * 0.5:
        df[col] = converted

# Strip any remaining whitespace-only strings → NaN
# pandas >= 2.1 uses .map(); older versions use .applymap()
_clean = lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x
try:
    df = df.map(_clean)
except AttributeError:
    df = df.applymap(_clean)

print(f"\n✓ Loaded {len(df)} records, {df.shape[1]} columns")
print(f"  Mortality rate: {df['in_hospital_mortality'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 2. SELECT CLINICALLY RELEVANT FEATURES
#    (Based on your feature selection + literature)
# ─────────────────────────────────────────────
FEATURES = [
    # Demographics
    "patient_age", "gender_id", "BMI",

    # Pre-op risk factors
    "diabetes", "hypertension", "dyslipidemia",
    "Active_tobacco_use", "f_history_cad",
    "Cerebovascular_disease", "chronic_lung_disease",
    "dialysis",

    # Cardiac status
    "ejection_fraction", "NYHA_class",
    "congestive_heart_failure_A",
    "cardiogenic_shock", "resuscitation",
    "myocardial_infarction",
    "arrhythmia", "AFibFlutter",
    "pulmonary_artery_hypertension",
    "Coronaries_diseased", "left_main_disease",
    "Aortic_regurgitation", "Mitral_regurgitation",

    # Labs
    "last_hematocrit", "last_cretenine_preop", "BPsystolic",

    # Intra-op
    "perfusion_time_min", "cross_clamp_time_min",
    "IABP", "intraop_blood_products",
    "Total_bypasses_grafted",

    # Post-op early indicators
    "initial_hours_ventilated", "initial_icu_hours",
    "drainage_at_12_hours", "drainage_at_24_hours",
    "post_op_creatinine", "reintubated_hospital_stay",
]

TARGET = "in_hospital_mortality"

# Keep only columns that actually exist in your data
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\n✓ Using {len(FEATURES)} features")

# Fix column name typo in source data
rename_map = {
    "Dyslipidemia": "dyslipidemia",
    "last_cretenine_preop": "last_cretenine_preop",  # keep as-is
}
df.rename(columns=rename_map, inplace=True)
FEATURES = [f for f in FEATURES if f in df.columns]

# ─────────────────────────────────────────────
# 3. CLEAN DATA
# ─────────────────────────────────────────────
# Replace sentinel values (1/1/1753 dates become NaN already as strings)
# Clip extreme numeric outliers
numeric_cols = df[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    q99 = df[col].quantile(0.99)
    q01 = df[col].quantile(0.01)
    df[col] = df[col].clip(lower=q01, upper=q99)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"  Missing values per feature (top 10):")
missing = X.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(10).to_string())

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5. BUILD PIPELINE
#    Impute → Scale → Select best features → Model
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ("imputer",   SimpleImputer(strategy="median")),
    ("scaler",    StandardScaler()),
    ("selector",  SelectKBest(f_classif, k=min(25, len(FEATURES)))),
    ("model",     RandomForestClassifier(
                      n_estimators=300,
                      max_depth=6,
                      min_samples_leaf=5,
                      class_weight="balanced",   # handles class imbalance
                      random_state=42,
                      n_jobs=-1
                  )),
])

# ─────────────────────────────────────────────
# 6. CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n── Cross-validation (5-fold, stratified) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv,
                          scoring="roc_auc", n_jobs=-1)
print(f"  AUC-ROC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ─────────────────────────────────────────────
# 7. FINAL FIT & TEST EVALUATION
# ─────────────────────────────────────────────
pipeline.fit(X_train, y_train)
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

auc   = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)
ap    = average_precision_score(y_test, y_prob)

print("\n── Test set performance ──")
print(f"  AUC-ROC : {auc:.3f}")
print(f"  Avg Prec: {ap:.3f}")
print(f"  Brier   : {brier:.3f}  (lower is better)")
print("\n── Classification report ──")
print(classification_report(y_test, y_pred, target_names=["Survived", "Died"]))
print("── Confusion matrix ──")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (top 15)
# ─────────────────────────────────────────────
selector   = pipeline.named_steps["selector"]
model      = pipeline.named_steps["model"]
sel_mask   = selector.get_support()
sel_feats  = np.array(FEATURES)[sel_mask]
importances = model.feature_importances_
top_idx    = np.argsort(importances)[::-1][:15]

print("\n── Top 15 features by importance ──")
for i in top_idx:
    print(f"  {sel_feats[i]:<35} {importances[i]:.4f}")

# ─────────────────────────────────────────────
# 9. SAVE MODEL & FEATURE INFO
# ─────────────────────────────────────────────
joblib.dump(pipeline, "model.pkl")
print("\n✓ Saved model.pkl")

feature_info = {
    "features": FEATURES,
    "target": TARGET,
    "model_auc": round(auc, 3),
    "cv_auc_mean": round(float(cv_auc.mean()), 3),
    "cv_auc_std":  round(float(cv_auc.std()), 3),
    "feature_descriptions": {
        "patient_age":              "Age (years)",
        "gender_id":                "Gender (1=Male, 2=Female)",
        "BMI":                      "Body Mass Index",
        "diabetes":                 "Diabetes (0=No, 1=Type1, 2=Type2)",
        "hypertension":             "Hypertension (0/1)",
        "dyslipidemia":             "Dyslipidemia (0/1)",
        "Active_tobacco_use":       "Active Tobacco Use (0/1)",
        "f_history_cad":            "Family History CAD (0/1)",
        "Cerebovascular_disease":   "Cerebrovascular Disease (0/1)",
        "chronic_lung_disease":     "Chronic Lung Disease (0/1)",
        "dialysis":                 "Pre-op Dialysis (0/1)",
        "ejection_fraction":        "Ejection Fraction (%)",
        "NYHA_class":               "NYHA Class (1–4)",
        "congestive_heart_failure_A": "Congestive Heart Failure (0/1)",
        "cardiogenic_shock":        "Cardiogenic Shock (0/1)",
        "resuscitation":            "Pre-op Resuscitation (0/1)",
        "myocardial_infarction":    "Recent MI (0/1)",
        "arrhythmia":               "Arrhythmia (0/1)",
        "AFibFlutter":              "AF/Flutter (0/1)",
        "pulmonary_artery_hypertension": "Pulm. Artery Hypertension (0/1)",
        "Coronaries_diseased":      "Number of Diseased Coronaries",
        "left_main_disease":        "Left Main Disease (0/1)",
        "Aortic_regurgitation":     "Aortic Regurgitation (0/1)",
        "Mitral_regurgitation":     "Mitral Regurgitation (0/1)",
        "last_hematocrit":          "Pre-op Hematocrit (%)",
        "last_cretenine_preop":     "Pre-op Creatinine (mg/dL)",
        "BPsystolic":               "Systolic BP (mmHg)",
        "perfusion_time_min":       "CPB Perfusion Time (min)",
        "cross_clamp_time_min":     "Aortic Cross-Clamp Time (min)",
        "IABP":                     "Intra-Aortic Balloon Pump (0/1)",
        "intraop_blood_products":   "Intraop Blood Products Used (0/1)",
        "Total_bypasses_grafted":   "Total Bypasses Grafted",
        "initial_hours_ventilated": "Hours Ventilated (ICU)",
        "initial_icu_hours":        "Initial ICU Hours",
        "drainage_at_12_hours":     "Drainage at 12h (mL)",
        "drainage_at_24_hours":     "Drainage at 24h (mL)",
        "post_op_creatinine":       "Post-op Creatinine (mg/dL)",
        "reintubated_hospital_stay": "Reintubated During Stay (0/1)",
    }
}

with open("feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)
print("✓ Saved feature_info.json")
print("\n✓ Training complete! Now run: uvicorn app:app --reload")
