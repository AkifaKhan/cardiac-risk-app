import pandas as pd
import numpy as np

# ======================
# 1. LOAD DATA
# ======================
df = pd.read_csv(r"C:\Users\DELL\Downloads\MS Thesis\Data/dataset.csv")
df.replace(" ", np.nan, inplace=True)
df.replace("", np.nan, inplace=True)
df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)
# ======================
# 2. DROP COLUMNS
# ======================
drop_cols = [
    # ID
    "TempSNO",

    # Dates
    "date_of_admission","date_of_surgery","date_of_discharge",
    "or_entry_date","or_entry_time","or_exit_date","or_exit_time",
    "intubation_date","intubation_time","extubation_date","extubation_time",
    "skin_incision_start_date","skin_incision_start_time",
    "skin_incision_closure_date","skin_incision_closure_time",
    "mortality_date",

    # Post-op leakage
    "icu_visit","initial_hours_ventilated","initial_icu_hours","initial_icu_stay",
    "readmission_to_icu","additional_icu_hours",
    "blood_products_used","red_blood_cell_units","fresh_frozen_plasma_units",
    "cryoprecipitate_units","platelet_units",
    "extubated_in_or","reintubated_hospital_stay","additional_hours_ventilated",
    "readmitted","drainage_at_12_hours","drainage_at_24_hours",
    "hospital_complications","Reopened_postCABG","post_op_creatinine",
    "prolonged_ventilation","pneumonia","pleural_effusion",
    "gastro_intestinal_complication","multi_system_failure",
    "Miscellaneous_complications","heart_failure","perioperative_mi",
    "heart_block","cardiac_arrest","ventricular_arrythmia",
    "inotropics_used","atrial_fibrillation","dialysis_newly_required",
    "post_operative_stroke","sternum_deep",
    "or_for_sternal_debridement","specify_other", "LM_specify", "Ethicity_language"
]


# Fix blanks
df.replace(" ", np.nan, inplace=True)
df.replace("", np.nan, inplace=True)

# Remove empty strings properly
df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)

# Convert or drop non-numeric columns
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        df.drop(columns=[col], inplace=True)
# ======================
# 3. HANDLE WEIRD VALUES
# ======================
df.replace("1/1/1753", np.nan, inplace=True)
print(df.shape)
# ======================
# 4. TARGET SPLIT
# ======================
X = df.drop("in_hospital_mortality", axis=1)
y = df["in_hospital_mortality"]

# ======================
# 5. IMPUTATION
# ======================
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")
X = imputer.fit_transform(X)

# ======================
# 6. TRAIN TEST SPLIT
# ======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ======================
# 7. MODELS
# ======================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

models = {
    "SVM": SVC(probability=True, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(
        scale_pos_weight=10,  # adjust later based on imbalance
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ======================
# 8. EVALUATION
# ======================
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results.append([name, acc, f1, recall, auc])

# ======================
# 9. RESULTS TABLE
# ======================
results_df = pd.DataFrame(results, columns=["Model","Accuracy","F1","Recall","AUC"])

print(results_df)
from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, preds)
#print(cm)