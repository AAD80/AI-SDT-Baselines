import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

def find_label_column(df):
    for c in df.columns:
        if 'label' in c.lower():
            return c
    return None

def clean_features(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna()
    return X

print("=" * 60)
print("SDN BASELINES (ITU-ML5G-PS-006)")
print("=" * 60)

# Note: Update the path if your file name is different
SDN_PATH = "data/test_data.csv"

try:
    sdn_df = pd.read_csv(SDN_PATH, nrows=500_000)
    label_col_sdn = find_label_column(sdn_df)
    
    y_raw_s = sdn_df[label_col_sdn].astype(str)
    X_s = sdn_df.drop(columns=[label_col_sdn]).select_dtypes(include=[np.number])
    X_s = clean_features(X_s)
    y_raw_s = y_raw_s.loc[X_s.index]
    y_s = LabelEncoder().fit_transform(y_raw_s)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )

    for name, clf in [
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', verbosity=0)),
    ]:
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        train_t = time.time() - t0
        t0 = time.time()
        y_pred = clf.predict(X_te)
        infer_ms = (time.time() - t0) / len(X_te) * 1000
        
        print(f"  {name}: Acc={accuracy_score(y_te, y_pred):.4f}, "
              f"F1={f1_score(y_te, y_pred, average='weighted', zero_division=0):.4f}, "
              f"Train={train_t:.1f}s, Infer={infer_ms:.4f} ms/sample")
except Exception as e:
    print(f"Failed to load or process SDN data. Error: {e}")
    print("Please ensure 'test_data.csv' is in the 'data/' folder.")