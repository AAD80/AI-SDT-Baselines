import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

def clean_features(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna()
    return X

print("=" * 60)
print("IIoT BASELINES (X-IIoTID)")
print("=" * 60)

# Note: Update the path if your file name is different
IIOT_PATH = "data/X-IIoTID dataset.csv"

try:
    iiot_df = pd.read_csv(IIOT_PATH, nrows=500_000, encoding='latin1', engine='python', on_bad_lines='skip')
    
    # Feature Engineering for X-IIoTID
    drop_cols = ['Date', 'Timestamp', 'Scr_IP', 'Des_IP']
    iiot_df = iiot_df.drop(columns=[c for c in drop_cols if c in iiot_df.columns])

    cat_cols = ['Protocol', 'Service', 'Conn_state', 'FIN or RST', 'Bad_checksum', 'is_SYN_with_RST', 'Scr_port', 'Des_port']
    for c in cat_cols:
        if c in iiot_df.columns:
            iiot_df[c] = LabelEncoder().fit_transform(iiot_df[c].astype(str))

    bool_cols = [c for c in iiot_df.columns if iiot_df[c].dtype == object and c not in ['class1', 'class2', 'class3']]
    for c in bool_cols:
        iiot_df[c] = iiot_df[c].map({'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0})
        if iiot_df[c].isna().any():
            iiot_df[c] = LabelEncoder().fit_transform(iiot_df[c].astype(str))

    iiot_df = iiot_df.replace('-', np.nan)
    
    label_col = 'class1' if 'class1' in iiot_df.columns else None
    if not label_col:
        raise ValueError("Label column 'class1' not found.")

    y_raw = iiot_df[label_col].astype(str)
    X = iiot_df.drop(columns=['class1', 'class2', 'class3'], errors='ignore').select_dtypes(include=[np.number])
    X = clean_features(X)
    y_raw = y_raw.loc[X.index]
    y = LabelEncoder().fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
    print(f"Failed to load or process IIoT data. Error: {e}")
    print("Please ensure 'X-IIoTID dataset.csv' is in the 'data/' folder.")