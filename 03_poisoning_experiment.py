import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Re-use the cleaning function from IIoT script
def clean_features(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna()
    return X

print("=" * 60)
print("SEMANTIC POISONING EXPERIMENT")
print("=" * 60)

IIOT_PATH = "data/X-IIoTID dataset.csv"

try:
    # Load and prepare data (same as 02_iiot_baseline.py)
    iiot_df = pd.read_csv(IIOT_PATH, nrows=500_000, encoding='latin1', engine='python', on_bad_lines='skip')
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
    y_raw = iiot_df['class1'].astype(str)
    X = iiot_df.drop(columns=['class1', 'class2', 'class3'], errors='ignore').select_dtypes(include=[np.number])
    X = clean_features(X)
    y_raw = y_raw.loc[X.index]
    y = LabelEncoder().fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    poison_rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    results = []

    for rate in poison_rates:
        y_tr_poisoned = y_tr.copy()
        n_poison = int(rate * len(y_tr_poisoned))
        if n_poison > 0:
            idx = np.random.choice(len(y_tr_poisoned), n_poison, replace=False)
            for i in idx:
                candidates = [c for c in np.unique(y_tr) if c != y_tr_poisoned[i]]
                y_tr_poisoned[i] = np.random.choice(candidates)
        
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr_poisoned)
        y_pred = rf.predict(X_te)
        
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        results.append({'rate': rate * 100, 'acc': acc, 'f1': f1})
        print(f"Poison rate {rate*100:5.0f}%: Acc={acc:.4f}, F1={f1:.4f}")

    # Plot and save Figure 5
    rates = [r['rate'] for r in results]
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(rates, accs, 'o-', label='Accuracy', color='#2E86AB', linewidth=2)
    plt.plot(rates, f1s, 's--', label='F1-Score', color='#A23B72', linewidth=2)
    plt.xlabel('Poisoning Rate (%)')
    plt.ylabel('Performance on Clean Test Data')
    plt.title('Figure 5. Effect of Label-Flipping Poisoning on IIoT Baseline')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('figure5_poisoning.png', dpi=300, bbox_inches='tight')
    print("\nSaved figure as 'figure5_poisoning.png'")

except Exception as e:
    print(f"An error occurred: {e}")