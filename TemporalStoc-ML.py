# ============================================================
# 1. Package Installations
# ============================================================
!pip install -q scikit-learn==1.2.2 imbalanced-learn==0.10.1 xgboost lightgbm catboost

# ============================================================
# 2. Imports
# ============================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
    roc_curve, average_precision_score  # <-- ROBUST PR-AUC
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# ============================================================
# 3. Load data
# ============================================================
input_file = 'temporal_stochastic.csv'
data = pd.read_csv(input_file)

X = data.drop(columns=['label', 'value'])
y = data['label'].astype(int)

n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
imbalance_ratio = n_neg / n_pos
print(f"Imbalance ratio: {imbalance_ratio:.1f}:1 (neg:{n_neg}, pos:{n_pos})")

# ============================================================
# 4. Train / Val / Test split (64 / 16 / 20)
# ============================================================
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=42
)

# ============================================================
# 5. Dynamic Sampling Ratios
# ============================================================
target_minority_ratio = min(0.3, 10 / imbalance_ratio)
smote_ratio       = min(0.5, target_minority_ratio * 2)
undersample_ratio = min(0.5, smote_ratio * 0.6)
smotetomek_ratio  = min(0.3, smote_ratio * 0.8)
scale_pos_weight  = max(1.0, imbalance_ratio * 0.9)

print(f"SMOTE:{smote_ratio:.3f} | Undersample:{undersample_ratio:.3f} | SMOTETomek:{smotetomek_ratio:.3f}")

# ============================================================
# 6. Model Pipelines
# ============================================================
models = {
    'RandomForest': ImbPipeline([
        ('scaler',   StandardScaler()),
        ('sampling', SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ('model',    RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced_subsample',
            random_state=42, n_jobs=-1
        ))
    ]),

    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='aucpr', random_state=42, n_jobs=-1, tree_method='hist'
    ),

    'LightGBM': LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1, reg_lambda=1.0,
        objective='binary', metric='average_precision',
        random_state=42, n_jobs=-1
    ),

    'CatBoost': CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.05,
        l2_leaf_reg=3, scale_pos_weight=scale_pos_weight,
        loss_function='Logloss', eval_metric='PRAUC',
        random_state=42, verbose=0, thread_count=-1
    ),

    'SVM': ImbPipeline([
        ('scaler',   StandardScaler()),
        ('sampling', RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=42)),
        ('model',    CalibratedClassifierCV(
            base_estimator=SVC(kernel='rbf', C=1.0, class_weight='balanced',
                               probability=True, random_state=42),
            method='isotonic', cv=3
        ))
    ]),

    'SMOTETomek_LGBM': ImbPipeline([
        ('scaler',   StandardScaler()),
        ('sampling', SMOTETomek(sampling_strategy=smotetomek_ratio, random_state=42)),
        ('model',    LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=42, n_jobs=-1
        ))
    ])
}

# ============================================================
# 7. Helpers
# ============================================================
def optimize_threshold(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * rec * prec / (rec + prec + 1e-12)
    best = np.argmax(f1)
    return thr[best] if len(thr) > best else 0.5

def cv_f1_score(model, X, y, name, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)):
    f1s = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        if name in ['XGBoost', 'LightGBM', 'CatBoost']:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]

        thr = optimize_threshold(y_va, proba)
        pred = (proba >= thr).astype(int)
        f1s.append(f1_score(y_va, pred, zero_division=0))
    return np.mean(f1s), np.std(f1s)

# ============================================================
# 8. Evaluation + Repeated CV
# ============================================================
results = []

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training & Evaluating: {name}")
    print(f"{'='*70}")

    model.fit(X_train, y_train)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    opt_thr = optimize_threshold(y_val, y_val_proba)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= opt_thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)

    cv_mean, cv_std = cv_f1_score(model, X_tr_full, y_tr_full, name)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC_AUC': roc,
        'PR_AUC': pr_auc,
        'Optimal_Threshold': opt_thr,
        'CV_F1_Mean': cv_mean,
        'CV_F1_Std': cv_std
    })

    print(f"Test F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | CV F1: {cv_mean:.4f} ± {cv_std:.4f}")

# ============================================================
# 9. Results
# ============================================================
results_df = pd.DataFrame(results).round(4).sort_values('F1', ascending=False)
print("\n" + "="*100)
print("FINAL RESULTS — lambda_stochastic_dual.csv")
print("="*100)
print(results_df[[
    'Model','F1','PR_AUC','ROC_AUC',
    'CV_F1_Mean','CV_F1_Std','Optimal_Threshold'
]].to_string(index=False))
