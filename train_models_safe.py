import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix
import shap
import os
import gc

# ==========================================
# 0. Resource Limit Security Lock (Optional)
# ==========================================
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Set the chart format
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Loading data and preprocessing
# ==========================================
print("Loading data...")
df = pd.read_excel('/home/data/t120623/ML.xlsx', sheet_name='Sheet1')

# Eliminate irrelevant features (to prevent data leakage)
columns_to_drop = ['Name', 'ID', '检测仪器']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df_model = df.drop(columns=columns_to_drop)

X = df_model.drop(columns=['GROUP'])
y = df_model['GROUP']

# Divide training set and test set (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data Loaded! Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================
# 2. Build a secure model library 
# ==========================================
models = {
    "Logistic_Regression": make_pipeline(
        StandardScaler(), 
        LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000, n_jobs=4)
    ),
    "SVM_RBF": make_pipeline(
        StandardScaler(), 
        SVC(probability=True, class_weight='balanced', random_state=42, 
            cache_size=1000, 
            max_iter=10000)  
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, 
        max_depth=5, n_jobs=4 
    ),
    "XGBoost": XGBClassifier(
        eval_metric='logloss', scale_pos_weight=1, random_state=42,
        tree_method='hist', 
        max_depth=4, n_jobs=4
    ),
    "LightGBM": LGBMClassifier(
        random_state=42, verbose=-1, max_depth=4, n_jobs=4
    ),
    "CatBoost": CatBoostClassifier(
        verbose=0, random_state=42, auto_class_weights='Balanced',
        depth=4, thread_count=4
    )
}

# ==========================================
# 3. Train and evaluate models one by one (loop automatically frees memory)
# ==========================================
results = []
plt.figure(figsize=(10, 8))

best_auc = 0
best_model_name = ""
best_model = None # Save the best model instance itself
for name, model in models.items():
    print(f"Training {name}...")
    
    # Training
    model.fit(X_train, y_train)
    
    # Predict probabilities and labels
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculation of evaluation indicators
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    results.append({
        "Model": name.replace('_', ' '),
        "AUC": roc_auc,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    })
    
    # ROC
    plt.plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ')} (AUC = {roc_auc:.3f})")
    
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model_name = name
        best_model = model
        
    gc.collect()

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves Comparison of Multiple Models', fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('ROC_Curves_Comparison.png', dpi=300)
plt.close()

df_results = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
df_results.to_csv('model_performance.csv', index=False)
print("\n--- Model Performance Ranking ---")
print(df_results.to_string(index=False))
print(f"\n>> Best Model Selected: {best_model_name}")

# ==========================================
# 4. SHAP analysis (for the best model)
# ==========================================
print(f"\nGenerating SHAP explanations for {best_model_name}...")

if best_model_name in ["Logistic_Regression", "SVM_RBF"]:
    core_model = best_model.named_steps[best_model_name.split('_')[0].lower() if 'SVM' not in best_model_name else 'svc']
    X_test_transformed = best_model.named_steps['standardscaler'].transform(X_test)
    
    # SVM / LR 
    explainer = shap.Explainer(core_model, X_train)
    shap_values = explainer(X_test_transformed)
    
    # Draw the SHAP swarming map
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll tasks completed successfully! Check the current directory for PNG and CSV files.")
