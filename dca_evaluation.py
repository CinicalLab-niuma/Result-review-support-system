import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

# ==========================================
# ==========================================
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

print("Loading data for DCA...")
df = pd.read_excel('/home/data/t120623/ML.xlsx', sheet_name='Sheet1')
columns_to_drop = ['Name', 'ID', '检测仪器']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df_model = df.drop(columns=columns_to_drop)

X = df_model.drop(columns=['GROUP'])
y = df_model['GROUP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. XGBoost
print("Training Champion Model (XGBoost)...")
model = XGBClassifier(
    eval_metric='logloss', random_state=42, max_depth=3, learning_rate=0.05, 
    subsample=0.7, colsample_bytree=0.7, reg_lambda=10, n_jobs=4, tree_method='hist'
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_true = y_test.values

# ==========================================
# 2. The core data of decision curve (DCA) were calculated
# ==========================================
def calculate_net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    n = len(y_true)
    # Calculate net income: Net Benefit = (TP / N) - (FP / N) * (Pt / (1 - Pt))
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

# The probability threshold was set from 0.01 to 0.99
thresholds = np.arange(0.01, 1.00, 0.01)

# Calculate the net benefits of the three strategies
nb_model = [calculate_net_benefit(y_true, y_prob, pt) for pt in thresholds]
nb_all = [calculate_net_benefit(y_true, np.ones_like(y_true), pt) for pt in thresholds] # 策略：所有结果均当做“真性高钙”（无拦截）
nb_none = np.zeros_like(thresholds) # 策略：所有结果均当做“假性高钙”（全部拦截，净收益为0）

# ==========================================
# 3. Plot the DCA
# ==========================================
plt.figure(figsize=(8, 6))

plt.plot(thresholds, nb_model, color='#E64B35', lw=2.5, label='XGBoost Model')
plt.plot(thresholds, nb_all, color='gray', lw=1.5, linestyle='--', label='Treat All (Trust all LIS results)')
plt.plot(thresholds, nb_none, color='black', lw=1.5, label='Treat None (Intercept all)')

plt.xlim([0.0, 1.0])
y_min = max(-0.2, min(nb_all) - 0.05)
plt.ylim([y_min, max(nb_model) + 0.05])

plt.xlabel('Threshold Probability', fontweight='bold')
plt.ylabel('Net Benefit', fontweight='bold')
plt.title('Decision Curve Analysis (DCA)', fontweight='bold', pad=15)
plt.legend(loc="upper right", frameon=True, edgecolor='black')

plt.fill_between(thresholds, np.maximum(nb_all, nb_none), nb_model, 
                 where=(nb_model > np.maximum(nb_all, nb_none)), 
                 color='#E64B35', alpha=0.15)

plt.savefig('Figure_3_DCA.png', dpi=300, bbox_inches='tight')
plt.close()
print(">> Saved: Figure_3_DCA.png")
