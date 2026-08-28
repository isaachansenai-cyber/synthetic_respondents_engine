"""
OLS regression: JS distance ~ question metadata.
n=28 so report as exploratory only — do not report p-values as inferential.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
 
master = pd.read_csv('outputs/master_results.csv')
js_cols = [c for c in master.columns if c.startswith('js_') and not c.startswith('js_avg')]
master['avg_js'] = master[js_cols].mean(axis=1)
 
master['attr_emotional']   = (master['attribute_type']=='emotional').astype(int)
master['lived_exp_yes']    = master['requires_lived_exp'].astype(int)
master['reverse_coded']    = master['is_reverse_coded'].astype(int)
for c in ['perceived_usefulness','ease_of_use','ai_capability','attitude']:
    master[f'construct_{c}'] = (master['tam_construct']==c).astype(int)
 
PREDICTORS = ['attr_emotional','lived_exp_yes','reverse_coded',
              'construct_perceived_usefulness','construct_ease_of_use',
              'construct_ai_capability','construct_attitude']
 
X = master[PREDICTORS].values
y = master['avg_js'].values
X_scaled = StandardScaler().fit_transform(X)
reg = LinearRegression().fit(X_scaled, y)
r2  = reg.score(X_scaled, y)
 
print('=== OLS REGRESSION: avg_JS ~ question metadata ===')
print(f'R² = {r2:.3f}  (n=28 — exploratory only; do not report p-values)\n')
print(f'{'Predictor':<42} Std Beta')
print('-'*52)
for name, beta in zip(PREDICTORS, reg.coef_):
    print(f'{name:<42} {beta:+.4f}')
 
pd.DataFrame({'predictor':PREDICTORS,'std_beta':reg.coef_,'r2':r2}
).to_csv('outputs/metrics/regression_results.csv', index=False)
print('\n✅ Regression results saved.')
