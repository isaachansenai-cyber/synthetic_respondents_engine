"""
Bias audit: AI Circularity Index and Cramers V stereotype amplification.
Includes arm-stratified versions of both.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency, ttest_1samp
from src.canonical_schema import (
    ALL_Q_IDS, CONSTRUCT_ITEMS, REVERSE_ITEMS, COL_SYNID, COL_ARM
)
 
MODELS        = ['gpt4','claude','gemini']
PROFILES_PATH = 'data/profiles/profiles.csv'
BASELINE_PATH = 'data/human_baseline/human_baseline_frozen_v1.csv'
OUT_DIR       = 'outputs/bias_audit'
os.makedirs(OUT_DIR, exist_ok=True)
 
POSITIVE_ITEMS = [q for q in ALL_Q_IDS if q not in REVERSE_ITEMS]
 
profiles     = pd.read_csv(PROFILES_PATH)
profiles_arm = profiles[[COL_SYNID, COL_ARM]]
baseline     = pd.read_csv(BASELINE_PATH)
 
# Human mean per item from baseline distributions
baseline['human_mean'] = (
    baseline[['p1','p2','p3','p4','p5']].values * np.array([1,2,3,4,5])
).sum(axis=1)
human_means = baseline.set_index('q_id')['human_mean']
 
def _cramers_v(df, dem_col, qid):
    ct = pd.crosstab(df[dem_col], df[qid])
    if ct.shape[0]<2 or ct.shape[1]<2: return None
    # Check expected cell counts >= 5
    expected = chi2_contingency(ct)[3]
    if (expected < 5).any(): return None  # invalid — skip
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum(); k = min(ct.shape)-1
    return float(np.sqrt(chi2/(n*k))) if n*k>0 else 0
 
SCALE_GROUPS = [
    ('scale_agree',  [q for q in POSITIVE_ITEMS if q.startswith(('q4_','q5_','q10_'))]),
    ('scale_useful', [q for q in POSITIVE_ITEMS if q.startswith('q7_')]),
    ('scale_likely', [q for q in POSITIVE_ITEMS if q.startswith('q9_')]),
]
DEM_PAIRS = [
    ('age_group', CONSTRUCT_ITEMS['behavioral_intention'][:3], 'age_x_BI'),
    ('sex',       CONSTRUCT_ITEMS['ease_of_use'][:3],          'sex_x_PEOU'),
]
 
aci_rows, cramers_rows = [], []
aci_arm_rows, cramers_arm_rows = [], []
 
for model in MODELS:
    path = f'outputs/{model}/{model}_full_responses.csv'
    if not os.path.exists(path): continue
    df = pd.read_csv(path)
    df = df.merge(profiles[[COL_SYNID,'age','sex']], on=COL_SYNID, how='left', validate='1:1')
    assert len(df)==len(pd.read_csv(path)), 'Merge changed row count'
    df['age_group'] = df['age'].apply(lambda a: '18-26' if a in ['18-21','22-26'] else '27plus')
 
    # ── ACI: aggregate ─────────────────────────────────
    for sg, items in SCALE_GROUPS:
        if not items: continue
        syn_mean   = df[items].values.mean()
        hmean      = human_means[items].mean()
        aci        = round(syn_mean - hmean, 4)
        t, pval    = ttest_1samp(df[items].values.flatten(), hmean)
        sig        = '⚠ SIGNIFICANT' if pval<0.05 and aci>0 else '✅ OK'
        print(f'  {model} | {sg}: ACI={aci:+.3f}  {sig}')
        aci_rows.append({'model':model,'scale_group':sg,'syn_mean':round(syn_mean,4),
                         'human_mean':round(hmean,4),'ACI':aci,
                         't_stat':round(t,4),'p_val':round(pval,4)})
 
    # ── Cramér's V: aggregate ─────────────────────────
    for dem_col, items, label in DEM_PAIRS:
        for qid in items:
            cv = _cramers_v(df, dem_col, qid)
            if cv is not None:
                cramers_rows.append({'model':model,'dem_var':dem_col,
                                     'q_id':qid,'cramers_v':round(cv,4),'label':label})
 
    # ── Arm-stratified ─────────────────────────────────
    df_arm = df.merge(profiles_arm, on=COL_SYNID, how='left', validate='1:1')
    assert len(df_arm)==len(df), 'Arm merge changed row count'
 
    for arm in ['demo_plus_traits','demo_only']:
        adf = df_arm[df_arm[COL_ARM]==arm].copy()
        if len(adf)<5: continue
 
        for sg, items in SCALE_GROUPS:
            if not items: continue
            syn_mean = adf[items].values.mean()
            hmean    = human_means[items].mean()
            aci      = round(syn_mean - hmean, 4)
            t, pval  = ttest_1samp(adf[items].values.flatten(), hmean)
            aci_arm_rows.append({'model':model,COL_ARM:arm,'scale_group':sg,
                                 'syn_mean':round(syn_mean,4),'human_mean':round(hmean,4),
                                 'ACI':aci,'t_stat':round(t,4),'p_val':round(pval,4),
                                 'n_arm':len(adf)})
 
        for dem_col, items, label in DEM_PAIRS:
            for qid in items:
                cv = _cramers_v(adf, dem_col, qid)
                if cv is not None:
                    cramers_arm_rows.append({'model':model,COL_ARM:arm,'dem_var':dem_col,
                                             'q_id':qid,'cramers_v':round(cv,4),'label':label})
 
pd.DataFrame(aci_rows).to_csv(f'{OUT_DIR}/ai_circularity_index.csv', index=False)
pd.DataFrame(cramers_rows).to_csv(f'{OUT_DIR}/cramers_v.csv', index=False)
if aci_arm_rows:
    pd.DataFrame(aci_arm_rows).to_csv(f'{OUT_DIR}/aci_by_arm.csv', index=False)
if cramers_arm_rows:
    pd.DataFrame(cramers_arm_rows).to_csv(f'{OUT_DIR}/cramers_by_arm.csv', index=False)
print(f'\n✅ Bias audit complete. Files in {OUT_DIR}/')
