"""
Compute validation metrics: JS distance, Frobenius norm, Cronbach alpha,
validity classifications, and arm-stratified comparisons.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
import pandas as pd, numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.linalg import norm
from src.canonical_schema import (
    ALL_Q_IDS, CONSTRUCT_ITEMS, COL_SYNID, COL_ARM
)
 
MODELS        = ['gpt4','claude','gemini']
Q_COLS        = ALL_Q_IDS
BASELINE_PATH = 'data/human_baseline/human_baseline_frozen_v1.csv'
PROFILES_PATH = 'data/profiles/profiles.csv'
OUT_DIR       = 'outputs/metrics'
os.makedirs(OUT_DIR, exist_ok=True)
 
baseline = pd.read_csv(BASELINE_PATH).set_index('q_id')
profiles_arm = pd.read_csv(PROFILES_PATH)[[COL_SYNID, COL_ARM]]
 
# ── Core helpers ──────────────────────────────────────
def to_distributions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid in Q_COLS:
        if qid not in df.columns:
            rows.append({'q_id':qid,'p1':0.2,'p2':0.2,'p3':0.2,'p4':0.2,'p5':0.2})
            continue
        counts = df[qid].dropna().astype(int).value_counts()
        counts = counts.reindex([1,2,3,4,5], fill_value=0)  # guard against missing levels
        total  = counts.sum()
        if total == 0:
            probs = {f'p{i}':0.2 for i in range(1,6)}
        else:
            probs = {f'p{i}':counts[i]/total for i in range(1,6)}
        rows.append({'q_id':qid, **probs})
    return pd.DataFrame(rows).set_index('q_id')
 
def js_dist(p, q):
    p = np.array(p,dtype=float)+1e-10; q=np.array(q,dtype=float)+1e-10
    p/=p.sum(); q/=q.sum()
    return float(jensenshannon(p,q))
 
def cronbach_alpha(df, items):
    k = len(items)
    if k < 2: return np.nan
    iv  = df[items].var(axis=0, ddof=1).sum()
    tv  = df[items].sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1-iv/tv) if tv!=0 else np.nan
 
def get_hp(qid):
    return baseline.loc[qid,['p1','p2','p3','p4','p5']].values
 
# ── Load model data ────────────────────────────────────
model_dfs = {}
for m in MODELS:
    p = f'outputs/{m}/{m}_full_responses.csv'
    if os.path.exists(p): model_dfs[m] = pd.read_csv(p)
    else: print(f'⚠ {m} not found — skipping')
 
# ── 1. Per-question JS: each model vs human ────────────
print('Computing JS distances...')
js_rows      = []
model_dists  = {}
for m, df in model_dfs.items():
    dist = to_distributions(df)
    model_dists[m] = dist
    for qid in Q_COLS:
        js_rows.append({'q_id':qid,'model':m,
                        'js_human_syn':js_dist(get_hp(qid),
                                               dist.loc[qid,['p1','p2','p3','p4','p5']].values)})
 
js_df = pd.DataFrame(js_rows)
 
# ── 2. Cross-model JS ─────────────────────────────────
cm_rows = []
mkeys   = list(model_dists.keys())
for i in range(len(mkeys)):
    for j in range(i+1, len(mkeys)):
        ma,mb = mkeys[i],mkeys[j]
        for qid in Q_COLS:
            pa = model_dists[ma].loc[qid,['p1','p2','p3','p4','p5']].values
            pb = model_dists[mb].loc[qid,['p1','p2','p3','p4','p5']].values
            cm_rows.append({'q_id':qid,'model_a':ma,'model_b':mb,'js_cross':js_dist(pa,pb)})
 
cm_df = pd.DataFrame(cm_rows)
 
# ── 3. Validity classification ────────────────────────
avg_hs   = js_df.groupby('q_id')['js_human_syn'].mean().rename('avg_js_human_syn')
avg_xm   = cm_df.groupby('q_id')['js_cross'].mean().rename('avg_js_cross')
val_df   = pd.concat([avg_hs, avg_xm], axis=1)
 
def classify(row):
    hi = row['avg_js_cross']  < 0.10
    lo = row['avg_js_human_syn'] < 0.15
    if hi and lo:       return 'robust_synthetic_candidate'
    if hi and not lo:   return 'genuine_llm_knowledge_gap'
    if not hi and lo:   return 'accidental_convergence_flag'
    return 'model_artifact'
 
val_df['validity_class'] = val_df.apply(classify, axis=1)
val_df['construct']      = [
    next(c for c,items in CONSTRUCT_ITEMS.items() if qid in items)
    for qid in val_df.index
]
 
# ── 4. Construct-level JS summary ─────────────────────
constr_js = (
    js_df.merge(val_df[['construct']].reset_index(), on='q_id')
    .groupby(['construct','model'])['js_human_syn']
    .mean().unstack('model')
)
print('\nConstruct-level JS distances:')
print(constr_js.round(3).to_string())
 
# ── 5. Cronbach's alpha & within-construct correlations ─
print('\nCronbach alpha and within-construct correlations:')
alpha_rows, wcc_rows = [], []
for m, df in model_dfs.items():
    ar = {'model':m}
    wr = {'model':m}
    for cname, items in CONSTRUCT_ITEMS.items():
        ar[cname] = round(cronbach_alpha(df, items), 3)
        c = df[items].corr().to_numpy(copy=True)
        np.fill_diagonal(c, np.nan)
        wr[cname] = round(float(np.nanmean(c)), 3)
    alpha_rows.append(ar); wcc_rows.append(wr)
    print(f'  {m} alpha: {ar}')
    print(f'  {m} within-r: {wr}')
 
alpha_df = pd.DataFrame(alpha_rows)
wcc_df   = pd.DataFrame(wcc_rows)
 
# ── 6. Arm-stratified metrics ────────────────────────
print('\nComputing arm-stratified metrics...')
arm_js_rows, arm_alpha_rows, cross_arm_rows = [], [], []
 
for m, df in model_dfs.items():
    df_arm = df.merge(profiles_arm, on=COL_SYNID, how='left', validate='1:1')
    assert len(df_arm)==len(df), 'Arm merge changed row count'
 
    for arm in ['demo_plus_traits','demo_only']:
        adf = df_arm[df_arm[COL_ARM]==arm]
        if len(adf)<5: print(f'  ⚠ {m}/{arm}: n={len(adf)} — skipping'); continue
 
        dist = to_distributions(adf)
        for qid in Q_COLS:
            arm_js_rows.append({'model':m,COL_ARM:arm,'q_id':qid,
                                'js_human_arm':js_dist(get_hp(qid),
                                               dist.loc[qid,['p1','p2','p3','p4','p5']].values),
                                'n_arm':len(adf)})
 
        row = {'model':m,COL_ARM:arm,'n':len(adf)}
        for cname, items in CONSTRUCT_ITEMS.items():
            row[cname] = round(cronbach_alpha(adf, items), 3)
        arm_alpha_rows.append(row)
 
    # Cross-arm JS: demo_plus_traits vs demo_only
    pt_df = df_arm[df_arm[COL_ARM]=='demo_plus_traits']
    do_df = df_arm[df_arm[COL_ARM]=='demo_only']
    if len(pt_df)>=5 and len(do_df)>=5:
        pt_dist = to_distributions(pt_df)
        do_dist = to_distributions(do_df)
        for qid in Q_COLS:
            cross_arm_rows.append({'model':m,'q_id':qid,
                                   'js_pt_vs_do':js_dist(
                                       pt_dist.loc[qid,['p1','p2','p3','p4','p5']].values,
                                       do_dist.loc[qid,['p1','p2','p3','p4','p5']].values)})
 
# ── Save all outputs ──────────────────────────────────
js_df.to_csv(f'{OUT_DIR}/js_distances.csv', index=False)
cm_df.to_csv(f'{OUT_DIR}/cross_model_js.csv', index=False)
val_df.reset_index().to_csv(f'{OUT_DIR}/validity_classifications.csv', index=False)
constr_js.reset_index().to_csv(f'{OUT_DIR}/construct_js_summary.csv')
alpha_df.to_csv(f'{OUT_DIR}/cronbach_alpha.csv', index=False)
wcc_df.to_csv(f'{OUT_DIR}/within_construct_corrs.csv', index=False)
 
if arm_js_rows:
    pd.DataFrame(arm_js_rows).to_csv(f'{OUT_DIR}/js_by_arm.csv', index=False)
if arm_alpha_rows:
    pd.DataFrame(arm_alpha_rows).to_csv(f'{OUT_DIR}/cronbach_alpha_by_arm.csv', index=False)
if cross_arm_rows:
    pd.DataFrame(cross_arm_rows).to_csv(f'{OUT_DIR}/js_pt_vs_do.csv', index=False)
 
print(f'\n✅ All metrics saved to {OUT_DIR}/')
print('Files:', [f for f in os.listdir(OUT_DIR)])
