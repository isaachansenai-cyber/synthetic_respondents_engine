"""
Prompt Funnel Experiment — Style Comparison Analysis.
 
Loops over PROMPT_STYLES (7 funnel batches) for each model, computing:
  - Mean JS distance vs human baseline (primary ranking metric)
  - JS distance by TAM construct
  - Cronbach alpha by construct
  - Between-profile variance
  - AI Circularity Index (positive items, scale_agree only)
  - Q7_6 reverse coding flag
 
Funnel interpretation of rank_by_js:
  Rank 1 (lowest JS) = optimal prompt engineering level.
  study1_legacy is the anchor; zero_shot is the floor.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd, numpy as np
from scipy.spatial.distance import jensenshannon
from src.canonical_schema import ALL_Q_IDS, CONSTRUCT_ITEMS, REVERSE_ITEMS
from src.prompt_builder import PROMPT_STYLES
 
MODELS        = ['gpt4','claude']
Q_COLS        = ALL_Q_IDS
POSITIVE_Q    = [q for q in Q_COLS if q not in REVERSE_ITEMS]
BASELINE_PATH = 'data/human_baseline/human_baseline_frozen_v1.csv'
OUT_DIR       = 'outputs/metrics'
os.makedirs(OUT_DIR, exist_ok=True)
 
baseline = pd.read_csv(BASELINE_PATH).set_index('q_id')
baseline['human_mean'] = (baseline[['p1','p2','p3','p4','p5']].values * np.array([1,2,3,4,5])).sum(axis=1)
 
def to_dist(df, qid):
    c = df[qid].dropna().astype(int).value_counts().reindex([1,2,3,4,5],fill_value=0)
    t = c.sum()
    return (c/t).values if t>0 else np.ones(5)/5
 
def js_dist(p, q):
    p=np.array(p)+1e-10; q=np.array(q)+1e-10
    return float(jensenshannon(p/p.sum(), q/q.sum()))
 
def cronbach(df, items):
    k=len(items)
    if k<2: return np.nan
    iv=df[items].var(axis=0,ddof=1).sum()
    tv=df[items].sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1-iv/tv) if tv!=0 else np.nan
 
summary_rows = []
 
for model in MODELS:
    for style in PROMPT_STYLES:
        path = f'outputs/{model}/{style}/{model}_{style}_full_responses.csv'
        if not os.path.exists(path):
            print(f'  SKIP (not found): {path}'); continue
        df = pd.read_csv(path)
        if len(df) < 10:
            print(f'  SKIP (too few rows): {path}  n={len(df)}'); continue
 
        # Overall JS distance
        js_vals = [js_dist(
            baseline.loc[qid,['p1','p2','p3','p4','p5']].values,
            to_dist(df, qid)) for qid in Q_COLS]
        mean_js = np.mean(js_vals)
 
        # JS by construct
        cjs = {}
        for cname, items in CONSTRUCT_ITEMS.items():
            cjs[f'js_{cname}'] = round(np.mean([js_dist(
                baseline.loc[q,['p1','p2','p3','p4','p5']].values,
                to_dist(df,q)) for q in items]),4)
 
        # Cronbach alpha
        calpha = {f'alpha_{c}': round(cronbach(df,items),3)
                  for c,items in CONSTRUCT_ITEMS.items()}
 
        # Between-profile variance
        mean_sd = df[Q_COLS].std().mean()
 
        # AI Circularity Index (scale_agree positive items only)
        agree_q  = [q for q in POSITIVE_Q if q.startswith(('q4_','q5_','q10_'))]
        syn_mean = df[agree_q].values.mean()
        hum_mean = baseline.loc[agree_q,'human_mean'].mean()
        aci      = round(syn_mean - hum_mean, 4)
 
        # Q7_6 reverse coding check
        r76 = df['q7_6'].corr(df[['q7_1','q7_2','q7_3','q7_4','q7_5']].mean(axis=1))
        rev_ok = bool(r76 < 0)
 
        row = {'model':model,'style':style,'n':len(df),
               'mean_js_all':round(mean_js,4),
               'mean_sd':round(mean_sd,3),
               'aci':aci,
               'q7_6_reverse_ok':rev_ok,
               **cjs, **calpha}
        summary_rows.append(row)
        rev_flag = '✅' if rev_ok else '❌ REVERSE FAIL'
        print(f'{model}/{style}: JS={mean_js:.4f} SD={mean_sd:.3f} ACI={aci:+.3f} {rev_flag}')
 
summary = pd.DataFrame(summary_rows)

# Rank within each model by mean JS (lower = better)
if len(summary) > 0:
    summary['rank_by_js'] = summary.groupby('model')['mean_js_all'].rank(method='min')
else:
    summary['rank_by_js'] = []
    print('No completed runs found — run generation first, then re-run compare_styles.py')
 
out = f'{OUT_DIR}/style_comparison.csv'
summary.to_csv(out, index=False)
print(f'\n✅ Saved: {out}')
 
print('\n=== FUNNEL RANKING (lowest JS distance to human baseline = rank 1) ===')
if len(summary) == 0:
    print('No completed runs found — run generation first, then re-run compare_styles.py')
else:
    for model in MODELS:
        sub = summary[summary['model']==model].sort_values('mean_js_all')
        if len(sub)==0: continue
        print(f'\n  {model.upper()}:')
        print(sub[['style','mean_js_all','mean_sd','aci','rank_by_js']].to_string(index=False))
 
# Flag any runs where Q7_6 reverse coding failed
if len(summary) > 0:
    bad_rev = summary[~summary['q7_6_reverse_ok']]
    if len(bad_rev) > 0:
        print('\n⚠ Q7_6 REVERSE CODING FAILED — discard these results:')
        print(bad_rev[['model','style']].to_string(index=False))