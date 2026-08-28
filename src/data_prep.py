"""
Data preparation: clean demographics, filter eligible respondents,
compute and freeze human baseline distributions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
import pandas as pd
import numpy as np
from src.canonical_schema import (
    COL_DATE, COL_ID, COL_AGE, COL_SEX, COL_COUNTRY, COL_CITY,
    COL_SRC_HUMAN_ID, RENAME_MAP, CONSTRUCT_ITEMS, ALL_Q_IDS,
    SCALE_TYPE, assert_schema, normalize_country
)
 
DEM_PATH = 'data/raw/SMA_2026_TAM_surveys_demographics.xlsx'
Q_PATH   = 'data/raw/SMA_2026_TAM_surveys_key_questions.xlsx'
OUT_BASE = 'data/human_baseline/human_baseline_frozen_v1.csv'
OUT_SUB  = 'data/human_baseline/human_baseline_subgroups_v1.csv'
OUT_DEM  = 'data/human_baseline/demographics_clean.csv'
ELIGIBLE_OUT = 'data/profiles/eligible_humans.csv'
 
os.makedirs('data/human_baseline', exist_ok=True)
os.makedirs('data/profiles', exist_ok=True)
os.makedirs('logs', exist_ok=True)
 
# ── 1. Load & fix column names ─────────────────────────
print('Loading demographics...')
raw = pd.read_excel(DEM_PATH)
print('Raw columns:', [repr(c) for c in raw.columns])
raw.columns = raw.columns.str.strip()
raw.rename(columns=RENAME_MAP, inplace=True)
assert_schema(raw, [COL_ID, COL_AGE, COL_SEX, COL_COUNTRY], 'demographics')
print(f'Loaded {len(raw)} respondents')
 
# ── 2. Country normalization ───────────────────────────
raw['country_raw'] = raw[COL_COUNTRY].copy()
raw[COL_COUNTRY] = raw['country_raw'].apply(normalize_country)
raw[[COL_ID,'country_raw',COL_COUNTRY]].to_csv('logs/country_normalization_log.csv', index=False)
print('Top 10 countries after normalization:')
print(raw[COL_COUNTRY].value_counts().head(10).to_string())
 
# ── 3. Clean age and sex ───────────────────────────────
raw[COL_AGE] = raw[COL_AGE].astype(str).str.strip()
raw.loc[raw[COL_AGE].isin(['nan','None','','#NULL!','#N/A']), COL_AGE] = 'unknown_age'
raw[COL_SEX] = raw[COL_SEX].astype(str).str.strip()
raw.loc[raw[COL_SEX].isin(['nan','None','']), COL_SEX] = 'unknown_sex'
 
raw.to_csv(OUT_DEM, index=False)
print(f'Clean demographics saved: {OUT_DEM}')
 
# ── 4. Eligibility filtering ───────────────────────────
MISSING_SENTINELS = {
    'nan','none','','unknown_age','unknown_sex','unknown_country',
    '#null!','#n/a','n/a','null','-','--','na','not available',
    'unknown','prefer not to say','prefer not to answer',
}
REQUIRED_DEMO_COLS = [COL_AGE, COL_SEX, COL_COUNTRY]
 
def is_eligible(row) -> bool:
    for col in REQUIRED_DEMO_COLS:
        if str(row[col]).strip().lower() in MISSING_SENTINELS: return False
    return True
 
raw['_eligible'] = raw.apply(is_eligible, axis=1)
eligible   = raw[raw['_eligible']].copy().reset_index(drop=True)
ineligible = raw[~raw['_eligible']].copy()
 
ELIGIBLE_SHUFFLE_SEED = 42
eligible = eligible.sample(frac=1, random_state=ELIGIBLE_SHUFFLE_SEED).reset_index(drop=True)
eligible[COL_SRC_HUMAN_ID] = ['H_' + str(i+1).zfill(4) for i in range(len(eligible))]
 
excl_rows = []
for _, row in ineligible.iterrows():
    reasons = [f'missing_{c}' for c in REQUIRED_DEMO_COLS
               if str(row[c]).strip().lower() in MISSING_SENTINELS]
    excl_rows.append({COL_ID: row[COL_ID], 'reasons': '|'.join(reasons)})
pd.DataFrame(excl_rows).to_csv('logs/eligibility_exclusions.csv', index=False)
 
print(f'\nEligibility: total={len(raw)} eligible={len(eligible)} excluded={len(ineligible)}')
print(f'Exclusion log: logs/eligibility_exclusions.csv')
 
eligible.drop(columns=['_eligible','country_raw'], errors='ignore').to_csv(ELIGIBLE_OUT, index=False)
print(f'Eligible humans saved: {ELIGIBLE_OUT}')
 
# ── 5. Human response distributions ───────────────────
# *** REPLACE THIS PLACEHOLDER with your actual Likert response data ***
# Load a separate Qualtrics export that has one column per question (q4_1 ... q10_4)
# with integer values 1-5. Map column names to lowercase q-id format.
# Then compute counts and probs below using real data.
#
print('\n⚠  Using placeholder response distributions — replace with real data!')
np.random.seed(0)
n_resp = len(raw)
placeholder = {qid: np.random.choice([1,2,3,4,5], size=n_resp, p=[0.04,0.08,0.18,0.38,0.32])
               for qid in ALL_Q_IDS}
resp_df = pd.DataFrame(placeholder)
resp_df.insert(0, COL_ID, raw[COL_ID].values)
# *** END PLACEHOLDER ***
 
baseline_rows = []
for qid in ALL_Q_IDS:
    vals = resp_df[qid].dropna().astype(int)
    counts = vals.value_counts().reindex([1,2,3,4,5], fill_value=0)
    probs = (counts / len(vals)).values
    construct = next(c for c,items in CONSTRUCT_ITEMS.items() if qid in items)
    baseline_rows.append({'q_id':qid,'p1':probs[0],'p2':probs[1],'p3':probs[2],
                          'p4':probs[3],'p5':probs[4],'n_valid':len(vals),
                          'mean':vals.mean(),'std':vals.std(),
                          'construct':construct,'scale_type':SCALE_TYPE[qid]})
 
if os.path.exists(OUT_BASE):
    raise FileExistsError(f'{OUT_BASE} already exists. Archive it before rerunning.')
pd.DataFrame(baseline_rows).to_csv(OUT_BASE, index=False)
print(f'Baseline saved: {OUT_BASE}')
 
sub_rows = []
for dim, label, mask in [
    (COL_AGE,'18-26',  raw[COL_AGE].isin(['18-21','22-26'])),
    (COL_AGE,'27plus', ~raw[COL_AGE].isin(['18-21','22-26','unknown_age'])),
    (COL_SEX,'Female', raw[COL_SEX]=='Female'),
    (COL_SEX,'Male',   raw[COL_SEX]=='Male'),
]:
    idx = raw.index[mask]
    sub = resp_df.loc[resp_df[COL_ID].isin(raw.loc[idx, COL_ID])]
    for qid in ALL_Q_IDS:
        vals = sub[qid].dropna().astype(int)
        if len(vals) < 20: continue
        counts = vals.value_counts().reindex([1,2,3,4,5], fill_value=0)
        probs = (counts/len(vals)).values
        sub_rows.append({'group_dim':dim,'group_label':label,'q_id':qid,
                         'p1':probs[0],'p2':probs[1],'p3':probs[2],
                         'p4':probs[3],'p5':probs[4],'n':len(vals)})
 
if os.path.exists(OUT_SUB): raise FileExistsError(f'{OUT_SUB} already exists.')
pd.DataFrame(sub_rows).to_csv(OUT_SUB, index=False)
print(f'Sub-baselines saved: {OUT_SUB}')
print('\n✅ data_prep.py complete.')
