"""
Build study2_profiles.csv for the 7-batch prompt funnel experiment.
 
Selects the FIRST 200 eligible respondents by source_human_id (H_0001–H_0200).
This is deterministic — not a random sample. The order was fixed in data_prep.py
with ELIGIBLE_SHUFFLE_SEED=42. All 7 funnel batches use these same 200 profiles.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd, numpy as np
from src.canonical_schema import (
    COL_SRC_HUMAN_ID, COL_ARM, COL_AGE, COL_SEX, COL_COUNTRY,
    COL_CITY, COL_SYNID, assert_schema
)
 
ELIGIBLE_PATH = 'data/profiles/eligible_humans.csv'
TRAITS_PATH   = 'data/profiles/latent_traits.csv'
PROFILES_OUT  = 'data/profiles/study2_profiles.csv'
N_SAMPLE      = 200
MIN_ARM_SIZE  = 10
PROFILE_SEED  = 456   # Used only for chatgpt_usage / travel_freq sampling
 
eligible = pd.read_csv(ELIGIBLE_PATH)
traits   = pd.read_csv(TRAITS_PATH)
 
assert len(eligible) >= N_SAMPLE, (
    f'Only {len(eligible)} eligible respondents — cannot take first {N_SAMPLE}. '
    'Check eligibility filtering in data_prep.py.'
)
 
# ── Deterministic first-N selection ────────────────────
sample = eligible.head(N_SAMPLE).copy().reset_index(drop=True)
 
# Verify expected source_human_id range
expected_ids = [f'H_{str(i+1).zfill(4)}' for i in range(N_SAMPLE)]
actual_ids   = sample[COL_SRC_HUMAN_ID].tolist()
assert actual_ids == expected_ids, (
    f'source_human_id mismatch. Expected H_0001..H_{str(N_SAMPLE).zfill(4)}, '
    f'got {actual_ids[:3]}...{actual_ids[-3:]}. '
    'Regenerate eligible_humans.csv with data_prep.py using ELIGIBLE_SHUFFLE_SEED=42.'
)
print(f'Selected: {actual_ids[0]} to {actual_ids[-1]} (deterministic)')
 
# ── Merge latent traits 1:1 ────────────────────────────
profiles = sample.merge(traits, on=COL_SRC_HUMAN_ID, validate='1:1', how='inner')
assert len(profiles) == N_SAMPLE, (
    f'Merge lost rows: expected {N_SAMPLE}, got {len(profiles)}. '
    'Ensure latent_traits.csv covers the full eligible frame.'
)
 
# ── Assign synthetic_ids ───────────────────────────────
profiles[COL_SYNID] = ['S2_' + str(i+1).zfill(4) for i in range(N_SAMPLE)]
 
# ── Assign conditioning_arm ────────────────────────────
n_traits = math.ceil(N_SAMPLE / 2)
profiles[COL_ARM] = (['demo_plus_traits'] * n_traits +
                     ['demo_only']        * (N_SAMPLE - n_traits))
for arm, n in profiles[COL_ARM].value_counts().items():
    assert n >= MIN_ARM_SIZE, f'Arm {arm} has only {n} profiles (min={MIN_ARM_SIZE})'
print(f'Arms: {profiles[COL_ARM].value_counts().to_dict()}')
 
# ── Sample chatgpt_usage and travel_freq ───────────────
rng = np.random.default_rng(PROFILE_SEED)
def _chatgpt(age):
    p = [0.25,0.42,0.33] if age in ['18-21','22-26'] else [0.45,0.38,0.17]
    return rng.choice(['never used','occasionally used','regularly used'], p=p)
def _travel(): return rng.choice(['rarely travels','occasional traveler','frequent traveler'], p=[0.30,0.50,0.20])
profiles['chatgpt_usage'] = [_chatgpt(r) for r in profiles[COL_AGE]]
profiles['travel_freq']   = [_travel() for _ in range(N_SAMPLE)]
 
# ── Reorder columns ────────────────────────────────────
front = [COL_SYNID, COL_SRC_HUMAN_ID, COL_ARM, COL_AGE, COL_SEX, COL_COUNTRY, 'chatgpt_usage','travel_freq']
if COL_CITY in profiles.columns: front.insert(6, COL_CITY)
profiles = profiles[front + [c for c in profiles.columns if c not in front]]
 
os.makedirs('data/profiles', exist_ok=True)
if os.path.exists(PROFILES_OUT):
    raise FileExistsError(f'{PROFILES_OUT} already exists. Delete it to rebuild.')
profiles.to_csv(PROFILES_OUT, index=False)
print(f'\n✅ study2_profiles.csv saved: {PROFILES_OUT}  (N={len(profiles)})')
print(profiles[[COL_SYNID, COL_SRC_HUMAN_ID, COL_ARM, COL_AGE, 'chatgpt_usage']].head(6).to_string())
