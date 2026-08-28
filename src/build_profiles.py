"""
Build profiles.csv: 1:1 merge of eligible human demographics with latent traits.
Assigns conditioning_arm and samples chatgpt_usage / travel_freq.
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
PROFILES_OUT  = 'data/profiles/profiles.csv'
MIN_ARM_SIZE  = 10
PROFILE_SEED  = 123
 
eligible = pd.read_csv(ELIGIBLE_PATH)
traits   = pd.read_csv(TRAITS_PATH)
 
assert_schema(eligible, [COL_SRC_HUMAN_ID, COL_AGE, COL_SEX, COL_COUNTRY], 'eligible')
assert_schema(traits,   [COL_SRC_HUMAN_ID], 'traits')
 
# ── 1:1 merge on source_human_id ───────────────────────
profiles = eligible.merge(traits, on=COL_SRC_HUMAN_ID, validate='1:1', how='inner')
N = len(profiles)
assert N == len(eligible), (
    f'Merge lost rows: eligible={len(eligible)} merged={N}. '
    f'Rerun data_prep.py then generate_traits.py then build_profiles.py in order.'
)
print(f'Merged profiles: N={N}')
 
# ── Assign synthetic_id ────────────────────────────────
profiles[COL_SYNID] = ['SYN_' + str(i+1).zfill(4) for i in range(N)]
 
# ── Assign conditioning_arm ────────────────────────────
# Row order is already stable (shuffled once in data_prep.py with seed 42)
n_traits_arm = math.ceil(N / 2)
profiles[COL_ARM] = (['demo_plus_traits'] * n_traits_arm +
                     ['demo_only']        * (N - n_traits_arm))
 
arm_counts = profiles[COL_ARM].value_counts()
for arm_name, arm_n in arm_counts.items():
    if arm_n < MIN_ARM_SIZE:
        raise ValueError(f'Arm "{arm_name}" has only {arm_n} profiles (min={MIN_ARM_SIZE})')
print(f'Arms: demo_plus_traits={arm_counts.get("demo_plus_traits",0)} demo_only={arm_counts.get("demo_only",0)}')
 
# ── Sample chatgpt_usage and travel_freq ───────────────
# These fields are not in the human data — sampled independently as synthetic priors.
# Age-stratified sampling for chatgpt_usage (young adults adopt more).
rng = np.random.default_rng(PROFILE_SEED)
 
def _chatgpt(age_bracket):
    young = age_bracket in ['18-21','22-26']
    p = [0.25,0.42,0.33] if young else [0.45,0.38,0.17]
    return rng.choice(['never used','occasionally used','regularly used'], p=p)
 
def _travel():
    return rng.choice(['rarely travels','occasional traveler','frequent traveler'],
                      p=[0.30,0.50,0.20])
 
profiles['chatgpt_usage'] = [_chatgpt(r) for r in profiles[COL_AGE]]
profiles['travel_freq']   = [_travel() for _ in range(N)]
 
# ── Reorder columns ────────────────────────────────────
front = [COL_SYNID, COL_SRC_HUMAN_ID, COL_ARM, COL_AGE, COL_SEX, COL_COUNTRY,
         'chatgpt_usage','travel_freq']
if COL_CITY in profiles.columns: front.insert(6, COL_CITY)
others = [c for c in profiles.columns if c not in front]
profiles = profiles[front + others]
 
os.makedirs('data/profiles', exist_ok=True)
profiles.to_csv(PROFILES_OUT, index=False)
print(f'\n✅ profiles.csv saved: {PROFILES_OUT}  ({N} rows)')
print(profiles[[COL_SYNID, COL_SRC_HUMAN_ID, COL_ARM, COL_AGE, 'chatgpt_usage']].head(8).to_string())
