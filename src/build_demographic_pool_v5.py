# build_demographic_pool_v5.py -- V5 Demographic Pool Construction
# Reads ONLY the demographic columns from the 850-person file.
# Builds three panel-weighted sampling pools (age, sex, location pairs).
# Generates 5,000 synthetic profiles by independent random draws.

import pandas as pd
import numpy as np
import os

# -- FILE PATHS ----------------------------------------------------------
PSYCHO_FILE  = 'data/psychographics/SMA 2026 TAM surveys virtual twins - 850 people with all demo and psycho.xlsx'
OUT_PROFILES = 'data/profiles/v5_synthetic_profiles.csv'
N_SYNTHETIC  = 5000
RANDOM_SEED  = 42

os.makedirs('data/profiles', exist_ok=True)

# -- LOAD DEMOGRAPHICS ONLY ----------------------------------------------
# usecols restricts to exactly four demographic columns.
# No psychographic or TAM data is loaded at any point in this script.
print('Loading 850-person file (demographics only)...')
df = pd.read_excel(
    PSYCHO_FILE,
    usecols=[
        'What is your age?',
        'What is your biological sex?',
        'What country is home for you?',
        'What state or city is home for you?',
    ]
)
print(f'  Loaded {len(df)} rows, {df.shape[1]} cols (expect 4)')

# Rename to short names
df = df.rename(columns={
    'What is your age?':                   'age',
    'What is your biological sex?':        'sex',
    'What country is home for you?':       'country',
    'What state or city is home for you?': 'city',
})

# Fill missing values
df['age']     = df['age'].fillna(df['age'].mode()[0])
df['sex']     = df['sex'].fillna(df['sex'].mode()[0])
df['country'] = df['country'].fillna('Unknown')
df['city']    = df['city'].fillna('Unknown')

print(f'  Nulls remaining: {df.isnull().sum().sum()} (must be 0)')
print(f'  Age unique values:     {df["age"].nunique()}')
print(f'  Sex unique values:     {df["sex"].nunique()}')
print(f'  Location unique pairs: {df.groupby(["country","city"]).ngroups}')

# -- BUILD THREE INDEPENDENT SAMPLING POOLS ------------------------------
# Each pool contains all 850 values, preserving the panel's marginal
# distribution. Sampling with replacement from each pool reproduces
# the observed demographic frequencies independently per dimension.
age_pool      = df['age'].tolist()                    # 850 age values
sex_pool      = df['sex'].tolist()                    # 850 sex values
location_pool = list(zip(df['country'], df['city']))  # 850 (country, city) pairs

print(f'  Pool sizes: age={len(age_pool)}, sex={len(sex_pool)}, location={len(location_pool)}')

# -- GENERATE 5,000 SYNTHETIC PROFILES ----------------------------------
# Three independent index draws. Age, sex, and location are sampled
# separately -- they are NOT correlated with each other in the synthetic pool.
print(f'Generating {N_SYNTHETIC} synthetic profiles (seed={RANDOM_SEED})...')
rng = np.random.default_rng(RANDOM_SEED)

idx_age = rng.choice(len(age_pool),      size=N_SYNTHETIC, replace=True)
idx_sex = rng.choice(len(sex_pool),      size=N_SYNTHETIC, replace=True)
idx_loc = rng.choice(len(location_pool), size=N_SYNTHETIC, replace=True)

profiles = pd.DataFrame({
    'synthetic_id': [f'S_{i+1:04d}' for i in range(N_SYNTHETIC)],
    'age':          [age_pool[i]         for i in idx_age],
    'sex':          [sex_pool[i]         for i in idx_sex],
    'country':      [location_pool[i][0] for i in idx_loc],
    'city':         [location_pool[i][1] for i in idx_loc],
})

profiles.to_csv(OUT_PROFILES, index=False)
print(f'  Saved to {OUT_PROFILES}')

# -- VERIFICATION SUMMARY ------------------------------------------------
print()
print('=== BUILD COMPLETE ===')
print(f'Total profiles:  {len(profiles)} (expect {N_SYNTHETIC})')
print(f'Nulls in output: {profiles.isnull().sum().sum()} (must be 0)')
print(f'ID range:        {profiles["synthetic_id"].iloc[0]} to {profiles["synthetic_id"].iloc[-1]}')
print()
print('Age distribution (top 5):')
print(profiles['age'].value_counts().head())
print()
print('Sex distribution:')
print(profiles['sex'].value_counts())
print()
print('Top 5 location pairs:')
print(profiles.groupby(['country','city']).size().sort_values(ascending=False).head())
