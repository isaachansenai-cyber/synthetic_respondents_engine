"""
Generate correlated TAM latent traits for all eligible respondents.
N is dynamic from eligible_humans.csv. Fixed seeds for reproducibility.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
import numpy as np, pandas as pd
from src.canonical_schema import COL_SRC_HUMAN_ID
 
RANDOM_SEED   = 42
ELIGIBLE_PATH = 'data/profiles/eligible_humans.csv'
TRAITS_OUT    = 'data/profiles/latent_traits.csv'
 
if os.path.exists(TRAITS_OUT):
    raise FileExistsError(
        f'{TRAITS_OUT} already exists. If you re-ran data_prep.py, delete this file '
        f'then re-run generate_traits.py and build_profiles.py in sequence.'
    )
 
eligible = pd.read_csv(ELIGIBLE_PATH)
N = len(eligible)
print(f'Generating latent traits for N={N} eligible respondents...')
 
TRAIT_NAMES = ['tau_PU','tau_PEOU','tau_BI','tau_ATT','tau_AI_CAP']
TRAIT_CORR  = np.array([
    [1.00, 0.45, 0.50, 0.52, 0.40],
    [0.45, 1.00, 0.35, 0.38, 0.30],
    [0.50, 0.35, 1.00, 0.55, 0.30],
    [0.52, 0.38, 0.55, 1.00, 0.35],
    [0.40, 0.30, 0.30, 0.35, 1.00],
])
 
def z_to_level(z): return 'LOW' if z<-0.5 else 'HIGH' if z>0.5 else 'MODERATE'
 
VERBALS = {
    'tau_PU':     {'LOW':'You are skeptical that AI tools genuinely improve task performance.',
                   'MODERATE':'You think AI tools can be somewhat useful for certain tasks.',
                   'HIGH':'You strongly believe AI tools improve efficiency and outcomes.'},
    'tau_PEOU':   {'LOW':'You find new technology interfaces difficult and frustrating.',
                   'MODERATE':'You find most apps usable but sometimes need time to learn.',
                   'HIGH':'You pick up new technology quickly and find it intuitive.'},
    'tau_BI':     {'LOW':'You rarely choose to adopt new digital tools in daily life.',
                   'MODERATE':'You sometimes try new technology when it seems relevant.',
                   'HIGH':'You actively seek out and adopt useful new technology tools.'},
    'tau_ATT':    {'LOW':'You feel somewhat uncomfortable or skeptical about AI activities.',
                   'MODERATE':'You have a neutral-to-positive attitude toward AI activities.',
                   'HIGH':'You feel enthusiastic and positive about AI-assisted activities.'},
    'tau_AI_CAP': {'LOW':'You doubt that current AI handles complex real-world tasks well.',
                   'MODERATE':'You think current AI has reasonable but limited capability.',
                   'HIGH':'You believe current AI is capable of complex information tasks.'},
}
 
np.random.seed(RANDOM_SEED)
z_scores = np.random.multivariate_normal(np.zeros(5), TRAIT_CORR, size=N)
 
rows = []
for i in range(N):
    row = {COL_SRC_HUMAN_ID: eligible.iloc[i][COL_SRC_HUMAN_ID]}
    for j, tname in enumerate(TRAIT_NAMES):
        z = z_scores[i, j]
        lvl = z_to_level(z)
        row[tname]             = z
        row[f'{tname}_level']  = lvl
        row[f'{tname}_verbal'] = VERBALS[tname][lvl]
    rows.append(row)
 
traits_df = pd.DataFrame(rows)
print('Realized trait correlations:')
print(traits_df[TRAIT_NAMES].corr().round(2).to_string())
 
os.makedirs('data/profiles', exist_ok=True)
traits_df.to_csv(TRAITS_OUT, index=False)
print(f'\n✅ Latent traits saved: {TRAITS_OUT}  (N={len(traits_df)})')
