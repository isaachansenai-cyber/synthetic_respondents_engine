# rag_retriever.py — V4 RAG Retrieval Module
# Returns the 2 most similar RAG corpus humans for a given generation profile.
 
import pandas as pd
import numpy as np
import json
import argparse
 
RAG_FILE = 'data/profiles/v4_rag_corpus.csv'
 
# ── FEATURE WEIGHTS ───────────────────────────────────────
# Psychographic groups are weighted 2x relative to demographics.
# Within psychographics, all items are weighted equally.
# Adjust these if you want to experiment — document final values in your paper.
DEMO_WEIGHT   = 1.0   # weight for demographic features (age_num, sex_num, generation_num)
PSYCHO_WEIGHT = 2.0   # weight for all psychographic normalised features
 
# ── DEMOGRAPHIC ENCODINGS ────────────────────────────────
AGE_ORDER = ['18-21','22-26','27-31','32-36','37-42','43-49','50-58','59-64','66-70','71-77']
GEN_ORDER = ['younger Gen Z','older Gen Z','younger Millennials','older Millennials',
             'younger Gen X','older Gen X','younger Baby Boomers','older Baby Boomer','Silent Generation']
 
def encode_age(age_str):
    try: return AGE_ORDER.index(str(age_str)) / max(len(AGE_ORDER)-1, 1)
    except ValueError: return 0.5  # unknown age → midpoint
 
def encode_sex(sex_str):
    return 0.0 if str(sex_str).strip().lower() == 'female' else 1.0
 
def encode_generation(gen_str):
    try: return GEN_ORDER.index(str(gen_str)) / max(len(GEN_ORDER)-1, 1)
    except ValueError: return 0.5
 
NORM_PSYCHO_COLS = [
    'fam_1_norm','fam_2_norm','fam_3_norm','fam_4_norm','fam_5_norm','fam_6_norm','fam_7_norm',
    'use_1_norm','use_2_norm','use_3_norm','use_4_norm','use_5_norm','use_6_norm',
    'use_7_norm','use_8_norm','use_9_norm','use_10_norm','use_11_norm',
    'risk_1_norm','risk_2_norm','risk_3_norm','risk_4_norm','risk_5_norm','risk_6_norm',
]
 
def profile_to_vector(row):
    """Convert a profile row (dict or Series) to a weighted feature vector."""
    demo = np.array([
        encode_age(row.get('age', '22-26') if isinstance(row, dict) else row['age']),
        encode_sex(row.get('sex', 'Female') if isinstance(row, dict) else row['sex']),
        encode_generation(row.get('generation', 'younger Gen Z') if isinstance(row, dict) else row['generation']),
    ]) * DEMO_WEIGHT
 
    if isinstance(row, dict):
        psycho = np.array([float(row.get(c, 0.5)) for c in NORM_PSYCHO_COLS])
    else:
        psycho = np.array([float(row[c]) if pd.notna(row[c]) else 0.5 for c in NORM_PSYCHO_COLS])
    psycho = psycho * PSYCHO_WEIGHT
 
    return np.concatenate([demo, psycho])
 
# ── LOAD AND CACHE RAG CORPUS ────────────────────────────
_rag_df   = None
_rag_vecs = None
 
def _load_rag():
    global _rag_df, _rag_vecs
    if _rag_df is not None:
        return
    _rag_df   = pd.read_csv(RAG_FILE)
    _rag_vecs = np.array([profile_to_vector(_rag_df.iloc[i]) for i in range(len(_rag_df))])
 
def get_exemplars(profile_row):
    """
    Given a generation profile (pandas Series or dict), return a list of 2
    RAG corpus dicts: the 2 nearest by weighted Euclidean distance.
    Each returned dict includes all profile fields + full_survey_responses.
    Also returns the retrieval distances for logging.
    """
    _load_rag()
    vec = profile_to_vector(profile_row)
    diffs = _rag_vecs - vec
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    top2_idx = np.argsort(distances)[:2]
    exemplars = []
    for idx in top2_idx:
        row = _rag_df.iloc[idx].to_dict()
        row['_retrieval_distance'] = float(distances[idx])
        exemplars.append(row)
    return exemplars
 
# ── COVERAGE CHECK MODE ──────────────────────────────────
def run_coverage_check():
    _load_rag()
    gen = pd.read_csv('data/profiles/v4_generation_profiles.csv')
    print(f'RAG corpus size:      {len(_rag_df)}')
    print(f'Generation pool size: {len(gen)}')
    print('Computing nearest-exemplar distances for all generation profiles...')
    distances = []
    for i in range(len(gen)):
        vec = profile_to_vector(gen.iloc[i])
        diffs = _rag_vecs - vec
        d = np.sqrt((diffs ** 2).sum(axis=1)).min()
        distances.append(d)
    distances = np.array(distances)
    mean_d = distances.mean()
    std_d  = distances.std()
    max_d  = distances.max()
    threshold = mean_d + 2 * std_d
    flagged = (distances > threshold).sum()
    print(f'Mean nearest-exemplar distance: {mean_d:.4f}')
    print(f'Std:                            {std_d:.4f}')
    print(f'Max:                            {max_d:.4f}')
    print(f'Flagged profiles (>2 SD):       {flagged} ({100*flagged/len(gen):.1f}%)')
    if flagged / len(gen) < 0.10:
        print('Coverage check: PASSED')
    else:
        print('Coverage check: WARNING — >10% of profiles have poor RAG coverage')
    return distances
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='coverage_check')
    args = parser.parse_args()
    if args.mode == 'coverage_check':
        run_coverage_check()
