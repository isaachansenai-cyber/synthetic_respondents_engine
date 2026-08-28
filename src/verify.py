"""
Quality verification for all generated response files.
N is loaded dynamically from profiles.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.canonical_schema import ALL_Q_IDS, CONSTRUCT_ITEMS, REVERSE_ITEMS

MODELS = ["gpt4", "claude", "gemini"]
POSITIVE_ITEMS = [q for q in ALL_Q_IDS if q not in REVERSE_ITEMS]

# Dynamic N from profiles
profiles_df = pd.read_csv("data/profiles/profiles.csv")
EXPECTED_N = len(profiles_df)
print(f"Expected N per model: {EXPECTED_N} (from profiles.csv)")

for model in MODELS:
    path = f"outputs/{model}/{model}_full_responses.csv"
    if not os.path.exists(path):
        print(f"\n⚠ {model}: file not found — skip")
        continue

    df = pd.read_csv(path)
    Q = ALL_Q_IDS

    print(f"\n{'=' * 55}")
    print(f"Model: {model} | n={len(df)}")

    # 1 — Row count
    assert len(df) == EXPECTED_N, f"Expected {EXPECTED_N} rows, got {len(df)}"
    print(f"✅ Row count: {len(df)}")

    # 2 — All 28 q_id columns present
    missing_cols = [q for q in Q if q not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    print("✅ All 28 question columns present")

    # 3 — All values 1-5
    invalid = df[Q].stack().loc[lambda s: ~s.isin([1, 2, 3, 4, 5])]
    assert len(invalid) == 0, f"Invalid values: {invalid.value_counts().to_dict()}"
    print("✅ All responses integers 1-5")

    # 4 — Circularity: no profile scores >20/28 items at 5
    max_fives = (df[Q] == 5).sum(axis=1).max()
    flag = "⚠ CIRCULARITY RISK" if max_fives > 20 else "✅"
    print(f"{flag} Max items scored 5 per profile: {max_fives} (threshold 20)")

    # 5 — Mean of positive items <= 3.8
    pos_mean = df[POSITIVE_ITEMS].values.mean()
    flag = "⚠ CIRCULARITY RISK" if pos_mean > 3.8 else "✅"
    print(f"{flag} Positive-item mean: {pos_mean:.3f} (target <=3.8)")

    # 6 — Q7_6 reverse coding (must be negative correlation with Q7_1-5 mean)
    r = df["q7_6"].corr(df[["q7_1", "q7_2", "q7_3", "q7_4", "q7_5"]].mean(axis=1))
    flag = "✅" if r < 0 else "❌ REVERSE CODING FAILED — STOP AND FIX"
    print(f"{flag} Q7_6 correlation with Q7_1-5 mean: {r:.3f} (must be negative)")

    if r >= 0:
        print("    ▶ Fix: strengthen reverse instruction in question_bank.py q7_6 text")
        print("    ▶ Then delete parsed_json files for this model and rerun generate.py")

    # 7 — Between-profile variance
    mean_sd = df[Q].std().mean()
    flag = "✅" if mean_sd >= 0.6 else "⚠ LOW VARIANCE — consider raising temperature in generate.py"
    print(f"{flag} Mean item SD across profiles: {mean_sd:.3f} (target >=0.6)")

    # 8 — Within-construct correlation (PU construct as bellwether)
    pu_items = CONSTRUCT_ITEMS["perceived_usefulness"]
    pu_corr = df[pu_items].corr().to_numpy(copy=True)
    np.fill_diagonal(pu_corr, np.nan)
    mean_pu = np.nanmean(pu_corr)
    flag = "✅" if mean_pu >= 0.30 else "⚠ LOW — single-call generation may not be preserving construct structure"
    print(f"{flag} Mean within-PU correlation: {mean_pu:.3f} (target >=0.30)")

print("\n✅ Verification complete.")