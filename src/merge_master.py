"""Build master results table joining all analytical outputs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
import pandas as pd
from src.canonical_schema import ALL_Q_IDS, CONSTRUCT_ITEMS, SCALE_TYPE, REVERSE_ITEMS, assert_schema
 
OUT_PATH = 'outputs/master_results.csv'
 
validity = pd.read_csv('outputs/metrics/validity_classifications.csv')
assert len(validity)==28, f'Expected 28 rows, got {len(validity)}'
 
js = pd.read_csv('outputs/metrics/js_distances.csv')
js_wide = js.pivot(index='q_id', columns='model', values='js_human_syn')
js_wide.columns = [f'js_{m}' for m in js_wide.columns]
js_wide = js_wide.reset_index()
assert len(js_wide)==28
 
# NaN check after pivot
nan_check = js_wide.isnull().sum()
if nan_check.any():
    print(f'⚠ NaN values in JS pivot — check that all models completed generation:')
    print(nan_check[nan_check>0].to_string())
 
meta_rows = []
for qid in ALL_Q_IDS:
    construct = next(c for c,items in CONSTRUCT_ITEMS.items() if qid in items)
    meta_rows.append({
        'q_id':             qid,
        'tam_construct':    construct,
        'scale_type':       SCALE_TYPE[qid],
        'is_reverse_coded': qid in REVERSE_ITEMS,
        'attribute_type':   'emotional' if construct=='attitude' else 'functional',
        'requires_lived_exp': construct in ['ease_of_use','attitude'],
    })
meta_df = pd.DataFrame(meta_rows)
 
master = meta_df.merge(validity, on='q_id', validate='1:1')
assert len(master)==28, f'After validity merge: {len(master)}'
master = master.merge(js_wide,  on='q_id', validate='1:1')
assert len(master)==28, f'After JS merge: {len(master)}'
 
master.to_csv(OUT_PATH, index=False)
print(f'\n✅ Master table: {OUT_PATH}  ({len(master)} rows, {len(master.columns)} cols)')
print('\nValidity class distribution:')
print(master['validity_class'].value_counts().to_string())
js_cols = [c for c in master.columns if c.startswith('js_')]
print('\nMean JS by construct:')
print(master.groupby('tam_construct')[js_cols].mean().round(3).to_string())
