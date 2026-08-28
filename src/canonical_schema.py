"""
Canonical schema contract. All column names, valid values, normalisation,
country mapping, and JSON extraction live here.
"""
import re, json
import pandas as pd
 
# ── Column constants ───────────────────────────────────
COL_DATE         = 'RecordedDate'
COL_ID           = 'ResponseId'
COL_AGE          = 'age'
COL_SEX          = 'sex'
COL_COUNTRY      = 'country'
COL_CITY         = 'city'
COL_QID          = 'question_id'
COL_RESP         = 'response'
COL_SYNID        = 'synthetic_id'
COL_SRC_HUMAN_ID = 'source_human_id'
COL_ARM          = 'conditioning_arm'
COL_CONF         = 'confidence'
COL_MODEL        = 'model'
COL_REASON       = 'reasoning'
 
# ── Rename map (fixes corrupted Qualtrics header) ──────
RENAME_MAP = {
    'beh_in+G:ACtent_1':  COL_CITY,
    ' beh_in+G:ACtent_1': COL_CITY,
}
 
VALID_OPTIONS = {'1','2','3','4','5'}
 
# ── TAM construct item groups ──────────────────────────
CONSTRUCT_ITEMS = {
    'behavioral_intention': [f'q4_{i}' for i in range(1,8)],
    'perceived_usefulness': [f'q5_{i}' for i in range(1,9)],
    'ease_of_use':          [f'q7_{i}' for i in range(1,7)],
    'ai_capability':        [f'q9_{i}' for i in range(1,4)],
    'attitude':             [f'q10_{i}' for i in range(1,5)],
}
ALL_Q_IDS = [q for items in CONSTRUCT_ITEMS.values() for q in items]
 
SCALE_TYPE = {
    **{q:'scale_agree'  for q in CONSTRUCT_ITEMS['behavioral_intention']},
    **{q:'scale_agree'  for q in CONSTRUCT_ITEMS['perceived_usefulness']},
    **{q:'scale_useful' for q in CONSTRUCT_ITEMS['ease_of_use']},
    **{q:'scale_likely' for q in CONSTRUCT_ITEMS['ai_capability']},
    **{q:'scale_agree'  for q in CONSTRUCT_ITEMS['attitude']},
}
 
SCALE_LABELS = {
    'scale_agree':  '1=Strongly Disagree  2=Disagree  3=Neutral  4=Agree  5=Strongly Agree',
    'scale_useful': '1=Not at all easy  2=Slightly easy  3=Moderately easy  4=Very easy  5=Extremely easy',
    'scale_likely': '1=Not at all likely  2=Unlikely  3=Somewhat likely  4=Likely  5=Extremely likely',
}
 
REVERSE_ITEMS = {'q7_6'}
 
# ── Country normalization ──────────────────────────────
COUNTRY_MAP = {
    'united states':'United States','usa':'United States','us':'United States',
    'u.s.':'United States','u.s.a':'United States','america':'United States',
    'united states of america':'United States','u.s of a baby':'United States',
    'u.s of a baby!':'United States','american':'United States',
    'philippines':'Philippines','ph':'Philippines','philipines':'Philippines',
    'philippine':'Philippines','philiphhines':'Philippines',
    'mexico':'Mexico','méxico':'Mexico',
    'tahiti':'French Polynesia','french polynesia, tahiti':'French Polynesia',
    'tahiti, french polynesia':'French Polynesia','tahiti/pirae':'French Polynesia',
    'papua new guinea':'Papua New Guinea','png':'Papua New Guinea',
    'mongolia':'Mongolia',
    'south korea':'South Korea','korea':'South Korea',
    'kiribati':'Kiribati','tarawa':'Kiribati',
    'new zealand':'New Zealand','nz':'New Zealand','aotearoa':'New Zealand',
    'japan':'Japan','jaoan':'Japan',
    'fiji':'Fiji','foji':'Fiji',
    'sri lanka':'Sri Lanka',
    'samoa':'Samoa','sāmoa':'Samoa',
    'canada':'Canada',
    'england':'United Kingdom','scotland':'United Kingdom',
    'uk':'United Kingdom','great britain':'United Kingdom',
    'indonesia':'Indonesia','china':'China','tonga':'Tonga',
    'vanuatu':'Vanuatu','solomon islands':'Solomon Islands',
}
 
def normalize_country(raw) -> str:
    if pd.isna(raw): return None
    cleaned = str(raw).strip().lower()
    if cleaned in COUNTRY_MAP: return COUNTRY_MAP[cleaned]
    try:
        from rapidfuzz import process
        match, score, _ = process.extractOne(cleaned, COUNTRY_MAP.keys())
        if score >= 85: return COUNTRY_MAP[match]
    except Exception: pass
    return str(raw).strip().title()
 
# ── Response normalisation ─────────────────────────────
def normalise_response(v) -> str:
    s = str(v).strip().lower()
    m = {'strongly disagree':'1','not at all easy':'1','not at all likely':'1',
         'not at all useful':'1','disagree':'2','slightly easy':'2','unlikely':'2',
         'slightly useful':'2','neutral':'3','moderately easy':'3',
         'somewhat likely':'3','moderately useful':'3','agree':'4','very easy':'4',
         'likely':'4','very useful':'4','strongly agree':'5','extremely easy':'5',
         'extremely likely':'5','extremely useful':'5'}
    if s in m: return m[s]
    hit = re.search(r'\b([1-5])\b', s)
    return hit.group(1) if hit else None
 
# ── Schema assertion ───────────────────────────────────
def assert_schema(df, required_cols, context=''):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise AssertionError(f'[{context}] Missing cols: {missing}\nHave: {list(df.columns)}')
 
# ── JSON extraction ────────────────────────────────────
def extract_json(raw: str) -> dict:
    cleaned = re.sub(r'```(?:json)?', '', raw).strip()
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if not m: raise ValueError(f'No JSON found in: {raw[:200]}')
    obj = json.loads(m.group())
    return {k.lower().replace('-','_'): v for k,v in obj.items()}
