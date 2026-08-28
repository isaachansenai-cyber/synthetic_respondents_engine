# generate_v5.py -- V5 Main Generation Loop
# Claude Sonnet only. Batch 7: minimal demographic profile + task description.
# Run: python src\generate_v5.py [--micro]

import pandas as pd
import json, os, time, argparse
import anthropic
from dotenv import load_dotenv
import sys
sys.path.insert(0, 'src')
from prompt_builder_v5 import build_v5_prompt

load_dotenv()

# -- CONFIG --------------------------------------------------------------
PROFILES_FILE = 'data/profiles/v5_synthetic_profiles.csv'
MODEL         = 'claude-sonnet-4-6'
TEMPERATURE   = 0.9
MAX_TOKENS    = 1000
SLEEP_BETWEEN = 3       # seconds between API calls -- increase if rate limited
MICRO_N       = 5       # profiles for micro-test gate

Q_KEYS = [
    'q4_1','q4_2','q4_3','q4_4','q4_5','q4_6','q4_7',
    'q5_1','q5_2','q5_3','q5_4','q5_5','q5_6','q5_7','q5_8',
    'q6_1','q6_2','q6_3','q6_4','q6_5','q6_6','q6_7',
    'q7_1','q7_2','q7_3','q7_4','q7_5','q7_6',
    'q8_1','q8_2','q8_3','q8_4','q8_5','q8_6','q8_7',
    'q9_1','q9_2','q9_3',
    'q10_1','q10_2','q10_3','q10_4',
]

# -- API CALL ------------------------------------------------------------
def call_claude(system_prompt, user_prompt):
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}]
    )
    return resp.content[0].text.strip()

# -- RESPONSE PARSING ----------------------------------------------------
def parse_response(raw_text):
    """Extract and validate JSON from model output. Returns dict or raises ValueError."""
    text = raw_text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1])
    parsed = json.loads(text)
    missing = [k for k in Q_KEYS if k not in parsed]
    if missing:
        raise ValueError(f'Missing keys: {missing}')
    for k in Q_KEYS:
        v = int(parsed[k])
        if v < 1 or v > 5:
            raise ValueError(f'{k} = {v} out of range 1-5')
        parsed[k] = v
    return parsed

# -- MAIN GENERATION LOOP ------------------------------------------------
def run_generation(micro=False):
    profiles = pd.read_csv(PROFILES_FILE)
    if micro:
        profiles = profiles.head(MICRO_N)
        print(f'[MICRO] Running {MICRO_N} profiles')
    else:
        print(f'[FULL] Running {len(profiles)} profiles')

    suffix   = 'micro' if micro else 'full'
    out_dir  = 'outputs/claude/v5'
    out_file = f'{out_dir}/claude_v5_{suffix}_responses.csv'
    err_file = f'{out_dir}/claude_v5_{suffix}_errors.csv'
    os.makedirs(out_dir, exist_ok=True)

    # Resume logic: load already-completed synthetic IDs
    done_ids = set()
    if os.path.exists(out_file):
        done = pd.read_csv(out_file)
        done_ids = set(done['synthetic_id'])
        print(f'Resume: {len(done_ids)} profiles already complete, skipping...')

    results = []
    errors  = []
    total   = len(profiles)

    for i, row in profiles.iterrows():
        sid = row['synthetic_id']
        if sid in done_ids:
            continue

        print(f'  [{i+1}/{total}] {sid}...', end=' ', flush=True)

        try:
            system_p, user_p = build_v5_prompt(row)
            raw    = call_claude(system_p, user_p)
            parsed = parse_response(raw)

            result_row = {
                'synthetic_id': sid,
                'age':          row['age'],
                'sex':          row['sex'],
                'city':         row['city'],
                'country':      row['country'],
            }
            for k in Q_KEYS:
                result_row[k] = parsed[k]
            results.append(result_row)

            # Write immediately after each profile (resume safety)
            out_df = pd.DataFrame(results)
            if os.path.exists(out_file):
                existing = pd.read_csv(out_file)
                out_df = pd.concat([existing, out_df], ignore_index=True)
            out_df.to_csv(out_file, index=False)
            results = []   # clear buffer after writing
            print('OK')

        except Exception as e:
            print(f'ERROR: {e}')
            errors.append({'synthetic_id': sid, 'error': str(e)})
            pd.DataFrame(errors).to_csv(err_file, index=False)

        time.sleep(SLEEP_BETWEEN)

    print()
    print(f'Generation complete. Output: {out_file}')
    if errors:
        print(f'Errors logged to: {err_file} ({len(errors)} profiles failed)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro', action='store_true', help='Run micro-test (5 profiles only)')
    args = parser.parse_args()
    run_generation(micro=args.micro)
