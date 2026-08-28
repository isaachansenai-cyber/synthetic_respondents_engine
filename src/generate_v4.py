# generate.py — V4 Main Generation Loop
# Run: python src\generate.py --model gpt4 [--micro]
#      python src\generate.py --model claude [--micro]
 
import pandas as pd
import json, os, time, argparse
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
import sys
sys.path.insert(0, 'src')
from rag_retriever import get_exemplars
from prompt_builder_v4 import build_v4_system_prompt
 
load_dotenv()
 
# ── CONFIG ───────────────────────────────────────────────
PROFILES_FILE = 'data/profiles/v4_generation_profiles.csv'
TEMPERATURE   = 0.9
MAX_TOKENS    = 2000
SLEEP_BETWEEN = 3   # seconds between API calls — increase if you hit rate limits
MICRO_N       = 5   # number of profiles for micro-test
 
Q_KEYS = [
    'q4_1','q4_2','q4_3','q4_4','q4_5','q4_6','q4_7',
    'q5_1','q5_2','q5_3','q5_4','q5_5','q5_6','q5_7','q5_8',
    'q6_1','q6_2','q6_3','q6_4','q6_5','q6_6','q6_7',
    'q7_1','q7_2','q7_3','q7_4','q7_5','q7_6',
    'q8_1','q8_2','q8_3','q8_4','q8_5','q8_6','q8_7',
    'q9_1','q9_2','q9_3',
    'q10_1','q10_2','q10_3','q10_4',
]
 
def call_gpt4(system_prompt, user_prompt):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    resp = client.chat.completions.create(
        model='gpt-4o',
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ]
    )
    return resp.choices[0].message.content.strip()
 
def call_claude(system_prompt, user_prompt):
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    resp = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}]
    )
    return resp.content[0].text.strip()
 
def parse_response(raw_text):
    """Extract and validate JSON from model output. Returns dict or raises ValueError."""
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1])
    parsed = json.loads(text)
    # Validate all required keys present
    missing = [k for k in Q_KEYS if k not in parsed]
    if missing:
        raise ValueError(f'Missing keys: {missing}')
    # Validate all values are integers 1-5
    for k in Q_KEYS:
        v = int(parsed[k])
        if v < 1 or v > 5:
            raise ValueError(f'{k} = {v} is out of range 1-5')
        parsed[k] = v
    if 'reasoning' not in parsed or len(str(parsed.get('reasoning',''))) < 20:
        raise ValueError('reasoning field missing or too short')
    return parsed
 
def run_generation(model_name, micro=False):
    profiles = pd.read_csv(PROFILES_FILE)
    if micro:
        profiles = profiles.head(MICRO_N)
        print(f'[MICRO] Running {MICRO_N} profiles for {model_name}')
    else:
        print(f'[FULL] Running {len(profiles)} profiles for {model_name}')
 
    suffix     = 'micro' if micro else 'full'
    out_dir    = f'outputs/{model_name}/v4'
    out_file   = f'{out_dir}/{model_name}_v4_{suffix}_responses.csv'
    err_file   = f'{out_dir}/{model_name}_v4_{suffix}_errors.csv'
    os.makedirs(out_dir, exist_ok=True)
 
    # Load already-completed IDs for resume logic
    done_ids = set()
    if os.path.exists(out_file):
        done = pd.read_csv(out_file)
        done_ids = set(done['source_human_id'])
        print(f'Resume: {len(done_ids)} profiles already complete, skipping...')
 
    call_fn = call_gpt4 if model_name == 'gpt4' else call_claude
 
    results = []
    errors  = []
    total   = len(profiles)
 
    for i, row in profiles.iterrows():
        sid = row['source_human_id']
        if sid in done_ids:
            continue
 
        print(f'  [{i+1}/{total}] {sid}...', end=' ', flush=True)
 
        try:
            exemplars = get_exemplars(row)
            system_p, user_p = build_v4_system_prompt(row, exemplars)
            raw = call_fn(system_p, user_p)
            parsed = parse_response(raw)
            rag_dist = (exemplars[0]['_retrieval_distance'] + exemplars[1]['_retrieval_distance']) / 2
            result_row = {
                'source_human_id': sid,
                'model': model_name,
                'rag_exemplar_1': exemplars[0].get('source_human_id','?'),
                'rag_exemplar_2': exemplars[1].get('source_human_id','?'),
                'rag_mean_distance': round(rag_dist, 4),
                'reasoning': parsed['reasoning'],
            }
            for k in Q_KEYS:
                result_row[k] = parsed[k]
            results.append(result_row)
            # Write after each profile (resume safety)
            out_df = pd.DataFrame(results)
            if os.path.exists(out_file):
                existing = pd.read_csv(out_file)
                out_df = pd.concat([existing, out_df], ignore_index=True)
            out_df.to_csv(out_file, index=False)
            results = []  # clear buffer after writing
            print('OK')
        except Exception as e:
            print(f'ERROR: {e}')
            errors.append({'source_human_id': sid, 'error': str(e)})
            pd.DataFrame(errors).to_csv(err_file, index=False)
 
        time.sleep(SLEEP_BETWEEN)
 
    print(f'Done. Output: {out_file}')
    if errors:
        print(f'Errors logged: {err_file} ({len(errors)} profiles failed)')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['gpt4','claude'])
    parser.add_argument('--micro', action='store_true')
    args = parser.parse_args()
    run_generation(args.model, micro=args.micro)
