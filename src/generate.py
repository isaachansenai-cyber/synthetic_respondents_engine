"""
Generation script — Study 1 compatible + Study 2 prompt funnel support.
One API call per profile, all 28 questions at once.
Supports --style (funnel batch), --profiles (custom profiles CSV), and --micro.
Outputs: outputs/{model}/{style}/{model}_{style}_{micro|full}_responses.csv
Resumes interrupted runs automatically.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from src.canonical_schema import ALL_Q_IDS, extract_json
from src.prompt_builder import (
    get_system_prompt, build_self_reflect_review_prompt,
    PROMPT_STYLES, build_user_prompt
)
from src.models import SyntheticResponse

load_dotenv()

TEMPERATURE = 0.9
MAX_TOKENS  = 900

MODEL_CONFIGS = {
    'gpt4':   {'model_id': 'gpt-4o',           'temp': TEMPERATURE},
    'claude': {'model_id': 'claude-sonnet-4-6', 'temp': TEMPERATURE},
    # 'gemini': {'model_id': 'gemini-1.5-pro',   'temp': TEMPERATURE},
}
MODEL_SLEEP = {'gpt4': 0.6, 'claude': 1.8, 'gemini': 1.0}

# ── API call functions ─────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=90))
def call_openai(sys_prompt, user_prompt, cfg):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    r = client.chat.completions.create(
        model=cfg['model_id'], temperature=cfg['temp'], max_tokens=MAX_TOKENS,
        messages=[{'role': 'system', 'content': sys_prompt},
                  {'role': 'user',   'content': user_prompt}],
        response_format={'type': 'json_object'},
    )
    return r.choices[0].message.content

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=90))
def call_claude(sys_prompt, user_prompt, cfg):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    r = client.messages.create(
        model=cfg['model_id'], max_tokens=MAX_TOKENS, temperature=cfg['temp'],
        system=sys_prompt + '\n\nRespond with valid JSON only. No markdown.',
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    return r.content[0].text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=90))
def call_gemini(sys_prompt, user_prompt, cfg):
    import google.generativeai as genai
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    try:
        model = genai.GenerativeModel(
            model_name=cfg['model_id'],
            system_instruction=sys_prompt + '\nReturn valid JSON only.',
            generation_config=genai.types.GenerationConfig(
                temperature=cfg['temp'], max_output_tokens=MAX_TOKENS,
                response_mime_type='application/json',
            ),
        )
        return model.generate_content(user_prompt).text
    except (TypeError, ValueError, AttributeError):
        model = genai.GenerativeModel(
            model_name=cfg['model_id'],
            system_instruction=sys_prompt + '\nReturn valid JSON only. No markdown.',
            generation_config=genai.types.GenerationConfig(
                temperature=cfg['temp'], max_output_tokens=MAX_TOKENS),
        )
        return model.generate_content(user_prompt).text

API_FUNCS = {
    'gpt4':   call_openai,
    'claude': call_claude,
    # 'gemini': call_gemini,
}

# ── Main generation loop ───────────────────────────────
def run_model(model_key: str, profiles_df, style: str = 'study1_legacy', micro_test: bool = False):
    cfg         = MODEL_CONFIGS[model_key]
    api_func    = API_FUNCS[model_key]
    user_prompt = build_user_prompt()
    sleep_t     = MODEL_SLEEP.get(model_key, 1.0)

    # Style-nested output paths (Study 2). Study 1 runs used outputs/{model}/
    # directly — those outputs are untouched since the path now includes style.
    out_dir    = f'outputs/{model_key}/{style}'
    raw_dir    = f'{out_dir}/raw_api'
    parsed_dir = f'{out_dir}/parsed_json'
    for d in [raw_dir, parsed_dir]:
        os.makedirs(d, exist_ok=True)

    profiles_to_run = profiles_df.head(3) if micro_test else profiles_df
    results, errors = [], []

    for idx, row in profiles_to_run.iterrows():
        syn_id      = row['synthetic_id']
        parsed_path = f'{parsed_dir}/{syn_id}.json'
        raw_path    = f'{raw_dir}/{syn_id}.txt'

        # Resume: skip if already successfully generated
        if os.path.exists(parsed_path):
            with open(parsed_path, encoding='utf-8') as f:
                results.append(json.load(f))
            continue

        profile_dict = row.to_dict()
        sys_prompt   = get_system_prompt(style, profile_dict)

        # Prompt length guard (~4 chars per token)
        approx_tokens = len(sys_prompt + user_prompt) // 4
        if approx_tokens > 4500:
            print(f'  ❌ {syn_id}: prompt ~{approx_tokens} tokens — too long, skipping')
            errors.append({'synthetic_id': syn_id, 'error': f'prompt_too_long:{approx_tokens}'})
            continue
        if approx_tokens > 3500:
            print(f'  ⚠ {syn_id}: prompt ~{approx_tokens} tokens (near limit)')

        try:
            raw_output = api_func(sys_prompt, user_prompt, cfg)
        except RetryError as e:
            print(f'  ❌ {syn_id}: API failed after 3 retries')
            errors.append({'synthetic_id': syn_id, 'error': str(e)})
            continue

        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(raw_output)

        try:
            parsed    = extract_json(raw_output)
            validated = SyntheticResponse.from_raw(syn_id, model_key, parsed)
            row_dict  = validated.to_row()
            row_dict['prompt_style'] = style  # tag each row with its batch
            with open(parsed_path, 'w', encoding='utf-8') as f:
                json.dump(row_dict, f)
            results.append(row_dict)
            print(f'  ✅ {syn_id}: {validated.reasoning[:55]}...')
        except (ValueError, json.JSONDecodeError) as e:
            print(f'  ❌ {syn_id}: {e}')
            errors.append({'synthetic_id': syn_id, 'error': str(e), 'raw': raw_output[:200]})

        time.sleep(sleep_t)

    tag = 'micro' if micro_test else 'full'
    if results:
        pd.DataFrame(results).to_csv(f'{out_dir}/{model_key}_{style}_{tag}_responses.csv', index=False)
    if errors:
        pd.DataFrame(errors).to_csv(f'{out_dir}/{model_key}_{style}_errors.csv', index=False)

    # End-of-run summary
    expected_n  = len(profiles_to_run)
    generated_n = len(results)
    error_n     = len(errors)
    print(f'\n=== {model_key.upper()} / {style.upper()} SUMMARY ===')
    print(f'  Expected: {expected_n}  Generated: {generated_n}  Errors: {error_n}')
    if generated_n + error_n < expected_n:
        print(f'  ⚠ WARNING: {expected_n - generated_n - error_n} profiles unaccounted for!')
    return results, errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    required=True,  choices=['gpt4', 'claude'])
    parser.add_argument('--style',    required=True,  choices=PROMPT_STYLES)
    parser.add_argument('--profiles', default='data/profiles/study2_profiles.csv',
                        help='Path to profiles CSV (default: study2_profiles.csv)')
    parser.add_argument('--micro',    action='store_true')
    args = parser.parse_args()

    profiles_df = pd.read_csv(args.profiles)
    print(f'Loaded {len(profiles_df)} profiles | model={args.model} style={args.style} micro={args.micro}')
    run_model(args.model, profiles_df, style=args.style, micro_test=args.micro)