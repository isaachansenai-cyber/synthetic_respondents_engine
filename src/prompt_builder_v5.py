# prompt_builder_v5.py -- V5 Prompt Construction
# Batch 7: Minimal demographic profile + task description.
# No CoT, no few-shot, no persona narrative, no trait block, no reasoning field.

def build_v5_prompt(profile):
    """
    Build the system and user prompts for one synthetic respondent.
    profile: pandas Series or dict with keys: age, sex, city, country
    Returns: (system_prompt_str, user_prompt_str)
    """
    if isinstance(profile, dict):
        age     = profile.get('age',     'unknown age')
        sex     = profile.get('sex',     'unknown sex')
        city    = profile.get('city',    'Unknown')
        country = profile.get('country', 'Unknown')
    else:
        age     = profile['age']
        sex     = profile['sex']
        city    = profile['city']
        country = profile['country']

    system = (
        f"You are a {age} {sex.lower()} from {city}, {country}. "
        f"You are completing a survey about ChatGPT and vacation/holiday planning. "
        f"Answer each question honestly on the scale provided."
    )

    user = _build_output_format_block()

    return system, user


def _build_output_format_block():
    """Build the JSON output schema for the 42-item TAM survey."""
    q_keys = [
        'q4_1','q4_2','q4_3','q4_4','q4_5','q4_6','q4_7',
        'q5_1','q5_2','q5_3','q5_4','q5_5','q5_6','q5_7','q5_8',
        'q6_1','q6_2','q6_3','q6_4','q6_5','q6_6','q6_7',
        'q7_1','q7_2','q7_3','q7_4','q7_5','q7_6',
        'q8_1','q8_2','q8_3','q8_4','q8_5','q8_6','q8_7',
        'q9_1','q9_2','q9_3',
        'q10_1','q10_2','q10_3','q10_4',
    ]
    header = [
        'Respond ONLY with a JSON object. No other text before or after.',
        'Use exactly this structure:',
        '{',
    ]
    body = []
    for i, k in enumerate(q_keys):
        comma = ',' if i < len(q_keys) - 1 else ''
        body.append(f'  "{k}": <integer 1-5>{comma}')
    footer = ['}']
    return '\n'.join(header + body + footer)
