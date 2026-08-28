#src\prompt_builder.py — Part 1: Helpers, Constants, Core Builder
"""
Prompt builder for Study 2 Prompt Funnel Experiment.
Contains all 7 funnel-batch builders plus shared helpers.
"""
import warnings
from src.canonical_schema import SCALE_LABELS, ALL_Q_IDS, COL_ARM
from src.question_bank import QUESTIONS
 
SCALE_OVERRIDE_Q7 = '1=Not at all easy  2=Slightly easy  3=Moderately easy  4=Very easy  5=Extremely easy'
 
# ── Shared: latent trait block (conditioned on arm) ────
def _build_traits_block(p: dict) -> str:
    if str(p.get(COL_ARM, 'demo_plus_traits')).strip() == 'demo_only':
        return ''
    def _t(key, default='MODERATE'):
        val = p.get(key)
        return str(val).strip() if val and str(val).strip().lower() not in ('nan','none','') else default
    if all(p.get(f'tau_{t}_level') is None for t in ['PU','PEOU','BI','ATT','AI_CAP']):
        warnings.warn(f"Profile {p.get('synthetic_id')} is demo_plus_traits but has no traits.")
    return (
        '\n=== YOUR PRIOR ATTITUDES (be consistent with these) ===\n'
        f"Perceived Usefulness belief: {_t('tau_PU_level')}\n"
        f"  -> {_t('tau_PU_verbal','You have moderate beliefs about AI usefulness.')}\n"
        f"Ease of Use disposition:     {_t('tau_PEOU_level')}\n"
        f"  -> {_t('tau_PEOU_verbal','You have moderate ease with technology.')}\n"
        f"Behavioral Intention:        {_t('tau_BI_level')}\n"
        f"  -> {_t('tau_BI_verbal','You have moderate intention to adopt AI tools.')}\n"
        f"Attitude toward AI:          {_t('tau_ATT_level')}\n"
        f"  -> {_t('tau_ATT_verbal','You have a moderate attitude toward AI activities.')}\n"
        f"AI Capability belief:        {_t('tau_AI_CAP_level')}\n"
        f"  -> {_t('tau_AI_CAP_verbal','You have moderate beliefs about AI capability.')}\n"
    )
 
# ── Shared: few-shot anchor examples ───────────────────
FEW_SHOT_EXAMPLES = '''
=== EXAMPLE 1 ===
Profile: Age 19, Female, Philippines, occasionally used ChatGPT, occasional traveler.
Prior attitudes: LOW perceived usefulness, LOW ease of use, LOW behavioral intention.
{"reasoning": "I have tried ChatGPT once but found it confusing for planning. I prefer asking friends.",
 "q4_1": 2, "q4_2": 2, "q4_3": 2, "q4_4": 3, "q4_5": 2, "q4_6": 2, "q4_7": 1,
 "q5_1": 2, "q5_2": 2, "q5_3": 2, "q5_4": 2, "q5_5": 2, "q5_6": 3, "q5_7": 2, "q5_8": 2,
 "q7_1": 2, "q7_2": 2, "q7_3": 2, "q7_4": 3, "q7_5": 2, "q7_6": 4,
 "q9_1": 3, "q9_2": 3, "q9_3": 3, "q10_1": 2, "q10_2": 2, "q10_3": 2, "q10_4": 2}
 
=== EXAMPLE 2 ===
Profile: Age 24, Male, United States, regularly used ChatGPT, frequent traveler.
Prior attitudes: HIGH perceived usefulness, HIGH ease of use, HIGH behavioral intention.
{"reasoning": "I use ChatGPT constantly for trip research. It is faster than Google for overviews.",
 "q4_1": 5, "q4_2": 5, "q4_3": 4, "q4_4": 4, "q4_5": 5, "q4_6": 4, "q4_7": 5,
 "q5_1": 5, "q5_2": 5, "q5_3": 4, "q5_4": 4, "q5_5": 5, "q5_6": 4, "q5_7": 4, "q5_8": 5,
 "q7_1": 5, "q7_2": 5, "q7_3": 5, "q7_4": 4, "q7_5": 5, "q7_6": 1,
 "q9_1": 5, "q9_2": 4, "q9_3": 4, "q10_1": 5, "q10_2": 4, "q10_3": 4, "q10_4": 5}
'''
 
def _few_shot_block() -> str:
    return (
        '=== REFERENCE EXAMPLES (for format only — do NOT copy these responses) ===\n'
        + FEW_SHOT_EXAMPLES.strip()
        + '\n=== END EXAMPLES ===\n\n'
        + 'Now complete the survey for YOUR profile. Reflect YOUR own attitudes, not the examples.\n'
    )
 
def _cot_instruction() -> str:
    return (
        '- BEFORE answering, briefly consider your stance on each construct:\n'
        '  Usefulness: how useful would ChatGPT realistically be for YOU for travel?\n'
        '  Ease: how easy would it be given your experience?\n'
        '  Intention: how likely are YOU to actually use it?\n'
        '  Attitude: how do YOU genuinely feel about AI for travel planning?\n'
        '- Capture this in the "reasoning" field (2-4 sentences).\n'
    )
 
def _output_format_block() -> str:
    return (
        '=== OUTPUT FORMAT ===\n'
        'Return a single JSON object with:\n'
        '- "reasoning": 1-2 sentences on your overall view of ChatGPT for travel.\n'
        '- q4_1 through q10_4: integer 1-5 for each question.\n'
        'Return ONLY the JSON object. No markdown fences. No other text.'
    )
 
def _scale_block() -> str:
    return (
        '=== SCALE INSTRUCTIONS ===\n'
        f"Most questions: {SCALE_LABELS['scale_agree']}\n"
        f'Q7 questions (ease of use — NOT usefulness): {SCALE_OVERRIDE_Q7}\n'
        f"Q9 questions: {SCALE_LABELS['scale_likely']}\n"
        'Q7_6 is REVERSE SCORED — a score of 1 means easy (you disagree it is hard).\n'
    )
 
def _variance_instructions() -> str:
    return (
        '- Real people disagree. Some answers will be low, some high.\n'
        '- Do NOT default to neutral or positive scores on everything.\n'
        '- A person who has never used ChatGPT will be more uncertain on ease-of-use items.\n'
        '- A skeptic on AI usefulness should score Q5 items low.\n'
        '- Be psychologically consistent across all 28 items.\n'
    )
 
def build_user_prompt() -> str:
    lines = ['Please answer all 28 questions based on your profile.', '']
    for qid in ALL_Q_IDS:
        stem, item = QUESTIONS[qid]
        lines.append(f'{qid}: [{stem}] {item}')
    lines += ['', 'Return your reasoning and all 28 responses as a single JSON object.']
    return '\n'.join(lines)


#src\prompt_builder.py — Part 2: Batch Builders (append directly after Part 1)
 
# ══════════════════════════════════════════════════════════
# BATCH 1: study1_legacy
# Exact Study-1 system prompt. DO NOT reconstruct by hand.
# This function is also the 'standard' baseline for Study 2.
# ══════════════════════════════════════════════════════════
def build_system_prompt(profile: dict) -> str:
    p = profile
    traits_section = _build_traits_block(p)
    # Ensure clean spacing whether or not traits section is present
    sep = '\n' if not traits_section else ''
 
    return (
        'You are a human survey respondent. You are NOT an AI.\n'
        'Do not reference AI behavior or training. Answer based on your personal experience.\n'
        '\n=== YOUR PROFILE ===\n'
        f"Age bracket: {p.get('age','unknown')}\n"
        f"Sex: {p.get('sex','unknown')}\n"
        f"Country: {p.get('country','unknown')}\n"
        f"ChatGPT experience: {p.get('chatgpt_usage','unknown')}\n"
        f"Travel habits: {p.get('travel_freq','unknown')}\n"
        f"{traits_section}{sep}"
        '=== SCALE INSTRUCTIONS ===\n'
        f"Most questions: {SCALE_LABELS['scale_agree']}\n"
        f"Q7 questions (ease of use — NOT usefulness): {SCALE_OVERRIDE_Q7}\n"
        f"Q9 questions: {SCALE_LABELS['scale_likely']}\n"
        'Q7_6 is REVERSE SCORED — a score of 1 means easy (you disagree it is hard).\n'
        '\n=== INSTRUCTIONS ===\n'
        '- Real people disagree. Some answers will be low, some high — based on your profile.\n'
        '- Do NOT default to neutral or positive scores on everything.\n'
        '- A person who has never used ChatGPT will be more uncertain on ease-of-use items.\n'
        '- A skeptic on AI usefulness should score Q5 items low.\n'
        '- Be psychologically consistent across all 28 items.\n'
        '\n=== OUTPUT FORMAT ===\n'
        'Return a single JSON object with:\n'
        '- "reasoning": 1-2 sentences on your overall view of ChatGPT for travel.\n'
        '- q4_1 through q10_4: integer 1-5 for each question.\n'
        'Example: {"reasoning": "I am skeptical...", "q4_1": 2, "q4_2": 3, ...}\n'
        'Return ONLY the JSON object. No markdown fences. No other text.'
    )

 
# Explicit alias so the dispatcher is unambiguous
def build_system_prompt_study1_legacy(profile: dict) -> str:
    """Batch 1 — passes through to the Study-1 build_system_prompt(). Never rewrite."""
    return build_system_prompt(profile)
 
 
# ── BATCH 2: full_stack_plus_fewshot_persona ───────────
def build_system_prompt_full_stack_plus_fewshot_persona(profile: dict) -> str:
    """Batch 2 — Study-1 legacy + few-shot examples + persona-deep narrative."""
    p = profile
    country = p.get('country','your country')
    age     = p.get('age','unknown')
    sex     = p.get('sex','unknown')
    usage   = p.get('chatgpt_usage','unknown')
    ai_ctx  = {'never used':'You have never used ChatGPT before.',
               'occasionally used':'You have tried ChatGPT a handful of times.',
               'regularly used':'You use ChatGPT regularly.'}.get(usage, f'ChatGPT experience: {usage}.')
    scenario = (
        f'Imagine you are a real person: {age} years old, {sex}, living in {country}. '
        f'{ai_ctx} Think about your actual life and genuine feelings — not abstract ideals.\n\n'
    )
    traits_section = _build_traits_block(p)
    sep = '\n' if not traits_section else ''
    return (
        scenario
        + 'You are a human survey respondent. You are NOT an AI.\n\n'
        + '=== YOUR PROFILE ===\n'
        + f"Age: {age}  Sex: {sex}  Country: {country}  ChatGPT: {usage}\n"
        + f"{traits_section}{sep}"
        + _few_shot_block()
        + _scale_block()
        + '\n=== INSTRUCTIONS ===\n'
        + _variance_instructions()
        + '\n' + _output_format_block()
    )
 
 
# ── BATCH 3: stack_traits_cot_persona_fewshot ──────────
def build_system_prompt_stack_traits_cot_persona_fewshot(profile: dict) -> str:
    """Batch 3 — TAM traits (demo_plus_traits) + persona-deep + CoT + few-shot."""
    p = profile
    country = p.get('country','your country')
    age     = p.get('age','unknown')
    sex     = p.get('sex','unknown')
    usage   = p.get('chatgpt_usage','unknown')
    ai_ctx  = {'never used':'You have never used ChatGPT before.',
               'occasionally used':'You have tried ChatGPT a handful of times.',
               'regularly used':'You use ChatGPT regularly.'}.get(usage, f'ChatGPT experience: {usage}.')
    scenario = (
        f'Imagine you are a real person: {age} years old, {sex}, living in {country}. '
        f'{ai_ctx} Think about your actual life — not abstract ideals.\n\n'
    )
    traits_section = _build_traits_block(p)
    sep = '\n' if not traits_section else ''
    return (
        scenario
        + 'You are a human survey respondent. You are NOT an AI.\n\n'
        + '=== YOUR PROFILE ===\n'
        + f"Age: {age}  Sex: {sex}  Country: {country}  ChatGPT: {usage}\n"
        + f"{traits_section}{sep}"
        + _few_shot_block()
        + _scale_block()
        + '\n=== INSTRUCTIONS ===\n'
        + _cot_instruction()
        + _variance_instructions()
        + '\n' + _output_format_block()
    )
 
 
# ── BATCH 4: stack_persona_fewshot_cot ─────────────────
def build_system_prompt_stack_persona_fewshot_cot(profile: dict) -> str:
    """Batch 4 — Persona-deep + few-shot + CoT. No explicit extra trait block."""
    p = profile
    country = p.get('country', 'your country')
    age     = p.get('age', 'unknown')
    sex     = p.get('sex', 'unknown')
    usage   = p.get('chatgpt_usage', 'unknown')
    ai_ctx  = {'never used': 'You have never used ChatGPT before.',
               'occasionally used': 'You have tried ChatGPT a handful of times.',
               'regularly used': 'You use ChatGPT regularly.'}.get(usage, f'ChatGPT experience: {usage}.')
    scenario = (
        f'Imagine you are a real person: {age} years old, {sex}, living in {country}. '
        f'{ai_ctx} Think about your actual life — not abstract ideals.\n\n'
    )
    return (
        scenario
        + 'You are a human survey respondent. You are NOT an AI.\n\n'
        + '=== YOUR PROFILE ===\n'
        + f"Age: {age}  Sex: {sex}  Country: {country}  ChatGPT: {usage}\n\n"
        + _few_shot_block()
        + _scale_block()
        + '\n=== INSTRUCTIONS ===\n'
        + _cot_instruction()
        + _variance_instructions()
        + '\n' + _output_format_block()
    )
 
 
# ── BATCH 5: stack_fewshot_cot ─────────────────────────
def build_system_prompt_stack_fewshot_cot(profile: dict) -> str:
    """Batch 5 — Few-shot + CoT. Minimal standard demographic profile only."""
    p = profile
    traits_section = _build_traits_block(p)
    sep = '\n' if not traits_section else ''
    return (
        'You are a human survey respondent. You are NOT an AI.\n'
        'Answer based on your personal experience.\n'
        '\n=== YOUR PROFILE ===\n'
        f"Age: {p.get('age','unknown')}  Sex: {p.get('sex','unknown')}  Country: {p.get('country','unknown')}\n"
        f"ChatGPT experience: {p.get('chatgpt_usage','unknown')}\n"
        f"{traits_section}{sep}"
        + _few_shot_block()
        + _scale_block()
        + '\n=== INSTRUCTIONS ===\n'
        + _cot_instruction()
        + _variance_instructions()
        + '\n' + _output_format_block()
    )
 
 
# ── BATCH 6: cot_only ──────────────────────────────────
def build_system_prompt_cot_only(profile: dict) -> str:
    """Batch 6 — CoT instruction only. Minimal demographic profile."""
    p = profile
    return (
        'You are a human survey respondent. You are NOT an AI.\n'
        '\n=== YOUR PROFILE ===\n'
        f"Age: {p.get('age','unknown')}  Sex: {p.get('sex','unknown')}  Country: {p.get('country','unknown')}\n"
        f"ChatGPT experience: {p.get('chatgpt_usage','unknown')}\n\n"
        + _scale_block()
        + '\n=== INSTRUCTIONS ===\n'
        + _cot_instruction()
        + _variance_instructions()
        + '\n' + _output_format_block()
    )
 
 
# ── BATCH 7: zero_shot ─────────────────────────────────
def build_system_prompt_zero_shot(profile: dict) -> str:
    """Batch 7 — Minimal demographic profile + task description. No engineering."""
    p = profile
    return (
        'You are completing a survey about using ChatGPT for travel planning.\n'
        f"You are {p.get('age','unknown')} years old, {p.get('sex','unknown')}, from {p.get('country','unknown')}.\n"
        f"ChatGPT experience: {p.get('chatgpt_usage','unknown')}.\n\n"
        + _scale_block()
        + 'Answer each question honestly on the scale provided.\n\n'
        + _output_format_block()
    )
 
 
# ══════════════════════════════════════════════════════════
# PROMPT_STYLES — canonical list of all funnel style keys.
# Must match exactly: dispatcher below, --style arg, PowerShell $funnel array.
# ══════════════════════════════════════════════════════════
PROMPT_STYLES = [
    'study1_legacy',
    'full_stack_plus_fewshot_persona',
    'stack_traits_cot_persona_fewshot',
    'stack_persona_fewshot_cot',
    'stack_fewshot_cot',
    'cot_only',
    'zero_shot',
]
 
# ── self_reflect pass-2 prompt (used by generate.py) ───
def build_self_reflect_review_prompt(profile: dict, initial_json: str) -> str:
    p = profile
    return (
        f"You just completed a survey as a {p.get('age','unknown')} {p.get('sex','unknown')} from {p.get('country','unknown')} "
        f"with ChatGPT experience: {p.get('chatgpt_usage','unknown')}.\n\n"
        f'Here are your initial responses:\n{initial_json}\n\n'
        'Review your responses critically:\n'
        '1. Are Q5 (usefulness) scores consistent with Q7 (ease of use)?\n'
        '2. Are Q4 (intention) scores consistent with Q10 (attitude)?\n'
        '3. Does Q7_6 score the REVERSE of your other Q7 scores?\n'
        'Revise any inconsistencies. Keep consistent answers.\n'
        'Return the COMPLETE revised JSON object (all 28 questions + reasoning). '
        'Same format. Return ONLY the JSON object. No markdown.'
    )
 
 
# ── Dispatcher ─────────────────────────────────────────
def get_system_prompt(style: str, profile: dict) -> str:
    """Route style key to the correct builder function."""
    dispatch = {
        'study1_legacy':                       build_system_prompt_study1_legacy,
        'full_stack_plus_fewshot_persona':     build_system_prompt_full_stack_plus_fewshot_persona,
        'stack_traits_cot_persona_fewshot':    build_system_prompt_stack_traits_cot_persona_fewshot,
        'stack_persona_fewshot_cot':           build_system_prompt_stack_persona_fewshot_cot,
        'stack_fewshot_cot':                   build_system_prompt_stack_fewshot_cot,
        'cot_only':                            build_system_prompt_cot_only,
        'zero_shot':                           build_system_prompt_zero_shot,
    }
    if style not in dispatch:
        raise ValueError(f'Unknown style: {repr(style)}. Valid styles: {list(dispatch.keys())}')
    return dispatch[style](profile)

