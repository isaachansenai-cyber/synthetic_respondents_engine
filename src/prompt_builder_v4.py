# prompt_builder.py — V4 Prompt Construction
# Builds the full system prompt for each synthetic respondent.
# Stack: Persona-Deep + RAG Exemplars + Chain-of-Thought
 
import json
 
# ── PSYCHOGRAPHIC LABEL MAPS ─────────────────────────────
# Human-readable labels for each psychographic item (for prompt text)
FAM_LABELS = {
    'fam_1': 'I know a lot about ChatGPT',
    'fam_2': 'I am familiar with ChatGPT',
    'fam_3': 'I have much knowledge about ChatGPT',
    'fam_4': 'I am more familiar than the average person regarding ChatGPT',
    'fam_5': 'I know how to use ChatGPT',
    'fam_6': 'I know how to interact with ChatGPT',
    'fam_7': 'I feel ChatGPT gives me more insights than other sources',
}
USE_LABELS = {
    'use_1': 'get more insights than I can normally find on the internet',
    'use_2': 'help with writing papers',
    'use_3': 'help solve problems',
    'use_4': 'plan out vacations/holidays',
    'use_5': 'write computer code',
    'use_6': 'get answers on a wide range of topics',
    'use_7': 'provide advice on relationships and career decisions',
    'use_8': 'generate ideas for creative projects',
    'use_9': 'entertainment (jokes, stories, games)',
    'use_10': 'learning new languages and improving writing',
    'use_11': 'personal assistant tasks (reminders, appointments)',
}
RISK_LABELS = {
    'risk_1': 'AI generates false information presented as fact',
    'risk_2': 'AI creates imaginary scenarios leading to misinterpretation',
    'risk_3': 'AI perpetuates cultural biases and stereotypes',
    'risk_4': 'AI makes erroneous predictions from flawed data',
    'risk_5': 'AI is vulnerable to hacking and misinformation',
    'risk_6': 'ChatGPT provides more accurate information than humans',
}
 
# ── TAM QUESTION TEXT (for RAG exemplar display) ─────────
TAM_LABELS = {
    'q4_1': 'Q4.1 I intend to use ChatGPT to help plan vacations in the future',
    'q4_2': 'Q4.2 I predict I will use ChatGPT for vacation planning in the future',
    'q4_3': 'Q4.3 I plan to use ChatGPT to learn more about vacation locations',
    'q4_4': 'Q4.4 ChatGPT could increase my likelihood of going to a specific location',
    'q4_5': 'Q4.5 I would consider checking vacation prices using ChatGPT',
    'q4_6': 'Q4.6 ChatGPT increases my willingness to use specific hotels or restaurants',
    'q4_7': 'Q4.7 I will likely recommend ChatGPT to friends and family for vacation planning',
    'q5_1': 'Q5.1 ChatGPT can speed up vacation planning',
    'q5_2': 'Q5.2 ChatGPT lets me research vacations more quickly',
    'q5_3': 'Q5.3 ChatGPT would improve my vacation research results',
    'q5_4': 'Q5.4 ChatGPT would improve effectiveness of my vacation research',
    'q5_5': 'Q5.5 ChatGPT can improve efficiency of vacation planning',
    'q5_6': 'Q5.6 ChatGPT can enhance completeness of vacation planning',
    'q5_7': 'Q5.7 ChatGPT can enable accurate vacation planning',
    'q5_8': 'Q5.8 ChatGPT is a useful source of insights for vacation planning',
    'q6_1': 'Q6.1 AI is useful for real-time updates on events, weather, traffic',
    'q6_2': 'Q6.2 AI provides comprehensive destination information',
    'q6_3': 'Q6.3 AI provides local insights about attractions and restaurants',
    'q6_4': 'Q6.4 AI helps find the best deals on flights and accommodation',
    'q6_5': 'Q6.5 AI tailors recommendations based on my interests and preferences',
    'q6_6': 'Q6.6 AI quickly processes information to present the most relevant results',
    'q6_7': 'Q6.7 AI helps find tips and insights not found on other websites',
    'q7_1': 'Q7.1 Interaction with ChatGPT is clear and understandable',
    'q7_2': 'Q7.2 Interacting with ChatGPT does not require much effort',
    'q7_3': 'Q7.3 I find ChatGPT easy to use',
    'q7_4': 'Q7.4 ChatGPT is flexible to interact with',
    'q7_5': 'Q7.5 I find it easy to access desired information through ChatGPT',
    'q7_6': 'Q7.6 It will be really HARD to learn how to use ChatGPT [REVERSE CODED]',
    'q8_1': 'Q8.1 Easy to research tourism destinations from anywhere at any time',
    'q8_2': 'Q8.2 Easy to create itineraries for vacations',
    'q8_3': 'Q8.3 Easy to type specific/detailed things I want it to find',
    'q8_4': 'Q8.4 Easy to find updated information on what to do at a destination',
    'q8_5': 'Q8.5 Easy to translate information into different languages',
    'q8_6': 'Q8.6 Has intuitive and easy-to-use interfaces',
    'q8_7': 'Q8.7 Easy to automate repetitive booking tasks',
    'q9_1': 'Q9.1 AI can provide contextual destination information (weather, customs, events)',
    'q9_2': 'Q9.2 AI-powered tools provide transparent pricing and availability information',
    'q9_3': 'Q9.3 AI can provide real-time feedback on travel decisions',
    'q10_1': 'Q10.1 Using ChatGPT to learn about vacation destinations is a good idea',
    'q10_2': 'Q10.2 ChatGPT can make my vacation experience more interesting',
    'q10_3': 'Q10.3 Using ChatGPT makes visiting vacation destinations more fun',
    'q10_4': 'Q10.4 I like using ChatGPT as part of planning vacations',
}
 
TAM_KEYS = list(TAM_LABELS.keys())
 
def _score_label(val):
    """Convert a 1-5 Likert score to a readable label."""
    labels = {1:'Strongly Disagree',2:'Disagree',3:'Neutral',4:'Agree',5:'Strongly Agree'}
    return labels.get(int(val), str(val))
 
def _build_persona_block(profile):
    """Build the persona-deep narrative from real profile data."""
    age       = profile.get('age', 'unknown age')
    sex       = profile.get('sex', 'unknown sex')
    gen_group = profile.get('generation', 'unknown generation')
    country   = profile.get('country', 'unknown country')
    city      = profile.get('city', 'unknown city')
 
    # ChatGPT familiarity summary
    fam_scores = {k: profile.get(k, 3) for k in FAM_LABELS}
    avg_fam = sum(fam_scores.values()) / len(fam_scores)
    if avg_fam >= 4.0:
        fam_text = 'highly familiar with and knowledgeable about ChatGPT'
    elif avg_fam >= 3.0:
        fam_text = 'moderately familiar with ChatGPT'
    else:
        fam_text = 'not very familiar with ChatGPT'
 
    # Usage summary — tasks where score >= 3
    use_scores = {k: profile.get(k, 1) for k in USE_LABELS}
    active_uses = [USE_LABELS[k] for k, v in use_scores.items() if float(v) >= 3]
    if active_uses:
        use_text = 'You regularly use ChatGPT to: ' + ', '.join(active_uses[:4]) + '.'
    else:
        use_text = 'You rarely use ChatGPT for most tasks.'
 
    # Risk perception summary
    risk_scores = {k: profile.get(k, 3) for k in RISK_LABELS}
    avg_risk = sum(risk_scores.values()) / len(risk_scores)
    if avg_risk >= 4.0:
        risk_text = 'You are highly concerned about AI risks such as misinformation, bias, and inaccuracy.'
    elif avg_risk >= 3.0:
        risk_text = 'You have moderate concerns about AI generating misinformation or perpetuating bias.'
    else:
        risk_text = 'You are relatively unconcerned about AI risk and misinformation.'
 
    block = f"""=== WHO YOU ARE ===
You are a {age}-year-old {sex.lower()} from {city}, {country}. You identify as {gen_group}.
You are {fam_text}.
{use_text}
{risk_text}
 
Your specific ChatGPT familiarity ratings (1=low, 5=high):
{chr(10).join(f'  - {FAM_LABELS[k]}: {int(profile.get(k,3))}' for k in FAM_LABELS)}
 
Your ChatGPT usage frequency across tasks (1=never, 5=very often):
{chr(10).join(f'  - {USE_LABELS[k]}: {int(profile.get(k,1))}' for k in USE_LABELS)}
 
Your perceived AI risk ratings (1=not likely, 5=very likely):
{chr(10).join(f'  - {RISK_LABELS[k]}: {int(profile.get(k,3))}' for k in RISK_LABELS)}
"""
    return block
 
def _build_rag_block(exemplars):
    """Format two RAG exemplars as calibration references."""
    block = """=== REFERENCE RESPONDENTS (For calibration only — do NOT copy these responses) ===
Below are two real survey respondents with a similar profile to yours.
Use their responses ONLY to calibrate how a person with your background might
engage with the scale — not as answers to copy.
"""
    for i, ex in enumerate(exemplars, 1):
        age    = ex.get('age', '?')
        sex    = ex.get('sex', '?')
        gen    = ex.get('generation', '?')
        country= ex.get('country', '?')
 
        block += f"\n--- Reference Respondent {i} ---\n"
        block += f"Profile: {age}-year-old {sex}, {gen}, from {country}\n"
        block += "Their survey responses:\n"
 
        try:
            responses = json.loads(ex.get('full_survey_responses', '{}'))
        except (json.JSONDecodeError, TypeError):
            responses = {}
 
        for key in TAM_KEYS:
            val = responses.get(key, '?')
            label_text = TAM_LABELS.get(key, key)
            block += f"  {label_text}: {val}\n"
 
    return block
 
def _build_cot_block():
    return """=== YOUR TASK ===
You are about to answer a survey about ChatGPT and vacation/holiday planning.
Before giving your final numeric answers, reason through each construct:
  1. Behavioral Intention (Q4): How likely are YOU to use ChatGPT for vacation planning?
  2. Perceived Usefulness (Q5): How useful do YOU find ChatGPT for vacation research?
  3. Perceived Usefulness - Specific (Q6): How useful is AI for specific planning tasks?
  4. Ease of Use (Q7 & Q8): How easy do YOU find ChatGPT to use?
  5. Trust (Q9): How much do YOU trust AI to provide accurate travel information?
  6. Attitude (Q10): What is YOUR overall attitude toward using ChatGPT for vacations?
 
Think about your profile, your experiences, and your attitudes before answering.
Your answers should reflect your personal views — not what you think is correct.
Show your reasoning in the 'reasoning' field before giving your numeric answers.
"""
 
def _build_scale_block():
    return """=== SCALE ===
All questions use a 1–5 scale:
  1 = Strongly Disagree (or Never / Not at all useful / Very hard)
  2 = Disagree
  3 = Neutral
  4 = Agree
  5 = Strongly Agree (or Always / Very useful / Very easy)
 
IMPORTANT: Question Q7.6 is REVERSE CODED.
  Q7.6 says 'It will be really HARD to learn how to use ChatGPT'
  If you find ChatGPT EASY to use, score Q7.6 LOW (1 or 2).
  If you find ChatGPT HARD to use, score Q7.6 HIGH (4 or 5).
  Do NOT follow your general positive/negative pattern for this item.
"""
 
def _build_output_format_block():
    return """=== OUTPUT FORMAT ===
Respond ONLY with a JSON object. No other text before or after.
Use exactly this structure:
{
  "reasoning": "Your step-by-step reasoning across all six TAM constructs (at least 50 words)",
  "q4_1": <integer 1-5>,
  "q4_2": <integer 1-5>,
  "q4_3": <integer 1-5>,
  "q4_4": <integer 1-5>,
  "q4_5": <integer 1-5>,
  "q4_6": <integer 1-5>,
  "q4_7": <integer 1-5>,
  "q5_1": <integer 1-5>,
  "q5_2": <integer 1-5>,
  "q5_3": <integer 1-5>,
  "q5_4": <integer 1-5>,
  "q5_5": <integer 1-5>,
  "q5_6": <integer 1-5>,
  "q5_7": <integer 1-5>,
  "q5_8": <integer 1-5>,
  "q6_1": <integer 1-5>,
  "q6_2": <integer 1-5>,
  "q6_3": <integer 1-5>,
  "q6_4": <integer 1-5>,
  "q6_5": <integer 1-5>,
  "q6_6": <integer 1-5>,
  "q6_7": <integer 1-5>,
  "q7_1": <integer 1-5>,
  "q7_2": <integer 1-5>,
  "q7_3": <integer 1-5>,
  "q7_4": <integer 1-5>,
  "q7_5": <integer 1-5>,
  "q7_6": <integer 1-5>,
  "q8_1": <integer 1-5>,
  "q8_2": <integer 1-5>,
  "q8_3": <integer 1-5>,
  "q8_4": <integer 1-5>,
  "q8_5": <integer 1-5>,
  "q8_6": <integer 1-5>,
  "q8_7": <integer 1-5>,
  "q9_1": <integer 1-5>,
  "q9_2": <integer 1-5>,
  "q9_3": <integer 1-5>,
  "q10_1": <integer 1-5>,
  "q10_2": <integer 1-5>,
  "q10_3": <integer 1-5>,
  "q10_4": <integer 1-5>
}
"""
 
def build_v4_system_prompt(profile, rag_exemplars):
    """
    Build the complete V4 system prompt for one synthetic respondent.
    profile: dict or pandas Series — the generation profile row
    rag_exemplars: list of 2 dicts — from rag_retriever.get_exemplars()
    Returns: (system_prompt_str, user_prompt_str)
    """
    system = (
        'You are simulating a real human survey respondent. '
        'Answer exactly as this specific person would, based on their profile and experiences. '
        'Do not answer as a helpful AI assistant. Do not add disclaimers. '
        'Respond only as this person.\n\n'
    )
    system += _build_persona_block(profile)
    system += '\n'
    system += _build_rag_block(rag_exemplars)
    system += '\n'
    system += _build_cot_block()
    system += '\n'
    system += _build_scale_block()
 
    user = _build_output_format_block()
 
    return system, user
