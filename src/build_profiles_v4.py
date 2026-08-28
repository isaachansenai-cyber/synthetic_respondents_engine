# build_profiles.py — V4 Data Preparation
# Reads the two Excel input files and outputs generation pool + RAG corpus CSVs.
 
import pandas as pd
import numpy as np
import os
 
# ── FILE PATHS ────────────────────────────────────────────
PSYCHO_FILE = 'data/psychographics/SMA 2026 TAM surveys virtual twins - 850 people with all demo and psycho.xlsx'
RAG_FILE    = 'data/psychographics/SMA 2026 TAM surveys virtual twins - 50 people with all answers.xlsx'
OUT_GEN     = 'data/profiles/v4_generation_profiles.csv'
OUT_RAG     = 'data/profiles/v4_rag_corpus.csv'
 
os.makedirs('data/profiles', exist_ok=True)
os.makedirs('data/psychographics', exist_ok=True)
 
# ── COLUMN NAME MAPPINGS ──────────────────────────────────
# Psychographic columns in the 850-person file — mapped to short names
FAMILIARITY_COLS = {
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I know a lot about ChatGPT': 'fam_1',
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I am familiar with ChatGPT': 'fam_2',
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I have much knowledge about ChatGPT': 'fam_3',
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I am more familiar than the average person regarding ChatGPT': 'fam_4',
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I know how to use ChatGPT': 'fam_5',
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I know how to interact with ChatGPT': 'fam_6',
    'Q. How familiar are you with ChatGPT (artificial intelligence software that you give a prompt and it will search the internet for you)? - I feel ChatGPT gives me more insights than other sources': 'fam_7',
}
 
USAGE_COLS = {
    'Q1. How often do you use ChatGPT to help you with these different tasks? - to get more insights than I can normally find on the internet': 'use_1',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - to help me with writing papers': 'use_2',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - to help me solve problems': 'use_3',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - to help me plan out vacations/holidays': 'use_4',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - to help me write computer code': 'use_5',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - to get answers to their questions on a wide range of topics, such as history, science, geography, and more': 'use_6',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - provide advice on a variety of topics, from relationships to career decisions, based on its training data': 'use_7',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - generate ideas for creative projects, writing prompts, marketing campaigns, and more.': 'use_8',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - for entertainment purposes, such as asking it to tell them a joke, a story, or to play a game': 'use_9',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - learning new languages, practicing their language skills, or improving their writing skills': 'use_10',
    'Q1. How often do you use ChatGPT to help you with these different tasks? - as a personal assistant by setting reminders, scheduling appointments, and providing helpful tips': 'use_11',
}
 
RISK_COLS = {
    'Q12. In your opinion, how likely is it that... - AI can generate false information and present it as factual, leading to incorrect decisions or actions.': 'risk_1',
    'Q12. In your opinion, how likely is it that... - AI can make up and create imaginary scenarios that have no basis in reality, leading to confusion and misinterpretation of data': 'risk_2',
    'Q12. In your opinion, how likely is it that... - AI can perpetuate biases and stereotypes of different cultures and customs and peoples, leading to discriminatory outcomes and reinforcing social inequalities.': 'risk_3',
    'Q12. In your opinion, how likely is it that... - AI can make erroneous predictions based on flawed or incomplete data, leading to incorrect assumptions and misguided travel and tourism decisions.': 'risk_4',
    'Q12. In your opinion, how likely is it that... - AI can be vulnerable to hacking or manipulation, leading to the dissemination of false or misleading information about travel and tourism.': 'risk_5',
    'Q12. In your opinion, how likely is it that... - ChatGPT provides more accurate information than human beings': 'risk_6',
}
 
ALL_PSYCHO_COLS = {**FAMILIARITY_COLS, **USAGE_COLS, **RISK_COLS}
SHORT_PSYCHO = list(FAMILIARITY_COLS.values()) + list(USAGE_COLS.values()) + list(RISK_COLS.values())
 
# TAM output columns in the RAG file — mapped to short names
TAM_COLS = {
    'Q4. To what extent do you agree that... - I intend to use ChatGPT to help me plan vacations/holidays in the future': 'q4_1',
    'Q4. To what extent do you agree that... - I predict I will use ChatGPT for vacation/holiday planning in the future': 'q4_2',
    'Q4. To what extent do you agree that... - I plan to use ChatGPT to learn more about vacation/holiday locations': 'q4_3',
    'Q4. To what extent do you agree that... - The use of ChatGPT could increase my likelihood of going to a specific location for a vacation/holiday': 'q4_4',
    'Q4. To what extent do you agree that... - I would consider checking the prices of a vacation/holiday using ChatGPT': 'q4_5',
    'Q4. To what extent do you agree that... - The use of ChatGPT increases my willingness to use specific hotels or restaurants at the vacation/holiday location': 'q4_6',
    'Q4. To what extent do you agree that... - It is very likely that I will recommend using ChatGPT to my friends and family for vacation/holiday planning': 'q4_7',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - can speed up vacation/holiday planning': 'q5_1',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - would let me research for a vacation/holiday more quickly': 'q5_2',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - would improve the results of my vacation/holiday planning research': 'q5_3',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - would improve the effectiveness of my vacation/holiday planning research': 'q5_4',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - can improve the efficiency of vacation/holiday planning': 'q5_5',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - can enhance the completeness of vacation/holiday planning': 'q5_6',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - can enable accurate vacation/holiday planning': 'q5_7',
    'Q5. In your opinion, to what extent do you agree that using ChatGPT... - is a useful source of insights for vacation/holiday planning': 'q5_8',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - providing real-time updates about events, weather, traffic, and other important information that could affect your vacation/holiday planning': 'q6_1',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - providing you with a comprehensive, wide range of information about a destination, including its history, culture, cuisine, and popular attractions': 'q6_2',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - providing \'local insights\' about popular or hidden attractions, restaurants, and experiences in a destination': 'q6_3',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - helping you find the best deals on flights, accommodation, and activities, saving you money on your trip': 'q6_4',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - tailoring recommendations based on your interests, preferences, and past behaviors, making your planning experience more personalized': 'q6_5',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - quickly scanning and processing vast amounts of information to present you with the most relevant information, saving you valuable time': 'q6_6',
    'Q6. In your opinion, how useful is AI for vacation/holiday planning when it comes to specifically... - helping you find valuable tips and insights that you cannot find on other websites': 'q6_7',
    'Q7. In your opinion, how easy to use is ChatGPT for researching tourism vacations/holidays when it comes to . . . - Interaction with ChatGPT is clear and understandable': 'q7_1',
    'Q7. In your opinion, how easy to use is ChatGPT for researching tourism vacations/holidays when it comes to . . . - interaction with ChatGPT does not require much effort': 'q7_2',
    'Q7. In your opinion, how easy to use is ChatGPT for researching tourism vacations/holidays when it comes to . . . - I find ChatGPT to be easy to use': 'q7_3',
    'Q7. In your opinion, how easy to use is ChatGPT for researching tourism vacations/holidays when it comes to . . . - ChatGPT is flexible to interact with': 'q7_4',
    'Q7. In your opinion, how easy to use is ChatGPT for researching tourism vacations/holidays when it comes to . . . - I find it easy to access the desired information through ChatGPT': 'q7_5',
    'Q7. In your opinion, how easy to use is ChatGPT for researching tourism vacations/holidays when it comes to . . . - It will be really hard to learn how to use ChatGPT': 'q7_6',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - researching tourism destinations from anywhere at any time': 'q8_1',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - creating itineraries of things to do while on the vacation at that place': 'q8_2',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - typing in specific/detailed things I want it to find information about': 'q8_3',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - finding updated and new (and not really old) information on what there is to do and when places are open at the destination': 'q8_4',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - translating information into different languages, making it easier for you to research destinations in countries where you may not speak the local language (or not speak it well)': 'q8_5',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - having intuitive and easy-to-use interfaces, making it simple for you to search for and find the information that you need': 'q8_6',
    'Q8. In your opinion, how easy to use is ChatGPT for researching vacations/holidays when it comes to... - automating repetitive tasks such as booking accommodation, flights, or tours, making the process more streamlined and efficient for you': 'q8_7',
    'Q9. In your opinion, how likely is it that... - AI can provide me with contextual information about a destination, such as weather conditions, local customs, and cultural events, helping them make more informed decisions about their trip.': 'q9_1',
    'Q9. In your opinion, how likely is it that... - AI-powered travel tools can provide me with transparent information about pricing, availability, and other factors that affect my travel decisions, giving me a greater sense of control over the booking process': 'q9_2',
    'Q9. In your opinion, how likely is it that... - AI can provide real-time feedback on my travel decisions, such as flight times and availability of accommodation, giving me greater control over my plans': 'q9_3',
    'Q10. To what extent do you agree that... - Using ChatGPT to learn about vacation/holiday destinations is a good idea': 'q10_1',
    'Q10. To what extent do you agree that... - ChatGPT can make my vacation/holiday experience more interesting': 'q10_2',
    'Q10. To what extent do you agree that... - Using ChatGPT can makes visiting vacation/holiday destinations more fun': 'q10_3',
    'Q10. To what extent do you agree that... - I like using ChatGPT as a part of planning vacations/holidays': 'q10_4',
}
 
SHORT_TAM = list(TAM_COLS.values())
 
# ── LOAD 850-PERSON FILE ─────────────────────────────────
print('Loading 850-person psychographic file...')
df850 = pd.read_excel(PSYCHO_FILE)
print(f'  Loaded {len(df850)} rows, {df850.shape[1]} cols')
 
# Rename columns
df850 = df850.rename(columns={
    'Response ID': 'response_id',
    'What is your age?': 'age',
    'Which generation would you describe yourself as belonging to?': 'generation',
    'What is your biological sex?': 'sex',
    'What country is home for you?': 'country',
    'What state or city is home for you?': 'city',
})
df850 = df850.rename(columns=ALL_PSYCHO_COLS)
 
# Fill missing demographic values
df850['age']     = df850['age'].fillna(df850['age'].mode()[0])
df850['country'] = df850['country'].fillna('Unknown')
df850['city']    = df850['city'].fillna('Unknown')
 
# Assign sequential source IDs
df850['source_human_id'] = [f'H_{i+1:04d}' for i in range(len(df850))]
 
# ── LOAD 49-PERSON RAG FILE ───────────────────────────────
print('Loading RAG file...')
df49 = pd.read_excel(RAG_FILE)
print(f'  Loaded {len(df49)} rows, {df49.shape[1]} cols')
 
df49 = df49.rename(columns={'Response ID': 'response_id'})
df49 = df49.rename(columns={
    'What is your age?': 'age',
    'Which generation would you describe yourself as belonging to?': 'generation',
    'What is your biological sex?': 'sex',
    'What country is home for you?': 'country',
    'What state or city is home for you?': 'city',
})
df49 = df49.rename(columns=ALL_PSYCHO_COLS)
df49 = df49.rename(columns=TAM_COLS)
 
# Mark RAG IDs
rag_ids = set(df49['response_id'])
print(f'  RAG corpus size: {len(rag_ids)} unique IDs')
 
# ── NORMALISE PSYCHOGRAPHIC COLUMNS ─────────────────────
# Compute min/max from the full 850 (all available data)
print('Normalising psychographic columns...')
norm_stats = {}
for col in SHORT_PSYCHO:
    col_min = df850[col].min()
    col_max = df850[col].max()
    norm_stats[col] = (col_min, col_max)
    rng = col_max - col_min
    df850[col + '_norm'] = ((df850[col] - col_min) / rng) if rng > 0 else 0.0
 
NORM_COLS = [c + '_norm' for c in SHORT_PSYCHO]
 
# ── BUILD GENERATION POOL ────────────────────────────────
print('Building generation pool (excluding RAG IDs)...')
gen = df850[~df850['response_id'].isin(rag_ids)].copy().reset_index(drop=True)
print(f'  Generation pool size: {len(gen)} profiles')
 
# Keep only needed columns
gen_cols = ['source_human_id','response_id','age','generation','sex','country','city'] + SHORT_PSYCHO + NORM_COLS
gen = gen[gen_cols]
gen.to_csv(OUT_GEN, index=False)
print(f'  Saved to {OUT_GEN}')
 
# ── BUILD RAG CORPUS ─────────────────────────────────────
print('Building RAG corpus...')
# Merge RAG psychographic data with their source_human_id from df850
id_map = df850[['response_id','source_human_id']].copy()
rag_merged = df49.merge(id_map, on='response_id', how='left')
 
# Add normalised psychographic columns (reuse stats from full 850)
for col in SHORT_PSYCHO:
    col_min, col_max = norm_stats[col]
    rng = col_max - col_min
    rag_merged[col + '_norm'] = ((rag_merged[col] - col_min) / rng) if rng > 0 else 0.0
 
# Build full_survey_responses as JSON string
import json
rag_merged['full_survey_responses'] = rag_merged[SHORT_TAM].apply(
    lambda row: json.dumps(row.to_dict()), axis=1
)
 
rag_cols = ['source_human_id','response_id','age','generation','sex','country','city'] + SHORT_PSYCHO + NORM_COLS + SHORT_TAM + ['full_survey_responses']
rag_merged = rag_merged[rag_cols]
rag_merged.to_csv(OUT_RAG, index=False)
print(f'  Saved to {OUT_RAG}')
 
# ── FINAL SUMMARY ────────────────────────────────────────
overlap = set(gen['response_id']) & set(rag_merged['response_id'])
print()
print('=== BUILD COMPLETE ===')
print(f'Generation pool: {len(gen)} profiles  -> {OUT_GEN}')
print(f'RAG corpus:      {len(rag_merged)} profiles  -> {OUT_RAG}')
print(f'Overlap check:   {len(overlap)} shared IDs (must be 0)')
if len(overlap) > 0:
    print('  ERROR: Overlap detected! Check RAG ID matching.')
else:
    print('  OK: No overlap.')
