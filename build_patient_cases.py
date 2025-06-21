import pandas as pd
import json
import random
from openai import OpenAI
import os

N_CASES = 5

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_patient_case(variant):
    prompt = f"You are a physician. Write a one-sentence realistic patient case description for this clinical scenario: {variant}. Make it sound like a clinical note with demographics and context. You may choose to paraphrase or use synonyms of the words in the variant."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

df = pd.read_csv('synthetic.csv', delimiter='|')

appropriate_cases = []
for _, row in df.iterrows():
    procedures = json.loads(row['procedure_json'])
    usually_appropriate = {proc: appr for proc, appr in procedures.items() if appr == "Usually appropriate"}
    
    if usually_appropriate:
        appropriate_cases.append({
            'variant': row['Variant'],
            'procedures': list(usually_appropriate.keys())
        })

selected_cases = random.sample(appropriate_cases, min(N_CASES, len(appropriate_cases)))

patient_cases = []
for case in selected_cases:
    variant = case['variant']
    procedures = case['procedures']
    
    synthetic_patient_condition = generate_patient_case(variant)
    appropriate_imaging = "; ".join(procedures)
    
    patient_cases.append({
        'original_variant': variant,
        'synthetic_patient_condition': synthetic_patient_condition,
        'appropriate_imaging': appropriate_imaging
    })

result_df = pd.DataFrame(patient_cases)
result_df.to_csv('synthetic_patient_cases.csv', index=False, sep='|')

print("Sample cases:")
print(result_df.head()) 