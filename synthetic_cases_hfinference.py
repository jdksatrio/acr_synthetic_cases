import pandas as pd
import os
import time
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def setup_client():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    ENDPOINT_URL = "https://ns9ck40mmcq4spq5.us-east-1.aws.endpoints.huggingface.cloud"
    return InferenceClient(ENDPOINT_URL, token=HF_TOKEN)

def generate_descriptions(client, variant):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a triage clinician writing brief clinical one-liners for patient presentations.\n"
                "Task: For each ACR condition/variant, create exactly THREE concise clinical notes:\n"
                "  1. Closest – similar to the original scenario\n"
                "  2. Moderate – same clinical picture, different wording\n"
                "  3. Distant – clearly different presentation style but same medical facts\n\n"
                "**Style Requirements:**\n"
                "- Write SHORT clinical one-liners (1-2 sentences max)\n"
                "- Use triage/ED note format: 'Age+Gender with chief complaint and key symptoms'\n"
                "- Include age, gender, and primary symptoms only\n"
                "- Use natural clinical abbreviations and terminology\n"
                "- Sound like real emergency/triage notes, not textbook descriptions\n\n"
                "**HARD RULE: You MUST NOT mention imaging tests, scans, CT, MRI, "
                "X-ray, or any diagnostic orders.** Focus only on clinical presentation.\n\n"
                "**Good examples** (natural clinical notes)\n"
                "Original: Suspected pelvic-origin lower-extremity varicose veins in females. Initial diagnosis.\n"
                "1. 36F with bilateral leg varicose veins and aching pain, worse after prolonged standing.\n"
                "2. 36-year-old woman presents with bulging leg veins and heaviness since second pregnancy.\n"
                "3. Female, 36, c/o throbbing lower extremity varicosities and leg discomfort.\n\n"
                "**Bad examples (do NOT do this)**\n"
                "- Long narrative descriptions\n"
                "- Mentions of imaging orders\n"
                "- Overly formal textbook language\n\n"
                "Return only the three numbered clinical one-liners."
            )
        },
        {
            "role": "user",
            "content": f"Variant: {variant}"
        }
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.4
        )
        
        content = response.choices[0].message.content.strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) >= 3:
            desc_1 = lines[0].lstrip('1. ')
            desc_2 = lines[1].lstrip('2. ')
            desc_3 = lines[2].lstrip('3. ')
            return desc_1, desc_2, desc_3
        else:
            return None, None, None
            
    except Exception as e:
        print(f"Error generating descriptions for variant: {variant[:50]}... - {e}")
        return None, None, None

def main():
    df = pd.read_csv('results/synthetic.csv', delimiter='|')
    client = setup_client()
    
    results = []
    
    for idx, row in df.iterrows():
        variant = row['Variant']
        procedure_json = row['procedure_json']
        
        print(f"Processing {idx+1}/{len(df)}: {variant[:50]}...")
        
        desc_1, desc_2, desc_3 = generate_descriptions(client, variant)
        
        if desc_1 and desc_2 and desc_3:
            results.append({
                'original_variant': variant,
                'desc_1': desc_1,
                'desc_2': desc_2,
                'desc_3': desc_3,
                'procedure': procedure_json
            })
        else:
            print(f"Failed to generate descriptions for: {variant[:50]}...")
        
        time.sleep(0.5)
    
    output_df = pd.DataFrame(results)
    output_df.to_csv('results/patient_cases_2.csv', sep='|', index=False)
    
    print(f"Generated {len(results)} patient cases saved to results/patient_cases_full.csv")

if __name__ == "__main__":
    main() 