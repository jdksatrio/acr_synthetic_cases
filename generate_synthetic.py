import pandas as pd
import json

df = pd.read_csv('dataset/acr.csv', delimiter='|')

synthetic_data = []
for variant, group in df.groupby('Variant'):
    procedures = {}
    for _, row in group.iterrows():
        procedures[row['Procedure']] = row['Appropriateness Category']
    
    procedure_json = json.dumps(procedures, separators=(',', ':'))
    synthetic_data.append({
        'Variant': variant,
        'procedure_json': procedure_json
    })

synthetic_df = pd.DataFrame(synthetic_data)
synthetic_df.to_csv('synthetic.csv', index=False, sep='|') 