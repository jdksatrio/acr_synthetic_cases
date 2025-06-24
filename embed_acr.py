import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

# Use PubMedBERT model for medical text
model = SentenceTransformer('neuml/pubmedbert-base-embeddings')

df = pd.read_csv('dataset/acr.csv', delimiter='|')

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    database="acr",
    user="postgres",
    password="password"
)
cur = conn.cursor()

cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS acr_embeddings;
CREATE TABLE acr_embeddings (
    id SERIAL PRIMARY KEY,
    condition TEXT,
    variant TEXT,
    procedure TEXT,
    appropriateness TEXT,
    combined_text TEXT,
    embedding vector(768)
);
""")

for idx, row in df.iterrows():
    # Embed the combined condition and variant text
    combined_text = f"Condition: {row['Condition']} | Clinical Scenario: {row['Variant']}"
    
    embedding = model.encode(combined_text)
    
    cur.execute("""
    INSERT INTO acr_embeddings (condition, variant, procedure, appropriateness, combined_text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        row['Condition'],
        row['Variant'], 
        row['Procedure'],
        row['Appropriateness Category'],
        combined_text,
        embedding.tolist()
    ))

conn.commit()
cur.close()
conn.close()
print("Done!") 