import psycopg2
from sentence_transformers import SentenceTransformer

# Use PubMedBERT model for medical text
model = SentenceTransformer('neuml/pubmedbert-base-embeddings')

def search(query, limit=5):
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        database="acr",
        user="postgres",
        password="password"
    )
    cur = conn.cursor()
    
    # Query using the raw variant text (no formatting)
    query_embedding = model.encode(query)
    
    cur.execute("""
    SELECT condition, variant, procedure, appropriateness, embedding <-> %s::vector as distance
    FROM acr_embeddings
    ORDER BY embedding <-> %s::vector
    LIMIT %s
    """, (query_embedding.tolist(), query_embedding.tolist(), limit))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    print(f"Query: '{query}'")
    print("-" * 80)
    for i, (condition, variant, procedure, appropriateness, distance) in enumerate(results, 1):
        print(f"{i}. {variant} | Distance: {distance:.6f}")
        print(f"   Condition: {condition}")
        print(f"   Procedure: {procedure} | {appropriateness}")
        print()

if __name__ == "__main__":
    # Test cross-modal exact match
    search("Device selection")  # Should find exact variant match 