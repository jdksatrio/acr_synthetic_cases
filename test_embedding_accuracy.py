import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class EmbeddingEvaluator:
    def __init__(self):
        # Use PubMedBERT model for medical text
        self.model = SentenceTransformer('neuml/pubmedbert-base-embeddings')
        self.conn = psycopg2.connect(
            host="127.0.0.1",
            port=5432,
            database="acr",
            user="postgres",
            password="password"
        )
        
    def evaluate_exact_match_retrieval(self):
        """
        Evaluates the embedding system's ability to retrieve exact matches
        for condition-variant pairs through vector similarity search.
        """
        print("Initializing embedding evaluation...")
        
        # Load reference dataset
        df = pd.read_csv('dataset/acr.csv', delimiter='|')
        unique_scenarios = df[['Condition', 'Variant']].drop_duplicates().reset_index(drop=True)
        
        print(f"Evaluating {len(unique_scenarios)} unique condition-variant pairs...")
        
        # Initialize results storage
        evaluation_results = []
        
        for idx, row in unique_scenarios.iterrows():
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(unique_scenarios)} ({idx/len(unique_scenarios)*100:.1f}%)")
            
            # Query using only the variant (clinical scenario)
            query_text = row['Variant']
            
            # Perform vector similarity search
            query_embedding = self.model.encode(query_text)
            
            cur = self.conn.cursor()
            cur.execute("""
            SELECT condition, variant, embedding <-> %s::vector as distance
            FROM acr_embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT 1
            """, (query_embedding.tolist(), query_embedding.tolist()))
            
            result = cur.fetchone()
            cur.close()
            
            if result:
                retrieved_condition, retrieved_variant, distance = result
                
                # Determine exact match
                exact_match = (retrieved_condition == row['Condition'] and 
                              retrieved_variant == row['Variant'])
                
                evaluation_results.append({
                    'query_id': idx + 1,
                    'true_condition': row['Condition'],
                    'true_variant': row['Variant'],
                    'retrieved_condition': retrieved_condition,
                    'retrieved_variant': retrieved_variant,
                    'exact_match': 'Yes' if exact_match else 'No',
                    'cosine_distance': round(distance, 6)
                })
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(evaluation_results)
        
        # Generate evaluation metrics
        total_queries = len(results_df)
        exact_matches = len(results_df[results_df['exact_match'] == 'Yes'])
        accuracy = exact_matches / total_queries if total_queries > 0 else 0
        
        # Create summary statistics
        summary_stats = {
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': total_queries,
            'exact_matches': exact_matches,
            'accuracy': round(accuracy, 4),
            'mean_distance_exact_matches': round(results_df[results_df['exact_match'] == 'Yes']['cosine_distance'].mean(), 6) if exact_matches > 0 else None,
            'mean_distance_all': round(results_df['cosine_distance'].mean(), 6)
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f'embedding_evaluation_results_{timestamp}.csv'
        results_df.to_csv(results_filename, index=False)
        
        # Save summary
        summary_filename = f'embedding_evaluation_summary_{timestamp}.csv'
        pd.DataFrame([summary_stats]).to_csv(summary_filename, index=False)
        
        # Print formal evaluation report
        self._print_evaluation_report(summary_stats, results_filename, summary_filename)
        
        return results_df, summary_stats
    
    def _print_evaluation_report(self, stats, results_file, summary_file):
        """Print formal evaluation report."""
        print("\n" + "="*60)
        print("EMBEDDING SYSTEM EVALUATION REPORT")
        print("="*60)
        print(f"Evaluation Date: {stats['evaluation_timestamp']}")
        print(f"Model: neuml/pubmedbert-base-embeddings")
        print(f"Embedding Method: Combined Condition + Variant")
        print(f"Query Method: Variant Only")
        print(f"Evaluation Method: Cross-Modal Exact Match Retrieval")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Total Test Cases: {stats['total_queries']:,}")
        print(f"  Exact Matches: {stats['exact_matches']:,}")
        print(f"  Accuracy (Top-1): {stats['accuracy']:.2%}")
        print(f"  Mean Distance (Exact): {stats['mean_distance_exact_matches']}")
        print(f"  Mean Distance (All): {stats['mean_distance_all']}")
        print()
        print("OUTPUT FILES:")
        print(f"  Detailed Results: {results_file}")
        print(f"  Summary Statistics: {summary_file}")
        print()
        
        if stats['accuracy'] == 1.0:
            print("ASSESSMENT: Perfect retrieval performance.")
        elif stats['accuracy'] >= 0.95:
            print("ASSESSMENT: Excellent retrieval performance.")
        elif stats['accuracy'] >= 0.90:
            print("ASSESSMENT: Good retrieval performance.")
        elif stats['accuracy'] >= 0.80:
            print("ASSESSMENT: Acceptable retrieval performance.")
        else:
            print("ASSESSMENT: Sub-optimal retrieval performance requires investigation.")
        
        print("="*60)
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    evaluator = EmbeddingEvaluator()
    try:
        results_df, summary_stats = evaluator.evaluate_exact_match_retrieval()
    finally:
        evaluator.close()

if __name__ == "__main__":
    main() 