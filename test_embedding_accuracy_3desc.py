import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class SyntheticDescriptionEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('neuml/pubmedbert-base-embeddings')
        self.conn = psycopg2.connect(
            host="127.0.0.1",
            port=5432,
            database="acr",
            user="postgres",
            password="password"
        )
        
    def evaluate_synthetic_descriptions(self):
        """
        Evaluates how well synthetic patient descriptions can retrieve 
        their corresponding original variants from the embedding database.
        """
        print("Loading synthetic patient cases...")
        
        # Load synthetic patient cases
        df = pd.read_csv('results/patient_cases_2.csv', delimiter='|')
        print(f"Loaded {len(df)} synthetic patient cases")
        
        # Test each description type
        results_summary = {}
        all_results = []
        
        for desc_type in ['desc_1', 'desc_2', 'desc_3']:
            print(f"\n=== Testing {desc_type} (Synthetic Descriptions) ===")
            
            correct_matches = 0
            total_queries = 0
            distances = []
            
            for idx, row in df.iterrows():
                original_variant = row['original_variant']
                synthetic_desc = row[desc_type]
                
                if pd.isna(synthetic_desc):
                    continue
                    
                total_queries += 1
                
                # Query with synthetic description
                query_embedding = self.model.encode(synthetic_desc)
                
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
                    distances.append(distance)
                    
                    # Check if we retrieved the correct original variant
                    exact_match = (retrieved_variant == original_variant)
                    if exact_match:
                        correct_matches += 1
                    
                    all_results.append({
                        'description_type': desc_type,
                        'query_id': idx + 1,
                        'original_variant': original_variant,
                        'synthetic_description': synthetic_desc,
                        'retrieved_condition': retrieved_condition,
                        'retrieved_variant': retrieved_variant,
                        'exact_match': 'Yes' if exact_match else 'No',
                        'euclidean_distance': round(distance, 6)
                    })
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} queries for {desc_type}")
            
            # Calculate metrics for this description type
            accuracy = correct_matches / total_queries if total_queries > 0 else 0
            mean_distance = np.mean(distances) if distances else 0
            
            results_summary[desc_type] = {
                'description_type': desc_type,
                'total_queries': total_queries,
                'exact_matches': correct_matches,
                'accuracy': round(accuracy, 4),
                'mean_distance': round(mean_distance, 6)
            }
            
            print(f"Results for {desc_type}:")
            print(f"  Accuracy: {accuracy:.2%} ({correct_matches}/{total_queries})")
            print(f"  Mean Distance: {mean_distance:.4f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(all_results)
        results_file = f"synthetic_desc_evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save summary
        summary_df = pd.DataFrame(list(results_summary.values()))
        summary_file = f"synthetic_desc_evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self._print_evaluation_report(results_summary, results_file, summary_file)
        
        return results_summary, results_file, summary_file
        
    def _print_evaluation_report(self, results_summary, results_file, summary_file):
        """Print comprehensive evaluation report."""
        print("\n" + "="*70)
        print("SYNTHETIC DESCRIPTION EMBEDDING EVALUATION REPORT")
        print("="*70)
        print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: neuml/pubmedbert-base-embeddings")
        print(f"Embedding Method: Combined Condition + Variant")
        print(f"Query Method: Synthetic Patient Descriptions")
        print(f"Distance Metric: Euclidean Distance (L2)")
        print(f"Evaluation Method: Cross-Modal Retrieval")
        print()
        
        print("PERFORMANCE COMPARISON:")
        print(f"{'Description Type':<15} {'Accuracy':<10} {'Exact Matches':<15} {'Mean Distance':<15}")
        print("-" * 55)
        
        for desc_type, stats in results_summary.items():
            desc_label = {
                'desc_1': 'Closest',
                'desc_2': 'Moderate', 
                'desc_3': 'Distant'
            }.get(desc_type, desc_type)
            
            print(f"{desc_label:<15} {stats['accuracy']:<10.2%} "
                  f"{stats['exact_matches']}/{stats['total_queries']:<10} "
                  f"{stats['mean_distance']:<15.6f}")
        
        print()
        print("KEY INSIGHTS:")
        
        # Find best and worst performing description types
        best_desc = max(results_summary.keys(), key=lambda x: results_summary[x]['accuracy'])
        worst_desc = min(results_summary.keys(), key=lambda x: results_summary[x]['accuracy'])
        
        best_label = {'desc_1': 'Closest', 'desc_2': 'Moderate', 'desc_3': 'Distant'}.get(best_desc)
        worst_label = {'desc_1': 'Closest', 'desc_2': 'Moderate', 'desc_3': 'Distant'}.get(worst_desc)
        
        print(f"- Best Performance: {best_label} descriptions ({results_summary[best_desc]['accuracy']:.2%})")
        print(f"- Worst Performance: {worst_label} descriptions ({results_summary[worst_desc]['accuracy']:.2%})")
        
        accuracy_range = results_summary[best_desc]['accuracy'] - results_summary[worst_desc]['accuracy']
        print(f"- Accuracy Range: {accuracy_range:.2%}")
        
        print()
        print("FILES GENERATED:")
        print(f"- Detailed Results: {results_file}")
        print(f"- Summary Statistics: {summary_file}")
        print("="*70)

def main():
    evaluator = SyntheticDescriptionEvaluator()
    try:
        results_summary, results_file, summary_file = evaluator.evaluate_synthetic_descriptions()
        return results_summary
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
    finally:
        evaluator.conn.close()

if __name__ == "__main__":
    main() 