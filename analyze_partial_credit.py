import pandas as pd
import json
import ast
from collections import defaultdict

def safe_parse_json(json_str):
    """Safely parse JSON string, handling various formats"""
    if pd.isna(json_str):
        return []
    
    try:
        # Try direct JSON parsing
        return json.loads(json_str)
    except:
        try:
            # Try ast.literal_eval for Python-like strings
            return ast.literal_eval(json_str)
        except:
            # Return empty list if parsing fails
            return []

def extract_procedures(procedure_json):
    """Extract procedure codes/names from procedure JSON"""
    procedures = safe_parse_json(procedure_json)
    if not procedures:
        return set()
    
    # Extract procedure identifiers (codes or names)
    proc_set = set()
    for proc in procedures:
        if isinstance(proc, dict):
            # Add procedure code if available
            if 'code' in proc:
                proc_set.add(proc['code'])
            # Add procedure name if available
            if 'name' in proc:
                proc_set.add(proc['name'])
            # Add the whole dict as string if no specific fields
            if not proc_set:
                proc_set.add(str(proc))
        else:
            proc_set.add(str(proc))
    
    return proc_set

def load_acr_mapping():
    """Load the original ACR dataset to create condition->variant and variant->procedure mappings"""
    print("Loading original ACR dataset...")
    
    # Load synthetic.csv to get the procedure mappings
    df_synthetic = pd.read_csv('results/synthetic.csv', delimiter='|')
    variant_to_procedure = {}
    for _, row in df_synthetic.iterrows():
        variant_to_procedure[row['Variant']] = row['procedure_json']
    
    # Load the main ACR dataset to get condition mappings
    try:
        df_acr = pd.read_csv('dataset/acr_appropriateness_criteria.csv')
        condition_to_variants = defaultdict(set)
        variant_to_condition = {}
        
        for _, row in df_acr.iterrows():
            condition = row['Condition']
            variant = row['Variant']
            condition_to_variants[condition].add(variant)
            variant_to_condition[variant] = condition
            
    except FileNotFoundError:
        print("Warning: Main ACR dataset not found, using synthetic data only")
        # Fallback: try to infer conditions from variant text
        variant_to_condition = {}
        condition_to_variants = defaultdict(set)
        
        for variant in variant_to_procedure.keys():
            # Simple heuristic: extract condition from variant text
            # This is imperfect but better than nothing
            condition = variant.split('.')[0] if '.' in variant else variant[:50]
            variant_to_condition[variant] = condition
            condition_to_variants[condition].add(variant)
    
    return variant_to_procedure, variant_to_condition, condition_to_variants

def analyze_partial_credit():
    """Analyze procedure-level and condition-level accuracy"""
    
    # Load mappings
    variant_to_procedure, variant_to_condition, condition_to_variants = load_acr_mapping()
    
    # Load evaluation results
    results_file = 'synthetic_desc_evaluation_results_20250625_192222.csv'
    df_results = pd.read_csv(results_file)
    
    print(f"Loaded {len(df_results)} evaluation results")
    
    # Analysis by description type
    analysis = {}
    
    for desc_type in ['desc_1', 'desc_2', 'desc_3']:
        desc_results = df_results[df_results['description_type'] == desc_type].copy()
        
        exact_matches = 0
        procedure_matches = 0
        condition_matches = 0
        total = len(desc_results)
        
        procedure_precision_scores = []
        procedure_recall_scores = []
        
        for _, row in desc_results.iterrows():
            original_variant = row['original_variant']
            retrieved_variant = row['retrieved_variant']
            
            # Exact match
            if original_variant == retrieved_variant:
                exact_matches += 1
                procedure_matches += 1
                condition_matches += 1
                procedure_precision_scores.append(1.0)
                procedure_recall_scores.append(1.0)
                continue
            
            # Procedure-level analysis
            original_procedures = extract_procedures(variant_to_procedure.get(original_variant, '[]'))
            retrieved_procedures = extract_procedures(variant_to_procedure.get(retrieved_variant, '[]'))
            
            if original_procedures and retrieved_procedures:
                # Calculate precision and recall for procedures
                intersection = original_procedures.intersection(retrieved_procedures)
                if intersection:
                    procedure_matches += 1
                
                precision = len(intersection) / len(retrieved_procedures) if retrieved_procedures else 0
                recall = len(intersection) / len(original_procedures) if original_procedures else 0
                
                procedure_precision_scores.append(precision)
                procedure_recall_scores.append(recall)
            else:
                procedure_precision_scores.append(0.0)
                procedure_recall_scores.append(0.0)
            
            # Condition-level analysis
            original_condition = variant_to_condition.get(original_variant)
            retrieved_condition = variant_to_condition.get(retrieved_variant)
            
            if original_condition and retrieved_condition and original_condition == retrieved_condition:
                condition_matches += 1
        
        # Calculate metrics
        exact_accuracy = exact_matches / total
        procedure_accuracy = procedure_matches / total
        condition_accuracy = condition_matches / total
        
        avg_procedure_precision = sum(procedure_precision_scores) / len(procedure_precision_scores)
        avg_procedure_recall = sum(procedure_recall_scores) / len(procedure_recall_scores)
        procedure_f1 = 2 * (avg_procedure_precision * avg_procedure_recall) / (avg_procedure_precision + avg_procedure_recall) if (avg_procedure_precision + avg_procedure_recall) > 0 else 0
        
        analysis[desc_type] = {
            'total_queries': total,
            'exact_matches': exact_matches,
            'procedure_matches': procedure_matches,
            'condition_matches': condition_matches,
            'exact_accuracy': exact_accuracy,
            'procedure_accuracy': procedure_accuracy,
            'condition_accuracy': condition_accuracy,
            'procedure_precision': avg_procedure_precision,
            'procedure_recall': avg_procedure_recall,
            'procedure_f1': procedure_f1
        }
    
    return analysis

def print_analysis_report(analysis):
    """Print comprehensive analysis report"""
    print("\n" + "="*80)
    print("PARTIAL CREDIT ANALYSIS REPORT")
    print("="*80)
    
    print(f"{'Metric':<25} {'Desc 1':<15} {'Desc 2':<15} {'Desc 3':<15}")
    print("-" * 80)
    
    # Exact accuracy
    print(f"{'Exact Accuracy':<25} ", end="")
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        print(f"{analysis[desc]['exact_accuracy']:<15.2%}", end="")
    print()
    
    # Procedure accuracy
    print(f"{'Procedure Accuracy':<25} ", end="")
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        print(f"{analysis[desc]['procedure_accuracy']:<15.2%}", end="")
    print()
    
    # Condition accuracy
    print(f"{'Condition Accuracy':<25} ", end="")
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        print(f"{analysis[desc]['condition_accuracy']:<15.2%}", end="")
    print()
    
    print("-" * 80)
    
    # Procedure precision/recall/F1
    print(f"{'Procedure Precision':<25} ", end="")
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        print(f"{analysis[desc]['procedure_precision']:<15.2%}", end="")
    print()
    
    print(f"{'Procedure Recall':<25} ", end="")
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        print(f"{analysis[desc]['procedure_recall']:<15.2%}", end="")
    print()
    
    print(f"{'Procedure F1':<25} ", end="")
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        print(f"{analysis[desc]['procedure_f1']:<15.2%}", end="")
    print()
    
    print("\n" + "="*80)
    print("INSIGHTS:")
    
    # Calculate improvements
    for desc in ['desc_1', 'desc_2', 'desc_3']:
        exact = analysis[desc]['exact_accuracy']
        procedure = analysis[desc]['procedure_accuracy']
        condition = analysis[desc]['condition_accuracy']
        
        proc_lift = procedure - exact
        cond_lift = condition - exact
        
        desc_label = {'desc_1': 'Closest', 'desc_2': 'Moderate', 'desc_3': 'Distant'}[desc]
        
        print(f"\n{desc_label} Descriptions:")
        print(f"  - Procedure-level lift: +{proc_lift:.2%} (from {exact:.1%} to {procedure:.1%})")
        print(f"  - Condition-level lift: +{cond_lift:.2%} (from {exact:.1%} to {condition:.1%})")

def main():
    try:
        analysis = analyze_partial_credit()
        print_analysis_report(analysis)
        
        # Save detailed analysis
        df_analysis = pd.DataFrame(analysis).T
        df_analysis.to_csv('partial_credit_analysis.csv')
        print(f"\nDetailed analysis saved to: partial_credit_analysis.csv")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 