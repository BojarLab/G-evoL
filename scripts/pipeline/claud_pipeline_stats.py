#!/usr/bin/env python3
"""
Runner script to execute pipeline with statistical validation
============================================================
This shows how to use the statistical validation with your existing pipeline.
"""

# Import your existing pipeline (adjust the import based on your file name)
from claud_pipeline import MotifAnalysisPipeline  # or whatever your main pipeline file is called
from claud_script7 import StatisticalValidator

def run_pipeline_with_statistics():
    """Run the complete pipeline with statistical validation"""
    
    print("=" * 70)
    print("RUNNING MOTIF ANALYSIS PIPELINE WITH STATISTICAL VALIDATION")
    print("=" * 70)
    
    # Step 1: Run your main pipeline as usual
    print("\nStep 1: Running main pipeline...")
    pipeline = MotifAnalysisPipeline()
    pipeline.run()
    
    # Step 2: Add statistical validation
    print("\n" + "=" * 70)
    print("Step 2: Adding statistical validation...")
    print("=" * 70)
    
    validator = StatisticalValidator(pipeline)
    stats_results = validator.run_complete_validation()
    
    # Step 3: Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    
    robust_motifs = stats_results['robust_motifs']
    print(f"\nTotal motifs analyzed: {len(pipeline.selected_motifs)}")
    print(f"Robust motifs identified: {len(robust_motifs)}")
    print(f"Highly robust (significant in ≥2 methods): "
          f"{len(robust_motifs[robust_motifs['n_methods_significant'] >= 2])}")
    
    print("\nTop 10 most robust motifs:")
    for i, row in robust_motifs.head(10).iterrows():
        print(f"  {i+1}. {row['motif']} - selected by {row['n_methods_selected']} methods")
    
    print(f"\nAll results saved to: output/")
    print("Key files:")
    print("  - output/selection/robust_motifs.csv - High confidence motif list")
    print("  - output/reports/motif_significance_matrix.csv - Full significance data")
    print("  - output/reports/statistical_validation_report.txt - Detailed report")
    
    return pipeline, stats_results


# Alternative: If you've already run the pipeline and just want to add stats
def add_stats_to_existing_results():
    """Add statistical validation to existing pipeline results"""
    
    print("Adding statistical validation to existing results...")
    
    # Create a minimal pipeline instance just to load the results
    # This assumes you've already run the pipeline and have the outputs
    
    import pandas as pd
    from ete3 import NCBITaxa
    
    class MinimalPipeline:
        def __init__(self):
            self.config = {
                'output_dirs': {
                    'base': 'output',
                    'selection': 'output/motif_selection',
                    'reports': 'output/reports'
                }
            }
            # Load binary data
            self.binary_df = pd.read_csv('output/binary_motif_table.csv', index_col=0)
            self.ncbi = NCBITaxa()
            
            # Load selected motifs
            self.selected_motifs = {}
            import os
            
            method_files = {
                'mutual_info': 'mutual_info_motifs.csv',
                'mrmr': 'mrmr_motifs.csv',
                'random_forest': 'random_forest_motifs.csv',
                'co_occurrence': 'co_occurrence_motifs.csv',
                'pca': 'pca_motifs.csv',
                'phylogenetic': 'phylogenetic_motifs.csv'
            }
            
            for method, filename in method_files.items():
                filepath = os.path.join('output/motif_selection', filename)
                if os.path.exists(filepath):
                    motifs = pd.read_csv(filepath, header=None)[0].tolist()
                    self.selected_motifs[method] = motifs
    
    # Create minimal pipeline and run stats
    pipeline = MinimalPipeline()
    validator = StatisticalValidator(pipeline)
    stats_results = validator.run_complete_validation()
    
    print("Statistical validation complete!")
    return stats_results


if __name__ == "__main__":
    # Option 1: Run complete pipeline with stats
    # run_pipeline_with_statistics()
    
    # Option 2: Add stats to existing results (if you've already run the pipeline)
    add_stats_to_existing_results()