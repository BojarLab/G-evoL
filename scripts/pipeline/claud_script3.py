import pandas as pd
import numpy as np
from ete3 import NCBITaxa
import matplotlib.pyplot as plt
import seaborn as sns
from claud_script2 import MotifSelector
from claud_script4 import MotifEvolutionAnalyzer, enrichment_by_lineage, plot_lineage_enrichment_heatmap

# Pipeline configuration
CONFIG = {
    'selection_methods': {
        'mutual_info': {'enabled': True, 'n_motifs': 50},
        'mrmr': {'enabled': True, 'n_motifs': 30},
        'random_forest': {'enabled': True, 'n_motifs': 50},
        'co_occurrence': {'enabled': True, 'min_correlation': 0.3},
        'pca': {'enabled': True, 'n_components': 20},
        'combined': {'enabled': True, 'n_motifs': 40}
    },
    'evolution_analysis': {
        'tax_levels': ['family', 'order', 'class'],
        'enrichment_tax_level': 'phylum',
        'min_samples_per_taxon': 5
    },
    'output_dir': 'output/evolution_analysis/'
}


def run_complete_pipeline(binary_df_path="output/binary_motif_table.csv"):
    """Run complete motif selection and evolution analysis pipeline"""

    # Create output directory
    import os
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Load data
    print("Loading data...")
    binary_df = pd.read_csv(binary_df_path, index_col=0)
    binary_df['taxid'] = pd.to_numeric(binary_df['taxid'], errors='coerce').dropna().astype(int)

    # Initialize tools
    ncbi = NCBITaxa()
    selector = MotifSelector(binary_df, taxonomy_col='taxid')

    # Step 1: Run multiple selection methods
    print("\n=== STEP 1: MOTIF SELECTION ===")
    selected_motifs = {}

    if CONFIG['selection_methods']['mutual_info']['enabled']:
        print("\nRunning Mutual Information selection...")
        mi_motifs, mi_scores = selector.method2_mutual_information()
        n = CONFIG['selection_methods']['mutual_info']['n_motifs']
        selected_motifs['mutual_info'] = mi_motifs[:n]
        pd.Series(mi_motifs[:n]).to_csv(f"{CONFIG['output_dir']}mutual_info_motifs.csv",
                                        header=False, index=False)

    if CONFIG['selection_methods']['mrmr']['enabled']:
        print("\nRunning mRMR selection...")
        n = CONFIG['selection_methods']['mrmr']['n_motifs']
        mrmr_motifs, _ = selector.method3_maximum_relevance_minimum_redundancy(n_motifs=n)
        selected_motifs['mrmr'] = mrmr_motifs
        pd.Series(mrmr_motifs).to_csv(f"{CONFIG['output_dir']}mrmr_motifs.csv",
                                      header=False, index=False)

    if CONFIG['selection_methods']['random_forest']['enabled']:
        print("\nRunning Random Forest selection...")
        rf_motifs, rf_scores = selector.method4_random_forest_importance()
        n = CONFIG['selection_methods']['random_forest']['n_motifs']
        selected_motifs['random_forest'] = rf_motifs[:n]
        pd.Series(rf_motifs[:n]).to_csv(f"{CONFIG['output_dir']}random_forest_motifs.csv",
                                        header=False, index=False)

    if CONFIG['selection_methods']['co_occurrence']['enabled']:
        print("\nRunning Co-occurrence module selection...")
        min_corr = CONFIG['selection_methods']['co_occurrence']['min_correlation']
        module_reps, modules = selector.method5_co_occurrence_modules(min_correlation=min_corr)
        selected_motifs['co_occurrence'] = module_reps
        pd.Series(module_reps).to_csv(f"{CONFIG['output_dir']}co_occurrence_motifs.csv",
                                      header=False, index=False)

    if CONFIG['selection_methods']['pca']['enabled']:
        print("\nRunning PCA-based selection...")
        n = CONFIG['selection_methods']['pca']['n_components']
        pca_motifs, variance_explained = selector.method6_pca_loadings(n_components=n)
        selected_motifs['pca'] = pca_motifs
        pd.Series(pca_motifs).to_csv(f"{CONFIG['output_dir']}pca_motifs.csv",
                                     header=False, index=False)

    if CONFIG['selection_methods']['combined']['enabled']:
        print("\nRunning Combined method selection...")
        combined_motifs, combined_scores = selector.combine_methods()
        n = CONFIG['selection_methods']['combined']['n_motifs']
        selected_motifs['combined'] = combined_motifs[:n]
        pd.Series(combined_motifs[:n]).to_csv(f"{CONFIG['output_dir']}combined_motifs.csv",
                                              header=False, index=False)

    # Step 2: Evolution analysis for each motif set
    print("\n=== STEP 2: EVOLUTION ANALYSIS ===")
    evolution_results = {}

    for method_name, motifs in selected_motifs.items():
        print(f"\n--- Analyzing {method_name} motifs ({len(motifs)} motifs) ---")

        # Initialize analyzer
        analyzer = MotifEvolutionAnalyzer(binary_df, ncbi)

        # Analyze at multiple taxonomic levels
        for tax_level in CONFIG['evolution_analysis']['tax_levels']:
            print(f"\nAnalyzing at {tax_level} level...")

            try:
                results, tree = analyzer.analyze_motif_evolution(motifs, tax_level=tax_level)

                if results:
                    # Save detailed results
                    results_df = pd.DataFrame(results).T
                    results_df.to_csv(f"{CONFIG['output_dir']}{method_name}_{tax_level}_evolution.csv")

                    # Plot summary
                    save_path = f"{CONFIG['output_dir']}{method_name}_{tax_level}_evolution_summary.png"
                    summary_df = analyzer.plot_evolution_summary(results, top_n=20, save_path=save_path)

                    # Store key metrics
                    evolution_results[f"{method_name}_{tax_level}"] = {
                        'total_gains': results_df['gains'].sum(),
                        'total_losses': results_df['losses'].sum(),
                        'avg_changes': results_df['total_changes'].mean()
                    }

            except Exception as e:
                print(f"Error in evolution analysis: {e}")
                continue

        # Lineage enrichment analysis
        print(f"\nRunning lineage enrichment for {method_name}...")
        enrichment_level = CONFIG['evolution_analysis']['enrichment_tax_level']
        enrichment_df = enrichment_by_lineage(binary_df, motifs, tax_level=enrichment_level)

        if not enrichment_df.empty:
            # Save enrichment results
            enrichment_df.to_csv(f"{CONFIG['output_dir']}{method_name}_lineage_enrichment.csv")

            # Plot heatmap
            plot_lineage_enrichment_heatmap(
                enrichment_df,
                min_samples=CONFIG['evolution_analysis']['min_samples_per_taxon'],
                save_path=f"{CONFIG['output_dir']}{method_name}_enrichment_heatmap.png"
            )

    # Step 3: Compare methods
    print("\n=== STEP 3: METHOD COMPARISON ===")
    compare_selection_methods(selected_motifs, binary_df, evolution_results)

    # Step 4: Generate summary report
    generate_summary_report(selected_motifs, evolution_results, binary_df)

    print("\n✅ Pipeline completed! Results saved to:", CONFIG['output_dir'])


def compare_selection_methods(selected_motifs, binary_df, evolution_results):
    """Compare different selection methods"""

    # 1. Overlap analysis
    print("\nMotif overlap between methods:")
    overlap_matrix = np.zeros((len(selected_motifs), len(selected_motifs)))
    method_names = list(selected_motifs.keys())

    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i <= j:
                set1 = set(selected_motifs[method1])
                set2 = set(selected_motifs[method2])
                overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                overlap_matrix[i, j] = overlap_matrix[j, i] = overlap

    # Plot overlap heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix, xticklabels=method_names, yticklabels=method_names,
                annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
    plt.title("Jaccard Similarity Between Selection Methods")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}method_overlap_heatmap.png", dpi=300)
    plt.close()

    # 2. Performance metrics comparison
    metrics_data = []
    for method in method_names:
        motifs = selected_motifs[method]

        # Coverage: fraction of species with at least one motif
        coverage = (binary_df[motifs].sum(axis=1) > 0).mean()

        # Average frequency
        avg_freq = binary_df[motifs].mean().mean()

        # Diversity: average pairwise distance between motifs
        motif_patterns = binary_df[motifs].T
        if len(motifs) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(motif_patterns.values, metric='jaccard')
            avg_distance = distances.mean()
        else:
            avg_distance = 0

        metrics_data.append({
            'Method': method,
            'Coverage': coverage,
            'Avg_Frequency': avg_freq,
            'Diversity': avg_distance,
            'N_Motifs': len(motifs)
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f"{CONFIG['output_dir']}method_comparison_metrics.csv", index=False)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Coverage
    axes[0, 0].bar(metrics_df['Method'], metrics_df['Coverage'])
    axes[0, 0].set_ylabel('Species Coverage')
    axes[0, 0].set_title('Coverage by Selection Method')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Average frequency
    axes[0, 1].bar(metrics_df['Method'], metrics_df['Avg_Frequency'])
    axes[0, 1].set_ylabel('Average Motif Frequency')
    axes[0, 1].set_title('Motif Frequency by Method')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Diversity
    axes[1, 0].bar(metrics_df['Method'], metrics_df['Diversity'])
    axes[1, 0].set_ylabel('Average Pairwise Distance')
    axes[1, 0].set_title('Motif Diversity by Method')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Evolution metrics (if available)
    if evolution_results:
        evo_data = []
        for key, values in evolution_results.items():
            method = key.split('_')[0]
            evo_data.append({
                'Method': method,
                'Total_Changes': values['total_gains'] + values['total_losses']
            })

        evo_df = pd.DataFrame(evo_data).groupby('Method')['Total_Changes'].mean().reset_index()
        axes[1, 1].bar(evo_df['Method'], evo_df['Total_Changes'])
        axes[1, 1].set_ylabel('Average Evolutionary Changes')
        axes[1, 1].set_title('Evolutionary Dynamics by Method')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}method_comparison_plots.png", dpi=300)
    plt.close()


def generate_summary_report(selected_motifs, evolution_results, binary_df):
    """Generate a summary report of the analysis"""

    with open(f"{CONFIG['output_dir']}analysis_summary_report.txt", 'w') as f:
        f.write("=== MOTIF EVOLUTION ANALYSIS SUMMARY ===\n\n")

        f.write("1. DATA OVERVIEW\n")
        f.write(f"   - Total species: {len(binary_df)}\n")
        f.write(f"   - Total motifs analyzed: {len([c for c in binary_df.columns if c != 'taxid'])}\n")
        f.write(f"   - Analysis date: {pd.Timestamp.now()}\n\n")

        f.write("2. SELECTION METHODS USED\n")
        for method, motifs in selected_motifs.items():
            f.write(f"   - {method}: {len(motifs)} motifs selected\n")

        f.write("\n3. TOP EVOLVING MOTIFS\n")
        if evolution_results:
            # Find motifs with most changes across all analyses
            all_changes = {}
            for key, values in evolution_results.items():
                method = key.split('_')[0]
                changes = values['total_gains'] + values['total_losses']
                if method not in all_changes:
                    all_changes[method] = 0
                all_changes[method] += changes

            sorted_methods = sorted(all_changes.items(), key=lambda x: x[1], reverse=True)
            for method, total_changes in sorted_methods[:5]:
                f.write(f"   - {method}: {total_changes} total gain/loss events\n")

        f.write("\n4. RECOMMENDATIONS\n")
        f.write("   - For phylogenetic studies: Use mRMR or combined method motifs\n")
        f.write("   - For classification: Use Random Forest selected motifs\n")
        f.write("   - For evolutionary dynamics: Focus on motifs with high gain/loss rates\n")
        f.write("   - Consider taxonomic level based on your research question\n")

        f.write("\n5. OUTPUT FILES GENERATED\n")
        f.write(f"   - Motif lists: *_motifs.csv\n")
        f.write(f"   - Evolution results: *_evolution.csv\n")
        f.write(f"   - Enrichment analyses: *_enrichment.csv\n")
        f.write(f"   - Visualizations: *.png\n")
        f.write(f"   - All files saved to: {CONFIG['output_dir']}\n")


if __name__ == "__main__":
    # Run the complete pipeline
    run_complete_pipeline()