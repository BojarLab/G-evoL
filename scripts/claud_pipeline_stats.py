#!/usr/bin/env python3
"""
Modified Main Pipeline Section to Integrate Statistical Validation
==================================================================
Add this to your main pipeline file to include statistical validation
"""

# Add to imports at the top of your main pipeline file:
import pandas as pd
import numpy as np
import re
import os
import pickle
from ete3 import NCBITaxa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Import your scripts (assuming they're in the same directory)
from claud_script2 import MotifSelector
from claud_script6 import (
    StatisticalValidator,
    CrossMethodAnalyzer,
    EnhancedPipelineExtension,
    add_statistical_validation
)


# Modify the MotifAnalysisPipeline class to store selection scores:

class MotifAnalysisPipeline:
    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        self.config = config or self.get_default_config()
        self.ncbi = NCBITaxa()
        self.results = {}
        self.selection_scores = {}  # ADD THIS: Store scores for validation

        # Update config to include statistical parameters
        if 'statistical_validation' not in self.config:
            self.config['statistical_validation'] = {
                'enabled': True,
                'n_bootstrap': 1000,
                'alpha': 0.05,
                'min_methods_for_robust': 3,
                'require_significance': True
            }

        # Create output directories
        for dir_path in self.config['output_dirs'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def step2_select_motifs(self):
        """Modified Step 2: Store scores for statistical validation"""
        print("\n" + "=" * 60)
        print("STEP 2: MOTIF SELECTION")
        print("=" * 60)

        selector = MotifSelector(self.binary_df, taxonomy_col='taxid')
        selected_motifs = {}

        # Run each enabled method and STORE SCORES
        methods_config = self.config['selection_methods']

        if methods_config['mutual_info']['enabled']:
            print("\n1. Mutual Information Selection...")
            mi_motifs, mi_scores = selector.method2_mutual_information()
            n = methods_config['mutual_info']['n_motifs']
            selected_motifs['mutual_info'] = mi_motifs[:n]
            self.selection_scores['mutual_info'] = dict(zip(mi_motifs, mi_scores))  # STORE SCORES
            self._save_motifs(mi_motifs[:n], 'mutual_info_motifs.csv')
            print(f"   Selected {n} motifs")

        # Similar modifications for other methods...
        if methods_config['random_forest']['enabled']:
            print("\n3. Random Forest Selection...")
            rf_motifs, rf_scores = selector.method4_random_forest_importance()
            n = methods_config['random_forest']['n_motifs']
            selected_motifs['random_forest'] = rf_motifs[:n]
            self.selection_scores['random_forest'] = dict(zip(rf_motifs, rf_scores))  # STORE SCORES
            self._save_motifs(rf_motifs[:n], 'random_forest_motifs.csv')
            print(f"   Selected {n} motifs")

        # ... continue for other methods ...

        self.selected_motifs = selected_motifs
        self.results['selection_summary'] = {
            method: len(motifs) for method, motifs in selected_motifs.items()
        }

        return selected_motifs

    def step4_statistical_validation(self):
        """NEW Step 4: Statistical validation and enhanced cross-method analysis"""
        if not self.config['statistical_validation']['enabled']:
            print("\nStatistical validation disabled in config")
            return None

        print("\n" + "=" * 60)
        print("STEP 4: STATISTICAL VALIDATION")
        print("=" * 60)

        # Initialize validator
        validator = StatisticalValidator(
            self.binary_df,
            self.ncbi,
            n_bootstrap=self.config['statistical_validation']['n_bootstrap'],
            alpha=self.config['statistical_validation']['alpha']
        )

        validation_results = {}

        # Validate each method
        print("\nValidating selection methods...")

        # Phylogenetic signal validation
        if 'phylogenetic' in self.selected_motifs and 'phylogenetic' in self.selection_scores:
            print("- Validating phylogenetic signal...")
            validation_results['phylogenetic'] = validator.validate_phylogenetic_signal(
                self.selection_scores['phylogenetic'],
                tax_level=self.config['selection_methods']['phylogenetic']['tax_level']
            )
            print(f"  Significant motifs: {validation_results['phylogenetic']['n_significant']}")

        # Mutual information validation
        if 'mutual_info' in self.selected_motifs and 'mutual_info' in self.selection_scores:
            print("- Validating mutual information...")
            validation_results['mutual_info'] = validator.validate_mutual_information(
                self.selection_scores['mutual_info']
            )
            print(f"  Significant motifs: {len(validation_results['mutual_info']['significant_motifs'])}")

        # Random forest validation
        if 'random_forest' in self.selected_motifs and 'random_forest' in self.selection_scores:
            print("- Validating random forest importance...")
            validation_results['random_forest'] = validator.validate_random_forest(
                self.selection_scores['random_forest']
            )
            print(f"  Significant motifs: {validation_results['random_forest']['n_significant']}")

        # Enhanced cross-method analysis
        print("\nPerforming enhanced cross-method analysis...")
        analyzer = CrossMethodAnalyzer(
            self.binary_df,
            self.selected_motifs,
            validation_results
        )

        # Create significance matrix
        sig_matrix = analyzer.create_significance_matrix()
        sig_file = os.path.join(self.config['output_dirs']['reports'], 'motif_significance_matrix.csv')
        sig_matrix.to_csv(sig_file)

        # Calculate method agreement
        agreement_matrix = analyzer.calculate_method_agreement(sig_matrix)
        agreement_file = os.path.join(self.config['output_dirs']['reports'], 'method_agreement_matrix.csv')
        agreement_matrix.to_csv(agreement_file)

        # Identify robust motifs
        robust_motifs = analyzer.identify_robust_motifs(
            sig_matrix,
            min_methods=self.config['statistical_validation']['min_methods_for_robust'],
            require_significance=self.config['statistical_validation']['require_significance']
        )
        robust_file = os.path.join(self.config['output_dirs']['reports'], 'robust_motifs.csv')
        robust_motifs.to_csv(robust_file, index=False)

        print(f"\nIdentified {len(robust_motifs)} statistically robust motifs")
        if len(robust_motifs) > 0:
            print("\nTop 10 most robust motifs:")
            print(robust_motifs[['motif', 'n_methods_selected', 'n_methods_significant', 'robustness_score']].head(10))

        # Create integrated ranking
        integrated_ranking = analyzer.create_integrated_ranking()
        ranking_file = os.path.join(self.config['output_dirs']['reports'], 'integrated_motif_ranking.csv')
        integrated_ranking.to_csv(ranking_file, index=False)

        # Create robust motif set for downstream analysis
        self.robust_motifs = robust_motifs['motif'].tolist()
        robust_motif_file = os.path.join(self.config['output_dirs']['selection'], 'robust_motifs.csv')
        pd.Series(self.robust_motifs).to_csv(robust_motif_file, header=False, index=False)

        # Store results
        self.validation_results = validation_results
        self.results['statistical_validation'] = {
            'n_robust_motifs': len(robust_motifs),
            'methods_validated': len(validation_results),
            'average_agreement': agreement_matrix.values[np.triu_indices_from(agreement_matrix.values, k=1)].mean()
        }

        # Create visualizations
        self._create_statistical_visualizations(sig_matrix, agreement_matrix, robust_motifs)

        return validation_results

    def _create_statistical_visualizations(self, sig_matrix, agreement_matrix, robust_motifs):
        """Create statistical validation visualizations"""

        # 1. Significance heatmap
        plt.figure(figsize=(12, 10))

        # Select top motifs for visualization
        motif_sums = sig_matrix.sum(axis=1)
        top_motifs = motif_sums.nlargest(50).index

        # Create custom colormap
        colors = ['white', 'lightblue', 'darkblue']
        n_bins = 3
        cmap = plt.cm.colors.ListedColormap(colors)

        ax = sns.heatmap(
            sig_matrix.loc[top_motifs],
            cmap=cmap,
            cbar_kws={'label': 'Selection Status', 'ticks': [0, 1, 2]},
            xticklabels=True,
            yticklabels=True,
            vmin=0,
            vmax=2
        )

        # Customize colorbar labels
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['Not Selected', 'Selected', 'Significant'])

        plt.title('Motif Selection and Statistical Significance Across Methods', fontsize=14)
        plt.xlabel('Selection Method')
        plt.ylabel('Motif')
        plt.tight_layout()

        plot_file = os.path.join(self.config['output_dirs']['reports'], 'motif_significance_heatmap.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Method agreement heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(agreement_matrix), k=1)

        sns.heatmap(
            agreement_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Agreement Score'}
        )

        plt.title('Statistical Agreement Between Selection Methods', fontsize=14)
        plt.tight_layout()

        plot_file = os.path.join(self.config['output_dirs']['reports'], 'method_agreement_heatmap.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Robustness distribution
        if len(robust_motifs) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Methods selected distribution
            robust_motifs['n_methods_selected'].value_counts().sort_index().plot(
                kind='bar', ax=ax1, color='skyblue'
            )
            ax1.set_xlabel('Number of Methods')
            ax1.set_ylabel('Number of Motifs')
            ax1.set_title('Distribution of Motif Selection Across Methods')

            # Significance distribution
            robust_motifs['n_methods_significant'].value_counts().sort_index().plot(
                kind='bar', ax=ax2, color='darkblue'
            )
            ax2.set_xlabel('Number of Methods (Significant)')
            ax2.set_ylabel('Number of Motifs')
            ax2.set_title('Distribution of Statistical Significance Across Methods')

            plt.suptitle('Robust Motif Analysis', fontsize=16)
            plt.tight_layout()

            plot_file = os.path.join(self.config['output_dirs']['reports'], 'robust_motif_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

    def step5_compare_methods(self):
        """Modified Step 5: Now includes statistical validation results"""
        print("\n" + "=" * 60)
        print("STEP 5: COMPREHENSIVE METHOD COMPARISON")
        print("=" * 60)

        # Original overlap analysis
        overlap_data = []
        methods = list(self.selected_motifs.keys())

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i <= j:
                    set1 = set(self.selected_motifs[method1])
                    set2 = set(self.selected_motifs[method2])

                    # Calculate various overlap metrics
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0
                    overlap_coef = len(set1.intersection(set2)) / min(len(set1), len(set2)) if min(len(set1),
                                                                                                   len(set2)) > 0 else 0

                    overlap_data.append({
                        'method1': method1,
                        'method2': method2,
                        'jaccard_similarity': jaccard,
                        'overlap_coefficient': overlap_coef,
                        'shared_motifs': len(set1.intersection(set2)),
                        'unique_to_method1': len(set1 - set2),
                        'unique_to_method2': len(set2 - set1)
                    })

        overlap_df = pd.DataFrame(overlap_data)

        # Add statistical validation info if available
        if hasattr(self, 'validation_results'):
            # Add significance counts
            for method in methods:
                if method in self.validation_results:
                    if 'significant_motifs' in self.validation_results[method]:
                        n_sig = len(self.validation_results[method]['significant_motifs'])
                    elif 'n_significant' in self.validation_results[method]:
                        n_sig = self.validation_results[method]['n_significant']
                    else:
                        n_sig = 0

                    overlap_df.loc[overlap_df['method1'] == method, 'method1_n_significant'] = n_sig
                    overlap_df.loc[overlap_df['method2'] == method, 'method2_n_significant'] = n_sig

        overlap_file = os.path.join(self.config['output_dirs']['reports'], 'method_overlap_detailed.csv')
        overlap_df.to_csv(overlap_file, index=False)

        # Enhanced performance comparison
        self._compare_method_performance_enhanced()

        return overlap_df

    def _compare_method_performance_enhanced(self):
        """Enhanced method performance comparison including statistical validation"""
        metrics = []

        for method, motifs in self.selected_motifs.items():
            # Basic metrics
            coverage = (self.binary_df[motifs].sum(axis=1) > 0).mean()
            avg_freq = self.binary_df[motifs].mean().mean()

            # Diversity
            if len(motifs) > 1:
                from scipy.spatial.distance import pdist
                patterns = self.binary_df[motifs].T.values
                diversity = pdist(patterns, metric='jaccard').mean()
            else:
                diversity = 0

            # Statistical validation metrics
            n_significant = 0
            avg_p_value = np.nan

            if hasattr(self, 'validation_results') and method in self.validation_results:
                val_res = self.validation_results[method]
                if 'significant_motifs' in val_res:
                    n_significant = len(val_res['significant_motifs'])
                elif 'n_significant' in val_res:
                    n_significant = val_res['n_significant']

                if 'p_adjusted' in val_res:
                    p_values = list(val_res['p_adjusted'].values())
                    avg_p_value = np.mean(p_values)

            # Robustness (how many motifs are in robust set)
            n_robust = 0
            if hasattr(self, 'robust_motifs'):
                n_robust = len(set(motifs).intersection(self.robust_motifs))

            # Evolutionary activity
            evo_activity = 0
            if hasattr(self, 'all_motif_results') and method in self.all_motif_results:
                for tax_level, results in self.all_motif_results[method].items():
                    for motif_info in results.values():
                        evo_activity += motif_info['gains'] + motif_info['losses']

            metrics.append({
                'method': method,
                'n_motifs': len(motifs),
                'n_significant': n_significant,
                'n_robust': n_robust,
                'significance_rate': n_significant / len(motifs) if len(motifs) > 0 else 0,
                'robustness_rate': n_robust / len(motifs) if len(motifs) > 0 else 0,
                'avg_p_value': avg_p_value,
                'species_coverage': coverage,
                'avg_frequency': avg_freq,
                'motif_diversity': diversity,
                'evolutionary_activity': evo_activity
            })

        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(self.config['output_dirs']['reports'], 'method_performance_enhanced.csv')
        metrics_df.to_csv(metrics_file, index=False)

        # Create enhanced visualization
        self._plot_enhanced_method_performance(metrics_df)

        print("\nEnhanced Method Performance Summary:")
        display_cols = ['method', 'n_motifs', 'n_significant', 'significance_rate',
                        'n_robust', 'species_coverage', 'evolutionary_activity']
        print(metrics_df[display_cols].to_string(index=False))

    def _plot_enhanced_method_performance(self, metrics_df):
        """Create enhanced performance visualization with statistical metrics"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()

        # 1. Number of motifs with significance
        ax = axes[0]
        x = range(len(metrics_df))
        width = 0.35
        ax.bar([i - width / 2 for i in x], metrics_df['n_motifs'], width,
               label='Total Selected', color='lightblue')
        ax.bar([i + width / 2 for i in x], metrics_df['n_significant'], width,
               label='Statistically Significant', color='darkblue')
        ax.set_ylabel('Number of Motifs')
        ax.set_title('Total vs Significant Motifs')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['method'], rotation=45)
        ax.legend()

        # 2. Significance rate
        axes[1].bar(metrics_df['method'], metrics_df['significance_rate'], color='green')
        axes[1].set_ylabel('Significance Rate')
        axes[1].set_title('Proportion of Significant Motifs')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)

        # 3. Robustness rate
        axes[2].bar(metrics_df['method'], metrics_df['robustness_rate'], color='orange')
        axes[2].set_ylabel('Robustness Rate')
        axes[2].set_title('Proportion in Robust Set')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 1)

        # 4. Species coverage
        axes[3].bar(metrics_df['method'], metrics_df['species_coverage'], color='purple')
        axes[3].set_ylabel('Species Coverage')
        axes[3].set_title('Coverage Across Species')
        axes[3].tick_params(axis='x', rotation=45)

        # 5. Motif diversity
        axes[4].bar(metrics_df['method'], metrics_df['motif_diversity'], color='brown')
        axes[4].set_ylabel('Average Pairwise Distance')
        axes[4].set_title('Motif Diversity')
        axes[4].tick_params(axis='x', rotation=45)

        # 6. Evolutionary activity
        axes[5].bar(metrics_df['method'], metrics_df['evolutionary_activity'], color='red')
        axes[5].set_ylabel('Total Gain/Loss Events')
        axes[5].set_title('Evolutionary Activity')
        axes[5].tick_params(axis='x', rotation=45)

        # 7. Combined performance score
        # Normalize metrics
        norm_df = metrics_df.copy()
        for col in ['significance_rate', 'robustness_rate', 'species_coverage',
                    'motif_diversity', 'evolutionary_activity']:
            if norm_df[col].max() > 0:
                norm_df[col] = norm_df[col] / norm_df[col].max()

        # Calculate weighted score
        norm_df['combined_score'] = (
                0.3 * norm_df['significance_rate'] +
                0.3 * norm_df['robustness_rate'] +
                0.2 * norm_df['species_coverage'] +
                0.1 * norm_df['motif_diversity'] +
                0.1 * norm_df['evolutionary_activity']
        )

        axes[6].bar(norm_df['method'], norm_df['combined_score'], color='darkgreen')
        axes[6].set_ylabel('Combined Score')
        axes[6].set_title('Overall Performance (Weighted)')
        axes[6].tick_params(axis='x', rotation=45)
        axes[6].set_ylim(0, 1)

        # 8. Radar chart for top methods
        top_methods = norm_df.nlargest(3, 'combined_score')
        categories = ['Significance', 'Robustness', 'Coverage', 'Diversity', 'Evolution']

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(2, 4, 8, projection='polar')

        for _, method_data in top_methods.iterrows():
            values = [
                method_data['significance_rate'],
                method_data['robustness_rate'],
                method_data['species_coverage'],
                method_data['motif_diversity'],
                method_data['evolutionary_activity']
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=method_data['method'])
            ax.fill(angles, values, alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Top Methods Profile')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.suptitle('Enhanced Method Performance Analysis with Statistical Validation', fontsize=16)
        plt.tight_layout()

        plot_file = os.path.join(self.config['output_dirs']['reports'],
                                 'method_performance_enhanced.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        """Modified run method with statistical validation"""
        print("STARTING COMPLETE MOTIF ANALYSIS PIPELINE WITH STATISTICAL VALIDATION")
        print("=" * 70)
        print("Pipeline includes:")
        print("- Data preparation with multi-sugar filtering")
        print("- 6 motif selection methods")
        print("- Statistical validation of selection results")
        print("- Multi-level evolutionary analysis")
        print("- Enhanced phylogenetic visualization")
        print("- Statistically-informed cross-method analysis")
        print("=" * 70)

        # Step 1: Prepare data
        self.step1_prepare_data()

        # Step 2: Select motifs (modified to store scores)
        self.step2_select_motifs()

        # Step 3: Statistical validation (NEW)
        if self.config['statistical_validation']['enabled']:
            self.step4_statistical_validation()

        # Step 4: Evolution analysis (was step 3)
        self.step3_evolution_analysis()

        # Step 5: Enhanced method comparison (was step 4)
        self.step5_compare_methods()

        # Step 6: Generate final report (was step 5)
        self.step6_generate_final_report()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All results saved to: {self.config['output_dirs']['base']}")
        print("\nKey outputs:")
        print(f"- Robust motifs: {self.config['output_dirs']['selection']}/robust_motifs.csv")
        print(f"- Statistical validation: {self.config['output_dirs']['reports']}/motif_significance_matrix.csv")
        print(f"- Integrated ranking: {self.config['output_dirs']['reports']}/integrated_motif_ranking.csv")
        print(f"- Enhanced performance: {self.config['output_dirs']['reports']}/method_performance_enhanced.csv")
        print("=" * 70)

    def step6_generate_final_report(self):
        """Enhanced final report with statistical validation results"""
        print("\n" + "=" * 60)
        print("STEP 6: GENERATING COMPREHENSIVE FINAL REPORT")
        print("=" * 60)

        report_file = os.path.join(self.config['output_dirs']['reports'],
                                   'analysis_summary_with_statistics.txt')

        with open(report_file, 'w') as f:
            f.write("MOTIF ANALYSIS PIPELINE SUMMARY WITH STATISTICAL VALIDATION\n")
            f.write("=" * 70 + "\n\n")

            # Data summary
            f.write("1. DATA OVERVIEW\n")
            f.write(f"   Input file: {self.config['input_file']}\n")
            f.write(f"   Total species: {self.results['data_summary']['n_species']}\n")
            f.write(f"   Total motifs (after filtering): {self.results['data_summary']['n_motifs']}\n")
            f.write(f"   Single sugars filtered: {self.results['data_summary']['filtered_single_sugars']}\n")
            f.write(f"   Frequency range: {self.config['filtering']['min_species_freq']}-"
                    f"{self.config['filtering']['max_species_freq']}\n\n")

            # Selection summary
            f.write("2. MOTIF SELECTION RESULTS\n")
            for method, n_motifs in self.results['selection_summary'].items():
                f.write(f"   {method}: {n_motifs} motifs")

                # Add statistical validation info
                if hasattr(self, 'validation_results') and method in self.validation_results:
                    val = self.validation_results[method]
                    if 'n_significant' in val:
                        f.write(f" ({val['n_significant']} statistically significant)")
                    elif 'significant_motifs' in val:
                        f.write(f" ({len(val['significant_motifs'])} statistically significant)")
                f.write("\n")

            # Statistical validation summary
            if hasattr(self, 'validation_results'):
                f.write("\n3. STATISTICAL VALIDATION SUMMARY\n")
                f.write(f"   Significance threshold (alpha): {self.config['statistical_validation']['alpha']}\n")
                f.write(f"   Bootstrap iterations: {self.config['statistical_validation']['n_bootstrap']}\n")
                f.write(f"   Multiple testing correction: FDR (Benjamini-Hochberg)\n")

                if 'statistical_validation' in self.results:
                    f.write(
                        f"   Robust motifs identified: {self.results['statistical_validation']['n_robust_motifs']}\n")
                    f.write(
                        f"   Average method agreement: {self.results['statistical_validation']['average_agreement']:.3f}\n")

            # Evolution summary
            f.write("\n4. EVOLUTIONARY ANALYSIS HIGHLIGHTS\n")
            if 'evolution_summary' in self.results:
                total_gains = sum(v['total_gains'] for v in self.results['evolution_summary'].values())
                total_losses = sum(v['total_losses'] for v in self.results['evolution_summary'].values())
                f.write(f"   Total gain events: {total_gains}\n")
                f.write(f"   Total loss events: {total_losses}\n")
                f.write(f"   Gain/Loss ratio: {total_gains / (total_losses + 1):.2f}\n")

            # Top robust motifs
            if hasattr(self, 'robust_motifs') and len(self.robust_motifs) > 0:
                f.write("\n5. TOP STATISTICALLY ROBUST MOTIFS\n")
                for i, motif in enumerate(self.robust_motifs[:10]):
                    f.write(f"   {i + 1}. {motif}\n")

            # Recommendations
            f.write("\n6. RECOMMENDATIONS\n")
            f.write("   - For high-confidence analysis: Use statistically robust motifs\n")
            f.write("   - For classification: Prioritize methods with high significance rates\n")
            f.write("   - For exploratory analysis: Consider the integrated ranking\n")
            f.write("   - For evolutionary studies: Focus on motifs with significant gain/loss patterns\n")
            f.write("   - Validate findings with independent datasets when possible\n")

            # Output files
            f.write("\n7. OUTPUT FILES GENERATED\n")
            f.write(f"   Base directory: {self.config['output_dirs']['base']}\n")
            f.write(f"   Key statistical outputs:\n")
            f.write(f"   - {self.config['output_dirs']['selection']}/robust_motifs.csv\n")
            f.write(f"   - {self.config['output_dirs']['reports']}/motif_significance_matrix.csv\n")
            f.write(f"   - {self.config['output_dirs']['reports']}/integrated_motif_ranking.csv\n")
            f.write(f"   - {self.config['output_dirs']['reports']}/method_performance_enhanced.csv\n")

        print(f"\nFinal report saved to: {report_file}")

        return report_file


# Example usage with statistical validation enabled
if __name__ == "__main__":
    # Configure pipeline with statistical validation
    config = MotifAnalysisPipeline.get_default_config()
    config['statistical_validation'] = {
        'enabled': True,
        'n_bootstrap': 1000,  # Increase for production use
        'alpha': 0.05,
        'min_methods_for_robust': 3,
        'require_significance': True
    }

    # Run pipeline
    pipeline = MotifAnalysisPipeline(config)
    pipeline.run()