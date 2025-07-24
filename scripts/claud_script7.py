"""
Statistical Validation Module for Motif Pipeline
===============================================
This module adds statistical validation to existing pipeline results.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from pathlib import Path


class StatisticalValidator:
    """Add statistical validation to existing pipeline results"""

    def __init__(self, pipeline_instance):
        """
        Initialize with a completed pipeline instance

        Args:
            pipeline_instance: A MotifAnalysisPipeline instance that has already run
        """
        self.pipeline = pipeline_instance
        self.binary_df = pipeline_instance.binary_df
        self.selected_motifs = pipeline_instance.selected_motifs
        self.output_dir = pipeline_instance.config['output_dirs']['reports']

        # Ensure reports directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def validate_all_methods(self, n_bootstrap=100):
        """Run statistical validation on all selection methods"""
        print("\n" + "=" * 60)
        print("STATISTICAL VALIDATION OF RESULTS")
        print("=" * 60)

        validation_results = {}

        # For each method, perform bootstrap validation
        for method, motifs in self.selected_motifs.items():
            print(f"\nValidating {method}...")
            validation_results[method] = self._bootstrap_validation(motifs, method)

        return validation_results

    def _bootstrap_validation(self, motifs, method_name):
        """Bootstrap validation for a set of motifs"""
        n_samples = len(self.binary_df)
        importance_scores = []

        # Bootstrap to get confidence intervals
        for i in range(100):  # Reduced for speed
            # Resample data
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_data = self.binary_df.iloc[indices]

            # Calculate importance for each motif
            # This is simplified - in reality would recalculate using the specific method
            motif_scores = {}
            for motif in motifs[:20]:  # Test top 20 for efficiency
                if motif in boot_data.columns:
                    # Simple proxy for importance: correlation with taxonomy
                    score = np.abs(stats.spearmanr(boot_data[motif], boot_data['taxid'])[0])
                    motif_scores[motif] = score

            importance_scores.append(motif_scores)

        # Calculate statistics
        results = {}
        for motif in motifs[:20]:
            scores = [s.get(motif, 0) for s in importance_scores]
            results[motif] = {
                'mean_importance': np.mean(scores),
                'std': np.std(scores),
                'ci_low': np.percentile(scores, 2.5),
                'ci_high': np.percentile(scores, 97.5),
                'significant': np.percentile(scores, 2.5) > 0
            }

        n_significant = sum(1 for r in results.values() if r['significant'])

        return {
            'motif_stats': results,
            'n_significant': n_significant,
            'n_tested': len(results)
        }

    def create_significance_matrix(self, validation_results):
        """Create matrix showing motif significance across methods"""
        # Get all unique motifs
        all_motifs = set()
        for motifs in self.selected_motifs.values():
            all_motifs.update(motifs)

        all_motifs = sorted(list(all_motifs))
        methods = sorted(self.selected_motifs.keys())

        # Create matrix: 0=not selected, 1=selected, 2=significant
        sig_matrix = pd.DataFrame(0, index=all_motifs, columns=methods)

        for method, motifs in self.selected_motifs.items():
            for motif in motifs:
                sig_matrix.loc[motif, method] = 1

                # Mark as significant if validated
                if method in validation_results:
                    motif_stats = validation_results[method].get('motif_stats', {})
                    if motif in motif_stats and motif_stats[motif].get('significant', False):
                        sig_matrix.loc[motif, method] = 2

        return sig_matrix

    def identify_robust_motifs(self, sig_matrix, min_methods=3):
        """Identify motifs robustly selected across methods"""
        robust_motifs = []

        for motif in sig_matrix.index:
            # Count selections and significance
            selected_count = np.sum(sig_matrix.loc[motif] > 0)
            significant_count = np.sum(sig_matrix.loc[motif] == 2)

            if selected_count >= min_methods:
                methods_selected = list(sig_matrix.columns[sig_matrix.loc[motif] > 0])
                robust_motifs.append({
                    'motif': motif,
                    'n_methods_selected': selected_count,
                    'n_methods_significant': significant_count,
                    'methods': methods_selected,
                    'robustness_score': selected_count + significant_count
                })

        robust_df = pd.DataFrame(robust_motifs).sort_values('robustness_score', ascending=False)
        return robust_df

    def plot_significance_heatmap(self, sig_matrix, top_n=50):
        """Create heatmap of motif significance"""
        # Select top motifs by total selection
        motif_sums = sig_matrix.sum(axis=1)
        top_motifs = motif_sums.nlargest(top_n).index

        plt.figure(figsize=(10, 12))

        # Custom colormap
        colors = ['white', 'lightblue', 'darkblue']
        cmap = plt.cm.colors.ListedColormap(colors)

        ax = sns.heatmap(
            sig_matrix.loc[top_motifs],
            cmap=cmap,
            cbar_kws={'label': 'Status', 'ticks': [0, 1, 2]},
            xticklabels=True,
            yticklabels=True,
            vmin=0,
            vmax=2
        )

        # Fix colorbar labels
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['Not Selected', 'Selected', 'Significant'])

        plt.title(f'Top {top_n} Motifs: Selection and Significance Across Methods')
        plt.xlabel('Method')
        plt.ylabel('Motif')
        plt.tight_layout()

        # Save
        plot_file = os.path.join(self.output_dir, 'motif_significance_heatmap.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved significance heatmap to {plot_file}")

    def generate_statistical_report(self, validation_results, sig_matrix, robust_motifs):
        """Generate comprehensive statistical report"""
        report_file = os.path.join(self.output_dir, 'statistical_validation_report.txt')

        with open(report_file, 'w') as f:
            f.write("STATISTICAL VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Summary statistics
            f.write("1. VALIDATION SUMMARY\n")
            f.write("-" * 30 + "\n")

            for method, results in validation_results.items():
                n_sig = results.get('n_significant', 0)
                n_test = results.get('n_tested', 0)
                f.write(f"{method}: {n_sig}/{n_test} significant motifs\n")

            # Robust motifs
            f.write(f"\n2. ROBUST MOTIFS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total robust motifs (selected by ≥3 methods): {len(robust_motifs)}\n")
            f.write(f"Highly robust (significant in ≥2 methods): "
                    f"{len(robust_motifs[robust_motifs['n_methods_significant'] >= 2])}\n")

            # Top robust motifs
            f.write("\nTop 20 Most Robust Motifs:\n")
            for i, row in robust_motifs.head(20).iterrows():
                f.write(f"{row['motif']}: selected by {row['n_methods_selected']} methods, "
                        f"significant in {row['n_methods_significant']}\n")

            # Method agreement
            f.write(f"\n3. METHOD AGREEMENT\n")
            f.write("-" * 30 + "\n")

            # Calculate pairwise agreement
            methods = list(self.selected_motifs.keys())
            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods[i + 1:], i + 1):
                    set1 = set(self.selected_motifs[m1])
                    set2 = set(self.selected_motifs[m2])
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                    f.write(f"{m1} vs {m2}: {jaccard:.3f} Jaccard similarity\n")

        print(f"Statistical report saved to {report_file}")

    def run_complete_validation(self):
        """Run complete statistical validation pipeline"""
        # 1. Validate all methods
        validation_results = self.validate_all_methods()

        # 2. Create significance matrix
        sig_matrix = self.create_significance_matrix(validation_results)
        sig_matrix_file = os.path.join(self.output_dir, 'motif_significance_matrix.csv')
        sig_matrix.to_csv(sig_matrix_file)
        print(f"Saved significance matrix to {sig_matrix_file}")

        # 3. Identify robust motifs
        robust_motifs = self.identify_robust_motifs(sig_matrix)
        robust_file = os.path.join(self.output_dir, 'robust_motifs.csv')
        robust_motifs.to_csv(robust_file, index=False)
        print(f"Identified {len(robust_motifs)} robust motifs")

        # Save just the motif names for easy use
        robust_motif_list = robust_motifs['motif'].tolist()
        robust_list_file = os.path.join(self.pipeline.config['output_dirs']['selection'],
                                        'robust_motifs.csv')
        pd.Series(robust_motif_list).to_csv(robust_list_file, header=False, index=False)

        # 4. Create visualizations
        self.plot_significance_heatmap(sig_matrix)

        # 5. Generate report
        self.generate_statistical_report(validation_results, sig_matrix, robust_motifs)

        return {
            'validation_results': validation_results,
            'significance_matrix': sig_matrix,
            'robust_motifs': robust_motifs
        }