@staticmethod
def get_default_config():
    """Default pipeline configuration"""
    return {
        'input_file': 'output/motif_counts_per_species_agg_mean.csv',
        'output_dirs': {
            'base': 'output',
            'selection': 'output/motif_selection',
            'evolution': 'output/evolution_analysis',
            'reports': 'output/reports',
            'phylo_viz': 'output/phylo_viz'
        },
        'filtering': {
            'min_species_freq': 0.05,  # 5% minimum
            'max_species_freq': 0.95,  # 95% maximum
            'filter_single_sugars': True,
            'min_motif_count': 30  # For clustering
        },
        'selection_methods': {
            'mutual_info': {'enabled': True, 'n_motifs': 50},
            'mrmr': {'enabled': True, 'n_motifs': 30},
            'random_forest': {'enabled': True, 'n_motifs': 50},
            'co_occurrence': {'enabled': True, 'min_correlation': 0.3},
            'pca': {'enabled': True, 'n_components': 20},
            'phylogenetic': {'enabled': True, 'tax_level': 'family', 'n_motifs': 30}
        },
        'evolution_analysis': {
            'tax_levels': ['family', 'order', 'class'],
            'enrichment_level': 'phylum',
            'min_samples_per_taxon': 5
        },
        'phylogenetic_viz': {
            'enabled': True,
            'create_individual_motif_trees': True,
            'max_individual_trees': 5
        }
    }  # !/usr/bin/env python3


"""
Complete Motif Analysis Pipeline
================================
This script integrates:
1. Data preparation with monosaccharide filtering
2. Advanced motif selection methods
3. Evolutionary gain/loss analysis
"""

import pandas as pd
import numpy as np
import re
import os
from ete3 import NCBITaxa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import your scripts (assuming they're in the same directory)
from claud_script2 import MotifSelector
from claud_script4 import MotifEvolutionAnalyzer, enrichment_by_lineage, plot_lineage_enrichment_heatmap
from claud_script5 import PhylogeneticMotifVisualizer, enhance_evolution_analysis


def is_multi_monosaccharide(motif_name):
    """Check if motif contains multiple monosaccharides"""
    # Count sugar-like names (capitalized words) in the motif name
    sugars = re.findall(r'[A-Z][a-z]+', motif_name)
    return len(sugars) > 1


class MotifAnalysisPipeline:
    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        self.config = config or self.get_default_config()
        self.ncbi = NCBITaxa()
        self.results = {}

        # Create output directories
        for dir_path in self.config['output_dirs'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_default_config():
        """Default pipeline configuration"""
        return {
            'input_file': 'output/motif_counts_per_species_agg_mean.csv',
            'output_dirs': {
                'base': 'output',
                'selection': 'output/motif_selection',
                'evolution': 'output/evolution_analysis',
                'reports': 'output/reports'
            },
            'filtering': {
                'min_species_freq': 0.05,  # 5% minimum
                'max_species_freq': 0.95,  # 95% maximum
                'filter_single_sugars': True,
                'min_motif_count': 30  # For clustering
            },
            'selection_methods': {
                'mutual_info': {'enabled': True, 'n_motifs': 50},
                'mrmr': {'enabled': True, 'n_motifs': 30},
                'random_forest': {'enabled': True, 'n_motifs': 50},
                'co_occurrence': {'enabled': True, 'min_correlation': 0.3},
                'pca': {'enabled': True, 'n_components': 20},
                'phylogenetic': {'enabled': True, 'tax_level': 'family', 'n_motifs': 30}
            },
            'evolution_analysis': {
                'tax_levels': ['family', 'order', 'class'],
                'enrichment_level': 'phylum',
                'min_samples_per_taxon': 5
            }
        }

    def step1_prepare_data(self):
        """Step 1: Load and prepare binary motif data with filtering"""
        print("=" * 60)
        print("STEP 1: DATA PREPARATION")
        print("=" * 60)

        # Load data
        print(f"\nLoading data from {self.config['input_file']}...")
        motif_count = pd.read_csv(self.config['input_file'], header="infer", index_col=0)
        motif_count.index = motif_count.index.str.replace("_", " ").str.strip()
        print(f"Loaded {motif_count.shape[0]} species × {motif_count.shape[1]} motifs")

        # Filter 1: Remove absent motifs
        motif_count_filt = motif_count.loc[:, (motif_count != 0).any(axis=0)]
        print(f"\nAfter removing absent motifs: {motif_count_filt.shape[1]} motifs")

        # Filter 2: Remove rare/common motifs
        motif_freq = (motif_count_filt > 0).mean(axis=0)
        freq_range = (motif_freq > self.config['filtering']['min_species_freq']) & \
                     (motif_freq < self.config['filtering']['max_species_freq'])
        motif_count_filt = motif_count_filt.loc[:, freq_range]
        print(f"After frequency filtering ({self.config['filtering']['min_species_freq']}-"
              f"{self.config['filtering']['max_species_freq']}): {motif_count_filt.shape[1]} motifs")

        # Filter 3: Multi-monosaccharide filtering
        if self.config['filtering']['filter_single_sugars']:
            multi_sugar_motifs = [m for m in motif_count_filt.columns if is_multi_monosaccharide(m)]
            motif_count_filt = motif_count_filt[multi_sugar_motifs]
            print(f"After filtering single sugars: {motif_count_filt.shape[1]} motifs")

        # Convert to binary
        binary_df = (motif_count_filt > 0).astype(int)

        # Add taxonomy
        print("\nAdding taxonomy information...")
        sp_list = binary_df.index.tolist()
        name_to_taxid = {}
        missing = []

        for sp in sp_list:
            name2taxid = self.ncbi.get_name_translator([sp])
            if sp in name2taxid:
                name_to_taxid[sp] = name2taxid[sp][0]
            else:
                missing.append(sp)

        print(f"Successfully mapped {len(name_to_taxid)}/{len(sp_list)} species")
        if missing:
            print(f"Missing species ({len(missing)}): {missing[:5]}..." if len(missing) > 5 else missing)

        binary_df["taxid"] = binary_df.index.map(name_to_taxid)
        binary_df["taxid"] = binary_df["taxid"].astype("Int64")
        binary_df = binary_df.dropna(subset=["taxid"])

        # Save
        output_file = os.path.join(self.config['output_dirs']['base'], 'binary_motif_table.csv')
        binary_df.to_csv(output_file)
        print(f"\nSaved binary table: {binary_df.shape[0]} species × {binary_df.shape[1] - 1} motifs")
        print(f"Location: {output_file}")

        self.binary_df = binary_df
        self.results['data_summary'] = {
            'n_species': binary_df.shape[0],
            'n_motifs': binary_df.shape[1] - 1,
            'filtered_single_sugars': self.config['filtering']['filter_single_sugars']
        }

        return binary_df

    def step2_select_motifs(self):
        """Step 2: Run multiple motif selection methods"""
        print("\n" + "=" * 60)
        print("STEP 2: MOTIF SELECTION")
        print("=" * 60)

        selector = MotifSelector(self.binary_df, taxonomy_col='taxid')
        selected_motifs = {}

        # Run each enabled method
        methods_config = self.config['selection_methods']

        if methods_config['mutual_info']['enabled']:
            print("\n1. Mutual Information Selection...")
            mi_motifs, mi_scores = selector.method2_mutual_information()
            n = methods_config['mutual_info']['n_motifs']
            selected_motifs['mutual_info'] = mi_motifs[:n]
            self._save_motifs(mi_motifs[:n], 'mutual_info_motifs.csv')
            print(f"   Selected {n} motifs")

        if methods_config['mrmr']['enabled']:
            print("\n2. mRMR Selection...")
            n = methods_config['mrmr']['n_motifs']
            mrmr_motifs, _ = selector.method3_maximum_relevance_minimum_redundancy(n_motifs=n)
            selected_motifs['mrmr'] = mrmr_motifs
            self._save_motifs(mrmr_motifs, 'mrmr_motifs.csv')
            print(f"   Selected {len(mrmr_motifs)} motifs")

        if methods_config['random_forest']['enabled']:
            print("\n3. Random Forest Selection...")
            rf_motifs, rf_scores = selector.method4_random_forest_importance()
            n = methods_config['random_forest']['n_motifs']
            selected_motifs['random_forest'] = rf_motifs[:n]
            self._save_motifs(rf_motifs[:n], 'random_forest_motifs.csv')
            print(f"   Selected {n} motifs")

        if methods_config['co_occurrence']['enabled']:
            print("\n4. Co-occurrence Module Selection...")
            min_corr = methods_config['co_occurrence']['min_correlation']
            module_reps, modules = selector.method5_co_occurrence_modules(min_correlation=min_corr)
            selected_motifs['co_occurrence'] = module_reps
            self._save_motifs(module_reps, 'co_occurrence_motifs.csv')
            print(f"   Selected {len(module_reps)} module representatives")

        if methods_config['pca']['enabled']:
            print("\n5. PCA-based Selection...")
            n = methods_config['pca']['n_components']
            pca_motifs, var_explained = selector.method6_pca_loadings(n_components=n)
            selected_motifs['pca'] = pca_motifs
            self._save_motifs(pca_motifs, 'pca_motifs.csv')
            print(f"   Selected {len(pca_motifs)} motifs")
            print(f"   Variance explained: {var_explained.sum():.2%}")

        if methods_config['phylogenetic']['enabled']:
            print("\n6. Phylogenetic Signal Selection...")
            tax_level = methods_config['phylogenetic']['tax_level']
            n = methods_config['phylogenetic']['n_motifs']
            phylo_motifs, phylo_scores = selector.method1_phylogenetic_signal(self.ncbi, tax_level)
            selected_motifs['phylogenetic'] = phylo_motifs[:n]
            self._save_motifs(phylo_motifs[:n], 'phylogenetic_motifs.csv')
            print(f"   Selected {n} motifs with strong {tax_level}-level signal")

        self.selected_motifs = selected_motifs
        self.results['selection_summary'] = {
            method: len(motifs) for method, motifs in selected_motifs.items()
        }

        return selected_motifs

    def step3_evolution_analysis(self):
        """Step 3: Analyze evolutionary gain/loss for each motif set"""
        print("\n" + "=" * 60)
        print("STEP 3: EVOLUTIONARY ANALYSIS")
        print("=" * 60)

        analyzer = MotifEvolutionAnalyzer(self.binary_df, self.ncbi)
        evolution_results = {}
        motif_results_by_level = {}  # Store for phylogenetic visualization

        for method_name, motifs in self.selected_motifs.items():
            print(f"\n--- Analyzing {method_name} motifs ({len(motifs)} motifs) ---")

            # Analyze at multiple taxonomic levels
            for tax_level in self.config['evolution_analysis']['tax_levels']:
                print(f"\n  At {tax_level} level:")

                try:
                    results, tree = analyzer.analyze_motif_evolution(motifs, tax_level=tax_level)

                    if results:
                        # Save results
                        results_df = pd.DataFrame(results).T
                        results_file = os.path.join(
                            self.config['output_dirs']['evolution'],
                            f'{method_name}_{tax_level}_evolution.csv'
                        )
                        results_df.to_csv(results_file)

                        # Plot summary
                        plot_file = os.path.join(
                            self.config['output_dirs']['evolution'],
                            f'{method_name}_{tax_level}_evolution_summary.png'
                        )
                        summary_df = analyzer.plot_evolution_summary(
                            results, top_n=20, save_path=plot_file
                        )

                        # Store metrics
                        key = f"{method_name}_{tax_level}"
                        evolution_results[key] = {
                            'total_gains': results_df['gains'].sum(),
                            'total_losses': results_df['losses'].sum(),
                            'avg_changes': results_df[results_df['gains'] + results_df['losses'] > 0].shape[0]
                        }

                        # Store for phylogenetic visualization
                        if tax_level not in motif_results_by_level:
                            motif_results_by_level[tax_level] = {}
                        motif_results_by_level[tax_level][method_name] = results

                        print(f"    Total gains: {evolution_results[key]['total_gains']}")
                        print(f"    Total losses: {evolution_results[key]['total_losses']}")
                        print(f"    Motifs with changes: {evolution_results[key]['avg_changes']}")

                        # Create phylogenetic visualizations
                        if self.config.get('phylogenetic_viz', {}).get('enabled', True):
                            self._create_phylogenetic_visualizations(
                                results, method_name, tax_level
                            )

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            # Lineage enrichment
            print(f"\n  Lineage enrichment at {self.config['evolution_analysis']['enrichment_level']} level:")
            enrichment_df = enrichment_by_lineage(
                self.binary_df, motifs,
                tax_level=self.config['evolution_analysis']['enrichment_level']
            )

            if not enrichment_df.empty:
                # Save enrichment
                enrichment_file = os.path.join(
                    self.config['output_dirs']['evolution'],
                    f'{method_name}_lineage_enrichment.csv'
                )
                enrichment_df.to_csv(enrichment_file)

                # Plot heatmap
                heatmap_file = os.path.join(
                    self.config['output_dirs']['evolution'],
                    f'{method_name}_enrichment_heatmap.png'
                )
                plot_lineage_enrichment_heatmap(
                    enrichment_df,
                    min_samples=self.config['evolution_analysis']['min_samples_per_taxon'],
                    save_path=heatmap_file
                )

                # Top enrichments
                top_enriched = enrichment_df.nlargest(5, 'enrichment_score')
                print(f"    Top enriched lineages:")
                for _, row in top_enriched.iterrows():
                    print(f"      {row['motif']} in {row['lineage']}: "
                          f"{row['enrichment_score']:.2f}x enriched")

        self.results['evolution_summary'] = evolution_results
        self.motif_results_by_level = motif_results_by_level
        return evolution_results

    def _create_phylogenetic_visualizations(self, results, method_name, tax_level):
        """Create phylogenetic visualizations for gain/loss events"""
        print(f"    Creating phylogenetic visualizations...")

        visualizer = PhylogeneticMotifVisualizer(self.binary_df, self.ncbi)
        viz_dir = os.path.join(self.config['output_dirs']['phylo_viz'], method_name, tax_level)
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Create annotated tree
        tree_file = os.path.join(viz_dir, 'annotated_tree.png')
        tree, node_events = visualizer.create_annotated_tree(results, tax_level, tree_file)

        if tree:
            # 2. Create heatmap with tree for top motifs
            top_motifs = sorted(results.items(),
                                key=lambda x: x[1]['gains'] + x[1]['losses'],
                                reverse=True)[:20]
            top_motif_names = [m[0] for m in top_motifs]

            heatmap_file = os.path.join(viz_dir, 'motif_heatmap_tree.png')
            visualizer.create_motif_heatmap_with_tree(top_motif_names, tax_level, heatmap_file)

            # 3. Create individual motif trees
            if self.config['phylogenetic_viz']['create_individual_motif_trees']:
                max_trees = self.config['phylogenetic_viz']['max_individual_trees']
                for i, (motif, info) in enumerate(top_motifs[:max_trees]):
                    motif_file = os.path.join(viz_dir, f'{motif}_tree.png')
                    visualizer.create_motif_specific_tree(motif, results, tax_level, motif_file)

            print(f"    ✓ Phylogenetic visualizations saved to {viz_dir}")

    def step4_compare_methods(self):
        """Step 4: Compare different selection methods and create integrated visualizations"""
        print("\n" + "=" * 60)
        print("STEP 4: METHOD COMPARISON & INTEGRATED ANALYSIS")
        print("=" * 60)

        # Calculate overlap between methods
        print("\nCalculating method overlap...")
        overlap_data = []
        methods = list(self.selected_motifs.keys())

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i <= j:
                    set1 = set(self.selected_motifs[method1])
                    set2 = set(self.selected_motifs[method2])
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0
                    overlap_data.append({
                        'method1': method1,
                        'method2': method2,
                        'jaccard_similarity': jaccard,
                        'shared_motifs': len(set1.intersection(set2))
                    })

        overlap_df = pd.DataFrame(overlap_data)
        overlap_file = os.path.join(self.config['output_dirs']['reports'], 'method_overlap.csv')
        overlap_df.to_csv(overlap_file, index=False)

        # Plot overlap heatmap
        self._plot_method_overlap(overlap_df, methods)

        # Compare performance
        self._compare_method_performance()

        # Create integrated phylogenetic visualizations
        if self.config['phylogenetic_viz']['enabled'] and hasattr(self, 'motif_results_by_level'):
            self._create_integrated_phylo_viz()

        return overlap_df

    def step5_generate_report(self):
        """Step 5: Generate final summary report"""
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING REPORT")
        print("=" * 60)

        report_file = os.path.join(self.config['output_dirs']['reports'], 'analysis_summary.txt')

        with open(report_file, 'w') as f:
            f.write("MOTIF ANALYSIS PIPELINE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

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
                f.write(f"   {method}: {n_motifs} motifs\n")

            # Evolution summary
            f.write("\n3. EVOLUTIONARY ANALYSIS HIGHLIGHTS\n")
            if 'evolution_summary' in self.results:
                total_gains = sum(v['total_gains'] for v in self.results['evolution_summary'].values())
                total_losses = sum(v['total_losses'] for v in self.results['evolution_summary'].values())
                f.write(f"   Total gain events: {total_gains}\n")
                f.write(f"   Total loss events: {total_losses}\n")
                f.write(f"   Gain/Loss ratio: {total_gains / (total_losses + 1):.2f}\n")

            # Recommendations
            f.write("\n4. RECOMMENDATIONS\n")
            f.write("   - For classification: Use Random Forest or mRMR selected motifs\n")
            f.write("   - For phylogenetic studies: Use phylogenetic signal or mutual information motifs\n")
            f.write("   - For evolutionary dynamics: Focus on motifs with high gain/loss rates\n")
            f.write("   - For comprehensive analysis: Use the combined/consensus motif set\n")

            # Output files
            f.write("\n5. OUTPUT FILES GENERATED\n")
            f.write(f"   Base directory: {self.config['output_dirs']['base']}\n")
            f.write(f"   - Binary motif table: binary_motif_table.csv\n")
            f.write(f"   - Selected motifs: {self.config['output_dirs']['selection']}/*_motifs.csv\n")
            f.write(f"   - Evolution results: {self.config['output_dirs']['evolution']}/*\n")
            f.write(f"   - Reports: {self.config['output_dirs']['reports']}/*\n")

        print(f"\nReport saved to: {report_file}")

        # Create consensus motif set
        self._create_consensus_motifs()

        return report_file

    def _save_motifs(self, motifs, filename):
        """Save motif list to file"""
        filepath = os.path.join(self.config['output_dirs']['selection'], filename)
        pd.Series(motifs).to_csv(filepath, header=False, index=False)

    def _plot_method_overlap(self, overlap_df, methods):
        """Plot method overlap heatmap"""
        # Create matrix
        n = len(methods)
        matrix = np.eye(n)

        for _, row in overlap_df.iterrows():
            i = methods.index(row['method1'])
            j = methods.index(row['method2'])
            matrix[i, j] = matrix[j, i] = row['jaccard_similarity']

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=methods, yticklabels=methods,
                    annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
        plt.title("Jaccard Similarity Between Selection Methods")
        plt.tight_layout()

        plot_file = os.path.join(self.config['output_dirs']['reports'], 'method_overlap_heatmap.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_integrated_phylo_viz(self):
        """Create integrated phylogenetic visualizations across all methods"""
        print("\nCreating integrated phylogenetic visualizations...")

        visualizer = PhylogeneticMotifVisualizer(self.binary_df, self.ncbi)

        for tax_level in self.config['evolution_analysis']['tax_levels']:
            if tax_level not in self.motif_results_by_level:
                continue

            # Combine results from all methods
            all_motif_results = {}
            for method_name, results in self.motif_results_by_level[tax_level].items():
                all_motif_results.update(results)

            # Create summary visualization
            summary_dir = os.path.join(self.config['output_dirs']['phylo_viz'], 'integrated')
            os.makedirs(summary_dir, exist_ok=True)

            summary_file = os.path.join(summary_dir, f'{tax_level}_gain_loss_summary.png')
            fig, node_summary = visualizer.create_gain_loss_summary_tree(
                self.motif_results_by_level[tax_level],
                tax_level=tax_level,
                top_n=15,
                output_file=summary_file
            )

            # Create clade-specific report
            self._create_clade_report(node_summary, tax_level, summary_dir)

        print("✓ Integrated phylogenetic analysis complete")

    def _create_clade_report(self, node_summary, tax_level, output_dir):
        """Create detailed report of gain/loss events by clade"""
        report_file = os.path.join(output_dir, f'{tax_level}_clade_evolution_report.txt')

        with open(report_file, 'w') as f:
            f.write(f"CLADE-SPECIFIC MOTIF EVOLUTION REPORT ({tax_level} level)\n")
            f.write("=" * 80 + "\n\n")

            # Sort clades by total evolutionary activity
            sorted_clades = sorted(
                node_summary.items(),
                key=lambda x: x[1]['total_gains'] + x[1]['total_losses'],
                reverse=True
            )

            for clade, info in sorted_clades[:20]:  # Top 20 most active clades
                total_events = info['total_gains'] + info['total_losses']
                if total_events == 0:
                    continue

                f.write(f"\n{clade.upper()}\n")
                f.write("-" * len(clade) + "\n")
                f.write(f"Total evolutionary events: {total_events}\n")
                f.write(f"  - Gains: {info['total_gains']}\n")
                f.write(f"  - Losses: {info['total_losses']}\n")
                f.write(f"  - Gain/Loss ratio: {info['total_gains'] / (info['total_losses'] + 1):.2f}\n")
                f.write(f"  - Unique motifs involved: {len(info['motifs'])}\n")

                # List some example motifs
                motif_list = list(info['motifs'])[:10]
                f.write(f"  - Example motifs: {', '.join(motif_list)}")
                if len(info['motifs']) > 10:
                    f.write(f" ... and {len(info['motifs']) - 10} more")
                f.write("\n")

            # Summary statistics
            f.write("\n\nSUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            total_clades = len([c for c, i in node_summary.items()
                                if i['total_gains'] + i['total_losses'] > 0])
            f.write(f"Total clades with evolutionary events: {total_clades}\n")
            f.write(f"Average events per active clade: "
                    f"{sum(i['total_gains'] + i['total_losses'] for i in node_summary.values()) / max(total_clades, 1):.1f}\n")

            # Identify expansion vs contraction clades
            expanding = [c for c, i in node_summary.items()
                         if i['total_gains'] > i['total_losses']]
            contracting = [c for c, i in node_summary.items()
                           if i['total_losses'] > i['total_gains']]

            f.write(f"\nClades with net motif expansion: {len(expanding)}\n")
            f.write(f"  Examples: {', '.join(expanding[:5])}\n")
            f.write(f"\nClades with net motif contraction: {len(contracting)}\n")
            f.write(f"  Examples: {', '.join(contracting[:5])}\n")

    def _compare_method_performance(self):
        """Compare method performance metrics"""
        metrics = []

        for method, motifs in self.selected_motifs.items():
            # Coverage
            coverage = (self.binary_df[motifs].sum(axis=1) > 0).mean()

            # Average frequency
            avg_freq = self.binary_df[motifs].mean().mean()

            # Diversity (average pairwise distance)
            if len(motifs) > 1:
                from scipy.spatial.distance import pdist
                patterns = self.binary_df[motifs].T.values
                diversity = pdist(patterns, metric='jaccard').mean()
            else:
                diversity = 0

            # Evolutionary activity (if available)
            evo_activity = 0
            if hasattr(self, 'motif_results_by_level'):
                for tax_level, level_results in self.motif_results_by_level.items():
                    if method in level_results:
                        for motif_info in level_results[method].values():
                            evo_activity += motif_info['gains'] + motif_info['losses']

            metrics.append({
                'method': method,
                'n_motifs': len(motifs),
                'species_coverage': coverage,
                'avg_frequency': avg_freq,
                'motif_diversity': diversity,
                'evolutionary_activity': evo_activity
            })

        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(self.config['output_dirs']['reports'], 'method_performance.csv')
        metrics_df.to_csv(metrics_file, index=False)

        # Create performance visualization
        self._plot_method_performance(metrics_df)

        print("\nMethod Performance Summary:")
        print(metrics_df.to_string(index=False))

    def _plot_method_performance(self, metrics_df):
        """Create comprehensive performance visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Coverage
        axes[0, 0].bar(metrics_df['method'], metrics_df['species_coverage'])
        axes[0, 0].set_ylabel('Species Coverage')
        axes[0, 0].set_title('Coverage by Selection Method')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Average frequency
        axes[0, 1].bar(metrics_df['method'], metrics_df['avg_frequency'])
        axes[0, 1].set_ylabel('Average Motif Frequency')
        axes[0, 1].set_title('Motif Frequency by Method')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Diversity
        axes[0, 2].bar(metrics_df['method'], metrics_df['motif_diversity'])
        axes[0, 2].set_ylabel('Average Pairwise Distance')
        axes[0, 2].set_title('Motif Diversity by Method')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 4. Number of motifs
        axes[1, 0].bar(metrics_df['method'], metrics_df['n_motifs'])
        axes[1, 0].set_ylabel('Number of Motifs')
        axes[1, 0].set_title('Motifs Selected by Method')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. Evolutionary activity
        axes[1, 1].bar(metrics_df['method'], metrics_df['evolutionary_activity'])
        axes[1, 1].set_ylabel('Total Gain/Loss Events')
        axes[1, 1].set_title('Evolutionary Activity by Method')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 6. Combined score (normalized)
        # Normalize each metric to 0-1
        norm_df = metrics_df.copy()
        for col in ['species_coverage', 'avg_frequency', 'motif_diversity', 'evolutionary_activity']:
            if norm_df[col].max() > 0:
                norm_df[col] = norm_df[col] / norm_df[col].max()

        # Combined score
        norm_df['combined_score'] = norm_df[['species_coverage', 'motif_diversity', 'evolutionary_activity']].mean(
            axis=1)

        axes[1, 2].bar(norm_df['method'], norm_df['combined_score'])
        axes[1, 2].set_ylabel('Combined Score')
        axes[1, 2].set_title('Overall Performance Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].set_ylim(0, 1)

        plt.suptitle('Comprehensive Method Performance Comparison', fontsize=16)
        plt.tight_layout()

        plot_file = os.path.join(self.config['output_dirs']['reports'], 'method_performance_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_consensus_motifs(self):
        """Create consensus motif set based on frequency across methods"""
        # Count how many methods selected each motif
        all_motifs = []
        for motifs in self.selected_motifs.values():
            all_motifs.extend(motifs)

        motif_counts = pd.Series(all_motifs).value_counts()

        # Select motifs appearing in multiple methods
        consensus_motifs = motif_counts[motif_counts >= 2].index.tolist()

        consensus_file = os.path.join(self.config['output_dirs']['selection'], 'consensus_motifs.csv')
        pd.Series(consensus_motifs).to_csv(consensus_file, header=False, index=False)

        print(f"\nConsensus motifs (appearing in ≥2 methods): {len(consensus_motifs)} motifs")
        print(f"Top consensus motifs: {consensus_motifs[:5]}")

    def run(self):
        """Run complete pipeline"""
        print("STARTING MOTIF ANALYSIS PIPELINE")
        print("=" * 60)

        # Step 1: Prepare data
        self.step1_prepare_data()

        # Step 2: Select motifs
        self.step2_select_motifs()

        # Step 3: Evolution analysis
        self.step3_evolution_analysis()

        # Step 4: Compare methods
        self.step4_compare_methods()

        # Step 5: Generate report
        self.step5_generate_report()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All results saved to: {self.config['output_dirs']['base']}")
        print("=" * 60)


if __name__ == "__main__":
    # Option 1: Run with default configuration
    pipeline = MotifAnalysisPipeline()
    pipeline.run()

    # Option 2: Run with custom configuration
    # custom_config = MotifAnalysisPipeline.get_default_config()
    # custom_config['selection_methods']['mrmr']['n_motifs'] = 50
    # custom_config['evolution_analysis']['tax_levels'] = ['genus', 'family', 'order']
    # pipeline = MotifAnalysisPipeline(custom_config)
    # pipeline.run()