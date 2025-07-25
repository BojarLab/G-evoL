#!/usr/bin/env python3
"""
Complete Motif Analysis Pipeline with Enhanced Phylogenetic Visualization
=========================================================================
This script integrates:
1. Data preparation with monosaccharide filtering
2. Advanced motif selection methods
3. Evolutionary gain/loss analysis
4. Enhanced phylogenetic visualizations
"""

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
from claud_script4 import MotifEvolutionAnalyzer, enrichment_by_lineage, plot_lineage_enrichment_heatmap
from claud_script5 import PhylogeneticMotifVisualizer, enhance_evolution_analysis


def is_multi_monosaccharide_(motif_name):
    """Check if motif contains multiple monosaccharides"""
    # Count sugar-like names (capitalized words) in the motif name
    sugars = re.findall(r'[A-Z][a-z]+', motif_name)
    return len(sugars) > 1


def is_multi_monosaccharide(motif_name):
    """Check if motif contains multiple monosaccharides"""
    # Define common monosaccharide abbreviations
    monosaccharides = {
        'Glc', 'Gal', 'Man', 'Fuc', 'Xyl', 'Rib', 'Ara', 'Rha',
        'GlcNAc', 'GalNAc', 'ManNAc', 'GlcA', 'GalA', 'ManA',
        'Neu5Ac', 'Neu5Gc', 'KDN', 'Kdo', 'Dha', 'Fru',
        'GlcN', 'GalN', 'ManN', 'IdoA', 'GlcUA'
    }

    # Find all capitalized words (potential sugar names)
    potential_sugars = re.findall(r'[A-Z][a-zA-Z0-9]*', motif_name)

    # Count only those that are actual monosaccharides
    actual_sugars = [sugar for sugar in potential_sugars if sugar in monosaccharides]

    return len(actual_sugars) > 1


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
                #'tax_levels': ['family', 'order', 'class'],
                'tax_levels': ["phylum", "kingdom", "superkingdom"],
                'enrichment_level': 'phylum',
                'min_samples_per_taxon': 5
            },
            'phylogenetic_viz': {
                'enabled': True,
                'create_individual_motif_trees': True,
                'max_individual_trees': 5,
                'create_integrated_analysis': True
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
        """Step 3: Analyze evolutionary gain/loss for each motif set with enhanced visualization"""
        print("\n" + "=" * 60)
        print("STEP 3: EVOLUTIONARY ANALYSIS WITH PHYLOGENETIC VISUALIZATION")
        print("=" * 60)

        analyzer = MotifEvolutionAnalyzer(self.binary_df, self.ncbi)
        evolution_results = {}
        all_motif_results = {}  # Store all results for integrated analysis

        for method_name, motifs in self.selected_motifs.items():
            print(f"\n--- Analyzing {method_name} motifs ({len(motifs)} motifs) ---")

            method_results = {}

            # Analyze at multiple taxonomic levels
            for tax_level in self.config['evolution_analysis']['tax_levels']:
                print(f"\n  At {tax_level} level:")

                try:
                    results, tree = analyzer.analyze_motif_evolution(motifs, tax_level=tax_level)

                    if results:
                        # Save raw results
                        results_df = pd.DataFrame(results).T
                        results_file = os.path.join(
                            self.config['output_dirs']['evolution'],
                            f'{method_name}_{tax_level}_evolution.csv'
                        )
                        results_df.to_csv(results_file)

                        # Save pickled results for visualization
                        pickle_file = os.path.join(
                            self.config['output_dirs']['evolution'],
                            f'{method_name}_{tax_level}_results.pkl'
                        )
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(results, f)

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

                        # Store results for method
                        method_results[tax_level] = results

                        print(f"    Total gains: {evolution_results[key]['total_gains']}")
                        print(f"    Total losses: {evolution_results[key]['total_losses']}")
                        print(f"    Motifs with changes: {evolution_results[key]['avg_changes']}")

                except Exception as e:
                    print(f"    Error in evolution analysis: {e}")
                    continue

            # Store all results for this method
            all_motif_results[method_name] = method_results

            # Create enhanced phylogenetic visualizations for this method
            if self.config['phylogenetic_viz']['enabled'] and method_results:
                print(f"\n  Creating phylogenetic visualizations for {method_name}...")
                for tax_level, results in method_results.items():
                    if results:  # Only if we have results for this tax level
                        viz_dir = os.path.join(
                            self.config['output_dirs']['phylo_viz'],
                            method_name
                        )
                        os.makedirs(viz_dir, exist_ok=True)

                        try:
                            enhance_evolution_analysis(
                                self.binary_df,
                                results,
                                tax_level=tax_level,
                                output_dir=viz_dir
                            )
                            print(f"    ✓ Created visualizations for {tax_level} level")
                        except Exception as e:
                            print(f"    Warning: Could not create visualizations for {tax_level}: {e}")

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

        # Create integrated phylogenetic analysis
        if self.config['phylogenetic_viz']['create_integrated_analysis'] and all_motif_results:
            self._create_integrated_phylo_analysis(all_motif_results)

        self.results['evolution_summary'] = evolution_results
        self.all_motif_results = all_motif_results
        return evolution_results

    def _create_integrated_phylo_analysis(self, all_motif_results):
        """Create integrated phylogenetic analysis across all methods"""
        print("\n" + "-" * 40)
        print("Creating integrated phylogenetic analysis...")

        integrated_dir = os.path.join(self.config['output_dirs']['phylo_viz'], 'integrated')
        os.makedirs(integrated_dir, exist_ok=True)

        # For each taxonomic level, combine results from all methods
        for tax_level in self.config['evolution_analysis']['tax_levels']:
            print(f"\n  Integrating results at {tax_level} level...")

            # Combine all motif results for this tax level
            combined_results = {}
            methods_with_results = []

            for method_name, method_results in all_motif_results.items():
                if tax_level in method_results and method_results[tax_level]:
                    combined_results.update(method_results[tax_level])
                    methods_with_results.append(method_name)

            if not combined_results:
                print(f"    No results available for {tax_level} level")
                continue

            print(f"    Combining results from {len(methods_with_results)} methods")
            print(f"    Total motifs analyzed: {len(combined_results)}")

            # Create comprehensive visualization
            try:
                visualizer = PhylogeneticMotifVisualizer(self.binary_df, self.ncbi)

                # 1. Create combined annotated tree
                tree_file = os.path.join(integrated_dir, f'{tax_level}_combined_tree.png')
                tree, node_events = visualizer.create_annotated_tree(
                    combined_results, tax_level, tree_file
                )

                if tree:
                    # 2. Create summary visualization
                    summary_file = os.path.join(integrated_dir, f'{tax_level}_gain_loss_summary.png')
                    all_results_dict = {'combined': combined_results}
                    fig, node_summary = visualizer.create_gain_loss_summary_tree(
                        all_results_dict,
                        tax_level=tax_level,
                        top_n=20,
                        output_file=summary_file
                    )

                    # 3. Create top motifs heatmap
                    top_motifs = sorted(combined_results.items(),
                                        key=lambda x: x[1]['gains'] + x[1]['losses'],
                                        reverse=True)[:30]
                    top_motif_names = [m[0] for m in top_motifs]

                    heatmap_file = os.path.join(integrated_dir, f'{tax_level}_top_motifs_heatmap.png')
                    visualizer.create_motif_heatmap_with_tree(
                        top_motif_names, tax_level, heatmap_file
                    )

                    # 4. Create detailed report
                    self._create_integrated_report(
                        combined_results, node_summary, tax_level,
                        methods_with_results, integrated_dir
                    )

                    print(f"    ✓ Created integrated visualizations for {tax_level}")

            except Exception as e:
                print(f"    Error creating integrated visualization: {e}")

    def _create_integrated_report(self, combined_results, node_summary, tax_level,
                                  methods_included, output_dir):
        """Create detailed integrated analysis report"""
        report_file = os.path.join(output_dir, f'{tax_level}_integrated_analysis.txt')

        with open(report_file, 'w') as f:
            f.write(f"INTEGRATED PHYLOGENETIC ANALYSIS REPORT ({tax_level} level)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Methods included: {', '.join(methods_included)}\n")
            f.write(f"Total unique motifs analyzed: {len(combined_results)}\n\n")

            # Overall statistics
            total_gains = sum(m['gains'] for m in combined_results.values())
            total_losses = sum(m['losses'] for m in combined_results.values())

            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total gain events: {total_gains}\n")
            f.write(f"Total loss events: {total_losses}\n")
            f.write(f"Gain/Loss ratio: {total_gains / (total_losses + 1):.2f}\n")
            f.write(f"Motifs with evolutionary changes: "
                    f"{sum(1 for m in combined_results.values() if m['gains'] + m['losses'] > 0)}\n\n")

            # Most dynamic motifs
            f.write("TOP 20 MOST EVOLUTIONARILY DYNAMIC MOTIFS\n")
            f.write("-" * 40 + "\n")
            dynamic_motifs = sorted(combined_results.items(),
                                    key=lambda x: x[1]['gains'] + x[1]['losses'],
                                    reverse=True)[:20]

            for motif, info in dynamic_motifs:
                total_changes = info['gains'] + info['losses']
                f.write(f"\n{motif}:\n")
                f.write(f"  Total changes: {total_changes}\n")
                f.write(f"  Gains: {info['gains']} | Losses: {info['losses']}\n")
                f.write(f"  Gain nodes: {', '.join(info['gain_nodes'][:5])}")
                if len(info['gain_nodes']) > 5:
                    f.write(f" ... ({len(info['gain_nodes']) - 5} more)")
                f.write("\n")

            # Most active clades
            f.write("\n\nTOP 15 MOST EVOLUTIONARILY ACTIVE CLADES\n")
            f.write("-" * 40 + "\n")

            sorted_clades = sorted(
                node_summary.items(),
                key=lambda x: x[1]['total_gains'] + x[1]['total_losses'],
                reverse=True
            )[:15]

            for clade, info in sorted_clades:
                if info['total_gains'] + info['total_losses'] == 0:
                    continue
                f.write(f"\n{clade}:\n")
                f.write(f"  Total events: {info['total_gains'] + info['total_losses']}\n")
                f.write(f"  Gains: {info['total_gains']} | Losses: {info['total_losses']}\n")
                f.write(f"  Unique motifs: {len(info['motifs'])}\n")
                f.write(f"  Gain/Loss ratio: {info['total_gains'] / (info['total_losses'] + 1):.2f}\n")

    def step4_compare_methods(self):
        """Step 4: Compare different selection methods"""
        print("\n" + "=" * 60)
        print("STEP 4: METHOD COMPARISON")
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

        return overlap_df

    def step5_generate_report(self):
        """Step 5: Generate comprehensive final report"""
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING FINAL REPORT")
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

            # Output structure
            f.write("\n4. OUTPUT FILES STRUCTURE\n")
            f.write(f"   Base directory: {self.config['output_dirs']['base']}\n")
            f.write(f"   ├── binary_motif_table.csv - Binary presence/absence data\n")
            f.write(f"   ├── motif_selection/\n")
            f.write(f"   │   ├── *_motifs.csv - Selected motifs by each method\n")
            f.write(f"   │   └── consensus_motifs.csv - Motifs selected by multiple methods\n")
            f.write(f"   ├── evolution_analysis/\n")
            f.write(f"   │   ├── *_evolution.csv - Gain/loss statistics\n")
            f.write(f"   │   ├── *_enrichment.csv - Lineage enrichment analysis\n")
            f.write(f"   │   └── *.png - Evolution summary plots\n")
            f.write(f"   ├── phylo_viz/\n")
            f.write(f"   │   ├── [method_name]/ - Method-specific visualizations\n")
            f.write(f"   │   │   └── [tax_level]_*.png - Phylogenetic trees and heatmaps\n")
            f.write(f"   │   └── integrated/ - Cross-method integrated analysis\n")
            f.write(f"   └── reports/\n")
            f.write(f"       ├── analysis_summary.txt - This report\n")
            f.write(f"       ├── method_overlap.csv - Method comparison data\n")
            f.write(f"       └── *.png - Comparison visualizations\n")

            # Recommendations
            f.write("\n5. RECOMMENDATIONS\n")
            f.write("   - For classification: Use Random Forest or mRMR selected motifs\n")
            f.write("   - For phylogenetic studies: Use phylogenetic signal or mutual information motifs\n")
            f.write("   - For evolutionary dynamics: Focus on motifs with high gain/loss rates\n")
            f.write("   - For comprehensive analysis: Use the consensus motif set\n")
            f.write("   - Review integrated phylogenetic visualizations in phylo_viz/integrated/\n")

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
            if hasattr(self, 'all_motif_results') and method in self.all_motif_results:
                for tax_level, results in self.all_motif_results[method].items():
                    for motif_info in results.values():
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
        print("STARTING COMPLETE MOTIF ANALYSIS PIPELINE")
        print("=" * 60)
        print("Pipeline includes:")
        print("- Data preparation with multi-sugar filtering")
        print("- 6 motif selection methods")
        print("- Multi-level evolutionary analysis")
        print("- Enhanced phylogenetic visualization")
        print("- Integrated cross-method analysis")
        print("=" * 60)

        # Step 1: Prepare data
        self.step1_prepare_data()

        # Step 2: Select motifs
        self.step2_select_motifs()

        # Step 3: Evolution analysis with visualization
        self.step3_evolution_analysis()

        # Step 4: Compare methods
        self.step4_compare_methods()

        # Step 5: Generate report
        self.step5_generate_report()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All results saved to: {self.config['output_dirs']['base']}")
        print("\nKey outputs:")
        print(f"- Binary motif table: {self.config['output_dirs']['base']}/binary_motif_table.csv")
        print(f"- Selected motifs: {self.config['output_dirs']['selection']}/*_motifs.csv")
        print(f"- Evolution results: {self.config['output_dirs']['evolution']}/")
        print(f"- Phylogenetic visualizations: {self.config['output_dirs']['phylo_viz']}/")
        print(f"- Summary report: {self.config['output_dirs']['reports']}/analysis_summary.txt")
        print("=" * 60)


if __name__ == "__main__":
    # Option 1: Run with default configuration
    pipeline = MotifAnalysisPipeline()
    pipeline.run()

    # Option 2: Run with custom configuration
     #custom_config = MotifAnalysisPipeline.get_default_config()
     #custom_config['selection_methods']['mrmr']['n_motifs'] = 50
     #custom_config['evolution_analysis']['tax_levels'] = ["phylum", "kingdom"]
     #custom_config['phylogenetic_viz']['max_individual_trees'] = 10
     #pipeline = MotifAnalysisPipeline(custom_config)
     #pipeline.run()