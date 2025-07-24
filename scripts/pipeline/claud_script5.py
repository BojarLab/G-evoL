import pandas as pd
import numpy as np
from ete3 import NCBITaxa, Tree, TreeStyle, NodeStyle, TextFace, CircleFace
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
import colorsys


class PhylogeneticMotifVisualizer:
    """Enhanced visualization of motif gain/loss on phylogenetic trees"""

    def __init__(self, binary_df, ncbi):
        self.binary_df = binary_df
        self.ncbi = ncbi
        self.color_schemes = {
            'gain': '#2ecc71',  # Green
            'loss': '#e74c3c',  # Red
            'present': '#3498db',  # Blue
            'absent': '#ecf0f1',  # Light gray
            'both': '#f39c12'  # Orange (gain and loss)
        }

    def create_annotated_tree(self, motif_results, tax_level="family", output_file=None):
        """Create phylogenetic tree annotated with gain/loss events"""

        # Build base tree
        tree, taxid_to_level, names = self._build_tree(tax_level)
        if tree is None:
            return None

        # Count events per node
        node_events = defaultdict(lambda: {'gains': [], 'losses': []})

        for motif, info in motif_results.items():
            for gain_node in info['gain_nodes']:
                node_events[gain_node]['gains'].append(motif)
            for loss_node in info['loss_nodes']:
                node_events[loss_node]['losses'].append(motif)

        # Style tree
        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.scale = 120
        ts.title.add_face(TextFace(f"Motif Gain/Loss at {tax_level} Level", fsize=20), column=0)

        # Annotate nodes
        for node in tree.traverse():
            nstyle = NodeStyle()

            if node.name in node_events:
                events = node_events[node.name]
                n_gains = len(events['gains'])
                n_losses = len(events['losses'])

                # Color based on dominant event type
                if n_gains > n_losses:
                    nstyle["bgcolor"] = self.color_schemes['gain']
                elif n_losses > n_gains:
                    nstyle["bgcolor"] = self.color_schemes['loss']
                else:
                    nstyle["bgcolor"] = self.color_schemes['both']

                # Add event count
                event_text = f"+{n_gains}/-{n_losses}"
                node.add_face(TextFace(event_text, fsize=8), column=1, position="branch-right")

                # Size based on total events
                nstyle["size"] = min(20, 5 + (n_gains + n_losses) * 2)
            else:
                nstyle["size"] = 5
                nstyle["bgcolor"] = self.color_schemes['absent']

            node.set_style(nstyle)

        # Add legend
        self._add_tree_legend(ts)

        if output_file:
            tree.render(output_file, tree_style=ts)

        return tree, node_events

    def create_motif_heatmap_with_tree(self, motifs, tax_level="family", output_file=None):
        """Create heatmap of motif presence/absence with phylogenetic tree"""

        # Filter motifs to only include those that exist in the dataframe
        available_columns = [col for col in self.binary_df.columns if col not in ['taxid', 'species_name']]
        valid_motifs = [m for m in motifs if m in available_columns]

        if not valid_motifs:
            print(f"None of the requested motifs found in dataframe. Available motifs: {available_columns[:10]}...")
            return None

        if len(valid_motifs) < len(motifs):
            print(f"Warning: Only {len(valid_motifs)} of {len(motifs)} motifs found in dataframe")
            print(f"Using: {valid_motifs}")

        # Build tree and get species grouping
        tree, taxid_to_level, names = self._build_tree(tax_level)
        if tree is None:
            return None

        # Calculate motif presence at tax_level
        tax_groups = {}
        for taxid, level_taxid in taxid_to_level.items():
            if level_taxid not in tax_groups:
                tax_groups[level_taxid] = []
            tax_groups[level_taxid].append(taxid)

        # Create presence/absence matrix
        matrix_data = []
        ordered_taxa = []

        for node in tree.iter_leaves():
            node_taxid = self.ncbi.get_name_translator([node.name])[node.name][0]
            if node_taxid in tax_groups:
                ordered_taxa.append(node.name)

                # Calculate motif frequencies for this group
                group_species = tax_groups[node_taxid]
                group_data = self.binary_df[self.binary_df['taxid'].isin(group_species)]

                if len(group_data) > 0:
                    motif_freqs = group_data[valid_motifs].mean()
                    matrix_data.append(motif_freqs.values)

        if not matrix_data:
            print("No data to plot")
            return None

        # Create figure with tree and heatmap
        fig = plt.figure(figsize=(20, max(10, len(ordered_taxa) * 0.4)))

        # Tree subplot
        ax_tree = plt.subplot2grid((1, 10), (0, 0), colspan=3)
        self._plot_dendrogram(tree, ordered_taxa, ax_tree)

        # Heatmap subplot
        ax_heatmap = plt.subplot2grid((1, 10), (0, 3), colspan=7)

        # Plot heatmap
        matrix = np.array(matrix_data)
        im = ax_heatmap.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax_heatmap.set_yticks(range(len(ordered_taxa)))
        ax_heatmap.set_yticklabels(ordered_taxa)
        ax_heatmap.set_xticks(range(len(valid_motifs)))
        ax_heatmap.set_xticklabels(valid_motifs, rotation=45, ha='right')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, pad=0.02)
        cbar.set_label('Motif Frequency', rotation=270, labelpad=20)

        plt.suptitle(f'Motif Distribution across {tax_level.capitalize()} Taxa', fontsize=16)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def create_gain_loss_summary_tree(self, all_results, tax_level="family", top_n=10, output_file=None):
        """Create tree showing summary of all gain/loss events"""

        # Aggregate all events
        node_summary = defaultdict(lambda: {'total_gains': 0, 'total_losses': 0, 'motifs': set()})

        for method_results in all_results.values():
            for motif, info in method_results.items():
                for gain_node in info['gain_nodes']:
                    node_summary[gain_node]['total_gains'] += 1
                    node_summary[gain_node]['motifs'].add(motif)
                for loss_node in info['loss_nodes']:
                    node_summary[loss_node]['total_losses'] += 1
                    node_summary[loss_node]['motifs'].add(motif)

        # Build tree
        tree, _, _ = self._build_tree(tax_level)
        if tree is None:
            return None

        # Find nodes with most changes
        nodes_by_changes = sorted(
            node_summary.items(),
            key=lambda x: x[1]['total_gains'] + x[1]['total_losses'],
            reverse=True
        )[:top_n]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

        # Left: Bar chart of top nodes
        node_names = [n[0] for n in nodes_by_changes]
        gains = [n[1]['total_gains'] for n in nodes_by_changes]
        losses = [n[1]['total_losses'] for n in nodes_by_changes]

        x = np.arange(len(node_names))
        width = 0.35

        ax1.bar(x - width / 2, gains, width, label='Gains', color=self.color_schemes['gain'])
        ax1.bar(x + width / 2, losses, width, label='Losses', color=self.color_schemes['loss'])
        ax1.set_xlabel(f'{tax_level.capitalize()} Taxa')
        ax1.set_ylabel('Number of Events')
        ax1.set_title(f'Top {top_n} Taxa by Evolutionary Changes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(node_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Right: Gain/Loss ratio
        ratios = [g / (l + 1) for g, l in zip(gains, losses)]  # +1 to avoid division by zero
        colors = [self.color_schemes['gain'] if r > 1 else self.color_schemes['loss'] for r in ratios]

        ax2.bar(x, ratios, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel(f'{tax_level.capitalize()} Taxa')
        ax2.set_ylabel('Gain/Loss Ratio')
        ax2.set_title('Evolutionary Tendency')
        ax2.set_xticks(x)
        ax2.set_xticklabels(node_names, rotation=45, ha='right')
        ax2.set_yscale('log')

        plt.suptitle(f'Summary of Motif Evolution at {tax_level} Level', fontsize=16)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        return fig, node_summary

    def create_motif_specific_tree(self, motif, motif_results, tax_level="family", output_file=None):
        """Create tree showing gain/loss for a specific motif"""

        # Get motif info
        if motif not in motif_results:
            print(f"Motif {motif} not found in results")
            return None

        motif_info = motif_results[motif]

        # Build tree
        tree, taxid_to_level, names = self._build_tree(tax_level)
        if tree is None:
            return None

        # Calculate current presence/absence
        presence_by_taxon = {}
        for level_taxid in set(taxid_to_level.values()):
            species_in_group = [k for k, v in taxid_to_level.items() if v == level_taxid]
            group_data = self.binary_df[self.binary_df['taxid'].isin(species_in_group)]
            if len(group_data) > 0 and motif in group_data.columns:
                presence_by_taxon[level_taxid] = group_data[motif].mean()

        # Style tree
        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.scale = 120
        ts.title.add_face(TextFace(f"Evolution of {motif}", fsize=20), column=0)

        # Color nodes based on events and presence
        for node in tree.traverse():
            nstyle = NodeStyle()

            # Get taxid for this node
            if node.name:
                node_taxid = self.ncbi.get_name_translator([node.name]).get(node.name, [None])[0]

                # Color based on gain/loss events
                if node.name in motif_info['gain_nodes']:
                    nstyle["bgcolor"] = self.color_schemes['gain']
                    nstyle["size"] = 15
                    node.add_face(TextFace("GAIN", fsize=10, fgcolor="white"),
                                  column=1, position="branch-right")
                elif node.name in motif_info['loss_nodes']:
                    nstyle["bgcolor"] = self.color_schemes['loss']
                    nstyle["size"] = 15
                    node.add_face(TextFace("LOSS", fsize=10, fgcolor="white"),
                                  column=1, position="branch-right")
                else:
                    # Color based on current presence
                    if node_taxid and node_taxid in presence_by_taxon:
                        freq = presence_by_taxon[node_taxid]
                        if freq > 0.5:
                            nstyle["bgcolor"] = self.color_schemes['present']
                        else:
                            nstyle["bgcolor"] = self.color_schemes['absent']
                        nstyle["size"] = 5 + freq * 10

                        # Add frequency label
                        node.add_face(TextFace(f"{freq:.2f}", fsize=8),
                                      column=2, position="branch-right")

            node.set_style(nstyle)

        if output_file:
            tree.render(output_file, tree_style=ts)

        return tree

    def _build_tree(self, tax_level):
        """Build phylogenetic tree at specified level"""
        taxids = self.binary_df['taxid'].unique()
        lineages = self.ncbi.get_lineage_translator(taxids)

        ranks = self.ncbi.get_rank(sum(lineages.values(), []))
        names = self.ncbi.get_taxid_translator(sum(lineages.values(), []))

        level_taxa = set()
        taxid_to_level = {}

        for taxid, lineage in lineages.items():
            for tid in lineage:
                if ranks.get(tid) == tax_level:
                    level_taxa.add(tid)
                    taxid_to_level[taxid] = tid
                    break

        if len(level_taxa) >= 2:
            tree = self.ncbi.get_topology(list(level_taxa))

            for node in tree.traverse():
                if node.name:
                    node.name = names.get(int(node.name), node.name)

            return tree, taxid_to_level, names

        return None, None, None

    def _plot_dendrogram(self, tree, ordered_taxa, ax):
        """Plot tree as dendrogram"""
        from scipy.cluster.hierarchy import dendrogram

        # Get pairwise distances
        n = len(ordered_taxa)
        dist_matrix = np.zeros((n, n))

        leaf_nodes = {leaf.name: leaf for leaf in tree.iter_leaves()}

        for i in range(n):
            for j in range(i + 1, n):
                if ordered_taxa[i] in leaf_nodes and ordered_taxa[j] in leaf_nodes:
                    dist = tree.get_distance(leaf_nodes[ordered_taxa[i]],
                                             leaf_nodes[ordered_taxa[j]])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist

        # Create linkage and plot
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage

        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method='average')

        dendrogram(Z, labels=ordered_taxa, ax=ax, orientation='left')
        ax.set_xlabel('Distance')
        ax.invert_xaxis()

    def _add_tree_legend(self, tree_style):
        """Add legend to tree"""
        legend_text = (
            "Node Colors:\n"
            "Green = More gains\n"
            "Red = More losses\n"
            "Orange = Equal gains/losses\n"
            "Gray = No events\n"
            "\nNode Size = Total events"
        )
        tree_style.legend.add_face(TextFace(legend_text, fsize=10), column=0)
        tree_style.legend_position = 4


def enhance_evolution_analysis(binary_df, motif_results, tax_level="family", output_dir="output/phylo_viz"):
    """Run enhanced phylogenetic visualization"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    ncbi = NCBITaxa()
    visualizer = PhylogeneticMotifVisualizer(binary_df, ncbi)

    # 1. Create annotated tree with all gain/loss events
    print("Creating annotated phylogenetic tree...")
    tree, node_events = visualizer.create_annotated_tree(
        motif_results,
        tax_level=tax_level,
        output_file=os.path.join(output_dir, f"{tax_level}_annotated_tree.png")
    )

    # 2. Create heatmap with tree for top motifs
    print("Creating motif heatmap with phylogeny...")
    top_motifs = sorted(motif_results.items(),
                        key=lambda x: x[1]['gains'] + x[1]['losses'],
                        reverse=True)[:20]
    top_motif_names = [m[0] for m in top_motifs]

    if top_motif_names:
        visualizer.create_motif_heatmap_with_tree(
            top_motif_names,
            tax_level=tax_level,
            output_file=os.path.join(output_dir, f"{tax_level}_motif_heatmap_tree.png")
        )

    # 3. Create summary visualization
    print("Creating gain/loss summary...")
    all_results = {'combined': motif_results}  # Can add multiple methods here
    fig, node_summary = visualizer.create_gain_loss_summary_tree(
        all_results,
        tax_level=tax_level,
        top_n=15,
        output_file=os.path.join(output_dir, f"{tax_level}_summary.png")
    )

    # 4. Create individual trees for most dynamic motifs
    print("Creating motif-specific trees...")
    most_dynamic = sorted(motif_results.items(),
                          key=lambda x: x[1]['gains'] + x[1]['losses'],
                          reverse=True)[:5]

    for motif, info in most_dynamic:
        print(f"  - {motif}")
        visualizer.create_motif_specific_tree(
            motif,
            motif_results,
            tax_level=tax_level,
            output_file=os.path.join(output_dir, f"{tax_level}_{motif}_tree.png")
        )

    # 5. Generate summary report
    report_file = os.path.join(output_dir, f"{tax_level}_phylo_summary.txt")
    with open(report_file, 'w') as f:
        f.write(f"PHYLOGENETIC ANALYSIS SUMMARY ({tax_level} level)\n")
        f.write("=" * 60 + "\n\n")

        f.write("TOP CLADES BY EVOLUTIONARY ACTIVITY:\n")
        for node, events in sorted(node_events.items(),
                                   key=lambda x: len(x[1]['gains']) + len(x[1]['losses']),
                                   reverse=True)[:10]:
            f.write(f"\n{node}:\n")
            f.write(f"  Gains ({len(events['gains'])}): {', '.join(events['gains'][:5])}")
            if len(events['gains']) > 5:
                f.write(f"... and {len(events['gains']) - 5} more")
            f.write(f"\n  Losses ({len(events['losses'])}): {', '.join(events['losses'][:5])}")
            if len(events['losses']) > 5:
                f.write(f"... and {len(events['losses']) - 5} more")
            f.write("\n")

        f.write("\n\nMOST EVOLUTIONARILY DYNAMIC MOTIFS:\n")
        for motif, info in most_dynamic[:10]:
            f.write(f"\n{motif}:\n")
            f.write(f"  Total changes: {info['gains'] + info['losses']}\n")
            f.write(f"  Gains in: {', '.join(info['gain_nodes'][:5])}")
            if len(info['gain_nodes']) > 5:
                f.write("...")
            f.write(f"\n  Losses in: {', '.join(info['loss_nodes'][:5])}")
            if len(info['loss_nodes']) > 5:
                f.write("...")
            f.write("\n")

    print(f"\nPhylogenetic visualizations saved to: {output_dir}")
    return node_events, node_summary


# Example integration with your pipeline
if __name__ == "__main__":
    # Load your data
    binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0)
    binary_df['taxid'] = pd.to_numeric(binary_df['taxid'], errors='coerce').dropna().astype(int)

    # Get actual motif columns from your dataframe
    motif_columns = [col for col in binary_df.columns if col not in ['taxid', 'species_name']]
    print(f"Found {len(motif_columns)} motifs in dataframe")
    print(f"First 10 motifs: {motif_columns[:10]}")

    # CREATE REAL motif_results from your actual analysis
    # This is just an example structure - you need to replace this with actual results
    # from your MotifEvolutionAnalyzer or load from a saved file

    # Option 1: Load from your actual analysis results
    # import pickle
    # with open('output/motif_evolution_results.pkl', 'rb') as f:
    #     motif_results = pickle.load(f)

    # Option 2: Create a simple example with actual motifs from your data
    # This is DUMMY data - replace with real evolution analysis results!
    example_motifs = motif_columns[:3]  # Just use first 3 motifs as example
    motif_results = {}

    # You need to replace this with actual gain/loss analysis results
    # This is just showing the expected structure
    for motif in example_motifs:
        motif_results[motif] = {
            'gains': 2,  # Replace with actual count
            'losses': 1,  # Replace with actual count
            'gain_nodes': ['Mammalia'],  # Replace with actual taxonomic nodes
            'loss_nodes': ['Reptilia']  # Replace with actual taxonomic nodes
        }

    print("\nWARNING: Using dummy motif_results. Replace with actual evolution analysis results!")
    print("Expected structure: {motif_name: {'gains': N, 'losses': N, 'gain_nodes': [...], 'loss_nodes': [...]}}")

    # Run enhanced visualization
    enhance_evolution_analysis(binary_df, motif_results, tax_level="order")