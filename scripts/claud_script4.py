import pandas as pd
import numpy as np
from ete3 import NCBITaxa, Tree
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class MotifEvolutionAnalyzer:
    def __init__(self, binary_df, ncbi):
        self.binary_df = binary_df
        self.ncbi = ncbi

    def build_phylogenetic_tree(self, tax_level="family"):
        """Build phylogenetic tree at specified taxonomic level"""
        # Get lineages for all species
        taxids = self.binary_df['taxid'].unique()
        lineages = self.ncbi.get_lineage_translator(taxids)

        # Extract taxa at specified level
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

        # Build tree
        if len(level_taxa) >= 2:
            tree = self.ncbi.get_topology(list(level_taxa))

            # Rename nodes with actual names
            for node in tree.traverse():
                if node.name:
                    node.name = names.get(int(node.name), node.name)

            return tree, taxid_to_level, names
        return None, None, None

    def ancestral_state_reconstruction(self, tree, motif, taxid_to_level):
        """Simple parsimony-based ancestral state reconstruction"""
        # Calculate motif presence for each taxonomic group
        motif_states = {}

        for level_taxid in set(taxid_to_level.values()):
            species_in_group = [k for k, v in taxid_to_level.items() if v == level_taxid]
            group_data = self.binary_df[self.binary_df['taxid'].isin(species_in_group)]

            if len(group_data) > 0:
                # Use majority rule: >50% presence = 1, else 0
                motif_states[level_taxid] = int(group_data[motif].mean() > 0.5)

        # Map states to tree leaves
        for leaf in tree.iter_leaves():
            leaf_taxid = self.ncbi.get_name_translator([leaf.name])[leaf.name][0]
            if leaf_taxid in motif_states:
                leaf.add_feature("state", motif_states[leaf_taxid])
            else:
                leaf.add_feature("state", 0)

        # Fitch parsimony for internal nodes
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                child_states = [child.state for child in node.children]
                if len(set(child_states)) == 1:
                    node.add_feature("state", child_states[0])
                else:
                    # Ambiguous - for simplicity, use majority
                    node.add_feature("state", int(sum(child_states) > len(child_states) / 2))

        return tree

    def detect_gains_losses(self, tree):
        """Detect gain/loss events by comparing parent-child states"""
        events = {"gains": [], "losses": []}

        for node in tree.traverse():
            if not node.is_root():
                parent_state = node.up.state
                node_state = node.state

                if parent_state == 0 and node_state == 1:
                    events["gains"].append(node.name)
                elif parent_state == 1 and node_state == 0:
                    events["losses"].append(node.name)

        return events

    def analyze_motif_evolution(self, motifs, tax_level="family"):
        """Analyze gain/loss for multiple motifs"""
        tree, taxid_to_level, names = self.build_phylogenetic_tree(tax_level)

        if tree is None:
            print("Failed to build tree")
            return None

        results = {}

        for motif in motifs:
            if motif not in self.binary_df.columns:
                continue

            # Reconstruct ancestral states
            tree_copy = tree.copy()
            tree_with_states = self.ancestral_state_reconstruction(tree_copy, motif, taxid_to_level)

            # Detect gains/losses
            events = self.detect_gains_losses(tree_with_states)

            results[motif] = {
                "gains": len(events["gains"]),
                "losses": len(events["losses"]),
                "gain_nodes": events["gains"],
                "loss_nodes": events["losses"]
            }

        return results, tree

    def plot_evolution_summary(self, results, top_n=20, save_path=None):
        """Plot summary of gains/losses across motifs"""
        # Convert to DataFrame
        data = []
        for motif, info in results.items():
            data.append({
                "motif": motif,
                "gains": info["gains"],
                "losses": info["losses"],
                "total_changes": info["gains"] + info["losses"]
            })

        df = pd.DataFrame(data).sort_values("total_changes", ascending=False).head(top_n)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Stacked bar plot
        x = np.arange(len(df))
        ax1.bar(x, df["gains"], label="Gains", color="green", alpha=0.7)
        ax1.bar(x, df["losses"], bottom=df["gains"], label="Losses", color="red", alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["motif"], rotation=45, ha="right")
        ax1.set_ylabel("Number of Events")
        ax1.set_title(f"Top {top_n} Motifs by Evolutionary Changes")
        ax1.legend()

        # Gain/Loss ratio
        df["gain_loss_ratio"] = df["gains"] / (df["losses"] + 1)  # +1 to avoid division by zero
        ax2.bar(x, df["gain_loss_ratio"], color="blue", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["motif"], rotation=45, ha="right")
        ax2.set_ylabel("Gain/Loss Ratio")
        ax2.set_title("Gain vs Loss Tendency")
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.show()

        return df


def enrichment_by_lineage(binary_df, motifs, tax_level="phylum"):
    """Test if certain lineages are enriched for specific motifs"""
    ncbi = NCBITaxa()

    # Map species to higher taxonomic level
    taxids = binary_df['taxid'].unique()
    lineages = ncbi.get_lineage_translator(taxids)
    ranks = ncbi.get_rank(sum(lineages.values(), []))
    names = ncbi.get_taxid_translator(sum(lineages.values(), []))

    # Create mapping
    taxid_to_lineage = {}
    for taxid, lineage in lineages.items():
        for tid in lineage:
            if ranks.get(tid) == tax_level:
                taxid_to_lineage[taxid] = names.get(tid, "Unknown")
                break

    binary_df['lineage'] = binary_df['taxid'].map(taxid_to_lineage)

    # Test each motif for lineage association
    results = []

    for motif in motifs:
        if motif not in binary_df.columns:
            continue

        # Create contingency table
        table = pd.crosstab(binary_df['lineage'], binary_df[motif])

        if table.shape[1] == 2:
            chi2, pval, _, _ = chi2_contingency(table)

            # Calculate enrichment per lineage
            for lineage in table.index:
                observed = table.loc[lineage, 1]
                total = table.loc[lineage].sum()
                expected = table[1].sum() * total / table.sum().sum()

                if expected > 0:
                    enrichment = observed / expected

                    results.append({
                        'motif': motif,
                        'lineage': lineage,
                        'enrichment_score': enrichment,
                        'pvalue': pval,
                        'present': observed,
                        'total': total
                    })

    return pd.DataFrame(results)


def plot_lineage_enrichment_heatmap(enrichment_df, min_samples=10, save_path=None):
    """Heatmap of motif enrichment across lineages"""
    # Filter lineages with sufficient samples
    filtered = enrichment_df[enrichment_df['total'] >= min_samples]

    # Pivot for heatmap
    pivot = filtered.pivot(index='lineage', columns='motif', values='enrichment_score')

    # Log transform enrichment scores
    pivot_log = np.log2(pivot + 0.1)  # +0.1 to avoid log(0)

    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot_log, cmap="RdBu_r", center=0,
                cbar_kws={'label': 'Log2 Enrichment Score'})
    plt.title("Motif Enrichment Across Lineages")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load data
    binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0).reset_index()

    # Prepare taxid column
    binary_df['taxid'] = pd.to_numeric(binary_df['taxid'], errors='coerce').dropna().astype(int)

    # Load motifs
    motifs_df = pd.read_csv("output/top_entropy_motifs.csv", header=None)
    motifs = motifs_df[0].tolist()[:50]  # Top 50 motifs

    # Initialize NCBI taxonomy
    ncbi = NCBITaxa()

    # 1. Evolutionary gain/loss analysis
    print("=== EVOLUTIONARY GAIN/LOSS ANALYSIS ===")
    analyzer = MotifEvolutionAnalyzer(binary_df, ncbi)

    # Analyze at family level
    results, tree = analyzer.analyze_motif_evolution(motifs, tax_level="family")

    if results:
        # Plot summary
        summary_df = analyzer.plot_evolution_summary(results, top_n=20,
                                                     save_path="output/motif_evolution_summary.png")

        # Show motifs with most gains
        print("\nMotifs with most gains:")
        for motif, info in sorted(results.items(), key=lambda x: x[1]['gains'], reverse=True)[:10]:
            print(f"{motif}: {info['gains']} gains in {info['gain_nodes']}")

    # 2. Lineage enrichment analysis
    print("\n=== LINEAGE ENRICHMENT ANALYSIS ===")
    enrichment_df = enrichment_by_lineage(binary_df, motifs, tax_level="phylum")

    # Show top enrichments
    top_enriched = enrichment_df.nlargest(20, 'enrichment_score')
    print("\nTop enriched motif-lineage pairs:")
    print(top_enriched[['motif', 'lineage', 'enrichment_score', 'present', 'total']])

    # Plot heatmap
    plot_lineage_enrichment_heatmap(enrichment_df, min_samples=5,
                                    save_path="output/lineage_enrichment_heatmap.png")