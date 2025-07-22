import pandas as pd
from ete3 import NCBITaxa
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from ete3 import TreeStyle, NodeStyle
from matplotlib import gridspec
import numpy as np
from scipy.cluster import hierarchy


def map_taxid_to_clade(binary_df, ncbi, tax_rank):
    # Ensure taxid is integer
    binary_df = binary_df.dropna(subset=["taxid"]).copy()
    binary_df["taxid"] = pd.to_numeric(binary_df["taxid"], errors="coerce")
    binary_df = binary_df.dropna(subset=["taxid"])
    binary_df["taxid"] = binary_df["taxid"].astype(int)

    species_taxids = binary_df["taxid"].tolist()

    # Special handling for species rank
    if tax_rank == "species":
        # Check if "Species" column already exists (your data has this)
        if "Species" in binary_df.columns:
            print("✅ Using existing 'Species' column")
            return binary_df, "Species"  # Return the existing column name

        # Otherwise map taxid to species name
        species_names = ncbi.get_taxid_translator(species_taxids)
        binary_df["species"] = binary_df["taxid"].map(species_names)
        return binary_df, "species"

    # For other ranks, proceed with lineage lookup
    try:
        lineages = ncbi.get_lineage_translator(species_taxids)
    except Exception as e:
        print(f"Error fetching lineage: {e}")
        raise

    all_taxids = [tid for lineage in lineages.values() for tid in lineage]
    ranks = ncbi.get_rank(all_taxids)
    names = ncbi.get_taxid_translator(all_taxids)

    taxid_to_clade = {}
    for taxid, lineage in lineages.items():
        rank_taxid = next((tid for tid in lineage if ranks.get(tid) == tax_rank), None)
        if rank_taxid:
            taxid_to_clade[taxid] = names.get(rank_taxid, "Unknown")
        else:
            taxid_to_clade[taxid] = "Unknown"

    binary_df[tax_rank] = binary_df["taxid"].map(taxid_to_clade)

    return binary_df, tax_rank


def test_motif_enrichment(binary_df, motifs, tax_col, fdr_thresh=0.05):
    results = []
    for motif in motifs:
        table = pd.crosstab(binary_df[tax_col], binary_df[motif])
        if table.shape[1] == 2:  # 0 and 1 both present
            chi2, pval, _, _ = chi2_contingency(table)
        else:
            pval = 1.0
        results.append((motif, pval, table))

    motifs_ = [r[0] for r in results]
    raw_pvals = [r[1] for r in results]
    _, fdr_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")

    enriched = pd.DataFrame({
        "motif": motifs_,
        "raw_pval": raw_pvals,
        "fdr_pval": fdr_pvals
    }).sort_values("fdr_pval")

    return enriched


def plot_combined_tree_heatmap(binary_df, motifs, tax_col="clade", ncbi=None, tree_tax_rank="kingdom", save_path=None):
    # Group motif presence per clade
    clade_summary = binary_df.groupby(tax_col)[motifs].mean()
    clade_names = clade_summary.index.tolist()

    # Map clade names to NCBI taxids
    name2taxid = ncbi.get_name_translator(clade_names)
    missing_clades = set(clade_names) - set(name2taxid.keys())
    if missing_clades:
        print("⚠️ Missing clades from taxonomy name-to-taxid mapping:", missing_clades)

    # Filter clades that are successfully mapped
    taxid_name_pairs = [(name2taxid[name][0], name) for name in clade_names if name in name2taxid]
    taxids = [tid for tid, name in taxid_name_pairs]
    taxid2name = {tid: name for tid, name in taxid_name_pairs}

    # Build tree
    try:
        tree = ncbi.get_topology(taxids)
    except Exception as e:
        print(f"❌ Error building tree: {e}")
        return

    # Rename tree leaves
    for leaf in tree.iter_leaves():
        leaf_taxid = int(leaf.name) if leaf.name.isdigit() else leaf.name
        leaf.name = taxid2name.get(leaf_taxid, f"Unknown_{leaf.name}")

    # Compute distances and cluster
    leaves = [leaf.name for leaf in tree.iter_leaves()]
    valid_leaves = [leaf for leaf in leaves if leaf in clade_summary.index]

    if len(valid_leaves) < 2:
        print("❌ Not enough valid leaves for clustering")
        return

    n = len(valid_leaves)
    dist_matrix = np.zeros((n, n))
    leaf_nodes = {leaf.name: leaf for leaf in tree.iter_leaves()}

    for i in range(n):
        for j in range(i + 1, n):
            dist = tree.get_distance(leaf_nodes[valid_leaves[i]], leaf_nodes[valid_leaves[j]])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    linkage = hierarchy.linkage(hierarchy.distance.squareform(dist_matrix), method='average')
    dendro_order = hierarchy.dendrogram(linkage, labels=valid_leaves, no_plot=True)['ivl']

    # Setup plots
    #fig = plt.figure(figsize=(18, 10))
    #gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    fig = plt.figure(figsize=(20, max(12, len(valid_leaves) * 0.3)))  # Height scales with number of species
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 2])

    # Dendrogram
    ax0 = plt.subplot(gs[0])
    hierarchy.dendrogram(linkage, labels=valid_leaves, orientation='left', ax=ax0)
    ax0.tick_params(axis='y', labelsize=10, labelleft=True, labelright=False)
    ax0.set_xlabel('Distance')
    ax0.set_ylabel(tree_tax_rank.capitalize())
    ax0.invert_xaxis()

    # Heatmap (reversed order to match dendrogram)
    ax1 = plt.subplot(gs[1])
    heatmap_data = clade_summary.reindex(reversed(dendro_order))
    sns.heatmap(heatmap_data, cmap="Greens", annot=False, linewidths=0.5, ax=ax1,
                vmin=0, vmax=1, cbar_kws={"shrink": 0.5}, yticklabels=True)
    ax1.set_ylabel('')
    ax1.set_xlabel('Motif')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10, labelleft=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()



"""Selected motifs based on:
 Entropy
 Hierarchical Clustering"""

binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0)
binary_df = binary_df.reset_index()

ncbi = NCBITaxa()

TAX_RANK = "phylum"  # Change here once for all

motif_files = [
    ("top_entropy_motifs.csv", f"output/motif_{TAX_RANK}_heatmap_entropy.png"),
    ("representative_motifs.csv", f"output/motif_{TAX_RANK}_heatmap_representative.png")
]

binary_df, tax_col = map_taxid_to_clade(binary_df, ncbi, tax_rank=TAX_RANK)

for file_name, heatmap_path in motif_files:
    df = pd.read_csv(f"output/{file_name}", header=None)
    motif_list = df[0].tolist()

    enrichment_df = test_motif_enrichment(binary_df, motif_list, tax_col=tax_col)
    print(f"Significant motifs for {file_name}:")
    print(enrichment_df.query("fdr_pval < 0.05"))

    sig_motifs = enrichment_df.query("fdr_pval < 0.05")["motif"].tolist()

    if sig_motifs:
        plot_combined_tree_heatmap(binary_df, sig_motifs, tax_col=tax_col, ncbi=ncbi,
                                   tree_tax_rank=TAX_RANK, save_path=heatmap_path)

    else:
        print(f"No significant motifs found in {file_name}")
