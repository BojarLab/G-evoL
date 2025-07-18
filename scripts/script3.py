import pandas as pd
from ete3 import NCBITaxa
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from ete3 import TreeStyle, NodeStyle
from matplotlib import gridspec
import matplotlib.image as mpimg
import tempfile


def map_taxid_to_clade(binary_df, ncbi, tax_rank):
    binary_df = binary_df.dropna(subset=["taxid"])
    binary_df["taxid"] = binary_df["taxid"].astype(int)
    species_taxids = binary_df["taxid"].tolist()

    lineages = ncbi.get_lineage_translator(species_taxids)
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

    col_name = tax_rank
    binary_df[col_name] = binary_df["taxid"].map(taxid_to_clade)

    return binary_df, col_name


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


def get_phylogenetic_order(clade_names, ncbi, tax_rank="domain"):
    name2taxid = ncbi.get_name_translator(clade_names)
    taxids = [name2taxid[name][0] for name in clade_names if name in name2taxid]
    try:
        tree = ncbi.get_topology(taxids)
    except Exception as e:
        print(f"Error building phylogeny tree: {e}")
        return clade_names

    taxid2name = {v[0]: k for k, v in name2taxid.items()}
    ordered_names = [taxid2name.get(int(leaf.name), f"Unknown_{leaf.name}") for leaf in tree.iter_leaves()]
    return ordered_names

def plot_motif_clade_heatmap(binary_df, motifs, tax_col="clade", save_path=None):
    #clade_summary = binary_df.groupby(tax_col)[motifs].mean()
    clade_summary = binary_df.groupby(tax_col)[motifs].mean()

    # Reorder rows by phylogenetic order
    ordered_clades = get_phylogenetic_order(clade_summary.index.tolist(), ncbi)
    clade_summary = clade_summary.loc[clade_summary.index.intersection(ordered_clades)]
    clade_summary = clade_summary.reindex(ordered_clades)

    print(clade_summary.columns.tolist())

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(clade_summary, cmap="Greens", annot=False, fmt=".2f", linewidths=0.5)

    plt.title("Motif Presence Across Clades")
    plt.ylabel(tax_col.capitalize())
    plt.xlabel("Motif")

    labels = clade_summary.columns.astype(str).tolist()
    ax.set_xticks([x + 0.5 for x in range(len(labels))])  # move ticks to center of bars
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"Figure saved to {save_path}")

    plt.show()

def plot_tree_and_heatmap(binary_df, motifs, tax_col, ncbi=None, tree_tax_rank=None, save_path=None):
    clade_summary = binary_df.groupby(tax_col)[motifs].mean()

    # Use clade names for the tree, but build tree using taxids at tree_tax_rank level
    # We assume tax_col corresponds to the same tax rank as tree_tax_rank or compatible
    # Get clade names
    clade_names = clade_summary.index.tolist()
    name2taxid = ncbi.get_name_translator(clade_names)
    taxids = [name2taxid[name][0] for name in clade_names if name in name2taxid]

    # Build tree topology at specified tax rank if given, else use current taxids
    try:
        tree = ncbi.get_topology(taxids)
    except Exception as e:
        print(f"Error building phylogeny tree: {e}")
        # fallback to no tree, just heatmap
        plot_motif_clade_heatmap(binary_df, motifs, tax_col=tax_col, save_path=save_path)
        return

    taxid2name = {v[0]: k for k, v in name2taxid.items()}
    for leaf in tree.iter_leaves():
        if int(leaf.name) in taxid2name:
            leaf.name = taxid2name[int(leaf.name)]
        else:
            leaf.name = f"Unknown_{leaf.name}"

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.scale = 120
    ts.branch_vertical_margin = 10

    for n in tree.traverse():
        nstyle = NodeStyle()
        nstyle["size"] = 0
        n.set_style(nstyle)

    tree_img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    tree.render(tree_img_path, w=300, units="px", tree_style=ts)

    ordered_clades = [leaf.name for leaf in tree.iter_leaves()]
    clade_summary = clade_summary.loc[clade_summary.index.intersection(ordered_clades)]
    clade_summary = clade_summary.reindex(ordered_clades)

    tree_img = mpimg.imread(tree_img_path)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(tree_img)
    ax0.axis("off")
    ax0.set_title("Phylogenetic Tree", fontsize=12)

    ax1 = plt.subplot(gs[1])
    sns.heatmap(clade_summary, cmap="Greens", annot=False, linewidths=0.5, ax=ax1)
    ax1.set_title("Motif Presence Across Clades")
    ax1.set_ylabel(tax_col.capitalize())
    ax1.set_xlabel("Motif")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Combined tree + heatmap saved to: {save_path}")

    plt.show()

import numpy as np
from scipy.cluster import hierarchy

def plot_combined_tree_heatmap(binary_df, motifs, tax_col="clade", ncbi=None, tree_tax_rank="domain", save_path=None):
    # Group motif presence per clade
    clade_summary = binary_df.groupby(tax_col)[motifs].mean()

    # Get clade -> taxid mapping
    clade_names = clade_summary.index.tolist()
    name2taxid = ncbi.get_name_translator(clade_names)
    taxids = [name2taxid[name][0] for name in clade_names if name in name2taxid]

    # Get topology tree
    tree = ncbi.get_topology(taxids)

    # Map leaves to clade names
    taxid2name = {v[0]: k for k, v in name2taxid.items()}
    for leaf in tree.iter_leaves():
        leaf.name = taxid2name.get(int(leaf.name), f"Unknown_{leaf.name}")

    # Extract pairwise distances between leaves (clades) from the tree
    leaves = [leaf.name for leaf in tree.iter_leaves()]
    n = len(leaves)
    dist_matrix = np.zeros((n, n))
    leaf_nodes = {leaf.name: leaf for leaf in tree.iter_leaves()}

    def get_distance(n1, n2):
        return tree.get_distance(n1, n2)

    for i in range(n):
        for j in range(i + 1, n):
            dist = get_distance(leaf_nodes[leaves[i]], leaf_nodes[leaves[j]])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Perform hierarchical clustering on the distances to get dendrogram order
    linkage = hierarchy.linkage(hierarchy.distance.squareform(dist_matrix), method='average')

    # Get dendrogram to extract leaf order
    dendro = hierarchy.dendrogram(linkage, labels=leaves, no_plot=True)
    ordered_leaves = dendro['ivl']

    # Reorder heatmap rows according to dendrogram order
    clade_summary = clade_summary.reindex(ordered_leaves)

    # Plot combined figure with dendrogram and heatmap
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    # Plot dendrogram
    ax0 = plt.subplot(gs[0])
    hierarchy.dendrogram(linkage, labels=ordered_leaves, orientation='left', ax=ax0)
    ax0.tick_params(axis='y', labelsize=10)
    ax0.set_xlabel('Distance')
    ax0.set_ylabel(tax_col.capitalize())
    ax0.invert_xaxis()  # Optional: flip dendrogram horizontally for better layout

    # Plot heatmap
    ax1 = plt.subplot(gs[1])
    sns.heatmap(clade_summary, cmap="Greens", annot=False, linewidths=0.5, ax=ax1,
                cbar_kws={"shrink": 0.5})
    ax1.set_ylabel('')
    ax1.set_xlabel('Motif')
    ax1.set_yticks([])  # Hide y tick labels because dendrogram shows clades
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
        print(f"Combined dendrogram + heatmap saved to: {save_path}")

    plt.show()



"""Selected motifs based on:
 Entropy
 Hierarchical Clustering"""

binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0)
ncbi = NCBITaxa()

TAX_RANK = "kingdom"  # Change here once for all
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
                                   tree_tax_rank="domain", save_path=heatmap_path)
    else:
        print(f"No significant motifs found in {file_name}")
