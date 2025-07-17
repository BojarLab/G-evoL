import pandas as pd
from ete3 import NCBITaxa
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt



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

    binary_df["clade"] = binary_df["taxid"].map(taxid_to_clade)
    return binary_df


def test_motif_enrichment(binary_df, motifs, tax_col="clade", fdr_thresh=0.05):
    """Perform chi-squared test for motif enrichment across clades."""
    results = []
    for motif in motifs:
        table = pd.crosstab(binary_df[tax_col], binary_df[motif])
        if table.shape[1] == 2:  # 0 and 1 both present
            chi2, pval, _, _ = chi2_contingency(table)
        else:
            pval = 1.0  # no variation
        results.append((motif, pval, table))

    # Adjust p-values
    motifs = [r[0] for r in results]
    raw_pvals = [r[1] for r in results]
    _, fdr_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")

    enriched = pd.DataFrame({
        "motif": motifs,
        "raw_pval": raw_pvals,
        "fdr_pval": fdr_pvals
    }).sort_values("fdr_pval")

    return enriched


def plot_motif_clade_heatmap(binary_df, motifs, tax_col="clade", save_path=None):
    clade_summary = binary_df.groupby(tax_col)[motifs].mean()
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
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()





"""Selected motifs based on:
 Entropy
 Hierarchical Clustering"""

binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0)
motif_files = [
    ("top_entropy_motifs.csv", "output/motif_domain_heatmap_entropy.png"),
    ("representative_motifs.csv", "output/motif_domain_heatmap_representative.png")
]

ncbi = NCBITaxa()
binary_df = map_taxid_to_clade(binary_df, ncbi, tax_rank="domain")

for file_name, heatmap_path in motif_files:
    # Read without header and no index column
    df = pd.read_csv(f"output/{file_name}", header=None)
    motif_list = df[0].tolist()
    enrichment_df = test_motif_enrichment(binary_df, motif_list)
    print(f"Significant motifs for {file_name}:")
    print(enrichment_df.query("fdr_pval < 0.05"))
    sig_motifs = enrichment_df.query("fdr_pval < 0.05")["motif"].tolist()

    if sig_motifs:
        plot_motif_clade_heatmap(binary_df, sig_motifs, tax_col="clade", save_path=heatmap_path)
    else:
        print(f"No significant motifs found in {file_name}")
