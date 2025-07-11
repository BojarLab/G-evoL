import pandas as pd
from ete3 import NCBITaxa
ncbi = NCBITaxa()
from ete3 import Tree
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

result_df = pd.read_csv("output/motif_counts_per_species.csv", header="infer", index_col=0)

#filter1
filt_df = result_df.loc[:, (result_df != 0).any(axis=0)] # remove zero-only columns
#filt_df = filt_df.loc[(filt_df != 0).any(axis=1), :] # remove zero-only rows
motif_freq= filt_df.mean(axis=0).sort_values(ascending=False)
filt_motifs= filt_df[motif_freq.index]

"Are certain motifs significantly more common in some species than in others?"

sp_list = (
    filt_motifs.index
    .drop_duplicates()
    .str.strip()
    .str.replace("_", " ")
    .tolist()
)
valid_sp_list = []
taxids = []
count = 0

for sp in sp_list:
    name2taxid = ncbi.get_name_translator([sp])
    if sp in name2taxid:
        taxids.append(name2taxid[sp][0])
        valid_sp_list.append(sp)
    else:
        print(f"Warning: Taxid not found for {sp}")
        count += 1

tree = ncbi.get_topology(taxids)
print(tree.get_ascii(show_internal=True))
print(f"Taxid not found: {count}")
tree.write(outfile="output/937_species_tree.nw")


binary_df = (filt_df > 0).astype(int)
binary_df.to_csv("output/binary_motif_table.csv", index=True, header=True)
tree = Tree("output/937_species_tree.nw", format=1)
#tree.prune(selected_taxids, preserve_branch_length=True)
#tree.write(outfile="output/pruned_tree.nw")
print(tree.get_ascii(show_internal=True))






"Option A: Cluster motifs by presence pattern"
motif_counts = binary_df.sum(axis=0)
# filter 2
filt2_motifs = motif_counts[(motif_counts >= 30) & (motif_counts <= 500)]
print(f"{len(filt2_motifs)} motifs selected based on frequency range")

# Transpose: rows=motifs, cols=species
motif_patterns = binary_df[filt2_motifs.index].T
# Optionally scale
Z = linkage(motif_patterns, method='ward')
# Pick e.g., 10 clusters
labels = fcluster(Z, t=10, criterion='maxclust')
# For each cluster, pick the motif with highest variance (most informative)
representative_motifs = []
for cluster in range(1, 11):
    cluster_motifs = motif_patterns[labels == cluster]
    if not cluster_motifs.empty:
        variances = cluster_motifs.var(axis=1)
        top_motif = variances.idxmax()
        representative_motifs.append(top_motif)

print("Representative motifs:", representative_motifs)






"Option B: Pick motifs with highest entropy"

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

motif_freqs = motif_counts / len(binary_df)
motif_entropy = entropy(motif_freqs).fillna(0)
top_entropy_motifs = motif_entropy.sort_values(ascending=False).head(10).index.tolist()
print("Top motifs by entropy:", top_entropy_motifs)