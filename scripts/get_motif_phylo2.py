import pandas as pd
from ete3 import NCBITaxa
ncbi = NCBITaxa()
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

norm_df = pd.read_csv("output/motif_counts_per_species.csv", header="infer", index_col=0)
norm_df.index = norm_df.index.str.replace("_", " ")

#filter1
filt_df = norm_df.loc[:, (norm_df != 0).any(axis=0)] # remove zero-only columns
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
name_to_taxid= {}

for sp in sp_list:
    name2taxid = ncbi.get_name_translator([sp])
    if sp in name2taxid:
        tid = name2taxid[sp][0]
        taxids.append(tid)
        valid_sp_list.append(sp)
        name_to_taxid[sp] = tid
    else:
        print(f"Warning: Taxid not found for {sp}")
        count += 1


tree = ncbi.get_topology(taxids)
tree.write(outfile="output/937_species_tree.nw")
print(tree.get_ascii(show_internal=True))
print(f"Taxid not found: {count}")

binary_df = (filt_df > 0.05).astype(int)
binary_df["taxid"] = binary_df.index.map(name_to_taxid)
binary_df.to_csv("output/binary_motif_table.csv")





"Option A: Cluster motifs by presence pattern"
motif_counts = binary_df.sum(axis=0)
# filter 2
filt2_motifs = motif_counts[(motif_counts >= 30)]
print(f"{len(filt2_motifs)} motifs selected based on frequency range")

# Transpose: rows=motifs, cols=species
motif_patterns = binary_df[filt2_motifs.index].T
jaccard_dist = pdist(motif_patterns.values, metric='jaccard') #only considers presence and ignores absence
Z = linkage(jaccard_dist, method='average')
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