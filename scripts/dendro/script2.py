import pandas as pd
from ete3 import NCBITaxa
import re
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

"1- Convert motif_count to binary 0/1 table"

motif_count = pd.read_csv("output/motif_counts_per_species_agg_mean.csv", header="infer", index_col=0)
motif_count.index = motif_count.index.str.replace("_", " ").str.strip()
# Filter 1: Remove motifs that are absent in all species
motif_count_filt1 = motif_count.loc[:, (motif_count != 0).any(axis=0)]
# Filter 2: Remove rare motifs (<5% of species)
motif_freq = motif_count_filt1.mean(axis=0).sort_values(ascending=False)
common_motifs = motif_freq[motif_freq > 0.05].index
binary_df = (motif_count_filt1[common_motifs] > 0).astype(int)

# TAXID mapping
sp_list = binary_df.index.tolist()
valid_sp_list = []
taxids = []
count = 0
name_to_taxid = {}
# Get taxids for species
for sp in sp_list:
    name2taxid = NCBITaxa().get_name_translator([sp])
    if sp in name2taxid:
        tid = name2taxid[sp][0]
        taxids.append(tid)
        valid_sp_list.append(sp)
        name_to_taxid[sp] = tid
    else:
        print(f"Warning: Taxid not found for {sp}")
        count += 1
binary_df["taxid"] = binary_df.index.map(name_to_taxid)
binary_df["taxid"] = binary_df["taxid"].astype("Int64")  # nullable integer dtype
binary_df.to_csv("output/binary_motif_table.csv")



def is_multi_monosaccharide(motif_name):
    # Count sugar-like names in the motif name
    sugars = re.findall(r'[A-Z][a-z]+', motif_name)
    return len(sugars) > 1


"2- Select informative motifs:"
"Option A: Cluster motifs by presence pattern"

motif_counts = binary_df.drop(columns=["taxid"]).sum(axis=0)  # exclude taxid
# Filter 4: Select motifs with frequency between 30
filt = motif_counts[motif_counts >= 30]
filt = filt[[is_multi_monosaccharide(m) for m in filt.index]]
print(f"{len(filt)} motifs selected based on frequency range")

motif_patterns = binary_df[filt.index]
motif_patterns = motif_patterns.apply(pd.to_numeric, errors='coerce').fillna(0)
motif_patterns = motif_patterns.T
jaccard_dist = pdist(motif_patterns.values, metric='jaccard')
Z = linkage(jaccard_dist, method='average')
labels = fcluster(Z, t=10, criterion='maxclust')

representative_motifs = []
for cluster in range(1, 11):
    cluster_mask = (labels == cluster)
    cluster_motifs = motif_patterns.loc[cluster_mask]
    if not cluster_motifs.empty:
        variances = cluster_motifs.var(axis=1)
        #top_motif = variances.idxmax()
        top_motif = variances.idxmin()
        representative_motifs.append(top_motif)

print("Representative motifs:", representative_motifs)
representative_motifs = pd.Series(representative_motifs, name="motif")
representative_motifs.to_csv("output/representative_motifs.csv",header=False, index=False)


"Option B: Pick motifs with least entropy"

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

motif_freqs = motif_counts / len(binary_df)
motif_entropy = entropy(motif_freqs).fillna(0)
motif_entropy = motif_entropy[[is_multi_monosaccharide(m) for m in motif_entropy.index]]
top_entropy_motifs = motif_entropy.sort_values(ascending=True).head(10).index.tolist() #low entroy
#top_entropy_motifs = motif_entropy.sort_values(ascending=False).head(10).index.tolist() #high entropy

print("Top motifs by entropy:", top_entropy_motifs)
top_entropy_motifs = pd.Series(top_entropy_motifs, name="motif")
top_entropy_motifs.to_csv("output/top_entropy_motifs.csv",header=False,  index=False)