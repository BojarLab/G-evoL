import pandas as pd
from glycowork.motif.annotate import annotate_dataset
from glycowork.glycan_data.loader import df_glycan, df_species
from ete3 import NCBITaxa
ncbi = NCBITaxa()


sp_list = (
    df_species["Species"]
    .drop_duplicates()
    .str.strip()
    .str.replace("_", " ")
    .tolist()
)

taxids = []
count = 0
for sp in sp_list:
    name2taxid = ncbi.get_name_translator([sp])
    if sp in name2taxid:
        taxids.append(name2taxid[sp][0])
    else:
        print(f"Warning: Taxid not found for {sp}")
        count += 1
tree = ncbi.get_topology(taxids)
print(tree.get_ascii(show_internal=True))
print(f"Taxid not found: {count}")
tree.write(outfile="output/2918_species_tree.nw")


df_gly_sp = df_species[["glycan", "Species"]]#[:100]
sp_gly_dic = df_gly_sp.groupby("Species")["glycan"].agg(list).to_dict()

species_dfs = []
for species, glycan_list in sp_gly_dic.items():
    df = annotate_dataset(glycan_list, feature_set=["known", "exhaustive"])
    df["Species"] = species
    species_dfs.append(df)

combined_df = pd.concat(species_dfs, ignore_index=True)

result_df = combined_df.groupby("Species").sum(numeric_only=True)
result_df.to_csv("output/motif_counts_per_species.csv", index=True, header=True)

print(result_df)

