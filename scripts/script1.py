import pandas as pd
from glycowork.motif.annotate import annotate_dataset
from glycowork.glycan_data.loader import df_glycan, df_species
from ete3 import NCBITaxa
ncbi = NCBITaxa()

unique_glycans = list(set(df_species["glycan"]))
annots = annotate_dataset(unique_glycans, feature_set=["known", "exhaustive"])
annots = annots.reset_index().rename(columns={"index": "glycan"}).set_index("glycan")

annots.to_csv("output/annots.csv", index=False)
#annots= pd.read_csv("output/annots.csv", index_col=0)


dfs = []
for sp in df_species["Species"].unique():
    df_sp = df_species[df_species["Species"] == sp]
    glycan_list = df_sp["glycan"]
    sub_annots = annots.loc[glycan_list].reset_index()
    df_sp_annot = pd.concat([
        df_sp.reset_index(drop=True),
        sub_annots.drop(columns=["glycan"]).reset_index(drop=True)
    ], axis=1)
    dfs.append(df_sp_annot)

df_annot = pd.concat(dfs, ignore_index=True)


# normalize motif counts by taking the average per species
df_agg = df_annot.groupby("Species").mean(numeric_only=True) #agg by mean
df_agg.to_csv("output/motif_counts_per_species_agg_mean.csv", index=True, header=True)