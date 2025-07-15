import pandas as pd
from glycowork.motif.annotate import annotate_dataset
from glycowork.glycan_data.loader import df_glycan, df_species
from ete3 import NCBITaxa
ncbi = NCBITaxa()

unique_glycans = list(set(df_species["glycan"]))
annots = annotate_dataset(unique_glycans, feature_set=["known", "exhaustive"])
annots = annots.reset_index().rename(columns={"index": "glycan"})
df_annot = df_species.merge(annots, on="glycan", how="left") # glycan, sp, motif1, motif2, ...

# normalize motif counts by taking the average per species
df_agg = df_annot.groupby("Species").mean(numeric_only=True) #agg by mean
df_agg.to_csv("output/motif_counts_per_species_agg_mean.csv", index=True, header=True)