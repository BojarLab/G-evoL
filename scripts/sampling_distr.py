import pandas as pd
import matplotlib.pyplot as plt
from ete3 import NCBITaxa
import numpy as np

# Load your data
df = pd.read_csv("output/binary_motif_table.csv", index_col=0)
print(f"Total species: {len(df)}")

# Get taxonomy info
ncbi = NCBITaxa()

# Get lineage for each species
species_list = df.index.tolist()
taxid_list = df['taxid'].dropna().astype(int).tolist()

# Get taxonomic info at different levels
taxonomy_info = []

for species, taxid in zip(df.index, df['taxid']):
    if pd.notna(taxid):
        lineage = ncbi.get_lineage(int(taxid))
        ranks = ncbi.get_rank(lineage)
        names = ncbi.get_taxid_translator(lineage)

        # Extract specific ranks
        info = {'species': species}
        for rank in ['phylum', 'class', 'order', 'family', 'genus']:
            for tid in lineage:
                if ranks.get(tid) == rank:
                    info[rank] = names.get(tid, 'Unknown')
                    break

        taxonomy_info.append(info)

# Convert to DataFrame
tax_df = pd.DataFrame(taxonomy_info)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Species per Phylum
phylum_counts = tax_df['phylum'].value_counts()
ax1 = axes[0, 0]
phylum_counts.plot(kind='bar', ax=ax1)
ax1.set_title(f'Number of Species per Phylum (n={len(phylum_counts)} phyla)')
ax1.set_xlabel('Phylum')
ax1.set_ylabel('Number of Species')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Species per Class (top 20)
class_counts = tax_df['class'].value_counts().head(20)
ax2 = axes[0, 1]
class_counts.plot(kind='bar', ax=ax2)
ax2.set_title(f'Top 20 Classes by Species Count')
ax2.set_xlabel('Class')
ax2.set_ylabel('Number of Species')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Distribution histogram
ax3 = axes[1, 0]
order_counts = tax_df['order'].value_counts()
counts_per_order = order_counts.values
ax3.hist(counts_per_order, bins=30, edgecolor='black')
ax3.set_title('Distribution of Sample Sizes Across Orders')
ax3.set_xlabel('Number of Species in Order')
ax3.set_ylabel('Number of Orders')
ax3.set_yscale('log')  # Log scale to see rare groups

# Plot 4: Pie chart of major groups
ax4 = axes[1, 1]
# Group small phyla into "Others"
top_phyla = phylum_counts.head(10)
others = phylum_counts[10:].sum()
if others > 0:
    plot_data = pd.concat([top_phyla, pd.Series({'Others': others})])
else:
    plot_data = top_phyla

ax4.pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%')
ax4.set_title('Proportion of Species by Phylum')

plt.tight_layout()
plt.savefig('output/sampling_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== SAMPLING SUMMARY ===")
print(f"Total species: {len(tax_df)}")
print(f"Number of phyla: {tax_df['phylum'].nunique()}")
print(f"Number of classes: {tax_df['class'].nunique()}")
print(f"Number of orders: {tax_df['order'].nunique()}")

print("\n=== SAMPLING IMBALANCE ===")
print(f"Most sampled phylum: {phylum_counts.index[0]} ({phylum_counts.iloc[0]} species)")
print(f"Least sampled phyla: {phylum_counts[phylum_counts == 1].count()} phyla with only 1 species")

print("\n=== TOP 5 MOST SAMPLED GROUPS ===")
for rank in ['phylum', 'class', 'order']:
    print(f"\n{rank.upper()}:")
    top5 = tax_df[rank].value_counts().head(5)
    for name, count in top5.items():
        print(f"  {name}: {count} species")