from glycowork.glycan_data.loader import df_species
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# This has the actual glycan-to-taxonomy mappings
print(f"Total glycan-species observations: {len(df_species)}")
print(f"Unique glycans: {df_species['glycan'].nunique()}")

# Count glycans per taxonomic level
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Total glycan observations per phylum
ax1 = axes[0, 0]
glycans_per_phylum = df_species['Phylum'].value_counts()
glycans_per_phylum.head(20).plot(kind='bar', ax=ax1, color='darkgreen')
ax1.set_title('Total Glycan Observations per Phylum\n(THIS is the real sampling depth)')
ax1.set_ylabel('Number of Glycan Records')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Unique glycans per phylum
ax2 = axes[0, 1]
unique_glycans_per_phylum = df_species.groupby('Phylum')['glycan'].nunique().sort_values(ascending=False)
unique_glycans_per_phylum.head(20).plot(kind='bar', ax=ax2, color='darkblue')
ax2.set_title('Unique Glycan Structures per Phylum')
ax2.set_ylabel('Number of Unique Glycans')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Distribution at genus level (more detailed)
ax3 = axes[1, 0]
glycans_per_genus = df_species['Genus'].value_counts()
# Show distribution of how many glycans each genus has
counts = glycans_per_genus.values
ax3.hist(counts, bins=50, edgecolor='black')
ax3.set_xlabel('Number of Glycan Observations per Genus')
ax3.set_ylabel('Number of Genera')
ax3.set_title('Distribution of Glycan Sampling Depth (Genus Level)')
ax3.set_yscale('log')
ax3.axvline(np.median(counts), color='red', linestyle='--',
            label=f'Median: {np.median(counts):.0f}')
ax3.legend()

# Plot 4: Top 20 most sampled genera
ax4 = axes[1, 1]
top_genera = glycans_per_genus.head(20)
top_genera.plot(kind='barh', ax=ax4)
ax4.set_xlabel('Number of Glycan Observations')
ax4.set_title('Top 20 Most Deeply Sampled Genera')

plt.tight_layout()
plt.savefig('output/true_glycan_sampling_depth.png', dpi=300, bbox_inches='tight')
plt.show()

# Key statistics
print("\n=== TRUE GLYCAN SAMPLING DEPTH ===")
print(f"Total glycan-species records: {len(df_species)}")
print(f"Unique glycan structures: {df_species['glycan'].nunique()}")
print(f"Taxonomic coverage:")
print(f"  Phyla: {df_species['Phylum'].nunique()}")
print(f"  Classes: {df_species['Class'].nunique()}")
print(f"  Orders: {df_species['Order'].nunique()}")
print(f"  Families: {df_species['Family'].nunique()}")
print(f"  Genera: {df_species['Genus'].nunique()}")

# Which groups dominate?
print("\n=== DATA CONCENTRATION ===")
top_5_phyla = glycans_per_phylum.head(5)
print(f"Top 5 phyla contain {top_5_phyla.sum()} of {len(df_species)} records ({top_5_phyla.sum()/len(df_species)*100:.1f}%)")

# Extreme sampling examples
print("\n=== SAMPLING EXTREMES ===")
print("Most sampled genera:")
for genus, count in glycans_per_genus.head(5).items():
    phylum = df_species[df_species['Genus'] == genus]['Phylum'].iloc[0]
    print(f"  {genus} ({phylum}): {count} glycan records")

print("\nPhyla with very few records:")
rare_phyla = glycans_per_phylum[glycans_per_phylum < 10]
print(f"  {len(rare_phyla)} phyla have fewer than 10 glycan records")