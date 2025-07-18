import pandas as pd
from ete3 import NCBITaxa, TreeStyle, NodeStyle, TextFace
from script3 import binary_df  # Assuming binary_df is defined in script3
import os
from collections import defaultdict

# Initialize NCBI taxonomy
ncbi = NCBITaxa()

# Clean and prepare the dataframe
binary_df = binary_df.dropna(subset=["taxid"])
binary_df["taxid"] = binary_df["taxid"].astype(int)

# Configuration - Choose taxonomic level
TAXONOMIC_LEVEL = "genus"  # Options: "genus", "family", "order", "class", "phylum"
motif_file = "output/top_entropy_motifs.csv"
output_image = f"output/tree_overlay_top_entropy_{TAXONOMIC_LEVEL}.png"

# Taxonomic rank mapping
RANK_LEVELS = {
    "genus": "genus",
    "family": "family",
    "order": "order",
    "class": "class",
    "phylum": "phylum"
}

# Ensure output directory exists
os.makedirs(os.path.dirname(output_image), exist_ok=True)


def get_higher_taxon(taxid, target_rank):
    """Get the taxid of a higher taxonomic level for a given species taxid"""
    try:
        lineage = ncbi.get_lineage(taxid)
        ranks = ncbi.get_rank(lineage)

        # Find the taxid corresponding to the target rank
        for tid in lineage:
            if ranks.get(tid) == target_rank:
                return tid
        return None
    except:
        return None


def get_taxon_name(taxid):
    """Get the scientific name for a taxid"""
    try:
        names = ncbi.get_taxid_translator([taxid])
        return names.get(taxid, f"Unknown_{taxid}")
    except:
        return f"Unknown_{taxid}"


try:
    # Load selected motifs from file (no header)
    selected_motifs = pd.read_csv(motif_file, header=None)[0].tolist()
    print(f"Loaded {len(selected_motifs)} selected motifs from {motif_file}")

    # Verify selected motifs exist in dataframe
    available_motifs = [m for m in selected_motifs if m in binary_df.columns]
    if len(available_motifs) != len(selected_motifs):
        missing = set(selected_motifs) - set(available_motifs)
        print(f"Warning: {len(missing)} selected motifs not found in dataframe: {missing}")

    if not available_motifs:
        raise ValueError("None of the selected motifs were found in the dataframe")

    print(f"Using {len(available_motifs)} motifs for analysis: {available_motifs}")
    motifs_to_analyze = available_motifs

    # Get higher taxonomic levels for each species
    print(f"Mapping species to {TAXONOMIC_LEVEL} level...")
    higher_taxon_data = defaultdict(list)  # higher_taxid -> list of species data

    for _, row in binary_df.iterrows():
        species_taxid = int(row["taxid"])
        higher_taxid = get_higher_taxon(species_taxid, TAXONOMIC_LEVEL)

        if higher_taxid:
            higher_taxon_data[higher_taxid].append(row)

    print(f"Found {len(higher_taxon_data)} unique {TAXONOMIC_LEVEL}-level taxa")

    if len(higher_taxon_data) == 0:
        raise ValueError(f"No taxa found at {TAXONOMIC_LEVEL} level")

    # Aggregate motif data at higher taxonomic level
    print(f"Aggregating motif data at {TAXONOMIC_LEVEL} level...")
    aggregated_data = []

    for higher_taxid, species_rows in higher_taxon_data.items():
        # Combine all species data for this higher taxon
        combined_row = {"taxid": higher_taxid, "name": get_taxon_name(higher_taxid)}

        # For each SELECTED motif, check if ANY species in this higher taxon has it (binary: 0 or 1)
        for motif in motifs_to_analyze:
            # Check if any species in this higher taxon has this motif present (value = 1)
            has_motif = any(row[motif] == 1 for row in species_rows if motif in row.index)
            combined_row[motif] = 1 if has_motif else 0

        # Add species count for this higher taxon
        combined_row["species_count"] = len(species_rows)
        aggregated_data.append(combined_row)

    # Convert to DataFrame
    agg_df = pd.DataFrame(aggregated_data)

    # Build tree using higher taxonomic level taxids
    higher_taxids = list(higher_taxon_data.keys())
    print(f"Building tree for {len(higher_taxids)} {TAXONOMIC_LEVEL}-level taxa...")

    try:
        tree = ncbi.get_topology(higher_taxids)
    except Exception as e:
        print(f"Error building tree: {e}")
        print("This might be due to invalid taxids or NCBI connectivity issues")
        raise

    # Style leaves by motif presence
    styled_leaves = 0
    motif_stats = defaultdict(int)

    for leaf in tree.iter_leaves():
        tid = int(leaf.name)  # taxid

        # Find corresponding row in aggregated data
        taxon_row = agg_df[agg_df["taxid"] == tid]
        if taxon_row.empty:
            # Style as gray if no data
            style = NodeStyle()
            style["fgcolor"] = "lightgray"
            style["size"] = 8
            leaf.set_style(style)
            continue

        row = taxon_row.iloc[0]

        # Count how many DIFFERENT selected motifs are present in this taxon
        present_motifs = []
        for motif in motifs_to_analyze:
            if row[motif] == 1:  # Motif is present (binary: 1)
                present_motifs.append(motif)
                motif_stats[motif] += 1

        count = len(present_motifs)  # Number of different selected motifs present
        species_count = row["species_count"]

        # Node styling based on motif count with bold colors
        style = NodeStyle()
        node_size = max(15, min(40, species_count * 3))

        if count == 0:
            # No motifs - gray with white background
            style["fgcolor"] = "#808080"
            style["bgcolor"] = "#F0F0F0"
            style["size"] = node_size
        elif count <= 2:
            # Few motifs - bright orange with light orange background
            style["fgcolor"] = "#FF4500"
            style["bgcolor"] = "#FFE4B5"
            style["size"] = node_size + 5
        elif count <= 5:
            # Many motifs - bright red with light red background
            style["fgcolor"] = "#DC143C"
            style["bgcolor"] = "#FFB6C1"
            style["size"] = node_size + 10
        else:
            # Very many motifs - dark red with pink background
            style["fgcolor"] = "#8B0000"
            style["bgcolor"] = "#FFC0CB"
            style["size"] = node_size + 15

        # Add border for better visibility
        style["hz_line_width"] = 3
        style["vt_line_width"] = 3

        leaf.set_style(style)

        # Replace leaf name with scientific name
        taxon_name = row["name"]
        leaf.name = f"{taxon_name} (n={species_count})"

        # Add motif labels with colored backgrounds
        if count > 0:
            # Truncate label if too long
            if len(present_motifs) > 4:
                label = f"{', '.join(present_motifs[:4])}... (+{count - 4})"
            else:
                label = ", ".join(present_motifs)

            text_face = TextFace(label, fsize=10, bold=True)
            text_face.margin_left = 15
            text_face.margin_right = 5
            text_face.margin_top = 2
            text_face.margin_bottom = 2
            # Add colored background to text
            text_face.background.color = "#FFFF99"  # Light yellow background
            text_face.border.width = 1
            text_face.border.color = "#000000"
            leaf.add_face(text_face, column=0, position="aligned")
            styled_leaves += 1
        else:
            # Add "No motifs" label for taxa without motifs
            text_face = TextFace("No motifs", fsize=9)
            text_face.margin_left = 15
            text_face.background.color = "#E0E0E0"  # Light gray background
            text_face.border.width = 1
            text_face.border.color = "#808080"
            leaf.add_face(text_face, column=0, position="aligned")

    print(f"Styled {styled_leaves} leaves with motif information")

    # Tree style configuration with better contrast
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.branch_vertical_margin = 25
    ts.scale = 180
    ts.optimal_scale_level = "full"

    # Set background color for better contrast
    ts.bgcolor = "#FFFFFF"

    # Add title
    title_face = TextFace(f"Motif Overlay at {TAXONOMIC_LEVEL.title()} Level: {os.path.basename(motif_file)}",
                          fsize=16, bold=True)
    ts.title.add_face(title_face, column=0)

    # Add legend with color examples
    legend_text = f"Legend: Node colors - Gray=No motifs, Orange=1-2, Red=3-5, Dark Red=6+ motifs"
    legend_face = TextFace(legend_text, fsize=12, bold=True)
    legend_face.background.color = "#FFFFFF"
    legend_face.border.width = 2
    legend_face.border.color = "#000000"
    ts.legend.add_face(legend_face, column=0)

    # Render to file with high resolution
    print(f"Rendering tree to {output_image}...")
    tree.render(output_image, w=3000, h=2400, units="px", tree_style=ts, dpi=600)
    print(f"Successfully saved phylogeny with motif overlay to: {output_image}")

    # Print summary statistics
    total_leaves = len(list(tree.iter_leaves()))
    print(f"\nSummary:")
    print(f"- Taxonomic level: {TAXONOMIC_LEVEL}")
    print(f"- Total {TAXONOMIC_LEVEL}-level taxa in tree: {total_leaves}")
    print(f"- Taxa with motif data: {styled_leaves}")
    print(f"- Total species represented: {agg_df['species_count'].sum()}")
    print(f"- Selected motifs analyzed: {len(motifs_to_analyze)}")

    print(f"\nSelected motif distribution across {TAXONOMIC_LEVEL}-level taxa:")
    print(f"(Only showing the {len(motifs_to_analyze)} selected motifs from {os.path.basename(motif_file)})")
    for motif in sorted(motif_stats.keys(), key=lambda x: motif_stats[x], reverse=True):
        print(f"  {motif}: present in {motif_stats[motif]} taxa ({motif_stats[motif] / total_leaves * 100:.1f}%)")

except FileNotFoundError:
    print(f"Error: Motif file '{motif_file}' not found")
except pd.errors.EmptyDataError:
    print(f"Error: Motif file '{motif_file}' is empty")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()