from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace, TextFace
import pandas as pd
from get_motif_phylo2 import name_to_taxid

from scripts.get_motif_phylo2 import name2taxid

#With a given selected motif, visualize its presence/absence across species in a phylogenetic tree.
Representative_motifs= ['Man(a1-3)Man', 'Glc(b1-6)Glc', 'Glc(a1-?)Glc', 'GlcN', 'Galf', 'GalNAc', 'GlcNAc(b1-2)Man', 'Xyl', 'Gal(b1-3)Gal', 'Gal(b1-4)GlcNAc']
Top_motifs= ['Glc', 'Gal', 'Man', 'GlcNAc', 'Rha', 'Fuc', 'Man(a1-6)Man', 'Man(a1-?)Man', 'Man(a1-3)Man', 'GalNAc']

binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0)
tree = Tree("output/2918_species_tree.nw", format=1)
motif = "Xyl"

print(f"{motif} present in {binary_df[motif].sum()} species")


# Add name-to-species map from taxid
taxid_to_name = {v: k for k, v in name_to_taxid.items()}
for leaf in tree:
    try:
        taxid = int(leaf.name)
        if taxid in taxid_to_name:
            leaf.name = taxid_to_name[taxid]  # Replace taxid with species name
    except ValueError:
        continue


# Set presence/absence as a custom attribute for each leaf node
for leaf in tree:
    species_id = leaf.name
    if species_id in binary_df.index:
        leaf.add_feature("motif_state", binary_df.loc[species_id, motif])
    else:
        leaf.add_feature("motif_state", -1)  # Missing data



# Set styles and add colored labels
for node in tree.traverse():
    nstyle = NodeStyle()
    nstyle["size"] = 0  # Hide node circles

    # Default gray
    color = "gray"
    symbol = "?"

    if hasattr(node, "motif_state"):
        if node.motif_state == 1:
            color = "green"
            symbol = "✓"
        elif node.motif_state == 0:
            color = "red"
            symbol = "×"

        nstyle["fgcolor"] = color
        nstyle["hz_line_color"] = color
        nstyle["vt_line_color"] = color
        nstyle["hz_line_width"] = 2
        nstyle["vt_line_width"] = 2

        node.set_style(nstyle)

        # Add text face for leaves only
        if node.is_leaf():
            label_face = TextFace(f"{node.name} ({symbol})", fsize=10, fgcolor=color)
            node.add_face(label_face, column=0, position="branch-right")

# Tree layout options
ts = TreeStyle()
ts.show_leaf_name = False
ts.title.add_face(TextFace(f"Motif: {motif}", fsize=14, bold=True), column=0)
ts.scale = 50

# Export the tree
tree.render(f"output/{motif}_phylogeny.pdf", w=800, tree_style=ts)
