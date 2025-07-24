import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import fisher_exact
import warnings

warnings.filterwarnings('ignore')


class MotifSelector:
    """Advanced methods for selecting representative motifs"""

    def __init__(self, binary_df, taxonomy_col='taxid'):
        self.binary_df = binary_df
        self.taxonomy_col = taxonomy_col
        self.motif_cols = [col for col in binary_df.columns if col != taxonomy_col]

    def method1_phylogenetic_signal(self, ncbi, tax_level='family'):
        """Select motifs with strong phylogenetic signal"""
        from scipy.stats import chi2_contingency

        # Map to higher taxonomic level
        lineages = ncbi.get_lineage_translator(self.binary_df[self.taxonomy_col].dropna())
        ranks = ncbi.get_rank(sum(lineages.values(), []))

        tax_mapping = {}
        for taxid, lineage in lineages.items():
            for tid in lineage:
                if ranks.get(tid) == tax_level:
                    tax_mapping[taxid] = tid
                    break

        self.binary_df['tax_group'] = self.binary_df[self.taxonomy_col].map(tax_mapping)

        # Test each motif for association with taxonomic groups
        phylo_scores = {}
        for motif in self.motif_cols:
            table = pd.crosstab(self.binary_df['tax_group'], self.binary_df[motif])
            if table.shape[1] == 2 and table.shape[0] > 1:
                chi2, pval, _, _ = chi2_contingency(table)
                # Use chi2 statistic as phylogenetic signal strength
                phylo_scores[motif] = chi2 / table.sum().sum()  # Normalize by sample size
            else:
                phylo_scores[motif] = 0

        # Select top motifs
        top_motifs = pd.Series(phylo_scores).nlargest(20).index.tolist()
        return top_motifs, phylo_scores

    def method2_mutual_information(self, target_col=None):
        """Select motifs with high mutual information with taxonomy or other target"""
        if target_col is None:
            target_col = self.taxonomy_col

        X = self.binary_df[self.motif_cols].values
        y = pd.Categorical(self.binary_df[target_col]).codes

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        mi_dict = dict(zip(self.motif_cols, mi_scores))

        # Select top motifs
        top_motifs = pd.Series(mi_dict).nlargest(20).index.tolist()
        return top_motifs, mi_dict

    def method3_maximum_relevance_minimum_redundancy(self, n_motifs=20):
        """mRMR: Select motifs that are relevant but not redundant"""
        # Step 1: Calculate relevance (mutual information with taxonomy)
        _, mi_scores = self.method2_mutual_information()

        # Step 2: Calculate redundancy (mutual information between motifs)
        motif_data = self.binary_df[self.motif_cols]
        n_features = len(self.motif_cols)
        redundancy_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = mutual_info_classif(motif_data.iloc[:, [i]], motif_data.iloc[:, j],
                                         discrete_features=True, random_state=42)[0]
                redundancy_matrix[i, j] = redundancy_matrix[j, i] = mi

        # mRMR selection
        selected = []
        remaining = list(range(n_features))

        # Select first feature with maximum relevance
        first_idx = max(remaining, key=lambda i: mi_scores[self.motif_cols[i]])
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Iteratively select features
        while len(selected) < n_motifs and remaining:
            mrmr_scores = {}
            for idx in remaining:
                relevance = mi_scores[self.motif_cols[idx]]
                redundancy = np.mean([redundancy_matrix[idx, s] for s in selected])
                mrmr_scores[idx] = relevance - redundancy

            best_idx = max(mrmr_scores, key=mrmr_scores.get)
            selected.append(best_idx)
            remaining.remove(best_idx)

        selected_motifs = [self.motif_cols[i] for i in selected]
        return selected_motifs, mrmr_scores

    def method4_random_forest_importance(self, n_estimators=100):
        """Use Random Forest feature importance"""
        X = self.binary_df[self.motif_cols].values
        y = pd.Categorical(self.binary_df[self.taxonomy_col]).codes

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Get feature importances
        importance_dict = dict(zip(self.motif_cols, rf.feature_importances_))

        # Select top motifs
        top_motifs = pd.Series(importance_dict).nlargest(20).index.tolist()
        return top_motifs, importance_dict

    def method5_co_occurrence_modules(self, min_correlation=0.3):
        """Find co-occurring motif modules"""
        # Calculate correlation matrix
        motif_data = self.binary_df[self.motif_cols]
        corr_matrix = motif_data.corr()

        # Create network of highly correlated motifs
        modules = []
        visited = set()

        for motif in self.motif_cols:
            if motif in visited:
                continue

            # Find all motifs correlated with this one
            module = [motif]
            correlated = corr_matrix[motif][corr_matrix[motif] > min_correlation].index.tolist()

            for cor_motif in correlated:
                if cor_motif != motif and cor_motif not in visited:
                    module.append(cor_motif)
                    visited.add(cor_motif)

            visited.add(motif)
            if len(module) > 1:
                modules.append(module)

        # Select representative from each module
        representatives = []
        for module in modules[:10]:  # Top 10 modules
            # Pick motif with highest average correlation within module
            module_corrs = corr_matrix.loc[module, module].mean()
            representative = module_corrs.idxmax()
            representatives.append(representative)

        return representatives, modules

    def method6_pca_loadings(self, n_components=10):
        """Select motifs with highest loadings on principal components"""
        # PCA on binary data
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(self.binary_df[self.motif_cols])

        # Get absolute loadings
        loadings = np.abs(pca.components_)

        # Select motif with highest loading for each PC
        selected_motifs = []
        for i in range(n_components):
            top_motif_idx = np.argmax(loadings[i])
            selected_motifs.append(self.motif_cols[top_motif_idx])

        # Remove duplicates while preserving order
        selected_motifs = list(dict.fromkeys(selected_motifs))

        return selected_motifs, pca.explained_variance_ratio_

    def method7_discriminative_motifs(self, group1_taxa, group2_taxa):
        """Find motifs that discriminate between two groups of taxa"""
        # Create binary labels
        labels = np.zeros(len(self.binary_df))
        labels[self.binary_df[self.taxonomy_col].isin(group1_taxa)] = 0
        labels[self.binary_df[self.taxonomy_col].isin(group2_taxa)] = 1

        # Test each motif
        discriminative_scores = {}
        for motif in self.motif_cols:
            # Fisher's exact test
            table = pd.crosstab(labels, self.binary_df[motif])
            if table.shape == (2, 2):
                _, pval = fisher_exact(table)
                discriminative_scores[motif] = -np.log10(pval + 1e-300)  # -log10(p-value)
            else:
                discriminative_scores[motif] = 0

        # Select top discriminative motifs
        top_motifs = pd.Series(discriminative_scores).nlargest(20).index.tolist()
        return top_motifs, discriminative_scores

    def method8_coverage_optimization(self, min_coverage=0.8):
        """Select minimum set of motifs that cover maximum species"""
        # Greedy set cover algorithm
        covered_species = set()
        selected_motifs = []
        total_species = len(self.binary_df)

        while len(covered_species) < min_coverage * total_species:
            best_motif = None
            best_new_coverage = 0

            for motif in self.motif_cols:
                if motif in selected_motifs:
                    continue

                # Species where this motif is present
                motif_species = set(self.binary_df[self.binary_df[motif] == 1].index)
                new_coverage = len(motif_species - covered_species)

                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_motif = motif

            if best_motif is None:
                break

            selected_motifs.append(best_motif)
            motif_species = set(self.binary_df[self.binary_df[best_motif] == 1].index)
            covered_species.update(motif_species)

        return selected_motifs, len(covered_species) / total_species

    def method9_evolutionary_rate_variation(self, ncbi):
        """Select motifs with variable evolutionary rates across clades"""
        # This would require phylogenetic tree and branch length calculations
        # Placeholder for more complex evolutionary analysis
        pass

    def combine_methods(self, methods_weights=None):
        """Combine multiple methods using weighted voting"""
        if methods_weights is None:
            methods_weights = {
                'mutual_information': 1.0,
                'random_forest': 1.0,
                'phylogenetic': 0.5,
                'pca': 0.5
            }

        all_scores = {}

        # Run each method and collect scores
        if 'mutual_information' in methods_weights:
            _, mi_scores = self.method2_mutual_information()
            for motif, score in mi_scores.items():
                all_scores[motif] = all_scores.get(motif, 0) + score * methods_weights['mutual_information']

        if 'random_forest' in methods_weights:
            _, rf_scores = self.method4_random_forest_importance()
            for motif, score in rf_scores.items():
                all_scores[motif] = all_scores.get(motif, 0) + score * methods_weights['random_forest']

        # Normalize and select top motifs
        final_scores = pd.Series(all_scores)
        top_motifs = final_scores.nlargest(20).index.tolist()

        return top_motifs, final_scores


# Example usage with your data
if __name__ == "__main__":
    # Load your binary motif table
    binary_df = pd.read_csv("output/binary_motif_table.csv", index_col=0)

    # Initialize selector
    selector = MotifSelector(binary_df, taxonomy_col='taxid')

    # Method 1: Mutual Information
    print("=== Mutual Information Selection ===")
    mi_motifs, mi_scores = selector.method2_mutual_information()
    print(f"Top 10 motifs: {mi_motifs[:10]}")

    # Method 2: mRMR
    print("\n=== mRMR Selection ===")
    mrmr_motifs, _ = selector.method3_maximum_relevance_minimum_redundancy(n_motifs=20)
    print(f"Selected motifs: {mrmr_motifs[:10]}")

    # Method 3: Random Forest
    print("\n=== Random Forest Importance ===")
    rf_motifs, rf_scores = selector.method4_random_forest_importance()
    print(f"Top 10 motifs: {rf_motifs[:10]}")

    # Method 4: Co-occurrence modules
    print("\n=== Co-occurrence Modules ===")
    module_reps, modules = selector.method5_co_occurrence_modules()
    print(f"Module representatives: {module_reps[:5]}")

    # Method 5: PCA loadings
    print("\n=== PCA-based Selection ===")
    pca_motifs, variance_explained = selector.method6_pca_loadings()
    print(f"PCA motifs: {pca_motifs[:10]}")
    print(f"Variance explained by first 10 PCs: {variance_explained.sum():.2%}")

    # Method 6: Coverage optimization
    print("\n=== Coverage Optimization ===")
    coverage_motifs, coverage = selector.method8_coverage_optimization(min_coverage=0.8)
    print(f"Motifs needed for 80% coverage: {len(coverage_motifs)}")
    print(f"Actual coverage: {coverage:.2%}")

    # Combined approach
    print("\n=== Combined Method ===")
    combined_motifs, combined_scores = selector.combine_methods()
    print(f"Top 10 combined: {combined_motifs[:10]}")

    # Save results
    pd.Series(mi_motifs[:20]).to_csv("output/mutual_info_motifs.csv", header=False, index=False)
    pd.Series(mrmr_motifs).to_csv("output/mrmr_motifs.csv", header=False, index=False)
    pd.Series(rf_motifs[:20]).to_csv("output/random_forest_motifs.csv", header=False, index=False)
    pd.Series(combined_motifs).to_csv("output/combined_method_motifs.csv", header=False, index=False)