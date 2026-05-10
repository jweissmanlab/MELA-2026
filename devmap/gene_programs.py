import numpy as np
import pandas as pd
from scipy import sparse
import scipy as sp
import hotspot


def create_knn_graph(distances, cell_index, sigma=2.0):
    if not sparse.issparse(distances):
        raise TypeError("Expected a scipy sparse matrix")

    distances = distances.tocsr()

    indptr = distances.indptr
    indices = distances.indices
    data = distances.data

    counts = np.diff(indptr)

    if not np.all(counts == counts[0]):
        # print counts distribution
        print("Counts distribution:", np.unique(counts, return_counts=True))
        raise ValueError("Sparse distance graph has variable numbers of neighbors per row")

    k = counts[0]
    n_cells = distances.shape[0]

    neighbors = indices.reshape(n_cells, k)
    weights = np.exp(-data / sigma).reshape(n_cells, k)

    neighbors = pd.DataFrame(neighbors, index=cell_index)
    weights = pd.DataFrame(weights, index=cell_index, columns=neighbors.columns)

    weights_nr = hotspot.knn.make_weights_non_redundant(neighbors.values, weights.values)
    weights = pd.DataFrame(weights_nr, index=neighbors.index, columns=neighbors.columns)
    return neighbors, weights

def get_clusters(Z, t=5, min_size=10):
    # similarity -> distance
    D = -Z.values
    D = D - D.min()
    np.fill_diagonal(D, 0)

    L = sp.cluster.hierarchy.linkage(sp.spatial.distance.squareform(D, checks=False), method="average")

    clusters = pd.Series(
        sp.cluster.hierarchy.fcluster(L, t=t, criterion="maxclust"),
        index=Z.index,
        name="cluster"
    )

    # merge tiny clusters into nearest non-tiny cluster
    sizes = clusters.value_counts()
    small = sizes[sizes < min_size].index
    large = sizes[sizes >= min_size].index

    if len(small) > 0 and len(large) > 0:
        for c in small:
            genes_c = clusters[clusters == c].index

            best_target = None
            best_dist = np.inf

            for target in large:
                genes_t = clusters[clusters == target].index
                dist = D[np.ix_(
                    Z.index.get_indexer(genes_c),
                    Z.index.get_indexer(genes_t)
                )].mean()

                if dist < best_dist:
                    best_dist = dist
                    best_target = target

            clusters.loc[genes_c] = best_target

        # relabel consecutively
        relabel = {old: i + 1 for i, old in enumerate(pd.unique(clusters))}
        clusters = clusters.map(relabel)
        clusters.name = "cluster"

    return clusters, L

def score_programs(
    adata,
    programs,
    genes_col="genes",
    program_col="program",
    n_bins=25,
    layer=None,
    use_raw=False,
    gene_pool=None,
):
    if use_raw:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names)
    elif layer is not None:
        X = adata.layers[layer]
        var_names = pd.Index(adata.var_names)
    else:
        X = adata.X
        var_names = pd.Index(adata.var_names)

    if gene_pool is None:
        gene_mask = np.ones(len(var_names), dtype=bool)
    else:
        gene_mask = var_names.isin(gene_pool)

    pool_genes = var_names[gene_mask]
    pool_idx = np.where(gene_mask)[0]

    # Mean expression per gene
    if sparse.issparse(X):
        gene_means = np.asarray(X.mean(axis=0)).ravel()
    else:
        gene_means = np.asarray(X.mean(axis=0)).ravel()

    means_pool = gene_means[pool_idx]
    ranked = pd.Series(means_pool, index=pool_genes).rank(method="first")
    bins = pd.qcut(ranked, q=min(n_bins, len(ranked)), labels=False, duplicates="drop")
    bins = bins.astype(int)

    gene_to_idx = pd.Series(np.arange(len(var_names)), index=var_names)
    gene_to_bin = pd.Series(bins.values, index=pool_genes)

    unique_bins = np.sort(gene_to_bin.unique())

    # Precompute per-cell mean expression for each bin
    bin_means = {}
    for b in unique_bins:
        genes_b = gene_to_bin.index[gene_to_bin == b]
        idx_b = gene_to_idx[genes_b].to_numpy()

        if sparse.issparse(X):
            bin_means[b] = np.asarray(X[:, idx_b].mean(axis=1)).ravel()
        else:
            bin_means[b] = X[:, idx_b].mean(axis=1)

    scores = {}

    print("Scoring programs...")

    for row in programs.itertuples(index=False):
        program_name = getattr(row, program_col)
        genes = [g for g in getattr(row, genes_col).split(", ") if g in gene_to_idx.index]
        genes = pd.Index(genes).intersection(pool_genes)

        if len(genes) == 0:
            scores[program_name] = np.full(adata.n_obs, np.nan)
            continue

        prog_idx = gene_to_idx[genes].to_numpy()

        if sparse.issparse(X):
            target_score = np.asarray(X[:, prog_idx].mean(axis=1)).ravel()
        else:
            target_score = X[:, prog_idx].mean(axis=1)

        # Count how many program genes fall in each bin
        bin_counts = gene_to_bin.loc[genes].value_counts().sort_index()

        control_score = np.zeros(adata.n_obs, dtype=float)
        total = bin_counts.sum()

        for b, count in bin_counts.items():
            control_score += count * bin_means[b]

        control_score /= total

        scores[program_name] = target_score - control_score

    return pd.DataFrame(scores, index=adata.obs_names)

def compute_modules_agglomerative(
    cluster_Z,
    min_gene_threshold=8,
    z_threshold=40,
    core_only=True,
):
    """
    Agglomerative module finding based on mean off-diagonal Z between clusters.

    Two clusters A and B are merged if their mean cross-cluster Z score is the
    highest among all active pairs and is >= z_threshold.

    Parameters
    ----------
    cluster_Z : pd.DataFrame
        Symmetric gene x gene local correlation Z matrix
    min_gene_threshold : int
        Minimum cluster size to keep as a module
    z_threshold : float
        Minimum mean cross-cluster Z required to merge two clusters
    core_only : bool
        Kept for compatibility. Small clusters are labeled -1.

    Returns
    -------
    modules : pd.Series
        Module labels for each gene
    linkage : np.ndarray
        Linkage-like matrix with columns:
        [cluster1, cluster2, merge_score, new_cluster_size]
    """

    if not isinstance(cluster_Z, pd.DataFrame):
        raise TypeError("cluster_Z must be a pandas DataFrame")

    genes = cluster_Z.index
    Z = cluster_Z.loc[genes, genes].values.astype(float)
    n = Z.shape[0]

    if Z.shape[0] != Z.shape[1]:
        raise ValueError("cluster_Z must be square")

    # active clusters: cluster_id -> {"members": [...], "size": int}
    clusters = {
        i: {"members": [i], "size": 1}
        for i in range(n)
    }
    active = set(clusters.keys())

    # mean cross-cluster Z for active pairs
    # for singletons this is just Z[i, j]
    scores = {}
    for i in range(n):
        for j in range(i + 1, n):
            scores[(i, j)] = Z[i, j]

    linkage_rows = []
    next_cluster_id = n

    while True:
        # find best active pair
        best_pair = None
        best_score = -np.inf

        for (a, b), score in scores.items():
            if a in active and b in active and score > best_score:
                best_score = score
                best_pair = (a, b)

        if best_pair is None or best_score < z_threshold:
            break

        a, b = best_pair
        size_a = clusters[a]["size"]
        size_b = clusters[b]["size"]

        # create merged cluster
        new_members = clusters[a]["members"] + clusters[b]["members"]
        new_size = size_a + size_b
        new_id = next_cluster_id
        next_cluster_id += 1

        clusters[new_id] = {
            "members": new_members,
            "size": new_size,
        }

        # linkage-like output
        linkage_rows.append([a, b, best_score, new_size])

        # update scores between new cluster and remaining active clusters
        for c in list(active):
            if c in (a, b):
                continue

            key_ac = (a, c) if a < c else (c, a)
            key_bc = (b, c) if b < c else (c, b)

            score_ac = scores[key_ac]
            score_bc = scores[key_bc]

            # weighted average cross-cluster Z
            # this is exact for mean pairwise cross-cluster Z
            new_score = (size_a * score_ac + size_b * score_bc) / (size_a + size_b)

            key_new = (c, new_id) if c < new_id else (new_id, c)
            scores[key_new] = new_score

        # deactivate old clusters, activate new one
        active.remove(a)
        active.remove(b)
        active.add(new_id)

    # assign module labels
    modules = pd.Series(-1, index=genes, dtype=int)
    kept_clusters = []

    for cid in active:
        size = clusters[cid]["size"]
        if size >= min_gene_threshold:
            kept_clusters.append(cid)

    kept_clusters = sorted(kept_clusters, key=lambda cid: -clusters[cid]["size"])

    for module_id, cid in enumerate(kept_clusters, start=1):
        gene_idx = clusters[cid]["members"]
        modules.iloc[gene_idx] = module_id

    linkage = np.array(linkage_rows, dtype=float)
    if linkage.size == 0:
        linkage = np.zeros((0, 4), dtype=float)

    return modules.rename("Module"), linkage