import numpy as np
import pandas as pd
from scipy import sparse

def impute_obs_from_connectivities(
    adata,
    obs_key,
    connectivity_key="connectivities",
    missing_values=(np.nan, None, ""),
    restrict_to_mask=None,
    add_result_key=None,
    fallback="keep",  # "keep", "unknown", or any scalar/string
    copy=False,
):
    """
    Impute missing categorical/string values in adata.obs[obs_key] using
    weighted neighbor voting from adata.obsp[connectivity_key].

    For each missing cell i:
      - find neighbors j with non-missing labels
      - optionally restrict donor neighbors with restrict_to_mask
      - sum connectivity weights by label
      - assign the label with maximum total weight

    Parameters
    ----------
    adata : AnnData
    obs_key : str
        Column in adata.obs to impute.
    connectivity_key : str
        Key in adata.obsp containing the neighbor graph.
    missing_values : tuple
        Values to treat as missing.
    restrict_to_mask : None, str, or boolean array-like
        Restrict donor cells.
        - None: all non-missing cells can donate
        - str: use adata.obs[restrict_to_mask].astype(bool)
        - array-like: boolean mask of length adata.n_obs
    add_result_key : str or None
        If provided, write imputed result there. Otherwise overwrite obs_key.
    fallback : str or scalar
        What to assign if a cell has no valid labeled neighbors.
        - "keep": leave as-is
        - anything else: fill with that value
    copy : bool
        If True, return a copy of adata with results written.

    Returns
    -------
    pd.Series or AnnData
        Imputed Series, or AnnData if copy=True.
    """
    if connectivity_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{connectivity_key}'] not found")

    ad = adata.copy() if copy else adata

    W = ad.obsp[connectivity_key]
    W = W.tocsr() if sparse.issparse(W) else sparse.csr_matrix(W)

    vals = ad.obs[obs_key].astype(object).copy()
    x = vals.to_numpy()

    # Missing mask
    missing = pd.isna(x)
    for mv in missing_values:
        if mv is np.nan:
            continue
        if mv is None:
            missing |= pd.isna(x)
        else:
            missing |= (x == mv)

    # Donor restriction
    if restrict_to_mask is None:
        donor_ok = ~missing
    elif isinstance(restrict_to_mask, str):
        donor_ok = ad.obs[restrict_to_mask].astype(bool).to_numpy() & (~missing)
    else:
        donor_ok = np.asarray(restrict_to_mask, dtype=bool) & (~missing)

    if donor_ok.shape[0] != ad.n_obs:
        raise ValueError("restrict_to_mask must have length adata.n_obs")

    out = x.copy()
    donor_idx = np.where(donor_ok)[0]

    for i in np.where(missing)[0]:
        row = W.getrow(i)
        if row.nnz == 0:
            if fallback != "keep":
                out[i] = fallback
            continue

        nbr_idx = row.indices
        nbr_w = row.data

        valid = donor_ok[nbr_idx]
        if not np.any(valid):
            if fallback != "keep":
                out[i] = fallback
            continue

        nbr_idx = nbr_idx[valid]
        nbr_w = nbr_w[valid]
        nbr_labels = x[nbr_idx]

        # Weighted vote by label
        weight_by_label = {}
        for label, w in zip(nbr_labels, nbr_w):
            weight_by_label[label] = weight_by_label.get(label, 0.0) + w

        # Break ties deterministically by:
        # 1. highest total weight
        # 2. highest raw count
        # 3. lexical order of label as string
        count_by_label = {}
        for label in nbr_labels:
            count_by_label[label] = count_by_label.get(label, 0) + 1

        best_label = sorted(
            weight_by_label.keys(),
            key=lambda lab: (-weight_by_label[lab], -count_by_label[lab], str(lab))
        )[0]

        out[i] = best_label

    result_key = add_result_key or obs_key
    ad.obs[result_key] = pd.Series(out, index=ad.obs_names, dtype="object")

    return ad if copy else ad.obs[result_key]