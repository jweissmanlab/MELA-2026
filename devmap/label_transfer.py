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
    fallback="keep",
    mode="impute",  # "impute" or "smooth"
    smooth_min_fold_change=1.0,
    copy=False,
):
    """
    Impute or smooth categorical labels in ``adata.obs`` by weighted neighbor voting.

    For each target cell, the labels of its neighbors in
    ``adata.obsp[connectivity_key]`` are tallied by summed edge weight, and
    the label with the highest total weight wins (ties broken by neighbor
    count, then by string order). Only donor cells contribute votes: donors
    must have a non-missing label and, if given, pass *restrict_to_mask*.

    Parameters
    ----------
    adata : AnnData
        Annotated data with a cell-cell graph in ``.obsp``.
    obs_key : str
        Column in ``adata.obs`` holding the labels to impute or smooth.
    connectivity_key : str
        Key in ``adata.obsp`` of the (weighted) neighbor graph. Defaults to
        "connectivities".
    missing_values : tuple
        Values treated as missing in addition to NA (``pd.isna``), e.g. "".
        Defaults to (np.nan, None, "").
    restrict_to_mask : str | array-like of bool | None
        Additional donor restriction: either a boolean column name in
        ``adata.obs`` or a boolean array of length ``adata.n_obs``. If None,
        all non-missing cells are donors.
    add_result_key : str | None
        Column to write results to. If None, *obs_key* is overwritten.
    fallback : object
        Value assigned when a target cell has no neighbors or no valid donor
        neighbors. "keep" (default) leaves the current value unchanged.
    mode : {"impute", "smooth"}
        "impute" only relabels cells with missing values. "smooth" considers
        every cell: missing cells always take the winning label; non-missing
        cells switch only if the winning label differs from the current one
        and its weight is at least ``smooth_min_fold_change`` times the
        weight supporting the current label. Defaults to "impute".
    smooth_min_fold_change : float
        Fold-change threshold for relabeling non-missing cells in "smooth"
        mode. Defaults to 1.0.
    copy : bool
        If True, operate on and return a copy of *adata*. Defaults to False.

    Returns
    -------
    AnnData | pd.Series
        If *copy* is True, the modified copy of *adata*; otherwise the
        resulting object-dtype ``adata.obs[add_result_key or obs_key]``
        column (written in place).

    Raises
    ------
    ValueError
        If *mode* is invalid or *restrict_to_mask* does not have length
        ``adata.n_obs``.
    KeyError
        If *connectivity_key* is not in ``adata.obsp``.
    """
    if mode not in {"impute", "smooth"}:
        raise ValueError("mode must be 'impute' or 'smooth'")

    if connectivity_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{connectivity_key}'] not found")

    ad = adata.copy() if copy else adata

    W = ad.obsp[connectivity_key]
    W = W.tocsr() if sparse.issparse(W) else sparse.csr_matrix(W)

    vals = ad.obs[obs_key].astype(object).copy()
    x = vals.to_numpy()

    missing = pd.isna(x)
    for mv in missing_values:
        if mv is np.nan:
            continue
        if mv is None:
            missing |= pd.isna(x)
        else:
            missing |= x == mv

    if restrict_to_mask is None:
        donor_ok = ~missing
    elif isinstance(restrict_to_mask, str):
        donor_ok = ad.obs[restrict_to_mask].astype(bool).to_numpy() & (~missing)
    else:
        donor_ok = np.asarray(restrict_to_mask, dtype=bool) & (~missing)

    if donor_ok.shape[0] != ad.n_obs:
        raise ValueError("restrict_to_mask must have length adata.n_obs")

    out = x.copy()

    target_idx = np.where(missing)[0] if mode == "impute" else np.arange(ad.n_obs)

    for i in target_idx:
        row = W.getrow(i)

        if row.nnz == 0:
            if fallback != "keep":
                out[i] = fallback
            continue

        nbr_idx = row.indices
        nbr_w = row.data

        valid = donor_ok[nbr_idx]

        # In smooth mode, optionally let the cell keep its own label if present
        # and self-edges exist in the graph. Otherwise it is smoothed purely
        # from neighbors.
        if not np.any(valid):
            if fallback != "keep":
                out[i] = fallback
            continue

        nbr_idx = nbr_idx[valid]
        nbr_w = nbr_w[valid]
        nbr_labels = x[nbr_idx]

        weight_by_label = {}
        count_by_label = {}

        for label, w in zip(nbr_labels, nbr_w, strict=False):
            weight_by_label[label] = weight_by_label.get(label, 0.0) + float(w)
            count_by_label[label] = count_by_label.get(label, 0) + 1

        best_label = sorted(
            weight_by_label.keys(),
            key=lambda lab: (
                -weight_by_label[lab],
                -count_by_label[lab],
                str(lab),
            ),
        )[0]

        if mode == "smooth":
            current_label = x[i]

            # Missing current labels can always be replaced.
            if missing[i]:
                out[i] = best_label
            else:
                current_weight = weight_by_label.get(current_label, 0.0)
                best_weight = weight_by_label[best_label]

                if best_label != current_label and best_weight >= smooth_min_fold_change * current_weight:
                    out[i] = best_label
                else:
                    out[i] = current_label
        else:
            out[i] = best_label

    result_key = add_result_key or obs_key
    ad.obs[result_key] = pd.Series(out, index=ad.obs_names, dtype="object")

    return ad if copy else ad.obs[result_key]
