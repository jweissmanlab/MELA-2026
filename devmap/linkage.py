import numpy as np
import pandas as pd


def symmetrize_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Symmetrize a square matrix by averaging mirrored off-diagonal entries.

    Each pair of entries ``(i, j)`` and ``(j, i)`` is replaced by their mean.
    The diagonal is left unchanged. Values are cast to float.

    Parameters
    ----------
    df : pd.DataFrame
        Square matrix to symmetrize.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix with the same index and columns as ``df``.

    Raises
    ------
    ValueError
        If ``df`` is not square.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("DataFrame must be square.")

    arr = df.values.astype(float).copy()
    n = arr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            m = (arr[i, j] + arr[j, i]) / 2.0
            arr[i, j] = m
            arr[j, i] = m
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def get_mean_linkage(results_path, groupby, embryos, weighted=False):
    """
    Aggregate per-embryo ancestral linkage statistics across embryos.

    Reads ``results_path / groupby / f"{embryo}_linkage.csv"`` for each embryo,
    computes ``norm_value = value - permuted_value``, and aggregates the
    statistics for each ``(source, target)`` pair.

    Parameters
    ----------
    results_path : Path
        Directory containing per-``groupby`` linkage result subfolders.
    groupby : str
        Grouping key used for linkage (e.g. ``"cell_type"``); also the name of
        the subfolder of ``results_path`` to read from.
    embryos : list of str
        Embryo IDs whose linkage CSVs are aggregated. Each CSV must contain
        ``source``, ``target``, ``value``, ``permuted_value``, ``z_score``,
        ``p_value``, ``source_n``, and ``target_n`` columns.
    weighted : bool
        If False, take the unweighted mean of ``value``, ``norm_value``, and
        ``z_score``, the maximum ``p_value``, and the variance of
        ``norm_value``. If True, take means of ``value``, ``norm_value``,
        ``z_score``, and ``p_value`` weighted by ``target_n`` (the weighted
        p-value is returned as ``p_value_mean``), and the unweighted variance of
        ``norm_value``.

    Returns
    -------
    pd.DataFrame
        One row per ``(source, target)`` pair with columns ``source``,
        ``target``, ``value``, ``norm_value``, ``z_score``, ``norm_value_var``,
        ``p_value`` (or ``p_value_mean`` if ``weighted``), and the summed
        ``source_n`` and ``target_n``.
    """
    embryo_stats = []

    for embryo in embryos:
        stats = pd.read_csv(results_path / groupby / f"{embryo}_linkage.csv", index_col=0)
        embryo_stats.append(stats.assign(embryo=embryo))

    embryo_stats = pd.concat(embryo_stats)

    embryo_stats["norm_value"] = embryo_stats["value"] - embryo_stats["permuted_value"]

    if not weighted:
        mean_stats = (
            embryo_stats.groupby(["source", "target"])
            .agg(
                value=("value", "mean"),
                norm_value=("norm_value", "mean"),
                z_score=("z_score", "mean"),
                norm_value_var=("norm_value", "var"),
                p_value=("p_value", "max"),
                source_n=("source_n", "sum"),
                target_n=("target_n", "sum"),
            )
            .reset_index()
        )

    else:
        value_cols = ["value", "norm_value", "z_score", "p_value"]
        w = embryo_stats["target_n"]
        for col in value_cols:
            mask = embryo_stats[col].notna() & w.notna()
            embryo_stats[f"{col}_wx"] = embryo_stats[col].where(mask, 0) * w.where(mask, 0)
            embryo_stats[f"{col}_w"] = w.where(mask, 0)

        grouped = embryo_stats.groupby(["source", "target"])

        sums = grouped[
            [f"{col}_wx" for col in value_cols] + [f"{col}_w" for col in value_cols] + ["source_n", "target_n"]
        ].sum()

        mean_stats = pd.DataFrame(index=sums.index)

        for col in value_cols:
            mean_stats[col if col != "p_value" else "p_value_mean"] = (sums[f"{col}_wx"] / sums[f"{col}_w"]).replace(
                [np.inf, -np.inf], np.nan
            )

        mean_stats["norm_value_var"] = grouped["norm_value"].var()
        mean_stats["source_n"] = sums["source_n"]
        mean_stats["target_n"] = sums["target_n"]
        mean_stats = mean_stats.reset_index()

    return mean_stats
