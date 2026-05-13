
import pandas as pd

def symmetrize_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace entries above/below diagonal with their mean,
    so the result is symmetric.
    The diagonal itself is left unchanged.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("DataFrame must be square.")

    arr = df.values.astype(float).copy()
    n = arr.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            m = (arr[i, j] + arr[j, i]) / 2.0
            arr[i, j] = m
            arr[j, i] = m
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def get_mean_linkage(results_path, groupby, embryos, weighted=False):
    embryo_stats = []

    for embryo in embryos:
        stats = pd.read_csv(results_path / groupby / f"{embryo}_linkage.csv", index_col=0)
        embryo_stats.append(stats.assign(embryo=embryo))

    embryo_stats = pd.concat(embryo_stats)

    embryo_stats["norm_value"] = -(embryo_stats["value"] - embryo_stats["permuted_value"]) / 2

    if not weighted:
        mean_stats = embryo_stats.groupby(["source", "target"]).agg(
            value=("value", "mean"),
            norm_value=("norm_value", "mean"),
            z_score=("z_score", "mean"),
            norm_value_var=("norm_value", "var"),
            p_value=("p_value", "mean"),
            source_n=("source_n", "sum"),
            target_n=("target_n", "sum"),
        ).reset_index()

    else:
        value_cols = ["value", "norm_value", "z_score", "p_value"]
        w = embryo_stats["target_n"]
        for col in value_cols:
            mask = embryo_stats[col].notna() & w.notna()
            embryo_stats[f"{col}_wx"] = embryo_stats[col].where(mask, 0) * w.where(mask, 0)
            embryo_stats[f"{col}_w"] = w.where(mask, 0)

        grouped = embryo_stats.groupby(["source", "target"])

        sums = grouped[
            [f"{col}_wx" for col in value_cols] +
            [f"{col}_w" for col in value_cols] +
            ["source_n", "target_n"]
        ].sum()

        mean_stats = pd.DataFrame(index=sums.index)

        for col in value_cols:
            mean_stats[col if col != "p_value" else "p_value_mean"] = (
                sums[f"{col}_wx"] / sums[f"{col}_w"]
            ).replace([np.inf, -np.inf], np.nan)

        mean_stats["norm_value_var"] = grouped["norm_value"].var()
        mean_stats["source_n"] = sums["source_n"]
        mean_stats["target_n"] = sums["target_n"]
        mean_stats = mean_stats.reset_index()

    return mean_stats



