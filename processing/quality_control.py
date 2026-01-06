# Setup
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycea as py
import scanpy as sc
import anndata as ad
import treedata as td
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import tracertools
import warnings

from tracertools.config import colors,sequential_cmap, set_theme, edit_ids, edit_palette

site_names = ["EMX1","HEK3","RNF2"]
set_theme()

#### Data loading functions ####
def load_tdata(path:Path,captures:dict) -> td.TreeData:
    """Load TreeData from multiple captures and concatenate them."""
    print("Loading 10x data...")
    tdata = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for capture, name in captures.items():
            capture_tdata = td.TreeData(sc.read_10x_h5(path / capture / f"{capture}_gex_filtered_counts.h5"))
            capture_tdata.var_names_make_unique()
            capture_tdata = capture_tdata[:, ~capture_tdata.var.index.str.startswith('intID')].copy()
            capture_tdata.obs["capture"] = name
            capture_tdata.obs["cellBC"] = (name + "-" + capture_tdata.obs.index)
            capture_tdata.obs["embryo"] = name.split("-T")[0]
            capture_tdata.obs.index = capture_tdata.obs["cellBC"].values
            tdata.append(capture_tdata)
        tdata = td.concat(tdata, join='outer', merge = "same")
        tdata.var.drop(columns=["feature_types","genome"], inplace=True)
    return tdata

def load_alleles(path:Path,captures:dict,tdata:td.TreeData) -> pd.DataFrame:
    """Load allele counts from multiple captures and concatenate them."""
    print("Loading target site allele counts...")
    alleles = []
    for capture, name in captures.items():
        capture_alleles = pd.read_csv(path / capture / f"{capture}_allele_counts.csv")
        capture_alleles["capture"] = name
        capture_alleles["cellBC"] = (name + "-" + capture_alleles["cellBC"])
        alleles.append(capture_alleles)
    alleles = pd.concat(alleles).query("cellBC in @tdata.obs_names").copy()
    tdata.obs['total_ts_counts'] = alleles.groupby("cellBC")["UMI"].sum()
    tdata.obs['total_ts_counts'] = tdata.obs['total_ts_counts'].fillna(0)
    for site in site_names:
        alleles[site] = alleles[site].str.replace("-","*")
    return alleles.reset_index(drop=True)

def load_snp_counts(path:Path,captures:dict) -> pd.DataFrame:
    """Load SNP counts from multiple captures and concatenate them."""
    print("Loading SNP counts...")
    alt_counts = []
    ref_counts = []
    barcodes = []
    for capture, name in captures.items():
        capture_barcodes = pd.read_csv(path / capture / f"{capture}_gex_barcodes.tsv.gz", header=None)[0].tolist()
        capture_alt = sc.read_mtx(path / capture / f"{capture}_alt_counts.mtx").T
        capture_ref = sc.read_mtx(path / capture / f"{capture}_ref_counts.mtx").T
        barcodes.extend([f"{name}-{bc}" for bc in capture_barcodes])
        alt_counts.append(capture_alt)
        ref_counts.append(capture_ref)
    alt_counts = ad.concat(alt_counts)
    alt_counts.obs_names = barcodes
    ref_counts = ad.concat(ref_counts)
    ref_counts.obs_names = barcodes
    snp_counts = ad.concat([ref_counts, alt_counts], axis=1, keys = ["ref","alt"],index_unique="_")
    return snp_counts

#### Quality control functions ####
def filter_cells(tdata:td.TreeData, plot_path: Path = None) -> pd.DataFrame:
    print("Filtering low quality cells...")
    tdata.obs["type"] = "host"
    tdata.var['mito'] = tdata.var_names.str.startswith('mt-')
    tdata.obs["total_mito_counts"] = tdata[:,tdata.var['mito']].X.sum(axis=1)
    sc.pp.calculate_qc_metrics(tdata, percent_top=None, log1p=False, inplace=True)
    tdata.obs["pct_mito_counts"] = tdata.obs["total_mito_counts"]/tdata.obs["total_counts"]*100 
    # Cells with abnormal mitochondrial percentage are low quality
    cells = tdata.obs.copy()
    mito_cutoff = tracertools.seq.sigma_threshold(cells.query("total_mito_counts > 5")["total_mito_counts"],log = True, max_sigma = 3)
    cutoff = tracertools.seq.sigma_threshold(cells.query("total_mito_counts > @mito_cutoff")["total_counts"],log = True, max_sigma = 3)
    cells["type"] = ((cells["total_mito_counts"] > mito_cutoff) & (cells["total_counts"] > cutoff)).map({True:"host", False:"low_quality"})
    # Scatter plot
    sns.scatterplot(data = cells.sample(min(10000, len(cells)),replace=True), x = "total_counts", y = "pct_mito_counts",hue = "type",
                legend=False,alpha = 0.2,s = 10)
    plt.xscale("log")
    plt.ylim(0,3)
    plt.savefig(plot_path / "low_quality_cells.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    cells.drop(columns=["pct_mito_counts"], inplace=True)
    return cells


def filter_ts_reads(alleles:pd.DataFrame, plot_path: Path = None) -> pd.DataFrame:
    """Filter target site reads based on UMI and read counts."""
    print("Filtering target site reads...")
    alleles["reads_per_umi"] = alleles["readCount"]/alleles["UMI"]
    filtered_alleles = alleles.query("UMI > 4 & reads_per_umi > 3").copy()
    filtered_alleles["relative_read_support"] = filtered_alleles["reads_per_umi"] / filtered_alleles.groupby(["intID","cellBC"])["reads_per_umi"].transform("max")
    filtered_alleles = filtered_alleles.query("relative_read_support > 0.25 | reads_per_umi >= 10").copy()
    filtered_alleles["relative_umi_support"] = filtered_alleles["UMI"] / filtered_alleles.groupby(["intID","cellBC"])["UMI"].transform("max")
    filtered_alleles = filtered_alleles.query("relative_umi_support > 0.25").copy()
    alleles["keep"] = False
    alleles.loc[filtered_alleles.index, "keep"] = True
    sns.scatterplot(data = alleles.sample(min(10000, len(alleles))), x = "UMI", y = "reads_per_umi",
                        hue = "keep",size = .1,alpha = .2,legend=False)
    plt.yscale("log")
    plt.xscale("log")
    total_reads = alleles['readCount'].sum()
    total_usable = alleles.loc[alleles['keep'], 'readCount'].sum()
    # add usable read fraction annotation
    plt.title(f'Usable Read Fraction: {total_usable/total_reads:.2%}')
    plt.savefig(plot_path / "ts_umi_vs_reads.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    return filtered_alleles.drop(columns=["relative_read_support","relative_umi_support"]).copy()

def distinguish_donor_vs_host(cells:pd.DataFrame, alleles:pd.DataFrame, snp_counts:td.TreeData, tdata:td.TreeData, plot_path:Path=None) -> pd.DataFrame:
    """Distinguish donor vs host cells based on SNPs and target site counts."""
    print("Distinguishing donor vs host cells...")
    snp_counts = snp_counts.copy()
    cells = cells.copy()
    cells['total_ts_counts'] = alleles.groupby("cellBC")["UMI"].sum()
    cells['total_ts_counts'] = cells['total_ts_counts'].fillna(0)
    subset = cells.query("type != 'low_quality'").copy()
    subset["type"] = "host"
    donor_seed = subset[subset["total_ts_counts"] > np.percentile(subset["total_ts_counts"], 95)].index
    host_seed = subset[subset["total_ts_counts"] <= np.percentile(subset["total_ts_counts"], 5)].index
    # Correct for ambient RNA
    snp_counts = snp_counts[subset.index].copy()
    snp_counts.X = np.array(snp_counts.X.toarray())
    correction = np.round(np.percentile(snp_counts.X, 95, axis = 0)/ 10)
    snp_counts.X = np.clip(snp_counts.X - correction, 0, None)
    # Quantify host SNP counts
    host_snps = snp_counts[host_seed].X.sum(axis=0) / (snp_counts[donor_seed].X.sum(axis=0) + 1)
    host_snps = np.where(host_snps > 10)[0]
    subset["host_snp_counts"] = snp_counts[:,host_snps].X.sum(axis=1)
    # Get Xist counts
    subset["Xist_counts"] = sc.get.obs_df(tdata, "Xist")
    # Identify donor cells
    subset["ts_ratio"] = ((subset["total_ts_counts"]/subset.groupby("capture",observed=False)["total_ts_counts"].transform("mean")) /
                          (subset["total_counts"]/subset.groupby("capture",observed=False)["total_counts"].transform("mean")))
    subset.loc[subset["ts_ratio"] > .2,"type"] = "donor"
    # Identify doublets using SNPs
    if subset["host_snp_counts"].mean() > 10:
        doublet_cutoff = tracertools.seq.sigma_threshold(subset.query("type == 'host' & host_snp_counts > 1")["host_snp_counts"],log = True, max_sigma = 3,n_components=1)
        if doublet_cutoff < 5:
            doublet_cutoff = 5
        subset.loc[(subset["host_snp_counts"] > doublet_cutoff) & (subset["type"] == "donor"),"type"] = "mixed_doublet"
        sns.scatterplot(subset, x = "total_ts_counts", y = "host_snp_counts", hue = "type", alpha = 0.5, s = 10)
        plt.yscale("log")
        plt.xscale("log")
    # Identify doublets using Xist  expression
    elif subset["Xist_counts"].max() > 20:
        doublet_cutoff = 5
        subset.loc[(subset["Xist_counts"] > doublet_cutoff) & (subset["type"] == "donor"),"type"] = "mixed_doublet"
        sns.scatterplot(data = subset.query("type != 'low_quality'"), y = "Xist_counts", x = "total_ts_counts", hue = "type", alpha = 0.5, s = 10)
        plt.xscale("log")
        plt.yscale("log")
    else:
        sns.histplot(data = subset.query("ts_ratio < 5"), x = "ts_ratio", hue = "capture", 
                    bins = 100,multiple="stack",linewidth=0,alpha = 1)
        plt.axvline(x=.1, color="black", linestyle="--")
        plt.xlabel("Normalized TS/GEX ratio");
        plt.ylabel("Frequency");
    cells["host_snp_counts"] = 0
    cells.update(subset[["type","host_snp_counts"]])
    plt.savefig(plot_path / "donor_vs_host_cells.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    return cells


def detection_rate_filter(cells: pd.DataFrame, alleles: pd.DataFrame,blacklist: set = {"intID888"}, min_detection: float = 0.75, plot_path: Path = None) -> pd.DataFrame:
    """Filter cells and integrations based on intBC detection rate."""
    print("Filtering cells and integrations based on detection rate...")
    cells = cells.copy()
    alleles = alleles.copy()
    donor_cells = cells.query("type == 'donor'").index
    alleles = alleles[alleles["cellBC"].isin(donor_cells)].copy()
    per_cell_counts = alleles.groupby(["cellBC","intID"])["UMI"].sum().unstack().fillna(0)
    intBC_detection = (per_cell_counts > 0).mean(axis = 0)
    intBC_whitelist = list(set(intBC_detection.index[intBC_detection > 0.4].tolist()) - blacklist)
    cells["detection_rate"] = (per_cell_counts.loc[:,intBC_whitelist] > 0).mean(axis = 1)
    intBC_detection = (per_cell_counts.loc[:,intBC_whitelist] > 0).mean(axis = 1)
    remove_cells = intBC_detection.index[intBC_detection < min_detection].tolist()
    sns.histplot(intBC_detection, bins = 15,linewidth=0,alpha = 1)
    plt.axvline(x=min_detection, color="black", linestyle="--")
    plt.xlabel("detection_rate");
    plt.savefig(plot_path / "detection_rate.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    return cells, alleles[alleles["intID"].isin(intBC_whitelist) & ~alleles["cellBC"].isin(remove_cells)].copy()


def identify_donor_doublets(cells: pd.DataFrame, alleles: pd.DataFrame, max_frac = .03, plots_path: Path = None) -> pd.DataFrame:
    """Identify donor doublets based on allele conflicts."""
    print("Identifying donor doublets...")
    cells = cells.copy()
    # Initial doublet detection
    total_reads = alleles.groupby(["cellBC","intID"], observed=True)["readCount"].sum().unstack().fillna(0)
    total_conflicts = alleles.sort_values("readCount").groupby(["cellBC","intID"], observed=True).head(-1).groupby(
        ["cellBC","intID"], observed=True)["readCount"].sum().unstack().reindex(total_reads.index).fillna(0)
    cell_conflicts = total_conflicts.sum(axis = 1) / total_reads.sum(axis = 1)
    low_conflict_cells = cell_conflicts[cell_conflicts < 0.1].index
    int_conflict = total_conflicts.loc[low_conflict_cells].sum(axis=0) / total_reads.loc[low_conflict_cells].sum(axis=0)
    low_conflict_ints = int_conflict[int_conflict < 0.01].index
    cell_conflicts = total_conflicts[low_conflict_ints].sum(axis = 1) / total_reads[low_conflict_ints].sum(axis = 1)
    cells["allele_conflicts"] = cell_conflicts
    sns.histplot(cells, x = "allele_conflicts",bins = 50,hue = "capture"
                ,multiple = "stack",linewidth=0,alpha = 1, legend = False)
    plt.ylim(0,1000)
    plt.axvline(x=max_frac, color="black", linestyle="--")
    plt.savefig(plots_path / "donor_doublets.png",bbox_inches='tight') if plots_path else plt.show()
    plt.clf()
    cells.loc[cells["allele_conflicts"] > max_frac,"type"] = "donor_doublet"
    return cells


def greedy_reconstruction(cells: pd.DataFrame, alleles: pd.DataFrame, plot_path: Path = None) -> pd.DataFrame:
    """Greedly reconstruct tree stump and flag doublets."""
    # Greedy reconstruction
    cells = cells.copy()
    donor_cells = cells.query("type == 'donor'").index
    unique_alleles = alleles.query("cellBC in @donor_cells").sort_values(
        "frac",ascending = False).groupby(["intID","cellBC"]).first().reset_index()
    characters = tracertools.seq.alleles_to_characters(unique_alleles, edit_ids=edit_ids)
    stump_tdata = td.TreeData(obsm = {"characters":characters}, obs= pd.DataFrame(index = characters.index))
    stump, greedy_doublets = tracertools.solver.n_mutation_greedy(characters,connect_leaves=True, 
                                                                jaccard_threshold = .9, overlap_threshold = 0.9)
    stump_tdata.obst["stump"] = stump
    stump_tdata.obs["doublet"] = stump_tdata.obs_names.isin(greedy_doublets)
    stump_tdata.obs["clade"] = py.get.node_df(stump_tdata, tree = "stump").query("doublet.isna()")['parent']
    cells.loc[greedy_doublets,"type"] = "donor_doublet"
    cells["clade"] = stump_tdata.obs["clade"]
    fig, ax = plt.subplots(figsize=(5,5), dpi = 500)
    py.pl.tree(stump_tdata,tree = "stump",keys = "characters", palette=edit_palette, ax =ax)
    py.pl.annotation(stump_tdata,keys = "clade", width = .3, legend = False)
    py.pl.annotation(stump_tdata,keys = "doublet",palette = ["lightgrey",colors[2]], width = .3, legend = False)
    plt.savefig(plot_path / "greedy_reconstruction.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    stump_tdata = stump_tdata[stump_tdata.obs.doublet == False].copy()
    name = cells.embryo.unique()[0]
    stumps = {f"{name}-C{i+1}":stump for i, stump in enumerate(tracertools.tree.split_tree(stump_tdata.obst["stump"]))}
    stump_tdata.obs["clone"] = pd.NA
    cells["clone"] = pd.NA
    for name, stump in stumps.items():
        leaves = tracertools.tree.get_leaves(stump)
        stump_tdata.obs.loc[leaves, "clone"] = name
        cells.loc[leaves, "clone"] = name
        stump_tdata.obst[name] = stump
    del stump_tdata.obst["stump"]
    stump_tdata.obs.drop(columns=["doublet"], inplace=True)
    py.pp.add_depth(stump_tdata)
    return cells, stump_tdata


def filter_clade_integrations(cells: pd.DataFrame,alleles: pd.DataFrame, max_frac = .1, plot_path: Path = None) -> pd.DataFrame:
    """For each clade, exclude integrations with high conflict fraction."""
    alleles = alleles.copy()
    donor_cells = cells.query("type == 'donor'").index
    alleles["clade"] = alleles["cellBC"].map(cells["clade"])
    alleles = alleles[alleles["cellBC"].isin(donor_cells)].copy()
    int_conflicts = (alleles.sort_values("readCount").groupby(["intID","cellBC","clade"],observed=True).head(-1).groupby(
                ["intID","clade"],observed=True)["readCount"].sum() / 
                alleles.groupby(["intID","clade"],observed=True)["readCount"].sum()).fillna(0).reset_index(name="conflict_frac")
    sns.heatmap(int_conflicts.pivot(index="intID", columns="clade", values="conflict_frac"), cmap = sequential_cmap,vmax = .2)
    plt.savefig(plot_path / "clade_integration_conflicts.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    alleles = alleles[~alleles.set_index(['intID', 'clade']).index.isin(
        int_conflicts.query("conflict_frac > @max_frac").set_index(['intID', 'clade']).index)].copy()
    return alleles

def identify_host_doublets(cells: pd.DataFrame, plot_path: Path = None) -> pd.DataFrame:
    """Dynamically determine host doublet UMI threshold based on donor doublets."""
    subset = cells.query("type != 'low_quality'").copy()
    subset["norm_counts"] = subset["total_counts"] / subset.groupby("capture")["total_counts"].transform("median")
    donor_cells = subset.query("type.isin(['donor','donor_doublet','mixed_doublet'])").copy()
    donor_cells["is_doublet"] =  donor_cells["type"].str.contains("doublet")
    donor_cells = donor_cells.sort_values("norm_counts", ascending=True)
    donor_cells["doublet_frac"] = (
        donor_cells["is_doublet"]
        .rolling(window=201, center=True, min_periods=1)
        .mean()
    )
    threshold = donor_cells[donor_cells["doublet_frac"] > 0.5].head(1).norm_counts.values[0]
    subset.loc[(subset["type"] == "host") & (subset["norm_counts"] > threshold), "type"] = "host_doublet"
    if "mixed_doublet" not in subset["type"].unique():
        subset.loc[(subset["type"] == "donor")  & (subset["norm_counts"] > threshold), "type"] = "mixed_doublet"
    sns.kdeplot(subset.query("type != 'low_quality'"), x = "total_counts",hue = "type",log_scale=True)
    plt.savefig(plot_path / "host_doublets.png",bbox_inches='tight') if plot_path else plt.show()
    plt.clf()
    cells.update(subset[["type"]])
    return cells


def plot_doublet_counts(cells: pd.DataFrame, plot_path: Path = None):
    cells.query("type != 'low_quality'").groupby(['capture', 'type'],observed=False).size().unstack(fill_value=0).plot(kind='bar', stacked=True)
    plt.legend(loc='center left', title='cell type', bbox_to_anchor=(1, 0.5))
    plt.savefig(plot_path / "doublet_counts.png", bbox_inches='tight') if plot_path else plt.show()
    plt.clf()

def main():
    parser = argparse.ArgumentParser(
        description="Embryo 10x scRNA-seq Quality Control"
    )
    parser.add_argument("-i", "--input",help="Input file path",required=True)
    parser.add_argument("-o", "--output",help="Output file path",required=True)
    parser.add_argument("-n", "--name",help="Sample name",required=True)
    parser.add_argument("-p", "--prefix",help="Sample prefix",required=True)
    parser.add_argument("-c", "--captures",help="Number of captures",type=int,required=True)

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    name_lower = args.name.lower().replace("-","_")
    output_path = output_path / name_lower
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path = output_path / f"qc_plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    # Load data
    captures = {f"{args.prefix}_t{i}":f"{args.name}-T{i}" for i in range(1, args.captures+1)}
    tdata = load_tdata(input_path, captures)
    alleles = load_alleles(input_path, captures, tdata)
    snp_counts = load_snp_counts(input_path, captures)

    # Quality control
    cells = filter_cells(tdata, plot_path=plots_path)
    alleles = alleles[alleles["cellBC"].isin(cells.query("type != 'low_quality'")["cellBC"])].copy()
    alleles = filter_ts_reads(alleles, plot_path=plots_path)
    cells = distinguish_donor_vs_host(cells, alleles, snp_counts, tdata, plot_path=plots_path)
    cells, alleles = detection_rate_filter(cells, alleles, plot_path=plots_path)
    alleles["frac"] = alleles["UMI"] / alleles.groupby(["cellBC", "intID"])["UMI"].transform("sum")
    alleles = tracertools.seq.resolve_alleles_parallel(alleles)
    cells = identify_donor_doublets(cells, alleles, max_frac = .03, plots_path=plots_path)
    cells, stump_tdata = greedy_reconstruction(cells, alleles, plot_path=plots_path)
    alleles = filter_clade_integrations(cells, alleles, plot_path=plots_path)
    cells = identify_host_doublets(cells,plot_path=plots_path)
    plot_doublet_counts(cells, plot_path=plots_path)

    # Save outputs
    cells.to_csv(output_path / f"{name_lower}_droplets.csv")
    cells.drop(columns=['total_mito_counts', 'total_ts_counts', 'host_snp_counts', 'allele_conflicts'], inplace=True)
    unique_alleles = alleles.sort_values("frac",ascending = False).groupby(["intID","cellBC"]).first().reset_index()
    characters = tracertools.seq.alleles_to_characters(unique_alleles, edit_ids=edit_ids)
    stump_tdata.obsm["characters"] = characters.loc[stump_tdata.obs_names]
    stump_tdata.obs = cells.loc[stump_tdata.obs_names].copy()
    stump_tdata = stump_tdata[stump_tdata.obs["type"] == "donor"].copy()
    stump_tdata.write_h5td(output_path / f"{name_lower}_stump.h5td")
    tdata.obs = cells.loc[tdata.obs_names].copy()
    tdata = tdata[tdata.obs["type"].isin(["donor","host"])].copy()
    tdata.write_h5td(output_path / f"{name_lower}_counts.h5td")
    alleles.to_csv(output_path / f"{name_lower}_alleles.csv", index=False)

    
if __name__ == "__main__":
    main()