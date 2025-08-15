# %% [markdown]
# # Preprocessing and Clustering

# %%
# Core scverse libraries
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation

sc.settings.set_figure_params(dpi=80, facecolor="white")

# %% [markdown]
# ## Load Dataset 

# %%
import os
import pathlib
def get_git_root(cwd=None):
    """
    Gets the first parent root with a a .git folder
    """
    if cwd is None:
        cwd = os.getcwd()
    # go back until we find the git directory, signifying project root
    while ".git" not in os.listdir(cwd) and os.path.realpath(cwd) != "/":
        cwd = os.path.dirname(cwd)

    return pathlib.Path(cwd)

def get_data():
    return get_git_root() / "data"


# %%
data = get_data() / "crispr_singlecell/cellranger_output/"
sample_names = ["ABL005","ABL006","ABL007"]

samples = {}
for sample_name in sample_names:
    samples[sample_name] = data / sample_name / "outs/filtered_feature_bc_matrix.h5"
adatas = {}
for sample_id, path in samples.items():
    sample_adata = sc.read_10x_h5(path)
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()
adata.var_names_make_unique()
adata

# %%
adata.obs

# %%
import pandas as pd

data = get_data()
# List of CSV file paths
csv_files = [
    data / "crispr_singlecell/cellranger_output/ABL005/outs/crispr_analysis/protospacer_calls_per_cell.csv",
    data / "crispr_singlecell/cellranger_output/ABL006/outs/crispr_analysis/protospacer_calls_per_cell.csv",
    data / "crispr_singlecell/cellranger_output/ABL007/outs/crispr_analysis/protospacer_calls_per_cell.csv"
]

# Read all CSV files into a single AnnData object
adata_list = []
for csv_file in csv_files:
    adata_list.append(pd.read_csv(csv_file))

# Concatenate the list of AnnData objects into a single AnnData object
df = pd.concat(adata_list)
df

# %%
# 1. Filter the DataFrame to include only the rows corresponding to the barcodes present in adata.obs
filtered_df = df[df['cell_barcode'].isin(adata.obs.index)]

# 2. Drop duplicate rows based on the "cell_barcode" column
filtered_df = filtered_df.drop_duplicates(subset=['cell_barcode'])

# 3. Set the index of the filtered DataFrame to match the index of adata.obs
filtered_df.set_index('cell_barcode', inplace=True)
filtered_df

# %%
for column in filtered_df.columns:
    # Add new columns from the filtered DataFrame to adata.obs
    adata.obs[column] = filtered_df[column]

# %%
adata

# %% [markdown]
# The raw dataset is composed of 61,263 cells x 57,128 genes. 
# 
# Check cells containing Cas9 and pBA900 respectively.

# %%
import numpy as np

gene1 = "CAS9"
gene2 = "pBA900"

gene1_expressed = adata[:, gene1].X > 0 
gene2_expressed = adata[:, gene2].X > 0

gene1_express_percent = np.mean(gene1_expressed) * 100
gene2_express_percent = np.mean(gene2_expressed) * 100

print(f"Percentage of cells expressing {gene1}: {gene1_express_percent:.2f}%")
print(f"Percentage of cells expressing {gene2}: {gene2_express_percent:.2f}%")


# %%
gene1_express_num = gene1_expressed.nnz
gene2_express_num = gene2_expressed.nnz

print(f"Number of cells expressing {gene1}: {gene1_express_num}")
print(f"Number of cells expressing {gene2}: {gene2_express_num}")

# %% [markdown]
# ## Quality Control

# %%
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("mt-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^Hbb")

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

# %%
# QC 메트릭 리스트
qc_metrics = [
    "n_genes_by_counts", "total_counts", "log1p_total_counts",
    "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"
]

uniform_color = '#4C72B0'

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
for i, metric in enumerate(qc_metrics):
    row = i // 3
    col = i % 3
    sc.pl.violin(
        adata,
        keys=metric,
        jitter=0.4,
        groupby = 'sample',
        palette=[uniform_color],
        ax=axes[row, col],
        show=False
    )
    
plt.tight_layout()
plt.show()



# %%
sns.displot(adata.obs['pct_counts_mt'], bins=100, kde=True)

# %% [markdown]
# * Most of cells located within 0-10%. 
# 
# * However, the distribution showed right (skewed distribution) long tail, showing some cells have high mitochondrial gene expression. 

# %%
mt_pct = adata.obs['pct_counts_mt']
mt_median = np.median(mt_pct)
mt_mad = np.median(np.abs(mt_pct - mt_median))
threshold_mad = mt_median + 3 * mt_mad
mt_fixed_threshold = 8  

print("Median value is:", mt_median)
print("MAD threshold is:", threshold_mad)

# 플롯
sns.displot(mt_pct, bins=100, kde=True)
plt.axvline(mt_median, color='orange', linestyle='--', label='Median')
plt.axvline(threshold_mad, color='red', linestyle='--', label='3 MAD')
plt.axvline(mt_fixed_threshold, color='green', linestyle='--', label='8% cutoff')
plt.legend()
plt.xlabel('pct_counts_mt')
plt.ylabel('Cell count')
plt.title('Distribution of pct_counts_mt with QC thresholds')
plt.show()

# %%
sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

# %%
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

# %%
adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_50_genes", 5)
)
adata.obs.outlier.value_counts()

# %%
adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
    adata.obs["pct_counts_mt"] > 8
)
adata.obs.mt_outlier.value_counts()

# %%
print(f"Total number of cells: {adata.n_obs}")
adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

# %%
p1 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

# %% [markdown]
# #### Check NaN count

# %%
# Count the number of NaN values in 'column_name'
nan_count = adata.obs["feature_call"].isna().sum()

# Calculate the total number of rows in the column
total_count = len(adata.obs["feature_call"])

# Calculate the percentage of NaN values
nan_percentage = (nan_count / total_count) * 100

print(f"NaN count: {nan_count}")
print(f"Percentage of NaN values: {nan_percentage:.2f}%")


# %%
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)

# %%
adata

# %% [markdown]
# ## Doublet detection

# %%
sc.pp.scrublet(adata, batch_key="sample")

# %%
adata.obs[adata.obs["predicted_doublet"] == True]

# %%
# scrublet 결과에서 doublet_score 추출
scores = adata.obs["doublet_score"]

# 시각화
plt.figure(figsize=(8, 5))
sns.histplot(scores, bins=50, color='#1E64C8', edgecolor='black')
plt.xlabel("Doublet Score")
plt.ylabel("Number of Cells")
plt.title("Distribution of Scrublet Doublet Scores")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Normalization

# %%
# Saving count data
adata.layers["counts"] = adata.X.copy()

# %% [markdown]
# Raw dataset 은 "count" layer 에 저장해놓기.

# %%
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)

# %% [markdown]
# ## Dimensionality Reduction

# %%
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

# %%
sc.pl.pca(
    adata,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)

# %% [markdown]
# ## Nearest neighbor graph construction and visualization

# %% [markdown]
# Default k = 15

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(
    adata,
    color="sample",
    # Setting a smaller point size to get prevent overlap
    size=2,
)

# %%
sc.pl.umap(adata, color='Clec1b')

# %% [markdown]
# #### Figure 1

# %%
adata.layers

# %%
adata_raw = adata.layers["counts"]

# %%
# QC 메트릭 리스트
qc_metrics = [
    "n_genes_by_counts", "total_counts", "log1p_total_counts",
    "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"
]

uniform_color = '#4C72B0'

fig, axes = plt.subplots(3, 3, figsize=(20, 20))  
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# (a)-(f): violin plots
for i, metric in enumerate(qc_metrics):
    row = i // 3
    col = i % 3
    sc.pl.violin(
        adata,
        keys=metric,
        jitter=0.4,
        groupby='sample',
        palette=[uniform_color],
        ax=axes[row, col],
        show=False
    )

# (g): total_counts vs n_genes_by_counts scatter (matplotlib 직접 사용)
x = adata.obs["total_counts"]
y = adata.obs["n_genes_by_counts"]
c = adata.obs["pct_counts_mt"]

sc_g = axes[2, 0]
sc_plot = sc_g.scatter(x, y, c=c, cmap='viridis', s=2, linewidths=0)
sc_g.set_xlabel("total_counts")
sc_g.set_ylabel("n_genes_by_counts")
sc_g.set_title("Total counts vs Number of genes")
cbar = fig.colorbar(sc_plot, ax=sc_g, orientation='vertical', shrink=0.8)
cbar.set_label("pct_counts_mt")

# (h): histogram plot with QC thresholds
mt_pct = adata.obs['pct_counts_mt']
mt_median = np.median(mt_pct)
mt_mad = np.median(np.abs(mt_pct - mt_median))
threshold_mad = mt_median + 3 * mt_mad
mt_fixed_threshold = 8

ax_h = axes[2, 1]
sns.histplot(mt_pct, bins=100, kde=True, ax=ax_h)
ax_h.axvline(mt_median, color='orange', linestyle='--', label='Median')
ax_h.axvline(threshold_mad, color='red', linestyle='--', label='3 MAD')
ax_h.axvline(mt_fixed_threshold, color='green', linestyle='--', label='8% cutoff')
ax_h.legend()
ax_h.set_xlabel('pct_counts_mt')
ax_h.set_ylabel('Cell count')
ax_h.set_title('Distribution of pct_counts_mt with QC thresholds')

# (i)
sc.pl.umap(
    adata,
    color="sample",
    size=5,
    ax=axes[2, 2],
    show=False
)


panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
for idx, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.1, panel_labels[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left')

plt.tight_layout()
plt.show() 

# %% [markdown]
# ## Clustering

# %%
sc.tl.leiden(adata, resolution=1)

# %%
sc.pl.umap(adata, color="leiden")


# %% [markdown]
# ## Re-assess quality control and cell filtering

# %%
sc.pl.umap(
    adata,
    color=["leiden", "predicted_doublet", "doublet_score"],
    legend_loc = "on data"
    # increase horizontal space between panels
)

# %% [markdown]
# ### Cell type annotation

# %%
import math
data = list(adata.obs["feature_call"].unique())
# Use a set to store unique values
unique_values = set()

# Iterate through each item in the list
for item in data:
    if isinstance(item, str):
        # Split the string by '|' and add each part to the set
        unique_values.update(item.split('|'))
    elif not (isinstance(item, float) and math.isnan(item)):
        # If the item is not NaN, add it to the set
        unique_values.add(item)

# Convert the set back to a list (if needed)
substrings = list(unique_values)

# Output the unique values
print(substrings)
# Define the function to check for the presence of the substring
def check_presence(feature_call, substring):
    try:
        parts = feature_call.split('|')
        for part in parts:
            if part.strip() == substring:
                return 'positive'
        return 'negative'
    except Exception as e:
        print(feature_call)
        return 'negative'

for substring in substrings:
    # Apply the function to create the new column
    adata.obs[substring] = adata.obs['feature_call'].apply(lambda x: check_presence(x, substring))
    

sc.pl.umap(
    adata,
    color=["gAlk1_1","gCd64_1", "gNeg_1", "gF480_1"],
    na_color = "green",
    ncols=4,
    wspace = 0.4
)

# %% [markdown]
# ### Marker gene set

# %%
score_markers = {
    'LSEC' : ['Ptprb', 'Pcdh17', 'Stab2', 'Lyve1'],
    'Stellate cells' : ["Dcn", "Lrat", "Ngfr", "Lama1", "Ptger2"],
    'Hepatocytes' : ["Alb", "Cyp2e1", "Cyp2f2", "Ttr", "Pck1"],
    'Hepatocytes_nuc' : ["Ptprd", "Magi1", "Ghr", "Sox5"],
    'Kupffer cell': ['Clec4f', 'Irf7', 'Spic', 'Vsig4', 'Id3'], 
    'Monocytes': ['Ccdc88a', 'Ccll9', 'Ccr2', 'Ccr2', 'CD11b', 'Cd209a', 'CD45', 'CD74', 'Chil3', 'Fn1', 'Gr-1', 'Ly6C', 'Ly6c1', 'Ly6c2', 'Mycl', 'S100a4', 'S100a8'],
    'T cells' : ["Cd3e", "Cd3d", "Lat"],
    'B cells' : ["Cd79a", "Cd79b", "Cd19"],
    'Dendtiric cells' : ["Flt3", "Xcr1", "Itgae", "Cd209a", 'Cacnb3', 'Ccr7', 'Nudt17'],
    'Neutrophils' : ["S100a8", "S100a9", "Retnlg"],
    'Cholangiocytes' : ["Epcam", "Spp1", "Krt8"],
    'Eosinophils': ['Ccr3', 'Siglecf'],
    'Cycling cell' : ["Cdk1", "Mki67"],
    'cycling_s' : ["Pcna", "Mcm3", "Mcm7", "Rad51b"],
    'cycling_g2m' : ["Birc5", "Top2a"],
    }


# %%
marker_genes_in_data = {}

for ct, markers in score_markers.items():
    markers_found = [marker for marker in markers if marker in adata.var.index]
    if markers_found:  # 비어있지 않은 경우만 추가
        marker_genes_in_data[ct] = markers_found


# %%
marker_genes_in_data

# %% [markdown]
# ### Dot plot showing gene marker score per leiden cluster

# %%
def group_max(adata: sc.AnnData, groupby: str) -> str:
    agg = sc.get.aggregate(adata, by=groupby, func="mean")
    return pd.Series(agg.layers["mean"].sum(1), agg.obs[groupby]).idxmax()

# %%
sc.pl.dotplot(adata, marker_genes_in_data, groupby="leiden")

# %% [markdown]
# ### Gene marker score UMAP visualization

# %%
for cell_type, gene_list in marker_genes_in_data.items():
    if gene_list:  
        sc.tl.score_genes(adata, gene_list, score_name=cell_type + ' score')
    else:
        print(f"Skipping {cell_type}: no valid genes.")


# %%
sc.pl.umap(adata, color=[f'{marker} score' for marker in marker_genes_in_data.keys()])

# %%
sc.pl.umap(adata, color=["Clec4f", "Vsig4", "Id3"])

# %% [markdown]
# ### Differentially-expressed Genes as Markers

# %%
# Obtain cluster-specific differentially expressed genes
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# %%
sc.pl.rank_genes_groups_dotplot(
    adata, groupby="leiden", standard_scale="var", n_genes=5
)

# %%
adata.obs["leiden"][adata.obs["predicted_doublet"] == True]

# %%
sc.pl.umap(
    adata,
    color=["log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts", "doublet_score"]
)

# %%
cl_annotation = {}

# -- 1. Dying clusters
cl_annotation["1"] = "Dying LSEC"
cl_annotation["3"] = "LSEC"
cl_annotation["18"] = "LSEC"
cl_annotation["5"] = "Dying Kupffer cell"
cl_annotation["20"] = "Dying LSEC"

# -- 2. Cycling Kupffer cells
for k in ["4", "8", "11", "14", "15"]:
    cl_annotation[k] = "Cycling Kupffer cell"

# -- 3. Transitioning cells
cl_annotation["10"] = "Transitioning Monocyte"

# -- 4. Neutrophils
cl_annotation["7"] = "KCs-LSEC doublets"

# -- 5. T cells
cl_annotation["16"] = "T cell"

# -- 6. Dendritic cells
cl_annotation["17"] = "Dendritic cell"

# -- 7. Monocytes
cl_annotation["12"] = "Monocyte"

# -- 8. Hepatocytes / Stellate / Cholangiocytes
cl_annotation["19"] = "Hepatocyte"
cl_annotation["21"] = "Stellate / Cholangiocyte"

# -- 9. LSECs
cl_annotation["9"] = "LSEC"
cl_annotation["0"] = "KCs-LSEC doublets"

# -- 10. Kupffer cells (default 상태: healthy)
remaining_kcs = {
    "2", "6", "13", "22", "23", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "33", "34"
}
for k in remaining_kcs:
    cl_annotation[k] = "Kupffer cell"


# %%
adata.obs["Manual cell type annotation"] = adata.obs.leiden.map(cl_annotation)

# %%
adata.obs["Manual cell type annotation"] = adata.obs["Manual cell type annotation"].astype("category")
labels = adata.obs["Manual cell type annotation"].cat.categories
colors = adata.uns["Manual cell type annotation_colors"]
celltype_colors = dict(zip(labels, colors))

# %%
fig, ax = plt.subplots(figsize=(6, 6))
sc.pl.umap(
    adata,
    color='Manual cell type annotation',
    ax=ax,
    show=False,
    size=5,
    title='UMAP of Annotated Liver Cells',
    legend_loc=None 
)

# 범례 패치 생성
legend_patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in celltype_colors.items()
]

# 하단에 범례 추가
fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=3,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.15)  # 아래쪽 중앙에 고정
)

plt.tight_layout()
plt.show()


# %%
sc.pl.umap(adata, color=["Vsig4", "Clec4f"])

# %% [markdown]
# Filter out unnecessary cells.

# %%
adata_filtered = adata[adata.obs['leiden'].isin([ "2", "6", "10", "12", "13", "22", "23", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "33", "34"])].copy()

# %% [markdown]
# ### Second QC and Filtering

# %%
sc.tl.pca(adata_filtered)
sc.pp.neighbors(adata_filtered)
sc.tl.umap(adata_filtered)

# %%
sc.pl.umap(adata_filtered, color='Manual cell type annotation')

# %%
sc.tl.leiden(adata_filtered, key_added="filter1_res0_25", resolution=0.25)
sc.tl.leiden(adata_filtered, key_added="filter1_res0_5", resolution=0.5)
sc.tl.leiden(adata_filtered, key_added="filter1_res1", resolution=1.0)

# %%
sc.pl.umap(
    adata_filtered,
    color=["filter1_res0_25", "filter1_res0_5", "filter1_res1"],
    legend_loc="on data",
)

# %% [markdown]
# #### Second Filtering - 1. QC metrics 

# %%
sc.pl.umap(
    adata_filtered,
    color=["log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts", "doublet_score"]
)

# %% [markdown]
# #### Second Filtering - 2. Marker gene expression 

# %%
sc.pl.umap(adata_filtered, color=[f'{marker} score' for marker in marker_genes_in_data.keys()])

# %% [markdown]
# #### Second Filering - 3. Dot plot

# %%
sc.pl.dotplot(adata_filtered, marker_genes_in_data, groupby="filter1_res1")

# %%
sc.tl.rank_genes_groups(adata_filtered, groupby="filter1_res1", method="wilcoxon")

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_filtered, groupby="filter1_res1", standard_scale="var", n_genes=5
)

# %% [markdown]
# #### Second Filtering - Conclusion

# %%
sc.pl.umap(adata_filtered, color="Clec4f")

# %% [markdown]
# * Cluster 1: Dying Kupffer cells
# 
# * Cluster 7: Dendritic cells 
# 
# * Cluster 12: Dying Monocytes 
# 
# 

# %%
cl_annotation_1 = {}

# 특정 클러스터 우선 정의
cl_annotation_1.update({
    "7": "Dendritic cells",
    "9": "Transitioning monocytes",
    "10": "Monocytes",
    "12": "Dying Monocytes"
})

for cl in adata_filtered.obs["filter1_res1"].cat.categories:
    if cl not in cl_annotation_1:
        cl_annotation_1[cl] = "Kupffer cells"


# %%
adata_filtered.obs["Manual celltype annotation_filter1"] = adata_filtered.obs.filter1_res1.map(cl_annotation_1)

# %%
sc.pl.umap(adata_filtered, color="Manual celltype annotation_filter1")

# %% [markdown]
# As Leiden clustering alone was insufficient to accurately segregate cycling cells, I applied an additional filtering step based on cell cycle scores to remove cells exhibiting active cycling signatures.

# %%
adata_filtered = adata_filtered[
    (adata_filtered.obs["Cycling cell score"] <= 0.2) &
    (adata_filtered.obs["cycling_s score"] <= 0.2) &
    (adata_filtered.obs["cycling_g2m score"] <= 0.2)
].copy()


# %%
sc.pl.umap(adata_filtered, color="Manual celltype annotation_filter1")

# %%
adata_filtered_2 = adata_filtered[~adata_filtered.obs['filter1_res1'].isin(['7', '12'])].copy()

# %% [markdown]
# ### Perform Third QC and Filtering (Final Check)

# %%
sc.tl.pca(adata_filtered_2)
sc.pp.neighbors(adata_filtered_2)
sc.tl.umap(adata_filtered_2)

# %%
sc.pl.umap(adata_filtered_2, color=['Manual celltype annotation_filter1', 'Clec4f', 'Ly6c2'])

# %%
sc.tl.leiden(adata_filtered_2, key_added="filter2_res0_25", resolution=0.25)
sc.tl.leiden(adata_filtered_2, key_added="filter2_res0_5", resolution=0.5)
sc.tl.leiden(adata_filtered_2, key_added="filter2_res1", resolution=1.0)

# %%
sc.pl.umap(
    adata_filtered_2,
    color=["filter2_res0_25", "filter2_res0_5", "filter2_res1"],
    legend_loc="on data",
)

# %% [markdown]
# #### Third Filtering - 1. QC metrics

# %%
sc.pl.umap(
    adata_filtered_2,
    color=["log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts", "doublet_score"]
)

# %% [markdown]
# #### Third Filtering - 2. Marker gene expression

# %%
sc.pl.umap(adata_filtered_2, color=[f'{marker} score' for marker in marker_genes_in_data.keys()])

# %% [markdown]
# #### Third Filtering - Dot plot

# %%
sc.pl.dotplot(adata_filtered_2, marker_genes_in_data, groupby="filter2_res0_5")

# %%
sc.tl.rank_genes_groups(adata_filtered_2, groupby="filter2_res0_5", method="wilcoxon")

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_filtered_2, groupby="filter2_res0_5", standard_scale="var", n_genes=5
)

# %% [markdown]
# #### Third Filtering - Conclusion

# %% [markdown]

# %%
adata_filtered_2 = adata_filtered_2[adata_filtered_2.obs["pct_counts_mt"] <= 3].copy()

# %%
sc.pl.umap(adata_filtered_2, color='filter2_res0_5')

# %%
sc.tl.pca(adata_filtered_2)
sc.pp.neighbors(adata_filtered_2)
sc.tl.umap(adata_filtered_2)

# %%
sc.pl.umap(adata_filtered_2, color='Manual celltype annotation_filter1')

# %%
adata_filtered_2.write('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/00_filtered_mono_kc.h5ad')


