# %%
import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt

sc.settings.set_figure_params(dpi=50, facecolor="white")

# %% [markdown]
# # Labeling gRNA

# %% [markdown]
# ## Load Data

# %%
adata = sc.read('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/00_filtered_mono_kc.h5ad')

# %%
sc.pl.umap(adata, color='filter2_res0_5', legend_loc="on data")

# %%
adata.obs

# %% [markdown]
# ## Dimension reduction and visualization

# %%
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color="feature_call")

# %% [markdown]
# ## Remove NaN value

# %%
nan_count = adata.obs["feature_call"].isna().sum()
nan_percentage = nan_count / len(adata.obs) * 100
print(f"Number of NaN values: {nan_count}")
print(f"Percentage of NaN values: {nan_percentage:.2f}%")

# %%
adata = adata[~adata.obs["feature_call"].isna()].copy()

# %%
nan_count = adata.obs["feature_call"].isna().sum()
nan_percentage = nan_count / len(adata.obs) * 100
print(f"Number of NaN values: {nan_count}")
print(f"Percentage of NaN values: {nan_percentage:.2f}%")

# %%
adata

# %%
sc.pl.umap(adata, color="feature_call")

# %% [markdown]
# Since there are cells with multiple gRNAs, split the dataset into 
# 
# (1) Dataset only containing single gRNA
# 
# (2) Dataset also containing cells with multiple gRNA but max gRNA 

# %%
sc.pl.umap(adata, color='Manual celltype annotation_filter1')

# %%
import anndata as ad

def preprocess(adata):
    # -----------------------
    # 1. Singlet 세포
    # -----------------------
    is_singlet = adata.obs["feature_call"].str.count("\|") == 0
    singlet_adata = adata[is_singlet].copy()

    # -----------------------
    # 2. Singlet + Max-gRNA 세포
    # -----------------------
    is_multi = ~is_singlet
    multi_adata = adata[is_multi].copy()

    multi_df = (
        multi_adata.obs[["feature_call", "num_umis"]]
        .copy()
        .assign(cell_id=multi_adata.obs_names)
        .assign(gRNA=lambda df: df["feature_call"].str.split("|"))
        .explode("gRNA")
    )

    top_gRNA_df = (
        multi_df.sort_values("num_umis", ascending=False)
        .drop_duplicates("cell_id")
        .set_index("cell_id")
    )

    multi_adata.obs["feature_call"] = top_gRNA_df["gRNA"]

    # -----------------------
    # 3. Singlet + Max-gRNA 합치기
    # -----------------------
    singlet_plus_max_adata = singlet_adata.concatenate(
        multi_adata, batch_key=None, index_unique=None
    )

    # -----------------------
    # 4. 공통 condition 생성
    # -----------------------
    def feature_call_to_condition(x):
        guides = x.split("|")
        targets = [
            s.replace("g", "").replace("_1", "")
            for s in guides if s != "gNeg_1"
        ]
        return "Control" if len(targets) == 0 else "|".join(targets) + "_KO"

    gRNA_list = ["gAlk1_1", "gCd64_1", "gF480_1", "gNeg_1"]

    for ad_obj in [singlet_adata, singlet_plus_max_adata]:
        ad_obj.obs["condition"] = ad_obj.obs["feature_call"].apply(feature_call_to_condition)
        ad_obj.obs["pseudo_replicate"] = (
            ad_obj.obs["sample"].astype(str) + "_" + ad_obj.obs["condition"].astype(str)
        )

        for gRNA in gRNA_list:
            # Check if the gRNA string is present in the 'feature_call' column
            ad_obj.obs[gRNA] = ad_obj.obs["feature_call"].apply(
                lambda x: "positive" if gRNA in x.split("|") else "negative"
            ).astype('category')

        ad_obj.obs["condition2"] = ad_obj.obs["gAlk1_1"].apply(
            lambda x: "gAlk1_1" if x == "positive" else "Control"
        ).astype('category')

    return singlet_adata, singlet_plus_max_adata

# %%
single_kc, multiple_kc = preprocess(adata)

# %%
single_kc

# %%
multiple_kc

# %%
sc.pl.umap(single_kc, color=['feature_call', 'gAlk1_1', 'Clec4f', 'Timd4'], wspace=0.3)

# %%
sc.pl.umap(multiple_kc, color=['feature_call', 'gAlk1_1'], wspace=0.3)

# %% [markdown]
# ## Custom Cluster

# %%
sc.tl.leiden(single_kc, resolution=1, key_added='single_res1')

# %%
sc.pl.umap(single_kc, color=['single_res1', 'Kupffer cell score', 'Monocytes score', 'Clec4f', 'Clec2f'], legend_loc="on data")

# %%
sc.tl.rank_genes_groups(single_kc, groupby="leiden", method="wilcoxon")

# %%
sc.pl.rank_genes_groups_dotplot(
    single_kc, groupby="single_res1", standard_scale="var", n_genes=5
)

# %%
kc_diff_number_map = {
    '10': 1,
    '6': 2,
    '2': 3, '11': 3,
    '0': 5, '1': 5,
    '7': 6
}

kc_diff_type_map = {
    '10': "Monocytes",
    '7': 'ALK1 KO cells',
    '6': "Transitioning Monocytes",
    '2': "Kupffer cells", '11': "Kupffer cells",
    '0': "Mature Kupffer cells", '1': "Mature Kupffer cells"
}


# %%
single_kc.obs["KCs differentiation (Initial)"] = single_kc.obs["leiden"].map(kc_diff_number_map).fillna(4).astype(int)
single_kc.obs["KCs differentiation (Initial)"] = single_kc.obs["KCs differentiation (Initial)"].astype("category")

single_kc.obs["KCs differentiation (cell)"] = single_kc.obs["leiden"].map(kc_diff_type_map).fillna("Kupffer cells")


# %%
sc.pl.umap(single_kc, color='KCs differentiation (Initial)', title='Monocyte - KCs differentiation')

# %%
sc.pl.umap(single_kc, color='KCs differentiation (cell)', title='Monocyte - KCs differentiation cluster')



# %%
single_kc.obs["feature_call"] = single_kc.obs["feature_call"].astype("category")

# %%
import matplotlib.pyplot as plt


umap_targets = [
    ("KCs differentiation (Initial)", "Monocyte - KCs differentiation"),
    ("feature_call", "guide RNA"),
    ("Monocytes score", "Monocytes score"),
    ("Kupffer cell score", "Kupffer cell score"),
    ("gAlk1_1", "gAlk1_1"),
    ("gCd64_1", "gCd64_1"),
    ("gNeg_1", "gNeg_1"),
    ("gF480_1", "gF480_1"),
]

fig, axes = plt.subplots(4, 2, figsize=(15, 24))
axes = axes.flatten()

for i, (col, title) in enumerate(umap_targets):
    sc.pl.umap(
        single_kc,
        color=col,
        ax=axes[i],
        title=title,
        show=False,
        legend_loc='right margin'
    )

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Save dataset

# %%
save_dir = "/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data"
os.makedirs(save_dir, exist_ok=True)

single_kc.write(os.path.join(save_dir, "01_kc_sgRNA.h5ad"))
multiple_kc.write(os.path.join(save_dir, "01_kc_multigRNA.h5ad"))


