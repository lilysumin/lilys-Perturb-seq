# %% [markdown]
# ## Reference Mapping - scVI

# %%
from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import scvi
import os


os.chdir('../')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



sc.settings.set_figure_params(dpi=50, facecolor="white")

# %%
adata = sc.read('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/04_ti.h5ad')

# %%
adata


# %%
import scanpy as sc
import pandas as pd

adata_145 = adata[adata.obs['leiden'].isin(['1','4','5'])].copy()
adata_145.obs['leiden'] = adata_145.obs['leiden'].astype('category')  # 카테고리화

sc.tl.rank_genes_groups(
    adata_145,
    groupby='leiden',
    groups=['5'],
    reference='rest',
    method='wilcoxon'
)

deg_df = sc.get.rank_genes_groups_df(adata_145, group='5')
app_row = deg_df.loc[deg_df['names'] == 'Ndufab1', ['names','logfoldchanges','pvals_adj']]
print(app_row)


# %%
genes = ['Spp1', 'Trem2', 'Fabp5', 'Nupr1', 'Folr2', 'Cd163', 'Mrc1',
 'Rnase1', 'Ctsd', 'Cstb', 'Apoc1', 'Gpnmb', 'Ctsb', 'Apoe',
 'Ctsz', 'Lgals1', 'Pltp', 'Cxcl3', 'Colec12', 'Hspa1a', 'Plin2', 'Scd',
 'Ctsl', 'Hspa1b', 'Mmp19', 'Pld3', 'Abl2', 'Cebpb']

# %%
sc.pl.umap(adata, color=genes,  ncols=5)

# %%
sc.pl.umap(adata, color='')

# %%
# 1. Control 세포만 추출하여 reference 모델 학습
adata_ref = adata[adata.obs["feature_call"] == "gNeg_1"].copy()
scvi.model.SCVI.setup_anndata(adata_ref, batch_key=None)
model = scvi.model.SCVI(adata_ref)
model.train()

# %%
SCVI_LATENT_KEY = "X_scVI"
adata_ref.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(adata_ref)
sc.tl.umap(adata_ref)

# %%
sc.pl.umap(adata_ref, color="feature_call", palette=['#1f77b4'])

# %%
sc.pl.umap(adata_ref, color=["Clec4f", "Clec1b"])

# %%
sc.tl.diffmap(adata_ref)
iroot_cell  = adata_ref.obs["Monocyte_score"].idxmax()          # 문자열 ID
iroot_index = adata_ref.obs_names.get_loc(iroot_cell) 
adata_ref.uns['iroot'] = iroot_index

sc.tl.dpt(adata_ref)
sc.pl.umap(adata_ref, color=["dpt_pseudotime", "Clec4f"])


# %%
adata_query = adata[adata.obs["feature_call"] == "gAlk1_1"].copy()
scvi.model.SCVI.prepare_query_anndata(adata_query, model)
query_model = scvi.model.SCVI.load_query_data(adata_query, model)
query_model.train(max_epochs=200, plan_kwargs={"weight_decay": 0.0})
adata_query.obsm[SCVI_LATENT_KEY] = query_model.get_latent_representation()

# %%
if "iroot" in adata_query.uns:
    del adata_query.uns["iroot"]
    
iroot_cell  = adata_query.obs["Monocyte_score"].idxmax()
iroot_index = adata_query.obs_names.get_loc(iroot_cell)
adata_query.uns["iroot"] = iroot_index     # 정수로 저장

# %%
sc.pp.neighbors(adata_query, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(adata_query)
sc.tl.umap(adata_query)

# %%
sc.pl.umap(
    adata_query,
    color="feature_call",
    palette=["#FF7F0E"]  # 또는 palette=["#FF7F0E"]
)


# %%
adata_query.obs

# %%


# %%
adata_combined = adata_ref.concatenate(adata_query, batch_key="mapping_batch")

# %%
adata_combined.obsm[SCVI_LATENT_KEY] = query_model.get_latent_representation(adata_combined)

# %%
print(SCVI_LATENT_KEY)
print(adata_combined.obsm_keys())

# %%
sc.pp.neighbors(adata_combined, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(adata_combined)
sc.tl.umap(adata_combined)

# %%
print("X_scVI" in adata_combined.obsm)  # True?
print(type(adata_combined.obsm["X_scVI"]))  # np.ndarray or sparse?
print(adata_combined.obsm["X_scVI"].shape)  # (n_cells, n_latent)

# %%
sc.pl.umap(adata_combined, color=["feature_call", "dpt_pseudotime"],wspace=0.3)

# %%
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 50
mpl.rcParams["figure.facecolor"] = "white"
umap = adata_combined.obsm["X_umap"]
mask_gAlk1 = adata_combined.obs["feature_call"] == "gAlk1_1"

plt.figure(figsize=(5, 5), dpi=50, facecolor="white")  # dpi, 배경색 명시

plt.scatter(
    umap[:, 0], umap[:, 1],
    color="lightgray",
    s=10,
    alpha=0.6,
    label="gNeg_1"
)

plt.scatter(
    umap[mask_gAlk1, 0], umap[mask_gAlk1, 1],
    color="#1E64C8",
    s=12,
    label="gAlk1_1"
)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("gAlk1 Cells Highlighted on UMAP")
plt.legend()
plt.grid(False)  
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 50
mpl.rcParams["figure.facecolor"] = "white"

umap = adata_combined.obsm["X_umap"]
dpt = adata_combined.obs["dpt_pseudotime"]
mask_gAlk1 = adata_combined.obs["feature_call"] == "gAlk1_1"

plt.figure(figsize=(5, 5), dpi=50, facecolor="white")

plt.scatter(
    umap[:, 0], umap[:, 1],
    c=dpt,
    cmap="viridis",   # or "viridis", "plasma", etc.
    s=10,
    alpha=0.6,
    label="gNeg_1"
)

plt.scatter(
    umap[mask_gAlk1, 0], umap[mask_gAlk1, 1],
    color="#FF7F0E",
    s=12,
    label="gAlk1_1"
)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# %%
adata_ref.obs

# %%
sc.pl.umap(adata_ref, color='dpt_pseudotime')

# %%
from sklearn.neighbors import NearestNeighbors
import pandas as pd

mask_ref = adata_combined.obs["feature_call"] == "gNeg_1"
mask_query = adata_combined.obs["feature_call"] == "gAlk1_1"

X_ref = adata_combined.obsm["X_umap"][mask_ref]
X_query = adata_combined.obsm["X_umap"][mask_query]
y_ref = adata_combined.obs.loc[mask_ref, "dpt_pseudotime"].values

knn = NearestNeighbors(n_neighbors=15)
knn.fit(X_ref)
_, indices = knn.kneighbors(X_query)

projected_pseudotime = y_ref[indices].mean(axis=1)

adata_combined.obs.loc[mask_query, "dpt_projected_umap"] = projected_pseudotime

df_plot = pd.DataFrame({
    "pseudotime": np.concatenate([
        adata_combined.obs.loc[mask_ref, "dpt_pseudotime"].values,
        projected_pseudotime
    ]),
    "group": ["gNeg_1"] * mask_ref.sum() + ["gAlk1_1"] * mask_query.sum()
})

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df_plot, x="pseudotime", hue="group", fill=True, common_norm=False,
            palette={"gNeg_1": "#1f77b4", "gAlk1_1": "#FF7F0E"}, alpha=0.4)
plt.xlabel("DPT Pseudotime")
plt.tight_layout()
plt.grid(False)
plt.show()

# %%
df_violin = pd.DataFrame({
    "pseudotime": np.concatenate([
        adata_combined.obs.loc[mask_ref, "dpt_pseudotime"].values,
        projected_pseudotime
    ]),
    "group": ["gNeg_1"] * mask_ref.sum() + ["gAlk1_1"] * mask_query.sum()
})

plt.figure(figsize=(4, 6))
sns.violinplot(
    data=df_violin,
    x="group",
    y="pseudotime",
    palette={"gNeg_1": "#1f77b4", "gAlk1_1": "#FF7F0E"},
    inner="box",  
    linewidth=1
)
plt.ylabel("DPT Pseudotime")
plt.xlabel("")
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

pseudotime_values = adata_combined.obs.loc[adata_combined.obs["feature_call"] == "gAlk1_1", "dpt_projected_umap"]

counts, bins = np.histogram(pseudotime_values, bins=20)

max_bin_idx = np.argmax(counts)
peak_range = (bins[max_bin_idx], bins[max_bin_idx + 1])

print(f"가장 pseudotime이 집중된 구간: {peak_range[0]:.2f} ~ {peak_range[1]:.2f}")

plt.figure(figsize=(6, 3))
plt.hist(pseudotime_values, bins=20, color="#FF7F0E", alpha=0.7)
plt.axvspan(*peak_range, color="red", alpha=0.2, label=f"Peak: {peak_range[0]:.2f}–{peak_range[1]:.2f}")
plt.xlabel("DPT Pseudotime")
plt.ylabel("Cell count")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()



