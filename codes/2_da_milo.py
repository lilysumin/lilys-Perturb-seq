# %% [markdown]
# # Differential Abundance Analysis - MILO

# %% [markdown]
# ## Environmental Setup

# %%
# Core scverse libraries
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from scipy.stats import median_abs_deviation

sc.settings.set_figure_params(dpi=100, facecolor="white")

# %% [markdown]
# ## Load Data

# %%
adata = sc.read('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/01_kc_sgRNA.h5ad')

# %%
adata

# %%
import pertpy as pt
milo = pt.tl.Milo()
mdata = milo.load(adata)
sc.pp.neighbors(mdata["rna"], n_neighbors=50)
milo.make_nhoods(mdata["rna"], prop=0.2)
mdata["rna"].obsm["nhoods"]
mdata["rna"][mdata["rna"].obs["nhood_ixs_refined"] != 0].obs[["nhood_ixs_refined", "nhood_kth_distance"]]


# %%
nhood_size = np.array(mdata["rna"].obsm["nhoods"].sum(0)).ravel()
plt.hist(nhood_size, bins=100)
plt.xlabel("Number of cells in nhood")
plt.ylabel("Number of nhoods");
plt.show()

# %%
import re
import matplotlib.pyplot as plt

# R-safe name 변환 함수
def r_safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)

# Count neighborhoods
mdata = milo.count_nhoods(mdata, sample_col="pseudo_replicate")

mdata["rna"].obs["condition"] = mdata["rna"].obs["feature_call"].astype(str).map(r_safe_name).astype("category")
mdata["milo"].obs["condition"] = mdata["rna"].obs["condition"]

conditions = mdata["rna"].obs["condition"].cat.categories.tolist()
control_group = "gNeg_1"
test_groups = [g for g in conditions if g != control_group]

da_results = {}

total_groups = len(test_groups)
fig, axes = plt.subplots(total_groups, 2, figsize=(12, 4 * total_groups))

for idx, test_group in enumerate(test_groups):
    contrast = f"condition{test_group} - condition{control_group}"
    print(f"🚀 Running DA: {contrast}")

    milo.da_nhoods(
        mdata,
        design="~ condition",
        model_contrasts=contrast
    )

    result_df = mdata["milo"].var.copy()
    da_results[contrast] = result_df

    axes[idx, 0].hist(result_df.PValue, bins=50)
    axes[idx, 0].set_xlabel("P-Values")
    axes[idx, 0].set_title(f"{test_group} vs {control_group} — P-Value")

    axes[idx, 1].plot(result_df.logFC, -np.log10(result_df.SpatialFDR), ".", alpha=0.5)
    axes[idx, 1].axhline(-np.log10(0.1), color="red", linestyle="--", label="FDR = 0.1")
    axes[idx, 1].set_xlabel("logFC")
    axes[idx, 1].set_ylabel("-log10(FDR)")
    axes[idx, 1].legend()
    axes[idx, 1].set_title(f"{test_group} vs {control_group} — Volcano")

plt.tight_layout()
plt.show()

# %%
# %% DA log-Fold Change - Alk1_KO
milo.da_nhoods(mdata, design="~condition", model_contrasts="conditiongAlk1_1-conditiongNeg_1")
milo.build_nhood_graph(mdata)

milo.plot_nhood_graph(
    mdata,
    alpha=0.1,  ## SpatialFDR level (1%)
    min_size=1, ## Size of smallest dot
)

plt.title("DA log-Fold Change: gAlk1_1 vs gNeg_1", fontsize=16)
plt.tight_layout()
plt.show()


# %%
# %% DA log-Fold Change - Cd64_KO
milo.da_nhoods(mdata, design="~condition", model_contrasts="conditiongCd64_1-conditiongNeg_1")
milo.build_nhood_graph(mdata)

milo.plot_nhood_graph(
    mdata,
    alpha=0.1,  ## SpatialFDR level (1%)
    min_size=1, ## Size of smallest dot
)


# %%
# %% DA log-Fold Change - F480_KO
milo.da_nhoods(mdata, design="~condition", model_contrasts="conditiongCd64_1-conditiongNeg_1")
milo.build_nhood_graph(mdata)

milo.plot_nhood_graph(
    mdata,
    alpha=0.1,  ## SpatialFDR level (1%)
    min_size=1, ## Size of smallest dot
)


