# %% [markdown]
# # Pseudotemporal ordering

# %%
from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


sc.settings.set_figure_params(dpi=50, facecolor="white")

# %% [markdown]
# ## Load Data

# %%
adata = sc.read('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/01_kc_sgRNA.h5ad')

# %%
adata.obs

# %%
sc.tl.leiden(adata, resolution=0.5)

# %%
sc.pl.umap(adata, color='leiden')

# %% [markdown]
# ## Pseudotime construction
# 

# %% [markdown]
# ### 1. Denoising the graph

# %%
sc.tl.diffmap(adata)

# %%
sc.tl.draw_graph(adata)
sc.pl.draw_graph(adata, color="leiden", legend_loc="on data")


# %% [markdown]
# <Drawing single-cell graph using layout 'fa'>
# 
# * 내부적으로 ForceAtlas2 (FA) 또는 Fruchterman-Reingold (FR) 과 같은 force-directed lyaout algorithm 을 사용해서 시각화 좌표 계산. 
# 
# * 클러스터 간 연결 구조를 직관적으로 볼 수 있음.

# %% [markdown]
# ###  2. Clustering and PAGA

# %%
sc.tl.paga(adata, groups='leiden')

# %%
sc.pl.paga(adata, color='leiden')

# %% [markdown]
# We can annotate the clusters using marker genes.
# 
# * Monocyte: ['Ccdc88a', 'Ccr2', 'Cd209a', 'Chil3', 'Fn1', 'Ly6c1', 'Ly6c2', 'Mycl', 'S100a4', 'S100a8']
# 
# * Kupffer cell: ['Clec4f', 'Clec2f', 'Vsig4']

# %%
sc.pl.paga(adata, color=['leiden', 'Ccr2', 'Ly6c2', 'Clec1b', 'Clec4f', 'Vsig4', 'Timd4'])

# %%
adata.obs['leiden'].cat.categories

# %%
adata.obs["leiden_anno"] = adata.obs["leiden"].cat.rename_categories(
    {
        "3": "3/Mono",
        "1": "1/Mature KCs",
        "5": "5/ALK1 KO Cells"
    }
)

# %%
sc.tl.paga(adata, groups="leiden_anno")

# %%
sc.pl.paga(adata, threshold=0.03, show=False)


# %% [markdown]
# ### 3. Recomputing the embedding using PAGA-initialization

# %%
sc.tl.draw_graph(adata, init_pos='paga')
sc.pl.draw_graph(
    adata, color='leiden_anno', legend_loc="on data"
)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (0, 0) UMAP - Monocyte score
sc.pl.umap(adata, color='Monocytes score', ax=axes[0, 0], show=False)
axes[0, 0].set_title('UMAP - Monocyte score')

# (0, 1) UMAP - Kupffer cell score
sc.pl.umap(adata, color='Kupffer cell score', ax=axes[0, 1], show=False)
axes[0, 1].set_title('UMAP - Kupffer cell score')

# UMAP - gAlk1_1
sc.pl.umap(adata, color='gAlk1_1', palette={"positive": "red", "negative": "lightgray"}, ax=axes[0, 2], legend_loc="on data", show=False)
axes[0, 2].set_title('UMAP - gAlk1_1 distribution')

# (1, 0) UMAP - leiden
sc.pl.umap(adata, color='leiden', ax=axes[1, 0], legend_loc="on data", show=False)
axes[1, 0].set_title('UMAP - Leiden Clustering')

# (1, 1) PAGA
sc.pl.paga(adata, color='leiden', threshold=0.03, ax=axes[1, 1], show=False)
axes[1, 1].set_title('PAGA Clustering')

# (1, 2) Draw graph with PAGA 
sc.tl.draw_graph(adata, init_pos='paga')
sc.pl.draw_graph(
    adata, color='leiden', legend_loc="on data", ax=axes[1, 2], show=False
)
axes[1, 2].set_title('Draw Graph (PAGA)')

plt.tight_layout
plt.show()


# %% [markdown]
# 
# 
# * Cluster 5 (gAlk1_1 KO macrophages) shows the strongest connectivity to Cluster 2 & 4, based on the PAGA graph.
# 
# * Notably, it has no connections to Clusters 0 and 1, which are predicted to represent mature Kupffer cells.
# 
#     -> This suggests that cells in Cluster 5 likely fail to fully mature into Kupffer cells.
# 
#     -> These findings support the hypothesis that Alk1 plays a crucial role in Kupffer cell maturation, and that its loss may disrupt normal differentiation trajectories.

# %% [markdown]
# #### Choose the color of the clusters consistently

# %%
plt.figure(figsize=(8, 2))
for i in range(28):
    plt.scatter(i, 1, c=sc.pl.palettes.zeileis_28[i], s=200)
plt.show()

# %%
zeileis_colors = np.array(sc.pl.palettes.zeileis_28)
new_colors = np.array(adata.uns["leiden_anno_colors"])

# %%
new_colors[[2, 3]] = zeileis_colors[[7, 6]] # Mono & early KCs [Green]
new_colors[[5]] = zeileis_colors[[18]] # ALK1 KO 
new_colors[[4]] = zeileis_colors[[8]]
new_colors[[0]] = zeileis_colors[[8]]
new_colors[[1]] = zeileis_colors[[10]]

# %%
adata.uns["leiden_anno_colors"] = new_colors

# %%
sc.pl.paga(adata, threshold=0.03)

# %% [markdown]
# ### 4. Reconstructing gene changes along PAGA paths for a given set of genes

# %%
sc.pl.umap(adata, color=['Monocytes score', 'Kupffer cell score'])

# %% [markdown]
# It looks like UMAP1 does a pretty good job for tracking with the developmental progression. 
# 
# Therefore, set the highest UMAP1 coordinate for the root cell. Then, plot that cell in red.

# %%
root_idx = np.argmax(adata.obsm['X_umap'][:, 0])
adata.uns['iroot'] = root_idx 

# %%
fig = sc.pl.umap(adata, color='Monocytes score', return_fig=True)
x, y = adata.obsm['X_umap'][root_idx]

ax = plt.gca()
ax.scatter(x, y, color='red', s=50, label='Root Cell')

plt.legend()
plt.show()

# %%
sc.tl.dpt(adata, n_branchings=1)

# %%
sc.pl.umap(adata, color=["dpt_pseudotime", "dpt_groups"], cmap="viridis")


# %%
sc.pl.violin(adata, keys="dpt_pseudotime", groupby="leiden")

# %%
sc.pl.violin(adata, keys="dpt_pseudotime", groupby="dpt_groups")

# %%
adata.obs['dpt_forward_pseudotime'] = adata.obs['dpt_pseudotime']

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 9))

sc.pl.umap(adata, color='Monocytes score', ax=axes[0, 0], show=False)
x, y = adata.obsm['X_umap'][root_idx]
axes[0, 0].scatter(x, y, color='red', s=50, label='Root Cell')
axes[0, 0].legend()
axes[0, 0].set_title('UMAP - Monocyte score with Root cell')

sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[0, 1], show=False)
axes[0, 1].set_title('UMAP - DPT Pseudotime')

sc.pl.draw_graph(adata, color='leiden', ax=axes[1, 0], show=False, legend_loc="on data")
axes[1, 0].set_title('Draw Graph - PAGA & Leiden)')

sc.pl.draw_graph(adata, color='dpt_pseudotime', ax=axes[1, 1], show=False)
axes[1, 1].set_title('Draw Graph - DPT Pseudotime')

plt.tight_layout()
plt.show()

# %%
sc.pl.draw_graph(adata, color=['leiden_anno', 'dpt_forward_pseudotime'], legend_loc="on data")

# %% [markdown]
# #### Path 설정

# %%
paths = [
    ('Monocytes - gAlk1_1 KO cluster', [3, 2, 5]),
    ('Monocytes - Mature KCs', [3, 2, 4, 0, 1])
]

# %%
gene_names = [
    *["Ly6c2", "Fcna", "Cbr2", "Slpi"],
    *["Id3", "Clec1b", "Clec4f", "Vsig4", 'Timd4']
]

# %%
# Cosmetic change
adata.obs["distance"] = adata.obs["dpt_forward_pseudotime"]
adata.obs["clusters"] = adata.obs["leiden_anno"]
adata.uns["clusters_colors"] = adata.uns["leiden_anno_colors"]

# %%
_, axs = plt.subplots(
    ncols=2, figsize=(10, 5), gridspec_kw={"wspace": 0.05, "left": 0.12}
)
plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)
for ipath, (descr, path) in enumerate(paths):
    data = sc.pl.paga_path(
        adata,
        path,
        gene_names,
        show_node_names=False,
        ax=axs[ipath],
        ytick_fontsize=12,
        left_margin=0.15,
        n_avg=50,
        annotations=["distance"],
        show_yticks=True if ipath == 0 else False,
        show_colorbar=False,
        color_map="Greys",
        groups_key="clusters",
        color_maps_annotations={"distance": "viridis"},
        title="{} path".format(descr),
        return_data=True,
        show=False,
    )
plt.show()

# %% [markdown]
# #### Option 1. Find DEGs between leiden cluster

# %%
import pandas as pd

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
deg_all = sc.get.rank_genes_groups_df(adata, group=None)
deg_filtered = deg_all[
    (deg_all['logfoldchanges'] >= 0.25) &
    (deg_all['pvals_adj'] <= 0.05)
]

import pandas as pd

# 1. DEG 수행
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# 2. 전체 DEG 결과 가져오기
deg_all = sc.get.rank_genes_groups_df(adata, group=None)

# 3. 필터링: logFC ≥ 0.25 and adj p-value ≤ 0.05
deg_filtered = deg_all[
    (deg_all['logfoldchanges'] >= 0.25) &
    (deg_all['pvals_adj'] <= 0.05)
]

# 4. 각 클러스터별로 logFC 기준 상위 5개 유전자 추출
top5_degs_filtered = (
    deg_filtered.sort_values(by=["group", "logfoldchanges"], ascending=[True, False])
    .groupby("group")
    .head(5)
)

# 5. 출력
for cluster in top5_degs_filtered['group'].unique():
    print(f"\n Cluster {cluster} - Top 5 DEGs (filtered):")
    print(top5_degs_filtered[top5_degs_filtered["group"] == cluster][["names", "logfoldchanges", "pvals_adj"]])

# 6. 원하는 클러스터 순서대로 유전자 이름 정리
desired_order = ['3', '2', '4', '0', '1', '5']
top5_genes_ordered = []

for cluster in desired_order:
    df = top5_degs_filtered[top5_degs_filtered['group'] == cluster]
    top5_genes_ordered.extend(df['names'].tolist())

# 7. 중복 제거하여 유전자 리스트 완성
top5_genes_unique = list(dict.fromkeys(top5_genes_ordered))

top5_degs = deg_all.groupby("group").head(5)

for cluster in top5_degs['group'].unique():
    print(f"\n Cluster {cluster} - Top 5 DEGs:")
    print(top5_degs[top5_degs["group"] == cluster][["names", "logfoldchanges", "pvals_adj"]])
    
desired_order = ['3', '2', '4', '0', '1', '5']
top5_genes_ordered = []

for cluster in desired_order:
    df = deg_all[deg_all['group'] == cluster].head(5)
    top5_genes_ordered.extend(df['names'].tolist())

top5_genes_unique = list(dict.fromkeys(top5_genes_ordered))


# %%
_, axs = plt.subplots(
    ncols=2, figsize=(10, 10), gridspec_kw={"wspace": 0.05, "left": 0.12}
)
plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)
for ipath, (descr, path) in enumerate(paths):
    data = sc.pl.paga_path(
        adata,
        path,
        top5_genes_unique,
        show_node_names=False,
        ax=axs[ipath],
        ytick_fontsize=12,
        left_margin=0.15,
        n_avg=50,
        annotations=["distance"],
        show_yticks=True if ipath == 0 else False,
        show_colorbar=False,
        color_map="Greys",
        groups_key="clusters",
        color_maps_annotations={"distance": "viridis"},
        title="{} path".format(descr),
        return_data=True,
        show=False,
    )
plt.show()

# %% [markdown]
# ### 5. Calculate branches using correlation

# %% [markdown]
# Diffusion pseudotime 에는 root cell -> 모든 세포까지의 거리를 기반으로 Pseudotime 을 결정 (forward time).
# 
# 그런데 분기점 (branch) 를 찾으려면, 
# 
# * reverse pseudotime (끝점에서 다시 모든 셀까지의 거리) 도 같이 계산해서, 
# 
# * forward 와 reverse pseudotime 간의 상관/역상관 (correlation/anti-correlation) 을 비교. 
# 
#     -> 이 상관관계가 극적으로 변하는 지점 = 분기점 

# %%
adata.obs

# %%
alk1_end_idx = np.argmax(adata.obsm['X_umap'][:, 1])
kc_end_idx = adata.obs.index.get_loc(adata.obs["Kupffer cell score"].idxmax())

# %%
fig = sc.pl.umap(adata, color='dpt_forward_pseudotime', return_fig=True)
x, y = adata.obsm['X_umap'][root_idx]
z, a = adata.obsm['X_umap'][alk1_end_idx]
b, c = adata.obsm['X_umap'][kc_end_idx]

ax = plt.gca()
ax.scatter(x, y, color='red', s=50, label='Root Cell')
ax.scatter(z, a, color='blue', s=50, label='ALK1 KO Cell')
ax.scatter(b, c, color='green', s=50, label='Mature Kupffer Cell')

plt.title('Forward DPT Pseudotime')
plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0))
plt.show()

# %%
adata.uns['iroot'] = kc_end_idx
sc.tl.dpt(adata)
adata.obs['dpt_reverse_kc_pseudotime'] = adata.obs['dpt_pseudotime']

# %%
fig = sc.pl.umap(adata, color='dpt_reverse_kc_pseudotime', return_fig=True)
x, y = adata.obsm['X_umap'][root_idx]
b, c = adata.obsm['X_umap'][kc_end_idx]

ax = plt.gca()
ax.scatter(x, y, color='red', s=50, label='Root Cell')
ax.scatter(b, c, color='yellow', s=50, label='Mature Kupffer Cell')

plt.title('Reverse DPT Pseudotime')
plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0))
plt.show()

# %%
adata.uns['iroot'] = alk1_end_idx
sc.tl.dpt(adata)
adata.obs['dpt_reverse_alk1_pseudotime'] = adata.obs['dpt_pseudotime']

# %%
fig = sc.pl.umap(adata, color='dpt_reverse_alk1_pseudotime', return_fig=True)
x, y = adata.obsm['X_umap'][root_idx]
z, a = adata.obsm['X_umap'][alk1_end_idx]

ax = fig.axes[0]
ax.scatter(x, y, color='red', s=50, label='Root Cell')
ax.scatter(z, a, color='blue', s=50, label='ALK1 KO Cell')


plt.title('Reverse DPT Pseudotime')
plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0))
plt.show()

# %%
configs = [
    ("dpt_forward_pseudotime", {"Root": "red", "ALK1": "blue", "KC": "green"}),
    ("dpt_reverse_kc_pseudotime", {"Root": "red", "KC": "green"}),
    ("dpt_reverse_alk1_pseudotime", {"Root": "red", "ALK1": "blue"}),
]

points = {
    "Root": adata.obsm["X_umap"][root_idx],
    "ALK1": adata.obsm["X_umap"][alk1_end_idx],
    "KC": adata.obsm["X_umap"][kc_end_idx],
}

titles = [
    "Forward DPT Pseudotime",
    "Reverse DPT Pseudotime (KC End)",
    "Reverse DPT Pseudotime (ALK1 KO End)",
]

# subplot 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

for i, (pseudotime_key, markers) in enumerate(configs):
    sc.pl.umap(adata, color=pseudotime_key, ax=axes[i], show=False)
    for label, color in markers.items():
        x, y = points[label]
        axes[i].scatter(x, y, color=color, s=50, label=f"{label} Cell")
    axes[i].set_title(titles[i])
    axes[i].legend(loc='upper left', bbox_to_anchor=(1.2, 1.0))

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Plotting forward vs reverse pseudotime

# %%
forward = adata.obs["dpt_forward_pseudotime"]
reverse_kc = adata.obs["dpt_reverse_kc_pseudotime"]

# %%
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(forward, reverse_kc, c=adata.obs["Kupffer cell score"], s=5)
ax.grid(False)


# %%
reverse_alk1 = adata.obs["dpt_reverse_alk1_pseudotime"]

fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(forward, reverse_alk1, c=adata.obs["Kupffer cell score"], s=5)
ax.grid(False)

# %% [markdown]
# #### Clustering in the forward vs. reverse pseudotime space

# %%
mono_kc = adata.obs[adata.obs['leiden'] != '5']

max_pseudotime = mono_kc["dpt_forward_pseudotime"].max()
max_idx = mono_kc["dpt_forward_pseudotime"].idxmax()
print("최대 Pseudotime:", max_pseudotime)
print("해당 세포:", max_idx)

# %%
adata.uns

# %%
import seaborn as sns

# 카테고리 → 정수로 인코딩
leiden_numeric = adata.obs['leiden'].astype('category').cat.codes

# 색상 팔레트 지정
palette = sns.color_palette("tab10", n_colors=adata.obs['leiden'].nunique())

# 색상 배열 생성
colors = [palette[i] for i in leiden_numeric]


# %%
import scanpy as sc
import seaborn as sns

x, y = forward["GCTTGGGGTGAATTAG-1"], reverse_kc["GCTTGGGGTGAATTAG-1"]
root_x, root_y = forward.iloc[root_idx], reverse_kc.iloc[root_idx]
kc_x, kc_y = forward.iloc[kc_end_idx], reverse_kc.iloc[kc_end_idx]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# DPT groups color 지정 
dpt_groups = adata.obs['dpt_groups'].unique()
dpt_palette = sns.color_palette("tab10", n_colors=len(dpt_groups)).as_hex()
group_colors = dict(zip(dpt_groups, dpt_palette))
dpt_groups_colors = adata.obs['dpt_groups'].map(group_colors)
adata.uns['dpt_groups_colors'] = [group_colors[g] for g in sorted(dpt_groups)]


# --- [0, 0] Forward vs Reverse pseudotime (기존 cell_colors 사용)
ax = axes[0, 0]
ax.scatter(forward, reverse_kc, c=colors, s=5)
ax.scatter(x, y, c='yellow', s=80, marker='*', label='Selected Cell')
ax.scatter(root_x, root_y, c='red', s=50, label='Root Cell')
ax.scatter(kc_x, kc_y, c='green', s=50, label='Mature KC Cell')
ax.set_xlabel("Forward pseudotime")
ax.set_ylabel("Reverse pseudotime")
ax.set_title("Forward vs Reverse Pseudotime")
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
ax.grid(False)

# --- [0, 1] UMAP colored by 'leiden'
sc.pl.umap(adata, color='leiden', ax=axes[0, 1], show=False)

# --- [1, 0] Forward vs Reverse pseudotime colored by 'dpt_groups'
ax = axes[1, 0]
ax.scatter(forward, reverse_kc, c=dpt_groups_colors, s=5)
ax.scatter(x, y, c='yellow', s=80, marker='*')
ax.scatter(root_x, root_y, c='red', s=50)
ax.scatter(kc_x, kc_y, c='green', s=50)
ax.set_xlabel("Forward pseudotime")
ax.set_ylabel("Reverse pseudotime")
ax.set_title("Colored by DPT Group")
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
ax.grid(False)

# --- [1, 1] UMAP colored by 'dpt_groups'
sc.pl.umap(adata, color='dpt_groups', ax=axes[1, 1], show=False)

# 전체 레이아웃 정리
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Option 1: DC1 vs DC2 scatter plot 

# %%
dc1 = adata.obsm['X_diffmap'][:, 0]
dc2 = adata.obsm['X_diffmap'][:, 1]
dpt = adata.obs['dpt_forward_pseudotime']

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(dc1, dc2, c=dpt, cmap='viridis')
plt.xlabel("DC1")
plt.ylabel("DC2")
plt.title("Diffusion Components colored by DPT")
plt.colorbar(sc, label="Pseudotime")
plt.grid(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# * x-axis: Diffusion Component 1
# 
# * y-axis: Diffusion Component 2
# 
# * No clear branching is observed in the diffusion map
# 
# * The structure appears to follow a single trunk-like trajectory
# 
# * The branching structure is ambiguous
# 
# * Pseudotime increases gradually along this trajectory
# 

# %% [markdown]
# #### Perform DPT again for Figure 

# %%
import scanpy as sc

# %%
root_idx = np.argmax(adata.obsm['X_umap'][:, 0])
adata.uns['iroot'] = root_idx 
sc.tl.dpt(adata, n_branchings=1)


# %% [markdown]
# ### Figure 

# %%
import matplotlib.pyplot as plt
import scanpy as sc

# --- Figure 설정 ---
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(22, 20), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

# --- 범례 helper 함수 ---
def move_legend_outside(ax, y_center=0.5, x_offset=1.02):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            bbox_to_anchor=(x_offset, y_center),
            loc='center left',
            frameon=False,
            fontsize=11,
            markerscale=2.2,
            labelspacing=1.2
        )

# ---------- Row 0 ----------

# (0, 0): UMAP by leiden
sc.pl.umap(adata, color='leiden', ax=axes[0, 0], show=False)
axes[0, 0].set_title("UMAP: Leiden Clusters")

# (0, 1): PAGA
sc.tl.paga(adata, groups="leiden_anno")
sc.pl.paga(adata, threshold=0.03, ax=axes[0, 1], show=False)
axes[0, 1].set_title("PAGA: Cluster Connectivity")

# (0, 2): Force-directed layout with PAGA init
sc.tl.draw_graph(adata, init_pos='paga')
sc.pl.draw_graph(adata, color='leiden_anno', legend_loc="on data", ax=axes[0, 2], show=False)
axes[0, 2].set_title("Force-directed Graph: Leiden Annotation")

# ---------- Row 1 ----------

# (1, 0): UMAP with Monocytes score & root cell
sc.pl.umap(adata, color='Monocytes score', ax=axes[1, 0], show=False)
x, y = adata.obsm['X_umap'][root_idx]
axes[1, 0].scatter(x, y, color='red', s=50, label='Root Cell')
move_legend_outside(axes[1, 0], x_offset=1.12)
axes[1, 0].set_title("UMAP: Monocyte Score & Root Cell")

# (1, 1): UMAP with DPT pseudotime
sc.pl.umap(adata, color="dpt_pseudotime", ax=axes[1, 1], show=False)
axes[1, 1].set_title("UMAP: DPT Pseudotime")

# (1, 2): Force-directed layout with DPT pseudotime
sc.pl.draw_graph(adata, color='dpt_pseudotime', ax=axes[1, 2], show=False)
axes[1, 2].set_title("Force-directed Graph: DPT Pseudotime")

# ---------- Row 2 ----------

# (2, 0): Forward vs Reverse pseudotime with cluster color
axes[2, 0].scatter(forward, reverse_kc, c=colors, s=5)
axes[2, 0].set_xlabel("Forward Pseudotime")
axes[2, 0].set_ylabel("Reverse Pseudotime")
axes[2, 0].set_title("Forward vs Reverse Pseudotime")
move_legend_outside(axes[2, 0], y_center=0.9)
axes[2, 0].grid(False)

# (2, 1): Forward vs Reverse pseudotime with KC score
sca = axes[2, 1].scatter(forward, reverse_kc, c=adata.obs["Kupffer cell score"], s=5)
axes[2, 1].set_title("Pseudotime vs KC Score")
axes[2, 1].set_xlabel("Forward Pseudotime")
axes[2, 1].set_ylabel("Reverse Pseudotime")
axes[2, 1].grid(False)
plt.colorbar(sca, ax=axes[2, 1], fraction=0.046, pad=0.04)

# (2, 2): Violin plot of DPT pseudotime per leiden cluster
sc.pl.violin(adata, keys="dpt_pseudotime", groupby="leiden", ax=axes[2, 2], show=False)
axes[2, 2].set_title("DPT Pseudotime by Leiden Cluster")

# --- 전체 출력 ---
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()


# %% [markdown]
# #### Option 2: Monocyte 제외 Only Kupffer cell cluster 로만 시작하기

# %%
only_kc = adata[adata.obs['leiden']!='3']

# %%
sc.pl.umap(only_kc, color='leiden')

# %%
sc.tl.pca(only_kc)
sc.pp.neighbors(only_kc, n_neighbors=50)

# %%
sc.tl.umap(only_kc)

# %%
sc.pl.umap(only_kc, color='gAlk1_1')

# %%
sc.tl.leiden(only_kc, resolution=1)

# %%
sc.pl.umap(only_kc, color=['leiden', 'Monocytes score'])

# %%
sc.tl.diffmap(only_kc) 

# %%
sc.tl.draw_graph(only_kc)
sc.pl.draw_graph(only_kc, color="leiden", legend_loc="on data")

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# (0, 0) Draw graph with leiden color
sc.pl.draw_graph(only_kc, color="leiden", legend_loc="on data", ax=axes[0, 0], show=False)
axes[0, 0].set_title('Draw Graph (leiden)')

# (0, 1) UMAP - Monocyte score
sc.pl.umap(only_kc, color='Monocytes score', ax=axes[0, 1], show=False)
axes[0, 1].set_title('UMAP - Monocyte score')

# (1, 0) UMAP - gAlk1_1
sc.pl.umap(only_kc, color='gAlk1_1', ax=axes[1, 0], legend_loc="on data", show=False)
axes[1, 0].set_title('UMAP - gAlk1_1 distribution')

# (1, 1) UMAP - leiden
sc.pl.umap(only_kc, color='leiden', ax=axes[1, 1], legend_loc="on data", show=False)
axes[1, 1].set_title('UMAP - Leiden Clustering')

plt.tight_layout
plt.show()


# %% [markdown]
# #### Set root cell 
# 

# %%
root_idx = np.argmin(only_kc.obsm['X_umap'][:, 0])
only_kc.uns['iroot'] = root_idx 


# %%
sc.tl.dpt(only_kc, n_branchings=1)

# %%
sc.pl.umap(only_kc, color=['dpt_pseudotime', 'Clec4f', 'dpt_groups', 'leiden'], legend_loc="on data")

# %%
only_kc.obs['dpt_forward_pseudotime'] = only_kc.obs['dpt_pseudotime']

# %%
sc.tl.paga(only_kc, groups='leiden')

# %%
import scanpy as sc

# %%
sc.pl.paga(only_kc, color='leiden', threshold=0.2)

# %% [markdown]
# #### Calculate branches using correlation

# %%
kc_end_idx = np.argmax(only_kc.obsm['X_umap'][:, 0])

# %%
cluster_4_cells = only_kc[only_kc.obs['leiden'] == '7']
idx_within_cluster = np.argmin(cluster_4_cells.obsm['X_umap'][:, 0])
alk1_end_idx = np.where(only_kc.obs_names == cluster_4_cells.obs_names[idx_within_cluster])[0][0]

# %%
fig = sc.pl.umap(only_kc, color='dpt_forward_pseudotime', return_fig=True)
x, y = only_kc.obsm['X_umap'][root_idx]
z, a = only_kc.obsm['X_umap'][alk1_end_idx]
b, c = only_kc.obsm['X_umap'][kc_end_idx]

ax = plt.gca()
ax.scatter(x, y, color='red', s=50, label='Root Cell')
ax.scatter(z, a, color='blue', s=50, label='ALK1 KO Cell')
ax.scatter(b, c, color='green', s=50, label='Mature Kupffer Cell')

plt.title('Forward DPT Pseudotime')
plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0))
plt.show()

# %%
only_kc.uns['iroot'] = kc_end_idx
sc.tl.dpt(only_kc, n_branchings=1)
only_kc.obs['dpt_reverse_kc_pseudotime'] = only_kc.obs['dpt_pseudotime']

# %%
only_kc.uns['iroot'] = alk1_end_idx
sc.tl.dpt(only_kc, n_branchings=1)
only_kc.obs['dpt_reverse_alk1_pseudotime'] = only_kc.obs['dpt_pseudotime']

# %%
forward = only_kc.obs["dpt_forward_pseudotime"]
reverse_kc = only_kc.obs["dpt_reverse_kc_pseudotime"]

# %%
sc.pl.umap(only_kc, color=["dpt_forward_pseudotime", "dpt_reverse_kc_pseudotime"])

# %%
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(forward, reverse_kc, c=only_kc.obs["Kupffer cell score"], s=5)
ax.grid(False)

# %%
import scanpy as sc

leiden_numeric = only_kc.obs['leiden'].astype(int)

fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(forward, reverse_kc, c=leiden_numeric, s=5)
ax.grid(False)

# %%
import scanpy as sc

sc.pl.umap(only_kc, color=['leiden', 'gAlk1_1', 'Id3', 'dpt_groups'])

# %%
# 1. scanpy가 저장한 leiden 색상 불러오기
leiden_colors = only_kc.uns['leiden_colors']  # 클러스터 색상 리스트
leiden_color_dict = {str(i): c for i, c in enumerate(leiden_colors)}  # 클러스터 번호(str) → 색상

# 2. 색상 맵핑 (cell 단위로 색 지정)
cell_colors = only_kc.obs['leiden'].map(leiden_color_dict)

# 3. Scatter plot
fig, ax = plt.subplots(figsize=(5, 5))
sca = ax.scatter(forward, reverse_kc, c=cell_colors, s=5)

# 4. 축 라벨 및 제목
ax.set_xlabel("Forward Pseudotime")
ax.set_ylabel("Reverse Pseudotime")
ax.set_title("Forward vs Reverse Pseudotime")

ax.grid(False)
plt.show()


# %%
dc1 = only_kc.obsm['X_diffmap'][:, 0]
dc2 = only_kc.obsm['X_diffmap'][:, 1]
dpt = only_kc.obs['dpt_forward_pseudotime']

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(dc1, dc2, c=dpt, cmap='viridis')
plt.xlabel("DC1")
plt.ylabel("DC2")
plt.title("Diffusion Components colored by DPT")
plt.colorbar(sc, label="Pseudotime")
plt.grid(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### 임의의 Path 설정

# %%
only_kc.obs["branch_1"] = only_kc.obs["leiden"].isin(["3", "7"])
only_kc.obs["branch_2"] = ~only_kc.obs["leiden"].isin(["7", "0", "8"])


# %%
only_kc.obs

# %%
# pseudotime 값
pt = only_kc.obs["dpt_forward_pseudotime"]

# branch 마스크
mask1 = only_kc.obs["branch_1"]
mask2 = only_kc.obs["branch_2"]

# -----------------------------------
#  Figure 1: Branch 1 강조
fig1, ax1 = plt.subplots(figsize=(7, 6))

# 전체 셀 회색 배경
ax1.scatter(forward, reverse_kc, c='black', s=3, alpha=0.4)

# branch 1만 pseudotime 색상으로 강조
sc1 = ax1.scatter(forward[mask1], reverse_kc[mask1],
                  c=pt[mask1], cmap='viridis', s=10)

ax1.set_title("Branch 1 highlighted (by pseudotime)")
ax1.grid(False)
cbar1 = plt.colorbar(sc1, ax=ax1, label="Pseudotime")

# -----------------------------------
#  Figure 2: Branch 2 강조
fig2, ax2 = plt.subplots(figsize=(7, 6))

# 전체 셀 회색 배경
ax2.scatter(forward, reverse_kc, c='black', s=3, alpha=0.4)

# branch 2만 pseudotime 색상으로 강조
sc2 = ax2.scatter(forward[mask2], reverse_kc[mask2],
                  c=pt[mask2], cmap='viridis', s=10)

ax2.set_title("Branch 2 highlighted (by pseudotime)")
ax2.grid(False)
cbar2 = plt.colorbar(sc2, ax=ax2, label="Pseudotime")

plt.show()

# %% [markdown]
# #### DPT again

# %%
import scanpy as sc

root_idx = np.argmin(only_kc.obsm['X_umap'][:, 0])
only_kc.uns['iroot'] = root_idx 

sc.tl.dpt(only_kc, n_branchings=1)

# %% [markdown]
# #### 통합 Figure 
# 

# %%
import matplotlib.pyplot as plt
import scanpy as sc

# --- Figure 설정 (오른쪽 범례 공간 확보를 위해 right=0.85 이하로 설정) ---
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(23, 6), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

# ---------- Row 0 ----------

# (0): UMAP by leiden with root cell 표시
sc.pl.umap(only_kc, color='leiden', ax=axes[0], show=False, legend_loc="on data")
x, y = only_kc.obsm['X_umap'][root_idx]
root_plot_0 = axes[0].scatter(x, y, color='red', s=50, label='Root Cell')  
axes[0].set_title("UMAP: Kupffer Cells")

# (1): PAGA on only_kc
sc.pl.paga(only_kc, color='leiden', threshold=0.2, ax=axes[1], show=False)
axes[1].set_title("PAGA (threshold=0.2)")

# (2): Forward pseudotime + 세 지점 표시
sc.pl.umap(only_kc, color='dpt_forward_pseudotime', ax=axes[2], show=False)
x, y = only_kc.obsm['X_umap'][root_idx]
z, a = only_kc.obsm['X_umap'][alk1_end_idx]
b, c = only_kc.obsm['X_umap'][kc_end_idx]

root_plot = axes[2].scatter(x, y, color='red', s=50, label='Root Cell')
alk1_plot = axes[2].scatter(z, a, color='blue', s=50, label='ALK1 KO Cell')
kc_plot = axes[2].scatter(b, c, color='green', s=50, label='Mature Kupffer Cell')
axes[2].set_title("Forward DPT Pseudotime")

# --- Figure 오른쪽 중앙에 전체 범례 추가 ---
fig.legend(
    handles=[root_plot_0, alk1_plot, kc_plot],
    labels=["Root Cell", "ALK1 KO Cell", "Mature Kupffer Cell"],
    loc='center left',
    bbox_to_anchor=(0.88, 0.5),
    frameon=False,
    fontsize=11,
    markerscale=2.2,
    labelspacing=1.2
)

# --- 출력 ---
plt.subplots_adjust(hspace=0.3, wspace=0.3, right=0.85)
plt.show()


# %%
adata.write('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/04_ti.h5ad')


