# %% [markdown]
# # Differentially Expressed Gene Analysis

# %% [markdown]
# ## Environmental Setup

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
import os


warnings.simplefilter(action="ignore", category=Warning)

# verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.verbosity = 2

import muon as mu
import pertpy as pt 
from scanpy.tools._utils import _choose_representation

# %% [markdown]
# ## Load Data

# %%
adata = sc.read('/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/01_kc_sgRNA.h5ad')

# %% [markdown]
# ## 1. gALk1 KO cluster Shows Macrophage Identity

# %%
adata.obs

# %%
all_ma_genes = ['Pecr', 'Tmem195', 'Ptplad2', '1810011H11Rik', 'Fert2', 'Tlr4', 'Pon3', 'Mr1', 'Arsg', 'Fcgr1', 'Camk1', 'Fgd4', 'Sqrdl', 'Csf3r', 'Plod1',
                'Tom1', 'Myo7a', 'A930039A15Rik', 'Pld3', 'Tpp1', 'Ctsd', 'Pla2g15', 'Lamp2', 'Pla2g4a', 'MerTK', 'Xrcc5', 'Gm4878', 'Slco2b1', 'Gpr77', 'Gpr160', 'P2ry13', 'Tanc2', 'Sepn1', 'Mafb', 'Itga9', 'Cmklr1', 'Fez2', 'Tspan4', 'Abcc3',
                'Nr1d1', 'Ptprm', 'Ctsf', 'Tfpi', 'Hgf', 'Pilrb2', 'Mgst1', 'Klra2', 'Rnasel', 'Fcgr4', 'Rhoq', 'Fpr1', 'Cd302', 'Slc7a2', 'Slc16a7', 'Slc16a10', 'Slpi', 'Mitf', 'Snx24', 'Lyplal1', 'St7', 'Cd151', 'Lonrf3', 'Acy1']

# %%
filtered_ma_genes = [gene for gene in all_ma_genes if gene in adata.var_names]

# %% [markdown]
# #### Figure 만들기 

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 각 UMAP subplot 그리기
sc.pl.umap(adata, color='KCs differentiation (Initial)', ax=axes[0, 0], show=False, title='KCs differentiation (Initial)')
sc.pl.umap(adata, color='Monocytes score', ax=axes[0, 1], show=False, title='Monocytes score')
sc.pl.umap(adata, color='Kupffer cell score', ax=axes[1, 0], show=False, title='Kupffer cell score')
sc.pl.umap(adata, color='gAlk1_1', ax=axes[1, 1], show=False, title='gAlk1_1')

# 레이아웃 조정
plt.tight_layout()
plt.show()


# %%
filtered_ma_genes

# %%
groupby_1 = 'KCs differentiation (Initial)'

# %%
# 1. 그룹별 평균 발현값 계산
sc.tl.rank_genes_groups(adata, groupby=groupby_1, method='wilcoxon')  # Optional
avg_expr = sc.get.obs_df(adata, keys=filtered_ma_genes + [groupby_1]).groupby(groupby_1).mean().T

# 2. z-score 정규화 (행 기준: 유전자별)
avg_expr_z = avg_expr.sub(avg_expr.mean(axis=1), axis=0).div(avg_expr.std(axis=1), axis=0)

# %%
cmap = 'RdYlBu_r' 

# %%
import seaborn as sns

sns.set(font_scale=0.9)
g = sns.clustermap(
    avg_expr_z,
    cmap=cmap,
    center=0,
    col_cluster=False,
    row_cluster=False,
    figsize=(8, 10),
    xticklabels=True,
    yticklabels=True)

plt.tight_layout()
plt.show()

# %%


sc.pl.heatmap(
    adata,
    filtered_ma_genes,
    groupby='KCs differentiation (Initial)',
    cmap='RdYlBu_r',
    vmin=0,               # 논문 스타일 z-score처럼 조정
    vmax=1,
    show_gene_labels=True,
    swap_axes=True, 
    show=False
)

plt.tight_layout()
plt.show()


# %% [markdown]
# #### Macrophage genes

# %%
ma_dict = {
    'all_ma_genes': ['Pecr', 'Tmem195', 'Ptplad2', '1810011H11Rik', 'Fert2', 'Tlr4', 'Pon3', 'Mr1', 'Arsg', 'Fcgr1', 'Camk1', 'Fgd4', 'Sqrdl', 'Csf3r'],
    'Peritoneal_ma': ['Xrcc5', 'Gm4878', 'Slco2b1', 'Gpr77', 'Gpr160', 'P2ry13', 'Tanc2', 'Sepn1'],
    'lung_ma': ['Mafb', 'Itga9', 'Cmklr1', 'Fez2', 'Tspan4', 'Abcc3', 'Nr1d1', 'Ptprm', 'Ctsf', 'Tfpi'],
    'microglia_ma': ['Hgf', 'Pilrb2', 'Mgst1', 'Klra2', 'Rnasel', 'Fcgr4', 'Rhoq', 'Fpr1', 'Cd302', 'Slc7a2', 'Slc16a7', 'Slc16a10', 'Slpi', 'Mitf', 'Snx24', 'Lyplal1', 'St7'],
    'splenic_red_pulp_ma': ['Cd151', 'Lonrf3', 'Acy1']
    
}

# %%
# filter only genes present in adata.var_names
for group in ma_dict:
    ma_dict[group] = [g for g in ma_dict[group] if g in adata.var_names]

# %%
ma_dict

# %%
{key: len(value) for key, value in ma_dict.items()}


# %%
gene_order = []
var_group_labels = []
var_group_positions = []

current_index = 0
for group, genes in ma_dict.items():
    gene_order.extend(genes)
    var_group_labels.append(group)
    start = current_index
    end = current_index + len(genes) - 1
    var_group_positions.append((start, end))
    current_index += len(genes)


# %%
var_group_labels = [
    "All MΦ populations",
    "-Peritoneal MΦ",
    "-Lung MΦ",
    "-Microglia",
    "-Splenic red-pulp MΦ"
]


# %%
sc.pl.heatmap(
    adata,
    var_names=gene_order,
    groupby="KCs differentiation (Initial)",
    var_group_labels=var_group_labels,
    var_group_positions=var_group_positions,
    vmax=1,
    cmap='RdYlBu_r',
    swap_axes=True
    )



# %%
MΦ_core_genes = ['Pecr',
                 'Tmem195',
                 'Ptplad2',
                 '1810011H11Rik',
                 'Fert2',
                 'Tlr4',
                 'Mr1',
                 'Arsg',
                 'Fcgr1',
                 'Fgd4',
                 'Sqrdl',
                 'Csf3r',
                 'Plod1',
                 'Tom1',
                 'Pld3',
                 'Tpp1',
                 'Ctsd',
                 'Lamp2',
                 'Pla2g4',
                 'Mertk',
                 'Tlr7',
                 'Cd14',
                 'Tbxas1',
                 'Fcgr3',
                 'Sepp1',
                 'Cd164',
                 'Tcn2',
                 'Dok3',
                 'Ctsl',
                 'Tspan14',
                 'Comt1',
                 'Tmem77',
                 'Abca1']


# %%
MΦ_core_genes = [gene for gene in MΦ_core_genes if gene in adata.var_names]

# %%
MΦ_core_genes

# %%
sc.pl.heatmap(
    adata,
    MΦ_core_genes,
    groupby="KCs differentiation (Initial)",
    vmax=1,
    cmap='RdYlBu_r',
    swap_axes=True
    )



# %% [markdown]
# ## 2. gAlk1 cluster loss KC identity - Scott et al, 2018

# %%
kupffer_identity_genes = [
    "Vsig4", "Id1", "Id3", "Clec4f", "Clec1b", "Il18bp", "C6", "Irf7", "Slc40a1", "Timd4",
    "Cdh5", "Dmpk", "Nr1h3", "Paqr9", "Pcolce2", "Kcna2", "Gbp8", "Helz2", 
    "Cd207", "Icos", "Adcy4", "Slc1a2", "Rsad2", "Slc16a9", "Cd209f", "Oasl1",
    "Ram167a", "Cd5l", "Cxcl13", "Fabp7"
]

filtered_kc_genes = [gene for gene in kupffer_identity_genes if gene in adata.var_names]

# heatmap 시각화
sc.pl.heatmap(
    adata,
    var_names=filtered_kc_genes,
    groupby="KCs differentiation (Initial)",
    swap_axes=True,
    show_gene_labels=True,
    vmax=1,
    cmap='RdYlBu_r',
    show=True
)


# %%
sc.pl.umap(adata, color=filtered_kc_genes, ncols=4, cmap='viridis')

# %%
cluster_key = 'KCs differentiation (Initial)'

sc.pl.dotplot(
    adata,
    var_names=filtered_kc_genes,
    groupby=cluster_key,
    swap_axes=True,
    dot_max=0.5,
    figsize=(4, 8),
    smallest_dot=0.0
)

# %% [markdown]
# ## 3. Mixscape

# %%
adata.obs

# %%
def prepare_mixscape_metadata(adata):
    adata.obs['perturbation'] = adata.obs['feature_call'].apply(
        lambda x: 'NT' if x == 'gNeg_1' else 'Perturbed'
    )
    
    adata.obs['gene_target'] = adata.obs['feature_call'].apply(
        lambda x: 'NT' if x == 'gNeg_1' else x
    )
    return adata

# %%
adata = prepare_mixscape_metadata(adata)

# %%
adata.obs

# %%
adata.obs['perturbation'].value_counts()

# %%
adata.obs['gene_target'].value_counts()

# %%
sc.pl.umap(adata, color=["condition", "sample"], title=['KO condition', 'gAlk KO'], show=False, ncols=1, color_map='viridis')

fig = plt.gcf()
fig.set_size_inches(5, 9)  
plt.tight_layout()
plt.show() 

# %%
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 8), gridspec_kw={'hspace': 0.25})
sc.pl.umap(
    adata[adata.obs['perturbation'] == 'Perturbed'],
    ax=axs[0],
    color='perturbation',
    palette=['blue'],
    title='Perturbed Cells',
    show=False)

sc.pl.umap(
    adata[adata.obs['perturbation'] == 'NT'],
    ax=axs[1],
    color='perturbation',
    palette=['grey'],
    title='NT (Control) Cells',
    show=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

# %% [markdown]
# ## Calculating local perturbation signatures mitigates confounding effects

# %%
ms = pt.tl.Mixscape()
ms.perturbation_signature(adata, "perturbation", "NT")

# %%
adata_pert = adata.copy()
adata_pert.X = adata_pert.layers["X_pert"]

# %%
adata_pert.X

# %%
sc.pp.pca(adata_pert)
sc.pp.neighbors(adata_pert)
sc.tl.umap(adata_pert)

# %%
sc.pl.umap(adata_pert, color=["condition", "gAlk1_1"], title=['KO condition', 'gAlk KO'], show=False, wspace=0.4)

fig = plt.gcf()
fig.set_size_inches(11, 4)  
plt.tight_layout()
plt.show() 

# %%
sc.pl.umap(adata_pert[adata_pert.obs['perturbation'] == 'Perturbed'], color='perturbation', title='Perturbed')

# %%
sc.pl.umap(adata_pert[adata_pert.obs['perturbation'] == 'NT'], color='perturbation', title='Control')

# %%
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 8), gridspec_kw={'hspace': 0.25})
sc.pl.umap(
    adata_pert[adata_pert.obs['perturbation'] == 'Perturbed'],
    ax=axs[0],
    color='perturbation',
    palette=['blue'],
    title='Perturbed Cells',
    show=False)

sc.pl.umap(
    adata_pert[adata_pert.obs['perturbation'] == 'NT'],
    ax=axs[1],
    color='perturbation',
    palette=['grey'],
    title='NT (Control) Cells',
    show=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

# %% [markdown]
# #### Figure construction (4 x 2 Figure)

# %% [markdown]
# KO condition & Sample palette 사전 정의

# %%
# KO condition 별 색상 (일관되게)
condition_palette = {
    'Alk1_KO': '#d62728', 
    'Cd64_KO': '#1f77b4',  # 파랑
    'F480_KO': '#2ca02c',  # 녹색
    'Control': '#17becf'   # 시안
}

# Sample은 다른 계열 색상
sample_palette = {
    'ABL005': '#ff7f0e',  # 오렌지
    'ABL006': '#9467bd',  # 보라
    'ABL007': '#8c564b'   # 갈색
}

pert_palette = {'Perturbed': '#f7b938', 'NT': '#7f7f7f'}  # 노랑, 회색

galk1_palette = {
    'positive': '#d62728', # 빨강
    'negative': '#7f7f7f'  # 회색
}


# %%
adata.obs

# %%
sc.pl.umap(adata_pert, color='gAlk1_1', palette=galk1_palette)

# %% [markdown]
# ## Mixscape identifies cells with no detectable perturbation

# %%
ms.mixscape(adata, control="NT", labels="gene_target", layer="X_pert")

# %%
adata.obs["mixscape_class"].value_counts()

# %%
adata_pert.obs["mixscape_class_global"] = adata.obs.loc[adata_pert.obs_names, "mixscape_class_global"]

def create_condition2_column(adata_pert):
    condition2 = []
    for i in adata_pert.obs.index:
        status = adata_pert.obs.loc[i, "mixscape_class_global"]
        gene = adata_pert.obs.loc[i, "gene_target"]

        if status == "KO":
            label = f"{gene}_KO"
        elif status == "NP":
            label = "Non-perturbed cell"
        elif status == "NT":
            label = "Control"
        else:
            label = "Unknown"

        condition2.append(label)
    
    adata_pert.obs["condition2"] = condition2

create_condition2_column(adata_pert)

# %%
import matplotlib.pyplot as plt
import scanpy as sc

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(26, 10), gridspec_kw={'hspace': 0.3, 'wspace': 0.6})

# helper: 오른쪽 바깥에 범례 두기 (y 위치 조절 가능)
def move_legend_outside(ax, y_center=0.5):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            bbox_to_anchor=(1.02, y_center),  # 오른쪽 바깥에 legend
            loc='center left',
            frameon=False,
            fontsize=11,
            markerscale=2.2,
            labelspacing=1.2  # 줄 간격
        )

# ---------------- UMAP plots + 범례 이동 ----------------
# KO condition
sc.pl.umap(adata, color="condition", palette=condition_palette, title='KO condition', size=20, ax=axs[0, 0], show=False)
move_legend_outside(axs[0, 0])

# Sample
sc.pl.umap(adata, color="sample", palette=sample_palette, title='Sample', size=20, ax=axs[1, 0], show=False)
move_legend_outside(axs[1, 0])

# Perturbed Cells
sc.pl.umap(adata[adata.obs['perturbation'] == 'Perturbed'],
           color='perturbation', palette=[pert_palette['Perturbed']],
           title='Perturbed Cells', size=20, ax=axs[0, 1], show=False)
move_legend_outside(axs[0, 1])

# NT Cells
sc.pl.umap(adata[adata.obs['perturbation'] == 'NT'],
           color='perturbation', palette=[pert_palette['NT']],
           title='NT (Control) Cells', size=20, ax=axs[1, 1], show=False)
move_legend_outside(axs[1, 1])

# KO condition in adata_pert
sc.pl.umap(adata_pert, color="condition", palette=condition_palette, title='KO condition', size=20, ax=axs[0, 2], show=False)
move_legend_outside(axs[0, 2])

# Sample in adata_pert
sc.pl.umap(adata_pert, color="sample", palette=sample_palette, title='Sample', size=20, ax=axs[1, 2], show=False)
move_legend_outside(axs[1, 2])

# Perturbed Cells in adata_pert
sc.pl.umap(adata_pert[adata_pert.obs['perturbation'] == 'Perturbed'],
           color='perturbation', palette=[pert_palette['Perturbed']],
           title='Perturbed Cells', size=20, ax=axs[0, 3], show=False)
move_legend_outside(axs[0, 3])

# NT Cells in adata_pert
sc.pl.umap(adata_pert[adata_pert.obs['perturbation'] == 'NT'],
           color='perturbation', palette=[pert_palette['NT']],
           title='NT (Control) Cells', size=20, ax=axs[1, 3], show=False)
move_legend_outside(axs[1, 3])

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

# %% [markdown]
# ## Inspecting mixscape results

# %%
subset = adata.obs[adata.obs["gene_target"] == "gAlk1_1"]
counts = subset["mixscape_class_global"].value_counts(normalize=True) * 100  

# 결과 출력
print("gAlk1_1 Perturbed 세포 중 Mixscape 분류 비율:")
print(counts[["KO", "NP"]])

# %% [markdown]
# #### Perturbation Score Palette Setting

# %%
def build_mix_palette(mix_labels, condition_palette):
    palette = {}
    for label in mix_labels:
        if label == 'NT':
            palette[label] = '#7d7d7d'
        elif label.endswith(' NP'):
            palette[label] = '#c9c9c9'
        elif label.endswith(' KO'):
            gene_full = label.split(' ')[0]       
            gene_core = gene_full.split('_')[0][1:]  
            condition_key = f"{gene_core}_KO"
            palette[label] = condition_palette.get(condition_key, '#000000')
        else:
            palette[label] = '#000000'  # 예외 처리
    return palette


# mix label 들로부터 palette 생성
mix_labels = adata.obs["mixscape_class"].unique()
perturb_score_palette = build_mix_palette(mix_labels, condition_palette)


# %%
perturb_score_palette

# %% [markdown]
# #### Before Mixscape

# %%
import inspect

print(inspect.getfile(ms.plot_perturbscore))

# %%
ms.plot_perturbscore(adata, labels="gene_target", target_gene="gAlk1_1", color='#d62728', before_mixscape=True)

# %% [markdown]
# #### After Mixscape

# %%
ms.plot_perturbscore(adata, labels="gene_target", target_gene="gAlk1_1", palette=perturb_score_palette)

# %%
ms.plot_violin(
    adata,
    target_gene_idents=["NT", "gAlk1_1 NP", "gAlk1_1 KO"],
    groupby="mixscape_class",
)

# %%
ms.plot_violin(
    adata,
    target_gene_idents=["NT", "gAlk1_1 NP", "gAlk1_1 KO"],
    groupby="mixscape_class",
    palette={
        "NT": "#7d7d7d",          # 회색
        "gAlk1_1 NP": "#c9c9c9",  # 연회색
        "gAlk1_1 KO": "#d62728",  # 빨강
    }
)


# %%
ms.plot_heatmap(
    adata,
    labels="gene_target",
    target_gene="gAlk1_1",
    layer="X_pert", 
    control='NT')

# %%
alk1_am_genes = ['Nanp', 'Zfyve21', 'Bivm', 'Nudcd1', 'Tbl2', 'Wtip', 'Gm10484']

# %%
sc.pl.umap(adata, color=alk1_am_genes)

# %%
adata.obs['gene_target']

# %%
adata_subset = adata[(adata.obs['gene_target'] == 'gAlk1_1') | (adata.obs['gene_target'] == 'NT')].copy()

sc.tl.rank_genes_groups(adata_subset, layer='X_pert', groupby="mixscape_class", method='wilcoxon')


# %%
adata_subset.uns["rank_genes_groups"]

# %%
group = 'NT'  # 예: mixscape_class 중 NT 그룹
degs = pd.DataFrame({
    "gene": adata_subset.uns["rank_genes_groups"]["names"][group],
    "logFC": adata_subset.uns["rank_genes_groups"]["logfoldchanges"][group],
    "pval_adj": adata_subset.uns["rank_genes_groups"]["pvals_adj"][group],
    "scores": adata_subset.uns["rank_genes_groups"]["scores"][group]
})
print(degs.head(20))


# %%
adata.obs

# %%
# ms.plot_perturbscore(adata, labels="gene_target", target_gene="gCd64_1", color="#1f77b4", before_mixscape=True)

# %%
# ms.plot_perturbscore(adata, labels="gene_target", target_gene="gCd64_1", palette=perturb_score_palette)

# %%
# ms.plot_heatmap(
#     adata,
#     labels="gene_target",
#     target_gene="gCd64_1",
#     layer="X_pert", 
#     control='NT'
# )

# %%
# ms.plot_violin(
#     adata=adata,
#     target_gene_idents=["NT", "gCd64_1 NP", "gCd64_1 KO"],
#     groupby="mixscape_class",
# )

# %%
# ms.plot_perturbscore(adata, labels="gene_target", target_gene="gF480_1", color="green", before_mixscape=True)

# %%
# ms.plot_perturbscore(adata, labels="gene_target", target_gene="gF480_1", color="green")

# %%
# ms.plot_violin(
#     adata,
#     target_gene_idents=["NT", "gF480_1 NP", "gF480_1 KO"],
#     groupby="mixscape_class",
# )


# %%
# ms.plot_heatmap(
#     adata,
#     labels="gene_target",
#     target_gene="gF480_1",
#     layer="X_pert", 
#     control='NT'
# )

# %% [markdown]
# ## Visualizing perturbation responses with Linear Discriminant Analysis (LDA)

# %%
ms = pt.tl.Mixscape()
ms.lda(adata, control="NT", labels="gene_target", layer="X_pert")

# %%
print(adata.uns["mixscape_lda"].shape)

# %%
print(adata.obs["mixscape_class"].value_counts())

# %% [markdown]
# ## 4. DEG Comparison

# %% [markdown]
# ### 4.1. Inpect perturbation marker (Standard Wilcoxon rank sum test)

# %%
import pickle

f_path = '/srv/data/allUsers/lilyh/crispyKC/code/perturbseq/th_ABL005-007/data/filtered_perturbation_markers_detailed.pkl'

# %%
with open(f_path, 'rb') as f:
    perturbation_markers = pickle.load(f)

# %%
perturbation_markers

# %%
std_results = {}

for (group, gene_target), data in perturbation_markers.items():
    gene_names = data["names"]
    logfc = data["logfoldchanges"]
    pvals = data["pvals_adj"]

    df = pd.DataFrame({
        "gene": gene_names,
        "logFC": logfc,
        "pval_adj": pvals
    })

    # Upregulated: logFC > 0
    up_df = df[df["logFC"] > 0].sort_values(by="logFC", ascending=False).reset_index(drop=True)
    # Downregulated: logFC < 0
    down_df = df[df["logFC"] < 0].sort_values(by="logFC").reset_index(drop=True)

    std_results[gene_target] = {
        "upregulated": up_df,
        "downregulated": down_df
    }

# %%
std_results['gAlk1_1']['upregulated'].head(20)

# %%
std_results['gAlk1_1']['downregulated'].head(20)

# %%
std_results["gAlk1_1"]

# %% [markdown]
# Standard: 252 upregulated genes & 1445 downregulated genes 

# %%
std_top40 = (
    std_results["gAlk1_1"]["downregulated"].head(20)["gene"].tolist() + 
    std_results["gAlk1_1"]["upregulated"].head(20)["gene"].tolist() 
)


# %% [markdown]
# #### Inpect Mixscape DEGs (NT & NP vs KO)

# %%
ko_cells = adata[adata.obs["mixscape_class"] == "gAlk1_1 KO"].copy()
ko_cells.obs["group"] = "KO"

ctrl_cells = adata[
    (adata.obs["mixscape_class"] == "gAlk1_1 NP") |
    (adata.obs["mixscape_class"] == "NT")
].copy()
ctrl_cells.obs["group"] = "CTRL"

deg_data = ko_cells.concatenate(ctrl_cells)

sc.tl.rank_genes_groups(
    deg_data,
    groupby="group",
    reference="CTRL",
    method="wilcoxon"
)


# %%
degs = sc.get.rank_genes_groups_df(deg_data, group="KO")

# 필터링 기준
logfc_thresh = 0.25
pval_thresh = 0.05
min_de_genes = 5

ms_results = degs[
    (degs["pvals_adj"] < pval_thresh) &
    (abs(degs["logfoldchanges"]) >= logfc_thresh)
]

ms_top_up = ms_results[ms_results["logfoldchanges"] > 0].sort_values(by="logfoldchanges", ascending=False)
ms_top_down = ms_results[ms_results["logfoldchanges"] < 0].sort_values(by="logfoldchanges")

ms_top40 = ms_top_down["names"].head(20).tolist() + ms_top_up["names"].head(20).tolist() 



# %%
ms_results

# %%
ms_top_up

# %%
ms_top_down

# %% [markdown]
# Mixscape DEG: 344 Upregulated genes & 863 Downregulated genes

# %% [markdown]
# #### Standard & Mixscape - Top 20 Upregulated & Downregulatd genes heatmap 

# %%
# 1. gAlk1_1 샘플 150개
alk1_cells = adata[adata.obs["gene_target"] == "gAlk1_1"]
alk1_sampled = alk1_cells[np.random.choice(alk1_cells.shape[0], min(150, alk1_cells.shape[0]), replace=False)].copy()

# 2. NT 샘플 150개
nt_cells = adata[adata.obs["gene_target"] == "NT"]
nt_sampled = nt_cells[np.random.choice(nt_cells.shape[0], min(150, nt_cells.shape[0]), replace=False)].copy()

# 3. 병합
alk1_sampled.obs["group"] = "gAlk1_1"
nt_sampled.obs["group"] = "NT"
sampled_adata = alk1_sampled.concatenate(nt_sampled)

# 4. 그룹 순서 고정
sampled_adata.obs["group"] = pd.Categorical(
    sampled_adata.obs["group"],
    categories=["NT", "gAlk1_1"],
    ordered=True
)

# 5. heatmap 그리기
g = sc.pl.heatmap(
    sampled_adata,
    var_names=std_top40,                # <- 너가 지정한 top 40 유전자
    groupby="group",
    use_raw=False,
    vmin=0,
    vmax=1,
    cmap=cmap,
    show=False,
    swap_axes=True
)

# 6. 유전자 이름 이탤릭 처리
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontstyle("italic")

plt.show()

# %%
### 2. gAlk1_1 세포에서 NT, NP, KO 각각 150개씩 샘플링 
alk1_cells = adata[adata.obs["gene_target"] == "gAlk1_1"].copy()
alk1_cells.obs["mixscape_class"] = alk1_cells.obs["mixscape_class"].astype(str)

cells = []
for cls in ["NP", "KO"]:
    subset = alk1_cells[alk1_cells.obs["mixscape_class_global"] == cls]
    sampled = subset[np.random.choice(subset.shape[0], min(150, subset.shape[0]), replace=False)]
    cells.append(sampled)

# NT: gene_target == "NT" + mixscape_class == "NT"
nt_cells = adata[(adata.obs["gene_target"] == "NT") & (adata.obs["mixscape_class_global"] == "NT")]
nt_sampled = nt_cells[np.random.choice(nt_cells.shape[0], min(150, nt_cells.shape[0]), replace=False)]
cells.insert(0, nt_sampled)  # NT를 앞에 넣기

# 3. 병합
sampled_adata = cells[0].concatenate(cells[1:])

# 4. mixscape_class 순서 강제 고정
sampled_adata.obs["mixscape_class_global"] = pd.Categorical(
    sampled_adata.obs["mixscape_class_global"],
    categories=["NT", "NP", "KO"],
    ordered=True
)

# 5. heatmap
g = sc.pl.heatmap(
    sampled_adata,
    var_names=ms_top40,
    groupby="mixscape_class_global",
    use_raw=False,
    vmin=0,
    vmax=1,
    cmap=cmap,
    show=False,
    swap_axes=True
)

# y-axis gene labels to italic
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontstyle("italic")

plt.show()

# %%
adata.obs

# %% [markdown]
# #### Figure 제작 

# %%
import seaborn as sns

# 1. Standard heatmap: gAlk1_1 vs NT
alk1_cells = adata[adata.obs["gene_target"] == "gAlk1_1"]
alk1_sampled = alk1_cells[np.random.choice(alk1_cells.shape[0], min(150, alk1_cells.shape[0]), replace=False)].copy()

nt_cells = adata[adata.obs["gene_target"] == "NT"]
nt_sampled = nt_cells[np.random.choice(nt_cells.shape[0], min(150, nt_cells.shape[0]), replace=False)].copy()

alk1_sampled.obs["group"] = "gAlk1_1"
nt_sampled.obs["group"] = "NT"
adata_std = alk1_sampled.concatenate(nt_sampled)
adata_std.obs["group"] = pd.Categorical(adata_std.obs["group"], categories=["NT", "gAlk1_1"], ordered=True)

# 2. Mixscape heatmap: gAlk1_1 KO/NP vs NT
alk1_mix = adata[adata.obs["gene_target"] == "gAlk1_1"].copy()
alk1_mix.obs["mixscape_class"] = alk1_mix.obs["mixscape_class"].astype(str)

cells = []
for cls in ["NP", "KO"]:
    subset = alk1_mix[alk1_mix.obs["mixscape_class_global"] == cls]
    sampled = subset[np.random.choice(subset.shape[0], min(150, subset.shape[0]), replace=False)]
    cells.append(sampled)

nt_mix = adata[(adata.obs["gene_target"] == "NT") & (adata.obs["mixscape_class_global"] == "NT")]
nt_sampled = nt_mix[np.random.choice(nt_mix.shape[0], min(150, nt_mix.shape[0]), replace=False)]
cells.insert(0, nt_sampled)
adata_mix = cells[0].concatenate(cells[1:])
adata_mix.obs["mixscape_class_global"] = pd.Categorical(
    adata_mix.obs["mixscape_class_global"], categories=["NT", "NP", "KO"], ordered=True
)

# 3. 유전자 리스트: std_top40 (왼쪽), ms_top40 (오른쪽)
# 너가 미리 정의한 리스트여야 함
# std_top40 = [...]
# ms_top40 = [...]

# 4. 발현 행렬
X_std = pd.DataFrame(adata_std[:, std_top40].X.toarray(), columns=std_top40)
X_ms = pd.DataFrame(adata_mix[:, ms_top40].X.toarray(), columns=ms_top40)

# 5. 그림 만들기
fig, axes = plt.subplots(1, 2, figsize=(20, 14), dpi=150)

# 왼쪽 heatmap (Standard)
sns.heatmap(X_std.T, ax=axes[0], cmap="viridis", vmin=0, vmax=1, cbar=False, yticklabels=False)
axes[0].set_title("Standard Wilcoxon DEGs", fontsize=16)
axes[0].set_xlabel("Cells")
axes[0].set_ylabel("")
axes[0].yaxis.tick_right()
axes[0].yaxis.set_label_position("right")
axes[0].set_yticks(np.arange(len(std_top40)) + 0.5)
axes[0].set_yticklabels(std_top40, rotation=0, fontsize=8)

# 오른쪽 heatmap (Mixscape)
sns.heatmap(X_ms.T, ax=axes[1], cmap="viridis", vmin=0, vmax=1, cbar=False, yticklabels=ms_top40)
axes[1].set_title("Mixscape DEGs", fontsize=16)
axes[1].set_xlabel("Cells")
axes[1].set_ylabel("")

# 연결선 및 빨간색/이탤릭 표시
for i, gene in enumerate(std_top40):
    label = axes[0].get_yticklabels()[i]
    if gene in ms_top40:
        j = ms_top40.index(gene)
        axes[0].plot([1.01, 1.03], [i + 0.5, j + 0.5], transform=axes[0].transData,
                     color="black", linewidth=0.6)
    else:
        label.set_color("red")
        label.set_fontstyle("italic")

for j, gene in enumerate(ms_top40):
    label = axes[1].get_yticklabels()[j]
    if gene not in std_top40:
        label.set_color("red")
        label.set_fontstyle("italic")

plt.tight_layout()
plt.show()

# %%
# AnnData에서 X 추출
X_std = pd.DataFrame(adata_std[:, std_top40].X.toarray(), columns=std_top40)
X_ms = pd.DataFrame(adata_mix[:, ms_top40].X.toarray(), columns=ms_top40)

# mixscape 결과 유전자 목록
ms_gene_set = set(ms_results["names"])
std_gene_set = set(pd.concat([
    std_results["gAlk1_1"]["upregulated"]["gene"],
    std_results["gAlk1_1"]["downregulated"]["gene"]
]))

# 그림 만들기
fig, axes = plt.subplots(1, 2, figsize=(20, 14), dpi=150)

# 왼쪽 heatmap (Standard)
sns.heatmap(X_std.T, ax=axes[0], cmap=cmap, vmin=0, vmax=1, cbar=False, yticklabels=False)
axes[0].set_title("Standard Wilcoxon DEGs", fontsize=16)
axes[0].set_xlabel("Cells")
axes[0].set_ylabel("")
axes[0].yaxis.tick_right()
axes[0].yaxis.set_label_position("right")
axes[0].set_yticks(np.arange(len(std_top40)) + 0.5)
axes[0].set_yticklabels(std_top40, rotation=0, fontsize=10, fontstyle="italic")

# 오른쪽 heatmap (Mixscape)
sns.heatmap(X_ms.T, ax=axes[1], cmap=cmap, vmin=0, vmax=1, cbar=False, yticklabels=ms_top40)
axes[1].set_title("Mixscape DEGs", fontsize=16)
axes[1].set_xlabel("Cells")
axes[1].set_ylabel("")
axes[1].set_yticks(np.arange(len(ms_top40)) + 0.5)
axes[1].set_yticklabels(ms_top40, rotation=0, fontsize=10, fontstyle="italic")

# Non-overlapped genes 
std_only = []
ms_only = []

# 빨간색 표시 - std top40 중 mixscape 결과에 없는 유전자
for i, gene in enumerate(std_top40):
    label = axes[0].get_yticklabels()[i]
    if gene not in ms_gene_set:
        label.set_color("red")
        std_only.append(gene)

# 빨간색 표시 - ms top40 중 std DEG에 없는 유전자
for j, gene in enumerate(ms_top40):
    label = axes[1].get_yticklabels()[j]
    if gene not in std_gene_set:
        label.set_color("red")
        ms_only.append(gene)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Inspect Overlapping and Non-overlapping DEGs

# %%
sc.pl.umap(adata, color=std_only)

# %%
sc.pl.umap(adata, color=ms_only)

# %%
# 1. 유전자 리스트
shared_genes = filtered_kc_genes  

# 2. AnnData에서 발현 행렬 추출
X_std = pd.DataFrame(adata_std[:, shared_genes].X.toarray(), columns=shared_genes)
X_ms = pd.DataFrame(adata_mix[:, shared_genes].X.toarray(), columns=shared_genes)

# 3. 그림 생성: 3열짜리 subplot → [Standard | Gene labels | Mixscape]
fig, axes = plt.subplots(1, 3, figsize=(22, 14), width_ratios=[1, 0.1, 1], dpi=150)

# 4. 왼쪽 히트맵 (Standard)
sns.heatmap(X_std.T, ax=axes[0], cmap="viridis", vmin=0, vmax=1, cbar=False, yticklabels=False)
axes[0].set_title("Standard Wilcoxon DEGs", fontsize=16)
axes[0].set_xlabel("Cells")
axes[0].set_ylabel("")
axes[0].tick_params(axis='y', left=False)

# 5. 가운데 축 (유전자 라벨만 표시)
axes[1].set_ylim(len(shared_genes), 0)
axes[1].set_xlim(0, 1)
axes[1].axis("off")
for i, gene in enumerate(shared_genes):
    axes[1].text(
        0.5, i + 0.5, gene,
        ha="center", va="center",
        fontsize=10, fontstyle="italic"
    )

# 6. 오른쪽 히트맵 (Mixscape)
sns.heatmap(X_ms.T, ax=axes[2], cmap="viridis", vmin=0, vmax=1, cbar=False, yticklabels=False)
axes[2].set_title("Mixscape DEGs", fontsize=16)
axes[2].set_xlabel("Cells")
axes[2].set_ylabel("")
axes[2].tick_params(axis='y', left=False)

plt.tight_layout()
plt.show()


# %%
# 1. 유전자 리스트
shared_genes = filtered_kc_genes  # std_top40 = ms_top40 = filtered_kc_genes

# 2. 발현 행렬 추출
X_std = pd.DataFrame(adata_std[:, shared_genes].X.toarray(), columns=shared_genes)
X_ms = pd.DataFrame(adata_mix[:, shared_genes].X.toarray(), columns=shared_genes)

# 3. 그룹 정보
group_std = adata_std.obs["group"]
group_mix = adata_mix.obs["mixscape_class_global"]

# 4. 그룹별 경계 및 중간값 (Standard)
group_counts_std = group_std.value_counts().loc[group_std.cat.categories]
boundaries_std = np.cumsum(group_counts_std.values)
midpoints_std = np.insert(boundaries_std[:-1], 0, 0) + group_counts_std.values / 2

# 5. 그룹별 경계 및 중간값 (Mixscape)
group_counts_mix = group_mix.value_counts().loc[group_mix.cat.categories]
boundaries_mix = np.cumsum(group_counts_mix.values)
midpoints_mix = np.insert(boundaries_mix[:-1], 0, 0) + group_counts_mix.values / 2

# 6. 그림 생성
fig, axes = plt.subplots(1, 3, figsize=(22, 14), width_ratios=[1, 0.1, 1], dpi=150)

# 7. 왼쪽 Heatmap (Standard)
sns.heatmap(X_std.T, ax=axes[0], cmap=cmap, vmin=0, vmax=1,
            cbar=False, yticklabels=False)
axes[0].set_title("Standard Wilcoxon DEGs", fontsize=16)
axes[0].set_xlabel("Cells")
axes[0].set_ylabel("")
axes[0].tick_params(axis='y', left=False)
for b in boundaries_std[:-1]:
    axes[0].axvline(b, color="white", linewidth=1)
axes[0].set_xticks(midpoints_std)
axes[0].set_xticklabels(group_std.cat.categories, fontsize=10)

# 8. 가운데 유전자 라벨
axes[1].set_ylim(len(shared_genes), 0)
axes[1].set_xlim(0, 1)
axes[1].axis("off")
for i, gene in enumerate(shared_genes):
    axes[1].text(0.5, i + 0.5, gene,
                 ha="center", va="center",
                 fontsize=10, fontstyle="italic")

# 9. 오른쪽 Heatmap (Mixscape)
sns.heatmap(X_ms.T, ax=axes[2], cmap=cmap, vmin=0, vmax=1,
            cbar=False, yticklabels=False)
axes[2].set_title("Mixscape DEGs", fontsize=16)
axes[2].set_xlabel("Cells")
axes[2].set_ylabel("")
axes[2].tick_params(axis='y', left=False)
for b in boundaries_mix[:-1]:
    axes[2].axvline(b, color="white", linewidth=1)
axes[2].set_xticks(midpoints_mix)
axes[2].set_xticklabels(group_mix.cat.categories, fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inspect Overlapping & Non-overlapping genes 

# %%
std_gene_set

# %% [markdown]
# #### Overlapped DEGs

# %%
common_genes = list(std_gene_set & ms_gene_set)
print(common_genes)

# %%
from matplotlib_venn import venn2

venn2(
    [std_gene_set, ms_gene_set],
    set_labels=('Standard Wilcoxon DEGs', 'Mixscape DEGs')
)
plt.title("Overlap of DEGs")
plt.show() 

# %%
std_results

# %%
std_top_up = std_results["gAlk1_1"]["upregulated"].copy()
std_top_down = std_results["gAlk1_1"]["downregulated"].copy()

# %%
ms_top_up

# %%
std_up = set(std_top_up["gene"])
ms_up = set(ms_top_up["names"])

# %%
plt.figure(figsize=(6, 6))
venn2(
    [std_up, ms_up],
    set_labels=('Standard Wilcoxon DEGs', 'Mixscape DEGs'),
    set_colors=('skyblue', 'lightcoral'),
    alpha=0.7
)
plt.title("Overlap of Upregulated DEGs")
plt.show()

# %% [markdown]
# #### Overlapped upregulated genes 

# %%
common_upregulated_genes = list(std_up & ms_up)
print(common_upregulated_genes)

# %% [markdown]
# #### Overlapped downregulated genes

# %%
std_down = set(std_top_down["gene"])
ms_down = set(ms_top_down["names"])

plt.figure(figsize=(6, 6))
venn2(
    [std_down, ms_down],
    set_labels=('Standard Wilcoxon DEGs', 'Mixscape DEGs'),
    set_colors=('skyblue', 'lightcoral'),
    alpha=0.7
)
plt.title("Overlap of Downregulated DEGs")
plt.show()

# %%
common_downregulated_genes = list(std_down & ms_down)
print(common_downregulated_genes)

# %% [markdown]
# #### 공통된 upregulated 유전자 logFC 비교 산점도 (Standard vs Mixscape)

# %%
std_up = std_top_up.rename(columns={"gene": "gene", "logFC": "logFC"})
ms_up = ms_top_up.rename(columns={"names": "gene", "logfoldchanges": "logFC"})


# %%
ms_top_up

# %%
from adjustText import adjust_text

merged_up = pd.merge(
    std_up,
    ms_up,
    on="gene",
    suffixes=("_std", "_mix")
)

# 3. 평균 logFC 기준 상위 20개 유전자 선택
merged_up["logFC_mean"] = (merged_up["logFC_std"] + merged_up["logFC_mix"]) / 2
top20 = merged_up.nlargest(20, "logFC_mean")

# 4. 산점도 그리기
plt.figure(figsize=(8, 8))
plt.scatter(
    merged_up["logFC_std"], merged_up["logFC_mix"],
    color="lightgrey", label="All common genes", s=30
)
plt.scatter(
    top20["logFC_std"], top20["logFC_mix"],
    color="blue", label="Top 20 (labeled)", s=40
)

# 기준선 y = x
min_val = min(merged_up["logFC_std"].min(), merged_up["logFC_mix"].min())
max_val = max(merged_up["logFC_std"].max(), merged_up["logFC_mix"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

# 5. 라벨 추가
texts = []
for _, row in top20.iterrows():
    texts.append(
        plt.text(row["logFC_std"], row["logFC_mix"], row["gene"], fontsize=9)
    )
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
)

# 6. 마무리
plt.xlabel("logFC (Standard DEG)")
plt.ylabel("logFC (Mixscape DEG)")
plt.title("Top 20 Common Upregulated Genes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
std_down = std_top_down.rename(columns={"gene": "gene", "logFC": "logFC"})
ms_down = ms_top_down.rename(columns={"names": "gene", "logfoldchanges": "logFC"})

# 2. 공통 유전자 merge
merged_down = pd.merge(
    std_down,
    ms_down,
    on="gene",
    suffixes=("_std", "_mix")
)

# 3. 평균 logFC 기준 상위 20개 선택
merged_down["logFC_mean"] = (merged_down["logFC_std"] + merged_down["logFC_mix"]) / 2
top20_down = merged_down.nsmallest(20, "logFC_mean")  # 가장 낮은 값 → 가장 강한 down

# 4. 산점도
plt.figure(figsize=(8, 8))
plt.scatter(
    merged_down["logFC_std"], merged_down["logFC_mix"],
    color="lightgrey", s=30, label="All common genes"
)
plt.scatter(
    top20_down["logFC_std"], top20_down["logFC_mix"],
    color="red", s=40, label="Top 20 (labeled)"
)

# y = x 기준선
min_val = min(merged_down["logFC_std"].min(), merged_down["logFC_mix"].min())
max_val = max(merged_down["logFC_std"].max(), merged_down["logFC_mix"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")

# 라벨
texts = []
for _, row in top20_down.iterrows():
    texts.append(
        plt.text(row["logFC_std"], row["logFC_mix"], row["gene"], fontsize=9)
    )
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
)

# 마무리
plt.xlabel("logFC (Standard DEG)")
plt.ylabel("logFC (Mixscape DEG)")
plt.title("Top 20 Common Downregulated Genes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# #### UMAP visualization

# %%
# 공통 Upregulated DEG (Top 20)
merged_up["logFC_mean"] = (merged_up["logFC_std"] + merged_up["logFC_mix"]) / 2
top20_up_genes = merged_up.nlargest(20, "logFC_mean")["gene"].tolist()
top100_up_genes = merged_up.nlargest(100, "logFC_mean")["gene"].tolist()


# 공통 Downregulated DEG (Top 20)
merged_down["logFC_mean"] = (merged_down["logFC_std"] + merged_down["logFC_mix"]) / 2
top20_down_genes = merged_down.nsmallest(20, "logFC_mean")["gene"].tolist()
top100_down_genes = merged_down.nsmallest(100, "logFC_mean")["gene"].tolist()


# %%
def plot_umap_for_genes(adata, gene_list, title_prefix=""):
    n = len(gene_list)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, gene in enumerate(gene_list):
        if gene not in adata.var_names:
            print(f"Gene {gene} not found in AnnData.")
            continue
        sc.pl.umap(
            adata, color=gene, ax=axes[i], show=False,
            title=gene, vmin=0, vmax='p99', cmap="viridis"
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title_prefix} Top 20 Genes", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %% [markdown]
# 

# %%
plot_umap_for_genes(adata, top20_up_genes, title_prefix="Upregulated")


# %%
plot_umap_for_genes(adata, top20_down_genes, title_prefix="Downregulated")


# %% [markdown]
# ## 5. Gene Set Analysis (GSA)

# %% [markdown]
# ### 9.1. Hypergeometric enrichment test (초기하분포 검정)

# %%
import gseapy as gp

gene_set_names = gp.get_library_name(organism='Mouse')

# %%
gp.get_library_name()

# %%
glist = common_genes

# %%
up_glist = common_upregulated_genes

# %%
down_glist = common_downregulated_genes

# %%
down_glist

# %% [markdown]
# #### 1. GO_Biological_Process_2025

# %%
enr_res1 = gp.enrichr(gene_list=up_glist, organism='Mouse', gene_sets='GO_Biological_Process_2025', cutoff = 0.05)
enr_res1.results.head()

# %%
gp.barplot(enr_res1.res2d,title='GO_Biological_Process_2025 (Upregulated)')

# %%
enr_res2 = gp.enrichr(gene_list=down_glist, organism='Mouse', gene_sets='GO_Biological_Process_2025', cutoff = 0.05)
enr_res2.results.head()

# %%
gp.barplot(enr_res2.res2d,title='GO_Biological_Process_2025')

# %% [markdown]
# ### 2. MGI_Mammalian_Phenotype_Level_4_2024

# %%
enr_res3 = gp.enrichr(gene_list=up_glist, organism='Mouse', gene_sets='MGI_Mammalian_Phenotype_Level_4_2024', cutoff = 0.5)
enr_res3.results.head()

# %%
gp.barplot(enr_res3.res2d,title='MGI_Mammalian_Phenotype_Level_4_2024')

# %%
enr_res4 = gp.enrichr(gene_list=down_glist, organism='Mouse', gene_sets='MGI_Mammalian_Phenotype_Level_4_2024', cutoff = 0.5)
enr_res4.results.head()

# %%
gp.barplot(enr_res4.res2d,title='MGI_Mammalian_Phenotype_Level_4_2024')

# %% [markdown]
# ### 3. Disease_Perturbations_from_GEO_up

# %%
enr_res5 = gp.enrichr(gene_list=up_glist, organism='Mouse', gene_sets='Disease_Perturbations_from_GEO_up', cutoff = 0.5)
enr_res5.results.head()

# %%
gp.barplot(enr_res5.res2d,title='Disease_Perturbations_from_GEO_up')

# %% [markdown]
# ### 4. Disease_Perturbations_from_GEO_down

# %%
enr_res6 = gp.enrichr(gene_list=down_glist, organism='Mouse', gene_sets='Disease_Perturbations_from_GEO_down', cutoff = 0.5)
enr_res6.results.head()

# %%
gp.barplot(enr_res6.res2d,title='Disease_Perturbations_from_GEO_down')

# %% [markdown]
# ### 5. WikiPathways_2024_Mouse

# %%
enr_res7 = gp.enrichr(gene_list=up_glist, organism='Mouse', gene_sets='WikiPathways_2024_Mouse', cutoff = 0.5)
enr_res7.results.head()

# %%
gp.barplot(enr_res7.res2d,title='WikiPathways_2024_Mouse (Upregulated)')

# %%
enr_res8 = gp.enrichr(gene_list=down_glist, organism='Mouse', gene_sets='WikiPathways_2024_Mouse', cutoff = 0.5)
enr_res8.results.head()

# %%
gp.barplot(enr_res8.res2d,title='WikiPathways_2024_Mouse (Downregulated)')

# %% [markdown]
# ### 7. WikiPathways_2024_Human

# %%
enr_res9 = gp.enrichr(gene_list=up_glist, organism='Human', gene_sets='WikiPathways_2024_Human', cutoff = 0.5)
enr_res9.results.head()

# %%
gp.barplot(enr_res9.res2d,title='WikiPathways_2024_Human (Upregulated)')

# %%
enr_res10 = gp.enrichr(gene_list=down_glist, organism='Human', gene_sets='WikiPathways_2024_Human', cutoff = 0.5)
enr_res10.results.head()

# %%
gp.barplot(enr_res10.res2d,title='WikiPathways_2024_Human (downregulated)')

# %% [markdown]
# ### 10. KEGG_2021_Mouse

# %%
enr_res11 = gp.enrichr(gene_list=up_glist, organism='Mouse', gene_sets='KEGG_2019_Mouse', cutoff = 0.5)
enr_res11.results.head()

# %%
gp.barplot(enr_res11.res2d,title='KEGG_2019_Mouse (Up)')

# %%
enr_res12 = gp.enrichr(gene_list=down_glist, organism='Mouse', gene_sets='KEGG_2019_Mouse', cutoff = 0.5)
enr_res12.results.head()

# %%
gp.barplot(enr_res12.res2d,title='KEGG_2019_Mouse (Down)')

# %% [markdown]
# ## 6. Gene Set Enrichment Analysis (GSEA)

# %%
gene_set_names = gp.get_library_name(organism='Mouse')
print(gene_set_names)

# %%
import time

# %%
adata.obs["mixscape_binary"] = adata.obs["mixscape_class_global"].map(lambda x: "Perturbed" if x == "KO" else "Non-perturbed")

# %%
bdata = adata.copy()

# %%
bdata.obs

# %%
adata.obs['condition2']

# %%
expression_df = bdata.to_df().T  # gene × sample
labels = adata.obs["mixscape_binary"].tolist() 
t1 = time.time()
res = gp.gsea(
    data=expression_df,
    gene_sets="GO_Biological_Process_2025",
    cls=labels,
    permutation_num=1000,
    permutation_type="phenotype",
    outdir=None,
    method='s2n',
    threads=16
)
t2 = time.time()

print(f"GSEA 수행 시간: {t2 - t1:.2f}초")
res.res2d.head()

# %%
res.ranking.shape # raking metric

# %%
import networkx as nx


# %%
nodes, edges = gp.enrichment_map(res.res2d)

# %%
# build graph
G = nx.from_pandas_edgelist(edges,
                            source='src_idx',
                            target='targ_idx',
                            edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes'])

# Add missing node if there is any
for node in nodes.index:
    if node not in G.nodes():
        G.add_node(node)
fig, ax = plt.subplots(figsize=(8, 8))

# init node cooridnates
pos=nx.layout.spiral_layout(G)
#node_size = nx.get_node_attributes()
# draw node
nx.draw_networkx_nodes(G,
                       pos=pos,
                       cmap=plt.cm.RdYlBu,
                       node_color=list(nodes.NES),
                       node_size=list(nodes.Hits_ratio *1000))
# draw node label
nx.draw_networkx_labels(G,
                        pos=pos,
                        labels=nodes.Term.to_dict())
# draw edge
edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
nx.draw_networkx_edges(G,
                       pos=pos,
                       width=list(map(lambda x: x*10, edge_weight)),
                       edge_color='#CDDBD4')
plt.show()


# %%
i = 0
genes = res.res2d.Lead_genes.iloc[i].split(";")
ax = gp.heatmap(df = res.heatmat.loc[genes],
           z_score=None,
           title=res.res2d.Term.iloc[i] + " (Raw expression)",
           figsize=(6,5),
           cmap=plt.cm.viridis,
           xticklabels=False)

# %%
expression_df = bdata.to_df().T  # gene × sample
labels = adata.obs["mixscape_binary"].tolist()  

t1 = time.time()
result = gp.gsea(
    data=expression_df,
    gene_sets="WikiPathways_2024_Human",
    cls=labels,
    permutation_num=1000,
    permutation_type="phenotype",
    outdir=None,
    method='s2n',
    threads=16
)
t2 = time.time()

print(f"GSEA 수행 시간: {t2 - t1:.2f}초")
result.res2d.head()

# %%
result.ranking.shape

# %%

i = 0
genes = result.res2d.Lead_genes.iloc[i].split(";")
ax = gp.heatmap(df = result.heatmat.loc[genes],
           z_score=None,
           title=result.res2d.Term.iloc[i] + " (Raw expression)",
           figsize=(6,5),
           cmap=plt.cm.viridis,
           xticklabels=False)

# %%
from sklearn.utils import resample
perturbed_cells = bdata.obs.query("mixscape_binary == 'Perturbed'").index

np_cells = bdata.obs.query("mixscape_binary == 'Non-perturbed'").index
np_sample = resample(np_cells, n_samples=len(perturbed_cells), random_state=0, replace=False)

# 합치기
balanced_cells = perturbed_cells.union(np_sample)
heatmap_df = result.heatmat.loc[genes, balanced_cells]

# heatmap
gp.heatmap(
    df=heatmap_df,
    z_score=None,
    title=f"{result.res2d.Term.iloc[i]} (Balanced groups)",
    figsize=(6, 5),
    cmap=plt.cm.viridis,
    xticklabels=False
)


# %%
term = result.res2d.Term
# gp.gseaplot(res.ranking, term=term[i], **res.results[term[i]])
axs = result.plot(terms=term[:5])

# %% [markdown]
# ### Over-representation analysis

# %%
up_glist

# %%
# Enricr API
enr_up = gp.enrichr(up_glist,
                    gene_sets='GO_Biological_Process_2025',
                    outdir=None)


# %%
# Enricr API
enr_up_top100 = gp.enrichr(top100_up_genes,
                    gene_sets='GO_Biological_Process_2025',
                    outdir=None)


# %%
enr_up_top100.res2d

# %%
enr_up

# %%
enr_up.res2d.Term = enr_up.res2d.Term.str.split(" \(GO").str[0]


# %%
enr_up

# %%
enr_up.res2d



# %%
top5 = enr_up_top100.res2d.head(10)

# 반복문으로 각 term의 유전자들 UMAP에 시각화
for i, row in top5.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    # Human → Mouse 유전자 변환 (첫 글자만 대문자)
    genes_mouse = [g.capitalize() for g in genes_human]
    
    # 실제 adata에 존재하는 유전자만 필터링
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=3,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
# dotplot
gp.dotplot(enr_up.res2d, figsize=(3,5), title="Up", cmap = plt.cm.autumn_r)

ax.set_title("Up", fontsize=5)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)



plt.tight_layout()
plt.show()

# %%
# dotplot
ax = gp.dotplot(
    enr_up.res2d,
    figsize=(6, 5),
    title="Up",
    cmap=plt.cm.autumn_r
)

# ✅ 이미 ax는 matplotlib의 Axes → 바로 접근
ax.set_title("Up", fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)



plt.tight_layout()
plt.show()

# %% [markdown]
# 

# %%
enr_dw = gp.enrichr(down_glist,
                    gene_sets='GO_Biological_Process_2025',
                    outdir=None)

# %%
enr_dw.res2d.Term = enr_dw.res2d.Term.str.split(" \(GO").str[0]
ax = gp.dotplot(enr_dw.res2d,
           figsize=(6,5),
           cmap = plt.cm.winter_r,
           size=5)

ax.set_title("Down", fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)



plt.tight_layout()
plt.show()

# %%
top5_down = enr_dw.res2d.head(5)

for i, row in top5_down.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    # Human → Mouse 유전자 변환 (첫 글자만 대문자)
    genes_mouse = [g.capitalize() for g in genes_human]
    
    # 실제 adata에 존재하는 유전자만 필터링
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=3,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
# Enricr API
enr_down_top100 = gp.enrichr(top100_down_genes,
                    gene_sets='GO_Biological_Process_2025',
                    outdir=None)


# %%
enr_down_top100.res2d

# %%
top100_down = enr_down_top100.res2d.head(10)

# 반복문으로 각 term의 유전자들 UMAP에 시각화
for i, row in top100_down.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    # Human → Mouse 유전자 변환 (첫 글자만 대문자)
    genes_mouse = [g.capitalize() for g in genes_human]
    
    # 실제 adata에 존재하는 유전자만 필터링
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=3,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
enr_up.res2d['UP_DW'] = "UP"
enrdw.res2d['UP_DW'] = "DOWN"


enr_res = pd.concat([enr_up.res2d.head(), enr_dw.res2d.head()])

# %%
enr_res

# %%
from gseapy.scipalette import SciPalette
sci = SciPalette()
NbDr = sci.create_colormap()
# NbDr

# %%
# --- 개선된 dotplot ---
ax = gp.dotplot(
    enr_res,                     
    figsize=(7, 5),              
    x='UP_DW',
    x_order=["UP", "DOWN"],
    title="GO_BP",
    cmap=NbDr.reversed(),       
    size=3,
    show_ring=True
)

# --- 폰트 크기 및 타이틀 ---
ax.set_title("GO Biological Process 2025", fontsize=14)
ax.set_xlabel("")  # x축 label 제거
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)


# --- 여백 자동 조정 ---
plt.tight_layout()
plt.show()

# %%
# barplot 생성
ax = gp.barplot(
    enr_res,
    figsize=(10, 5),                
    group='UP_DW',
    title="GO_BP",
    color=['blue', 'red']               # UP = 빨강, DOWN = 파랑
)

ax.set_title("GO_Biological Process 2025", fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.set_xlabel("")

# 🔧 범례 조정
legend = ax.get_legend()
if legend:
    legend.set_title("Group", prop={'size': 10})
    for text in legend.get_texts():
        text.set_fontsize(9)
    legend.set_bbox_to_anchor((1.05, 1))  # 오른쪽으로 밀기

# 🔧 여백 자동 조정
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 세컨드츄라이

# %%
enr_up_top100.res2d['UP_DW'] = "UP"
enr_down_top100.res2d['UP_DW'] = "DOWN"


enr_res_top100 = pd.concat([enr_up_top100.res2d.head(), enr_down_top100.res2d.head()])

# %%
# display multi-datasets
ax = gp.dotplot(enr_res_top100,figsize=(3,5),
                x='UP_DW',
                x_order = ["UP","DOWN"],
                title="GO_BP",
                cmap = NbDr.reversed(),
                size=3,
                show_ring=True)
ax.set_xlabel("")
plt.show()

# %%
# dotplot
enr_up_top100.res2d.Term = enr_up_top100.res2d.Term.str.split(" \(GO").str[0]

# dotplot (Up-regulated terms only)
ax = gp.dotplot(
    enr_up_top100.res2d,       
    figsize=(2, 5),           
    title="Up",
    cmap=plt.cm.autumn_r        
)

# 🔧 타이틀 및 폰트 크기 조정
ax.set_title("Up", fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.set_xlabel("")


# 🔧 레이아웃 정리
plt.tight_layout()
plt.show()

# %%

enr_down_top100.res2d.Term = enr_down_top100.res2d.Term.str.split(" \(GO").str[0]

ax = gp.dotplot(enr_down_top100.res2d,
           figsize=(2,6),
           title="Down",
           cmap = plt.cm.winter_r,
           size=5)

ax.set_title("Down", fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.set_xlabel("")




plt.tight_layout()
plt.show()


# %% [markdown]
# ## KEGG

# %%
# Enricr API
kegg_enr_up = gp.enrichr(up_glist,
                    gene_sets='KEGG_2019_Mouse',
                    outdir=None)


# %%
# Enricr API
kegg_enr_up_top100 = gp.enrichr(top100_up_genes,
                    gene_sets='KEGG_2019_Mouse',
                    outdir=None)


# %%
kegg_enr_up.res2d

# %%
kegg_enr_up_top100.res2d

# %%
top5 = kegg_enr_up.res2d.head(10)

for i, row in top5.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    genes_mouse = [g.capitalize() for g in genes_human]
    
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=3,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
# dotplot
gp.dotplot(kegg_enr_up.res2d, figsize=(3,5), title="Up", cmap = plt.cm.autumn_r)
plt.show()




# %%
ax = gp.dotplot(
    kegg_enr_up.res2d,
    figsize=(5, 6),                   
    title="KEGG Up",
    cmap=plt.cm.autumn_r,             
    size=3,
    show_ring=True
)

# --- 폰트 및 타이틀 설정 ---
ax.set_title("KEGG Up", fontsize=14)
ax.set_xlabel("")  # x축 레이블 제거
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)


# --- 여백 자동 정리 ---
plt.tight_layout()
plt.show()

# %%
top5 = kegg_enr_up_top100.res2d.head(10)

for i, row in top5.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    # Human → Mouse 유전자 변환 (첫 글자만 대문자)
    genes_mouse = [g.capitalize() for g in genes_human]
    
    # 실제 adata에 존재하는 유전자만 필터링
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=3,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
# dotplot
gp.dotplot(kegg_enr_up_top100.res2d, figsize=(3,5), title="Up", cmap = plt.cm.autumn_r)
plt.show()

# %%
kegg_enr_dw = gp.enrichr(down_glist,
                    gene_sets='KEGG_2019_Mouse',
                    outdir=None)

# %%
kegg_enr_down_top100 = gp.enrichr(top100_down_genes,
                    gene_sets='KEGG_2019_Mouse',
                    outdir=None)

# %%
kegg_enr_dw.res2d

# %%
kegg_enr_down_top100.res2d

# %%
kegg_enr_dw.res2d.Term = kegg_enr_dw.res2d.Term.str.split(" \(GO").str[0]
ax = gp.dotplot(kegg_enr_dw.res2d,
           figsize=(2,6),
           title="Down",
           cmap = plt.cm.winter_r,
           size=5)
# --- 타이틀 및 폰트 조정 ---
ax.set_title("KEGG Down", fontsize=14)
ax.set_xlabel("")  # x축 label 제거
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)


plt.show()

# %%
top5_down = kegg_enr_dw.res2d.head(5)

# 반복문으로 각 term의 유전자들 UMAP에 시각화
for i, row in top5_down.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    # Human → Mouse 유전자 변환 (첫 글자만 대문자)
    genes_mouse = [g.capitalize() for g in genes_human]
    
    # 실제 adata에 존재하는 유전자만 필터링
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=5,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
kegg_top100_down = kegg_enr_dw.res2d.head(10)

for i, row in kegg_top100_down.iterrows():
    term = row["Term"]
    genes_human = row["Genes"].split(";")
    
    genes_mouse = [g.capitalize() for g in genes_human]
    
    genes_in_data = [g for g in genes_mouse if g in adata.var_names]
    
    if not genes_in_data:
        print(f"[!] No genes found in adata for: {term}")
        continue

    print(f"Plotting UMAP for: {term} ({len(genes_in_data)} genes)")
    
    # Plot
    sc.pl.umap(
        adata,
        color=genes_in_data,
        cmap="viridis",
        size=20,
        ncols=3,
        title=[f"{term}: {g}" for g in genes_in_data],
        show=True
    )

# %%
kegg_enr_up.res2d['UP_DW'] = "UP"
kegg_enr_dw.res2d['UP_DW'] = "DOWN"


kegg_enr_res = pd.concat([kegg_enr_up.res2d.head(), kegg_enr_dw.res2d.head()])

# %%
# dotplot
gp.dotplot(kegg_enr_up.res2d, figsize=(3,5), title="Up", cmap = plt.cm.autumn_r)
plt.show()

# %%
kegg_enr_dw.res2d.Term = kegg_enr_dw.res2d.Term.str.split(" \(GO").str[0]
gp.dotplot(kegg_enr_dw.res2d,
           figsize=(3,5),
           title="Down",
           cmap = plt.cm.winter_r,
           size=5)
plt.show()

# %%
# --- dotplot for multi-dataset (UP/DOWN) ---
ax = gp.dotplot(
    kegg_enr_res,
    figsize=(6, 6),                    
    x='UP_DW',
    x_order=["UP", "DOWN"],       
    title="KEGG Enrichment",
    cmap=NbDr.reversed(),             
    size=3,
    show_ring=True
)

ax.set_title("KEGG Enrichment", fontsize=14)
ax.set_xlabel("")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)


plt.tight_layout()
plt.show()

# %%
# --- barplot for multi-dataset (UP/DOWN) ---
ax = gp.barplot(
    kegg_enr_res,
    figsize=(9, 5),            
    group='UP_DW',             
    title="KEGG Enrichment",
    color=['blue', 'red']       

ax.set_title("KEGG Enrichment", fontsize=14)
ax.set_xlabel("")  # x축 label 제거
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

legend = ax.get_legend()
if legend:
    legend.set_bbox_to_anchor((1.05, 1))           
    legend.set_title("Group", prop={'size': 10})
    for text in legend.get_texts():
        text.set_fontsize(8)

plt.tight_layout()
plt.show()

# %%



