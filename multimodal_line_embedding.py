# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
multimodal_line_embedding.py
============================
Train multimodal POI-related node embeddings using a LINE-style negative
sampling objective on heterogeneous graphs constructed from LBSN check-in data.

The model jointly trains five embedding tables:
    user_emb   [U+1, D]
    cat_emb    [C+1, D]
    loc_emb    [L+1, D]
    time_emb   [12+1, D]
    macro_emb  [M+1, D]

The macro-category embeddings are used as training-only hub nodes to inject
category hierarchy information. After training, only user, category, location,
and time embeddings are exported and used by the downstream vMF-HMM model.

Subgraphs
---------
1. user-category co-occurrence graph
2. user-time co-occurrence graph
3. user-location co-occurrence graph
4. category-time co-occurrence graph
5. category-location graph
6. time-location co-occurrence graph
7. time-time cycle graph over 12 daily time bins
8. category-macro-category hierarchy graph
9. location-location spatial proximity graph

For subgraphs 1--6, low-frequency edges are removed using a theta-percentile
filter. The time-time graph is fixed as a 12-cycle, the category-macro graph is
constructed from a predefined macro-category mapping, and the location-location
graph connects POIs within a geographical distance threshold.

Outputs
-------
The script saves:
    user_emb.npy
    cat_emb.npy
    loc_emb.npy
    time_emb.npy
    embeddings.pt
    meta.json
    cat_tsne.png
    loc_tsne.png

Usage
-----
python multimodal_line_embedding.py \\
    --pkl_path data/NYC_data.pkl \\
    --macro_csv_path data/category_cluster_map.csv \\
    --save_dir embeddings/multimodal_line_runs \\
    --emb_dim 128 \\
    --theta_percent 50 \\
    --neg_samples 30 \\
    --loc_dist_km 0.1 \\
    --epochs 200
"""

import os
import json
import math
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.spatial
from tqdm import tqdm

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

plt.rcParams["font.family"] = "DejaVu Sans"

BASE = "."

# ── Node type constants ───────────────────────────────────────────────────────
TYPE_USER  = 0
TYPE_CAT   = 1
TYPE_LOC   = 2
TYPE_TIME  = 3
TYPE_MACRO = 4          # macro-category hub nodes
TYPE_NAMES = ["user", "cat", "loc", "time", "macro"]

# (src_type, dst_type) for graph indices 0-8
GRAPH_TYPES = [
    (TYPE_USER,  TYPE_CAT),   # 0 → graph 1: user-cat
    (TYPE_USER,  TYPE_TIME),  # 1 → graph 2: user-time
    (TYPE_USER,  TYPE_LOC),   # 2 → graph 3: user-loc
    (TYPE_CAT,   TYPE_TIME),  # 3 → graph 4: cat-time
    (TYPE_CAT,   TYPE_LOC),   # 4 → graph 5: cat-loc
    (TYPE_TIME,  TYPE_LOC),   # 5 → graph 6: time-loc
    (TYPE_TIME,  TYPE_TIME),  # 6 → graph 7: time-time cycle
    (TYPE_CAT,   TYPE_MACRO), # 7 → graph 8: cat → macro-cat hub
    (TYPE_LOC,   TYPE_LOC),   # 8 → graph 9: loc-loc
]


# =========================================================
# Helpers
# =========================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def theta_filter(count_dict, theta_percent):
    """Remove lowest theta_percent of edges by count. Returns list of (src, dst)."""
    if not count_dict:
        return []
    edges    = sorted(count_dict.items(), key=lambda x: x[1])
    n_remove = int(len(edges) * theta_percent / 100)
    return [pair for pair, _ in edges[n_remove:]]


# =========================================================
# Subgraph builders
# =========================================================

def build_cooccurrence_graphs(train_trajectories, theta_percent):
    """
    Subgraphs 1-6: data-driven co-occurrence edges with theta% filter.
    Returns dict g_id (0..5) → list of (src_1indexed, dst_1indexed).
    """
    counts = {g: defaultdict(int) for g in range(6)}

    for traj in tqdm(train_trajectories, desc="Building co-occurrence graphs"):
        uid = traj["user_id"]
        if uid == -1:
            continue
        for pid, cid0, hb in zip(traj["poi_seq"],
                                  traj["cat_seq"],
                                  traj["hour_bin_seq"]):
            u  = uid
            c  = cid0 + 1   # 1-indexed cat
            l  = pid         # 1-indexed loc
            ti = hb          # 1..12 time bin

            counts[0][(u, c)]  += 1   # user-cat
            counts[1][(u, ti)] += 1   # user-time
            counts[2][(u, l)]  += 1   # user-loc
            counts[3][(c, ti)] += 1   # cat-time
            counts[4][(c, l)]  += 1   # cat-loc
            counts[5][(ti, l)] += 1   # time-loc

    graphs = {}
    for g_id in range(6):
        raw   = len(counts[g_id])
        edges = theta_filter(counts[g_id], theta_percent)
        graphs[g_id] = edges
        st, dt = GRAPH_TYPES[g_id]
        print(f"  Graph {g_id+1} ({TYPE_NAMES[st]}-{TYPE_NAMES[dt]}): "
              f"{raw} → {len(edges)} edges (θ={theta_percent}% filter)")
    return graphs


def build_time_cycle_graph():
    """Subgraph 7: undirected 12-cycle on time nodes 1..12."""
    return [(ti, ti % 12 + 1) for ti in range(1, 13)]   # 1→2, ..., 12→1


def build_cat_macro_graph(cid1_to_macro, macro_list, cat_size):
    """
    Subgraph 8: every cat node is connected to its macro-cat hub node.
    macro nodes are 1-indexed by position in macro_list (sorted order).
    No threshold — every cat with a known macro gets one edge.
    Returns list of (cat_1indexed, macro_1indexed).
    """
    macro_to_idx1 = {m: i + 1 for i, m in enumerate(macro_list)}
    edges = []
    for cid1 in range(1, cat_size + 1):
        macro = cid1_to_macro.get(cid1)
        if macro is None:
            continue
        edges.append((cid1, macro_to_idx1[macro]))
    print(f"  Graph 8 (cat-macro hub): {len(edges)} edges  "
          f"({len(macro_list)} macro nodes, no threshold)")
    return edges


def build_loc_loc_graph(pid_lat_lon, loc_size, max_dist_km):
    """
    Subgraph 9: loc-loc edges for POIs within max_dist_km (haversine).
    cKDTree in radian space used for candidate pairs; haversine verifies.
    """
    pids       = [pid for pid in range(1, loc_size + 1) if pid in pid_lat_lon]
    coords     = np.array([[pid_lat_lon[p][0], pid_lat_lon[p][1]]
                           for p in pids], dtype=np.float64)
    coords_rad = np.radians(coords)

    r_rad = max_dist_km / 6371.0
    tree  = scipy.spatial.cKDTree(coords_rad)
    pairs = tree.query_ball_tree(tree, r=r_rad)

    edges = []
    for i, neighbors in enumerate(
            tqdm(pairs, desc="Building loc-loc graph", leave=False)):
        for j in neighbors:
            if j <= i:
                continue
            pid_i, pid_j = pids[i], pids[j]
            d = haversine_km(pid_lat_lon[pid_i][0], pid_lat_lon[pid_i][1],
                             pid_lat_lon[pid_j][0], pid_lat_lon[pid_j][1])
            if d < max_dist_km:
                edges.append((pid_i, pid_j))

    print(f"  Graph 9 (loc-loc): {len(edges)} edges (dist < {max_dist_km} km)")
    return edges


# =========================================================
# Master directed edge list
# =========================================================

def build_master_edge_list(graphs):
    """
    Expand all 9 undirected subgraphs into directed edges (both directions).
    Returns four numpy arrays: src_types [N], src_idxs [N], dst_types [N], dst_idxs [N].
    """
    all_st, all_si, all_dt, all_di = [], [], [], []

    for g_id, edges in sorted(graphs.items()):
        src_type, dst_type = GRAPH_TYPES[g_id]
        for src_idx, dst_idx in edges:
            all_st.append(src_type);  all_si.append(src_idx)
            all_dt.append(dst_type);  all_di.append(dst_idx)
            all_st.append(dst_type);  all_si.append(dst_idx)   # reverse
            all_dt.append(src_type);  all_di.append(src_idx)

    return (np.array(all_st, dtype=np.int8),
            np.array(all_si, dtype=np.int32),
            np.array(all_dt, dtype=np.int8),
            np.array(all_di, dtype=np.int32))


# =========================================================
# Model
# =========================================================

class MultimodalLineEmbedding(nn.Module):
    """
    Five embedding tables (user/cat/loc/time/macro) trained jointly via LINE
    negative sampling.  All nodes share one embedding (no separate src/dst).
    macro_emb is a training-only hub; only the first four tables are exported.
    """
    def __init__(self, num_users, num_cats, num_locs, num_macros, emb_dim=128):
        super().__init__()
        self.user_emb  = nn.Embedding(num_users  + 1, emb_dim, padding_idx=0)
        self.cat_emb   = nn.Embedding(num_cats   + 1, emb_dim, padding_idx=0)
        self.loc_emb   = nn.Embedding(num_locs   + 1, emb_dim, padding_idx=0)
        self.time_emb  = nn.Embedding(12          + 1, emb_dim, padding_idx=0)
        self.macro_emb = nn.Embedding(num_macros  + 1, emb_dim, padding_idx=0)
        # Dispatch by TYPE_* index (0-4); modules already registered above
        self._emb_list   = [self.user_emb, self.cat_emb,
                             self.loc_emb,  self.time_emb, self.macro_emb]
        self.vocab_sizes = [num_users, num_cats, num_locs, 12, num_macros]

    def _lookup(self, type_idx, indices):
        return self._emb_list[type_idx](indices)

    def line_loss_batch(self, src_type, src_idxs, dst_type, dst_idxs, neg_samples):
        """
        LINE 2nd-order proximity loss for a homogeneous batch of (src, dst).
        src_idxs / dst_idxs : 1D LongTensor [B].
        """
        src_emb   = self._lookup(src_type, src_idxs)          # [B, D]
        dst_emb   = self._lookup(dst_type, dst_idxs)          # [B, D]
        pos_score = (src_emb * dst_emb).sum(-1)                # [B]
        pos_loss  = F.logsigmoid(pos_score).mean()

        B          = src_idxs.shape[0]
        vocab_size = self.vocab_sizes[dst_type]
        neg_idxs   = t.randint(1, vocab_size + 1,
                               (B, neg_samples),
                               device=src_idxs.device)         # [B, N]
        neg_emb    = self._lookup(dst_type, neg_idxs)          # [B, N, D]
        neg_score  = t.bmm(neg_emb,
                           src_emb.unsqueeze(-1)).squeeze(-1)  # [B, N]
        neg_loss   = F.logsigmoid(-neg_score).mean()

        return -(pos_loss + neg_loss)


# =========================================================
# Training loop
# =========================================================

def train_line(model, src_types, src_idxs, dst_types, dst_idxs,
               epochs, batch_size, lr, device, neg_samples):
    """Train all 4 embedding tables jointly. Returns per-epoch avg loss list."""
    optimizer    = t.optim.Adam(model.parameters(), lr=lr)
    N            = len(src_types)
    loss_history = []

    for epoch in range(1, epochs + 1):
        perm    = np.random.permutation(N)
        st_p    = src_types[perm]
        si_p    = src_idxs[perm]
        dt_p    = dst_types[perm]
        di_p    = dst_idxs[perm]

        epoch_loss = 0.0
        n_steps    = 0

        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            b_st = st_p[start:end]
            b_si = si_p[start:end]
            b_dt = dt_p[start:end]
            b_di = di_p[start:end]

            group_losses = []
            for st_val in range(4):
                for dt_val in range(4):
                    mask = (b_st == st_val) & (b_dt == dt_val)
                    if not mask.any():
                        continue
                    s_idx = t.from_numpy(
                        b_si[mask].astype(np.int64)).to(device)
                    d_idx = t.from_numpy(
                        b_di[mask].astype(np.int64)).to(device)
                    group_losses.append(
                        model.line_loss_batch(
                            st_val, s_idx, dt_val, d_idx, neg_samples))

            if not group_losses:
                continue

            total_loss = sum(group_losses)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_steps    += 1

        avg = epoch_loss / max(n_steps, 1)
        loss_history.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  avg_loss={avg:.4f}")

    return loss_history


# =========================================================
# Post-training normalization
# =========================================================

@t.no_grad()
def normalize_embeddings(model):
    """L2-normalize rows 1.. in-place; row 0 (padding) stays zero."""
    for emb in [model.user_emb, model.cat_emb, model.loc_emb, model.time_emb]:
        w = emb.weight.data
        norms = w[1:].norm(dim=1, keepdim=True).clamp_min(1e-12)
        w[1:] = w[1:] / norms


def get_numpy_embs(model):
    return {
        "user_emb": model.user_emb.weight.data.cpu().numpy(),
        "cat_emb":  model.cat_emb.weight.data.cpu().numpy(),
        "loc_emb":  model.loc_emb.weight.data.cpu().numpy(),
        "time_emb": model.time_emb.weight.data.cpu().numpy(),
    }


# =========================================================
# Macro-category mapping
# =========================================================

def load_macro_mapping(macro_csv_path, cid_cat_dict):
    """
    Returns (cid1_to_macro, macro_list, macro_to_color).
    cid1_to_macro: {1-indexed cid → macro_cat str}
    """
    df = pd.read_csv(macro_csv_path)

    cid1_to_macro = {}
    if not pd.api.types.is_numeric_dtype(df["category"]):
        cat_str_to_cid1 = {str(v).strip(): int(k) + 1
                           for k, v in cid_cat_dict.items()}
        for _, row in df.iterrows():
            cid1 = cat_str_to_cid1.get(str(row["category"]).strip())
            if cid1 is not None:
                cid1_to_macro[cid1] = str(row["macro_cat"]).strip()
    else:
        for _, row in df.iterrows():
            cid1_to_macro[int(row["category"]) + 1] = str(row["macro_cat"]).strip()

    macro_list     = sorted(set(cid1_to_macro.values()))
    tab10          = plt.cm.get_cmap("tab10")
    macro_to_color = {m: tab10(i) for i, m in enumerate(macro_list)}
    macro_to_color["Unknown"] = (0.6, 0.6, 0.6, 1.0)

    return cid1_to_macro, macro_list, macro_to_color


# =========================================================
# t-SNE helpers
# =========================================================

def _run_tsne(X, perplexity, seed):
    tsne = TSNE(n_components=2, perplexity=perplexity,
                init="pca", random_state=seed,
                n_iter=1000, learning_rate="auto")
    return tsne.fit_transform(X.astype(np.float64))


def plot_cat_tsne(cat_emb_np, cat_size, cid1_to_macro, macro_list, macro_to_color,
                  cid_cat_dict, train_trajectories, top_n_cats,
                  perplexity, seed, save_path):
    """t-SNE of all C cat embeddings, coloured by macro, top-freq cats labelled."""
    # Training visit frequency per 1-indexed cid
    cat_freq = defaultdict(int)
    for traj in train_trajectories:
        for cid0 in traj.get("cat_seq", []):
            cat_freq[cid0 + 1] += 1

    cat_vecs    = cat_emb_np[1: cat_size + 1]   # [C, D]
    cat_macros  = [cid1_to_macro.get(c1, "Unknown")
                   for c1 in range(1, cat_size + 1)]

    print(f"\n[t-SNE cat] {cat_size} categories, perplexity={perplexity} ...")
    coords = _run_tsne(cat_vecs, perplexity, seed)

    fig, ax = plt.subplots(figsize=(12, 10))

    legend_handles = []
    for macro in macro_list:
        mask  = np.array([m == macro for m in cat_macros])
        color = macro_to_color[macro]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=20, c=[color], alpha=0.65, edgecolor="none", zorder=2)
        legend_handles.append(
            mpatches.Patch(color=color, label=f"{macro} (n={mask.sum()})"))

    # Annotate top-freq cats
    top_cids1 = sorted(cat_freq, key=lambda c: -cat_freq[c])[:top_n_cats]
    for cid1 in top_cids1:
        if 1 <= cid1 <= cat_size:
            i    = cid1 - 1   # 0-indexed into coords
            name = str(cid_cat_dict.get(str(cid1 - 1),
                       cid_cat_dict.get(cid1 - 1, str(cid1)))).strip()
            if len(name) > 18:
                name = name[:16] + "…"
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, style="italic", color="dimgray", zorder=3)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Category embedding (n={cat_size} cats, perplexity={perplexity:.0f})",
                 fontsize=13)
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True,
              title="Macro category")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_loc_tsne(loc_emb_np, loc_size, pid_cid_dict, cid1_to_macro,
                  macro_list, macro_to_color, n_sample_pois,
                  perplexity, seed, save_path):
    """Stratified POI sample coloured by macro; macro centres as stars."""
    # pid → macro
    pid_to_macro = {}
    for pid in range(1, loc_size + 1):
        cid0 = pid_cid_dict.get(pid)
        if cid0 is None:
            cid0 = pid_cid_dict.get(str(pid))
        pid_to_macro[pid] = (cid1_to_macro.get(int(cid0) + 1, "Unknown")
                             if cid0 is not None else "Unknown")

    # Stratified sample
    macro_pids  = {m: [] for m in macro_list}
    for pid in range(1, loc_size + 1):
        m = pid_to_macro.get(pid, "Unknown")
        if m in macro_pids:
            macro_pids[m].append(pid)

    total_known = sum(len(v) for v in macro_pids.values())
    alloc       = {m: round(n_sample_pois * len(macro_pids[m]) / max(total_known, 1))
                   for m in macro_list}
    diff        = n_sample_pois - sum(alloc.values())
    biggest     = max(macro_list, key=lambda m: len(macro_pids[m]))
    alloc[biggest] += diff

    rng          = random.Random(seed)
    sampled_pids = []
    for m in macro_list:
        pool = macro_pids[m]
        k    = min(alloc[m], len(pool))
        sampled_pids.extend(rng.sample(pool, k))
    sampled_pids = np.array(sampled_pids, dtype=np.int64)

    # Macro centres: mean of POI embs, L2-normalised
    D = loc_emb_np.shape[1]
    macro_centers = {}
    for m in macro_list:
        pids_m = macro_pids[m]
        if pids_m:
            vecs = loc_emb_np[np.array(pids_m, dtype=np.int64)]
            mean = vecs.mean(axis=0)
            n    = np.linalg.norm(mean)
            macro_centers[m] = (mean / n if n > 0 else mean).astype(np.float32)
        else:
            macro_centers[m] = np.zeros(D, dtype=np.float32)

    macro_mat = np.array([macro_centers[m] for m in macro_list])
    X         = np.vstack([loc_emb_np[sampled_pids], macro_mat])
    n_s       = len(sampled_pids)

    print(f"\n[t-SNE loc] {n_s} POIs + {len(macro_list)} macro centres, "
          f"perplexity={perplexity} ...")
    coords    = _run_tsne(X, perplexity, seed)
    poi_xy    = coords[:n_s]
    center_xy = coords[n_s:]

    fig, ax = plt.subplots(figsize=(12, 10))

    legend_handles = []
    for i_m, macro in enumerate(macro_list):
        mask  = np.array([pid_to_macro[int(pid)] == macro for pid in sampled_pids])
        color = macro_to_color[macro]
        ax.scatter(poi_xy[mask, 0], poi_xy[mask, 1],
                   s=12, c=[color], alpha=0.5, edgecolor="none", zorder=2)
        ax.scatter(center_xy[i_m, 0], center_xy[i_m, 1],
                   marker="*", s=500, c=[color], edgecolor="black",
                   linewidth=1.5, zorder=5)
        ax.annotate(macro, (center_xy[i_m, 0], center_xy[i_m, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.8, edgecolor="gray"),
                    zorder=6)
        legend_handles.append(
            mpatches.Patch(color=color,
                           label=f"{macro} (n={len(macro_pids[macro])})"))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"POI embedding ({n_sample_pois} POIs sampled, perplexity={perplexity:.0f})",
        fontsize=13)
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True,
              title="Macro category")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal LINE embedding on 9 heterogeneous subgraphs")
    parser.add_argument("--pkl_path",        type=str,
        default=os.path.join(BASE, "NYC_data.pkl"))
    parser.add_argument("--save_dir",        type=str,
        default=os.path.join(BASE, "multimodal_line_runs_16"))
    parser.add_argument("--emb_dim",         type=int,   default=16)
    parser.add_argument("--theta_percent",   type=int,   default=50)
    parser.add_argument("--neg_samples",     type=int,   default=30)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=1024)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--loc_dist_km",     type=float, default=0.1)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_seed",       type=int,   default=42)
    parser.add_argument("--top_n_cats",      type=int,   default=20)
    parser.add_argument("--n_sample_pois",   type=int,   default=1000)
    parser.add_argument("--macro_csv_path",  type=str,
        default=os.path.join(BASE, "category_cluster_map.csv"))
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--device",          type=str,
        default="cuda" if t.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = t.device(args.device)
    print(f"Device: {device}  emb_dim={args.emb_dim}  epochs={args.epochs}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading pkl ...")
    data_obj        = load_pickle(args.pkl_path)
    train_trajs     = data_obj["train_trajectories"]
    stats           = data_obj["stats"]
    cat_size        = stats["num_cats"]
    loc_size        = stats["num_pois"]
    pid_cid_dict    = data_obj["pid_cid_dict"]
    cid_cat_dict    = data_obj.get("cid_cat_dict", {})
    pid_lat_lon_raw = data_obj.get("pid_lat_lon", {})
    pid_lat_lon     = {int(k): v for k, v in pid_lat_lon_raw.items()}

    all_uids  = sorted(set(traj["user_id"]
                           for traj in train_trajs if traj["user_id"] != -1))
    num_users = max(all_uids) if all_uids else 1
    print(f"  num_users={num_users}  cat_size={cat_size}  loc_size={loc_size}")

    # ── Macro-category mapping (needed for graph 8) ────────────────────────────
    print("\nLoading macro-category mapping ...")
    cid1_to_macro, macro_list, macro_to_color = load_macro_mapping(
        args.macro_csv_path, cid_cat_dict)
    num_macros = len(macro_list)
    print(f"  {num_macros} macro categories: {macro_list}")

    # ── Build subgraphs ────────────────────────────────────────────────────────
    print("\nBuilding subgraphs ...")
    graphs = build_cooccurrence_graphs(train_trajs, args.theta_percent)

    graphs[6] = build_time_cycle_graph()
    print(f"  Graph 7 (time-time cycle): {len(graphs[6])} edges")

    graphs[7] = build_cat_macro_graph(cid1_to_macro, macro_list, cat_size)

    graphs[8] = build_loc_loc_graph(pid_lat_lon, loc_size, args.loc_dist_km)

    edge_counts = {g: len(graphs[g]) for g in range(9)}
    print("\nSubgraph edge counts (undirected):")
    for g in range(9):
        st, dt = GRAPH_TYPES[g]
        print(f"  Graph {g+1:2d} ({TYPE_NAMES[st]}-{TYPE_NAMES[dt]}): "
              f"{edge_counts[g]:>8,} edges")

    # ── Master directed edge list ──────────────────────────────────────────────
    print("\nBuilding master directed edge list ...")
    src_types, src_idxs, dst_types, dst_idxs = build_master_edge_list(graphs)
    print(f"  Total directed training edges: {len(src_types):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining on {device} ...")
    model = MultimodalLineEmbedding(
        num_users=num_users, num_cats=cat_size,
        num_locs=loc_size, num_macros=num_macros, emb_dim=args.emb_dim,
    ).to(device)

    loss_history = train_line(
        model, src_types, src_idxs, dst_types, dst_idxs,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, neg_samples=args.neg_samples,
    )

    # ── L2 normalise ──────────────────────────────────────────────────────────
    print("\nL2-normalising embeddings ...")
    normalize_embeddings(model)
    embs = get_numpy_embs(model)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Saving ...")
    for name, arr in embs.items():
        np.save(os.path.join(args.save_dir, f"{name}.npy"), arr)
    t.save(embs, os.path.join(args.save_dir, "embeddings.pt"))

    meta = dict(
        num_users=num_users, cat_size=cat_size, loc_size=loc_size,
        args={k: str(v) for k, v in vars(args).items()},
        edge_counts_undirected=edge_counts,
        total_directed_edges=int(len(src_types)),
        loss_history=loss_history,
    )
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Visualize ─────────────────────────────────────────────────────────────

    plot_cat_tsne(
        embs["cat_emb"], cat_size, cid1_to_macro, macro_list, macro_to_color,
        cid_cat_dict, train_trajs, args.top_n_cats,
        args.tsne_perplexity, args.tsne_seed,
        os.path.join(args.save_dir, "cat_tsne.png"),
    )

    plot_loc_tsne(
        embs["loc_emb"], loc_size, pid_cid_dict, cid1_to_macro,
        macro_list, macro_to_color, args.n_sample_pois,
        args.tsne_perplexity, args.tsne_seed,
        os.path.join(args.save_dir, "loc_tsne.png"),
    )

    print(f"\nDone. All outputs in: {args.save_dir}/")


if __name__ == "__main__":
    main()
