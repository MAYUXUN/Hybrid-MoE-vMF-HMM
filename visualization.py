# -*- coding: utf-8 -*-
"""
analysis_hybrid_vmf.py
======================
Interpretability analysis for a trained MixtureHybridVMFHMM checkpoint.
Supports any K >= 1.

Produces:
  1) state_macro_heatmap[_k{k}].png      — P(macro|state) from μ_base
  1b) state_macro_excess_heatmap[_k{k}].png
  1c) state_macro_contrast_heatmap[_k{k}].png
  2) state_geo_maps/state_{tag}_geo_top{N}.png
  3) poi_nearest_state_geo[_k{k}].png
  4) transition_matrix[_k{k}].png
  5) residual_displacement_k{k}_s{s}.png
  6) temporal_heatmaps/temporal_{tag}.png — 7×12 weekday×2h-slot heatmap per state
"""

import os
import csv
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from tqdm import tqdm

import torch as t
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from mixture_hybrid_vmf_hmm import (
    MixtureHybridVMFHMM,
    load_pickle,
    load_line_embeddings,
    infer_sizes_from_pkl,
)


# =========================================================
# 1. CLI
# =========================================================
def parse_args():
    BASE = "/content/drive/MyDrive/Colab Notebooks/phd_research/ABM/"
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str,
                   default=os.path.join(BASE, "mhvmf_K4_S4_angle_005_t01_10penalty_NYC/best_model.pt"))
    p.add_argument("--pkl_path",  type=str,
                   default=os.path.join(BASE, "NYC_getnext_ready.pkl"))
    p.add_argument("--pretrained_emb_path", type=str,
                   default=os.path.join(BASE, "multimodal_line_emb_NYC"))
    p.add_argument("--macro_csv_path", type=str,
                   default=os.path.join(BASE, "category_cluster_map.csv"))
    p.add_argument("--out_dir",          type=str,
                   default=os.path.join(BASE, "mhvmf_K4_S4_angle_005_t01_10penalty_NYC//analysis"))
    p.add_argument("--tsne_perplexity",  type=float, default=30.0)
    p.add_argument("--tsne_seed",        type=int, default=42)
    p.add_argument("--n_traj",           type=int, default=10)
    p.add_argument("--target_state",     type=int, default=None)
    p.add_argument("--target_k",         type=int, default=None,
                   help="Which mixture group k to use for residual displacement "
                        "(default: auto-select the group with most trajectories)")
    p.add_argument("--geo_top_n",        type=int, default=100,
                   help="Top-N nearest POIs per state for geo map")
    p.add_argument("--cat_top_n",        type=int, default=10,
                   help="Top-N micro categories to show per state in bar charts")
    p.add_argument("--n_grid",           type=int, default=16,
                   help="Grid resolution for geo grid heatmap (n_grid × n_grid cells)")
    p.add_argument("--device",           type=str,
                   default="cuda" if t.cuda.is_available() else "cpu")
    return p.parse_args()


# =========================================================
# 2. Model loading
# =========================================================
def load_model(ckpt_path, device):
    ckpt = t.load(ckpt_path, map_location="cpu", weights_only=False)
    a    = ckpt["args"]
    sizes     = ckpt["sizes"]
    user_size = ckpt["user_size"]
    emb_dim   = ckpt["emb_dim"]

    loc_size = sizes["loc_size"]
    cat_size = sizes["cat_size"]
    th_size  = sizes["th_size"]
    K        = a["num_classes"]
    S        = a["num_states"]

    line_ckpt_dir = (os.path.dirname(a["line_ckpt_dir"])
                     if os.path.isfile(a["line_ckpt_dir"])
                     else a["line_ckpt_dir"])

    user_np, loc_np, cat_np, time_np = load_line_embeddings(
        line_ckpt_dir, user_size, loc_size, cat_size, th_size)

    model = MixtureHybridVMFHMM(
        loc_size=loc_size, cat_size=cat_size, th_size=th_size,
        user_size=user_size, emb_dim=emb_dim,
        num_classes=K, num_states=S,
        seq_hidden=a.get("seq_hidden", 128),
        hist_hidden=a.get("hist_hidden", 128),
        dropout=0.0,
        max_angle=a.get("max_angle", 0.05),
        log_kappa_max=a.get("log_kappa_max", 4.0),
        log_kappa_min=a.get("log_kappa_min", None),
        line_user_emb_np=user_np,
        line_loc_emb_np=loc_np,
        line_cat_emb_np=cat_np,
        line_time_emb_np=time_np,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded model: K={K}  S={S}  D={emb_dim}  "
          f"loc={loc_size}  cat={cat_size}  "
          f"max_angle={model.max_angle:.4f}  "
          f"log_kappa_max={model.log_kappa_max:.3f}  "
          f"log_kappa_min={model.log_kappa_min:.3f}")
    return model, sizes, K, S, emb_dim


# =========================================================
# 3. Macro-category mapping helpers
# =========================================================
def build_macro_mapping(pkl_data, macro_csv_path, loc_size):
    pid_cid_dict = pkl_data.get("pid_cid_dict", {})
    cid_cat_dict = pkl_data.get("cid_cat_dict", {})

    cat_to_macro = {}
    with open(macro_csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat_to_macro[row["category"].strip()] = row["macro_cat"].strip()

    cluster_names = sorted(set(cat_to_macro.values()))
    macro_to_idx  = {n: i for i, n in enumerate(cluster_names)}
    n_macro       = len(cluster_names)

    cmap  = cm.get_cmap("tab10", n_macro)
    macro_to_color = {n: cmap(i) for i, n in enumerate(cluster_names)}

    pid_to_macro = np.zeros(loc_size, dtype=np.int64)
    for pid_raw, cid_raw in pid_cid_dict.items():
        pid = int(pid_raw)
        cid = int(cid_raw)
        if pid < 1 or pid > loc_size:
            continue
        cat_name = cid_cat_dict.get(cid) or cid_cat_dict.get(str(cid))
        if cat_name is None:
            continue
        macro = cat_to_macro.get(str(cat_name).strip())
        if macro is None:
            continue
        pid_to_macro[pid - 1] = macro_to_idx.get(macro, 0)

    return pid_to_macro, cluster_names, macro_to_color, n_macro


def build_pid_coords(pkl_data):
    coords = {}
    LAT_KEYS = ("lat_seq", "latitude_seq")
    LON_KEYS = ("lon_seq", "longitude_seq", "lng_seq")

    for split in ("train_trajectories", "val_trajectories", "test_trajectories"):
        for traj in pkl_data.get(split, []):
            poi_seq = traj.get("poi_seq", [])
            lat_seq = next((traj[k] for k in LAT_KEYS if k in traj), None)
            lon_seq = next((traj[k] for k in LON_KEYS if k in traj), None)
            if lat_seq is None or lon_seq is None:
                continue
            for pid, lat, lon in zip(poi_seq, lat_seq, lon_seq):
                if pid > 0 and pid not in coords:
                    coords[pid] = (float(lat), float(lon))

    return coords


def build_cat_mapping(pkl_data, macro_csv_path, loc_size):
    """
    Returns
    -------
    pid_to_cat_idx : np.ndarray [loc_size]  0-indexed fine-grained cat id
    cat_names      : list[str]  sorted fine-grained category names
    cat_to_color   : dict name → RGBA colour (inherits macro colour)
    """
    pid_cid_dict = pkl_data.get("pid_cid_dict", {})
    cid_cat_dict = pkl_data.get("cid_cat_dict", {})

    cat_to_macro = {}
    with open(macro_csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat_to_macro[row["category"].strip()] = row["macro_cat"].strip()

    macro_names  = sorted(set(cat_to_macro.values()))
    macro_cmap   = cm.get_cmap("tab10", len(macro_names))
    macro_to_color = {n: macro_cmap(i) for i, n in enumerate(macro_names)}

    cat_names  = sorted(set(cat_to_macro.keys()))
    cat_to_idx = {n: i for i, n in enumerate(cat_names)}
    cat_to_color = {
        c: macro_to_color.get(cat_to_macro.get(c), (0.5, 0.5, 0.5, 1.0))
        for c in cat_names
    }

    pid_to_cat_idx = np.zeros(loc_size, dtype=np.int64)
    for pid_raw, cid_raw in pid_cid_dict.items():
        pid = int(pid_raw)
        cid = int(cid_raw)
        if pid < 1 or pid > loc_size:
            continue
        cat_name = cid_cat_dict.get(cid) or cid_cat_dict.get(str(cid))
        if cat_name is None:
            continue
        cat_name = str(cat_name).strip()
        idx = cat_to_idx.get(cat_name)
        if idx is None:
            continue
        pid_to_cat_idx[pid - 1] = idx

    return pid_to_cat_idx, cat_names, cat_to_color


# =========================================================
# 4. Product 1 — state-macro heatmap (static μ_base emission)
#    Loops over all K groups; returns {k: p_macro_tensor [S, n_macro]}
# =========================================================
def _draw_heatmap(matrix_np, xtick_labels, ytick_labels, cmap_name,
                  vmin, vmax, title, xlabel, ylabel, fmt, out_path,
                  cbar_label="Probability", font_scale=1.0, cell_fontsize=10):
    S_  = matrix_np.shape[0]
    M_  = matrix_np.shape[1]
    fig, ax = plt.subplots(figsize=(max(14, M_ * 1.5), max(9, S_ * 1.0)))
    norm_obj = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix_np, cmap=cmap_name, norm=norm_obj, aspect="auto")
    ax.set_xticks(range(M_))
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=11 * font_scale)
    ax.set_yticks(range(S_))
    ax.set_yticklabels(ytick_labels, fontsize=11 * font_scale)
    ax.set_xlabel(xlabel, fontsize=13 * font_scale)
    ax.set_ylabel(ylabel, fontsize=13 * font_scale)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=12 * font_scale)
    cbar.ax.tick_params(labelsize=10 * font_scale)

    cmap_obj = plt.cm.get_cmap(cmap_name)
    for i in range(S_):
        for j in range(M_):
            v = matrix_np[i, j]
            r, g, b, _ = cmap_obj(norm_obj(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if lum < 0.5 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=text_color, fontsize=cell_fontsize, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_state_macro_heatmap(model, pid_to_macro, cluster_names, n_macro, K, S, out_dir):
    print("=== Product 1: state-macro heatmap ===")
    all_loc_emb = F.normalize(model.loc_emb.weight[1:], p=2, dim=-1)   # [L, D]
    pid_to_macro_t = t.from_numpy(pid_to_macro)

    mu_base_norm = F.normalize(model.mu_base.detach(), p=2, dim=-1)     # [K, S, D]
    lk_base      = model.log_kappa_base.detach()                        # [K, S]

    ytick_labels = [f"s{s}" for s in range(S)]
    p_macro_dict = {}

    for k in range(K):
        p_macro_mat = t.zeros(S, n_macro)
        for s in range(S):
            kappa_s = lk_base[k, s].exp()
            mu_s    = mu_base_norm[k, s]
            scaled  = kappa_s * (mu_s @ all_loc_emb.T)
            p_l     = t.softmax(scaled, dim=0)
            for m in range(n_macro):
                p_macro_mat[s, m] = p_l.cpu()[pid_to_macro_t == m].sum()

        p_macro_np = p_macro_mat.numpy()
        p_macro_dict[k] = p_macro_mat

        tag = f"_k{k}" if K > 1 else ""
        title = f"Static emission P(macro | state) from μ_base"
        if K > 1:
            title += f"  [k={k}]"

        out_path = out_dir / f"state_macro_heatmap{tag}.png"
        _draw_heatmap(p_macro_np, cluster_names, ytick_labels,
                      "Reds", 0.0, 1.0,
                      title, "Macro category", "State",
                      "{:.3f}", out_path, cbar_label="Probability")
        np.save(out_dir / f"state_macro_matrix{tag}.npy", p_macro_np)
        print(f"  Saved: {out_path}")

        print(f"  State → top-3 macro preferences (k={k}):")
        for s in range(S):
            top3 = p_macro_mat[s].argsort(descending=True)[:3]
            parts = [f"{cluster_names[i.item()]}({p_macro_mat[s, i].item():.2f})"
                     for i in top3]
            print(f"    s{s:2d}: " + ", ".join(parts))

    return p_macro_dict


# =========================================================
# 4b. P(macro|state) − P(macro) excess heatmap
# =========================================================
def make_state_macro_excess_heatmap(p_macro_dict, pid_to_macro,
                                    cluster_names, n_macro, K, S, out_dir):
    print("=== Product 1b: P(macro|state) − P(macro) excess heatmap ===")
    counts = np.array([(pid_to_macro == m).sum() for m in range(n_macro)],
                      dtype=np.float64)
    p_marginal = counts / counts.sum()
    ytick_labels = [f"s{s}" for s in range(S)]

    for k in range(K):
        p_macro_np = p_macro_dict[k].numpy()
        excess     = p_macro_np - p_marginal[np.newaxis, :]
        abs_max    = max(np.abs(excess).max(), 1e-9)

        tag = f"_k{k}" if K > 1 else ""
        title = "P(macro | state) − P(macro)   [red = enriched, blue = depleted]"
        if K > 1:
            title += f"  [k={k}]"

        out_path = out_dir / f"state_macro_excess_heatmap{tag}.png"
        _draw_heatmap(excess, cluster_names, ytick_labels,
                      "RdBu_r", -abs_max, abs_max,
                      title, "Macro category", "State",
                      "{:+.3f}", out_path, cbar_label="P(macro|state) − P(macro)")
        np.save(out_dir / f"state_macro_excess{tag}.npy", excess)
        print(f"  Saved: {out_path}")

        print(f"  State → top-3 enriched macros (k={k}):")
        for s in range(S):
            top3 = excess[s].argsort()[::-1][:3]
            parts = [f"{cluster_names[j]}({excess[s,j]:+.3f})" for j in top3]
            print(f"    s{s:2d}: " + ", ".join(parts))


# =========================================================
# 4c. P(macro|state) − mean_s P(macro|state=s) contrast heatmap
# =========================================================
def make_state_macro_contrast_heatmap(p_macro_dict, cluster_names, n_macro, K, S, out_dir):
    print("=== Product 1c: P(macro|state) − mean_s P(macro|state) contrast heatmap ===")
    ytick_labels = [f"s{s}" for s in range(S)]

    for k in range(K):
        p_macro_np       = p_macro_dict[k].numpy()
        mean_over_states = p_macro_np.mean(axis=0)
        contrast         = p_macro_np - mean_over_states[np.newaxis, :]
        abs_max          = max(np.abs(contrast).max(), 1e-9)

        tag = f"_k{k}" if K > 1 else ""
        title = (r"P(macro | state) $-$ $\overline{s}$P(macro | state)"
                 "   [red = above avg, blue = below avg]")
        if K > 1:
            title += f"  [k={k}]"

        out_path = out_dir / f"state_macro_contrast_heatmap{tag}.png"
        _draw_heatmap(contrast, cluster_names, ytick_labels,
                      "RdBu_r", -abs_max, abs_max,
                      title, "Macro category", "State",
                      "{:+.3f}", out_path, cbar_label="Contrast")
        np.save(out_dir / f"state_macro_contrast{tag}.npy", contrast)
        print(f"  Saved: {out_path}")

        print(f"  State → top-3 macros above cross-state mean (k={k}):")
        for s in range(S):
            top3 = contrast[s].argsort()[::-1][:3]
            parts = [f"{cluster_names[j]}({contrast[s,j]:+.3f})" for j in top3]
            print(f"    s{s:2d}: " + ", ".join(parts))


# =========================================================
# 4d. Per-state top-N micro category bar charts
# =========================================================
def make_state_cat_barcharts(model, pid_to_cat_idx, cat_names, cat_to_color,
                             K, S, top_n, out_dir):
    """
    For each (k, s): compute P(cat|state) from μ_base via softmax over loc embeddings,
    aggregate by fine-grained category using scatter_add, then plot top-{top_n} as a
    horizontal bar chart coloured by the category's macro group.
    Saves one PNG per (k, s) under state_cat_barcharts/.
    """
    print(f"=== Product 1d: per-state top-{top_n} micro category bar charts ===")

    all_loc_emb  = F.normalize(model.loc_emb.weight[1:], p=2, dim=-1)   # [L, D]
    mu_base_norm = F.normalize(model.mu_base.detach(), p=2, dim=-1)      # [K, S, D]
    lk_base      = model.log_kappa_base.detach()                         # [K, S]

    pid_to_cat_t = t.from_numpy(pid_to_cat_idx).long()                   # [L]
    n_cats       = len(cat_names)

    cat_out = out_dir / "state_cat_barcharts"
    cat_out.mkdir(exist_ok=True)

    for k in range(K):
        for s in range(S):
            kappa_s = lk_base[k, s].exp()
            mu_s    = mu_base_norm[k, s]
            scaled  = kappa_s * (mu_s @ all_loc_emb.T)                  # [L]
            p_l     = t.softmax(scaled, dim=0)                           # [L]

            # Aggregate probability per fine-grained category
            p_cat = t.zeros(n_cats).scatter_add(0, pid_to_cat_t, p_l.cpu())  # [n_cats]
            p_cat_np = p_cat.numpy()

            top_idx    = p_cat_np.argsort()[::-1][:top_n]
            top_cats   = [cat_names[i] for i in top_idx]
            top_probs  = [float(p_cat_np[i]) for i in top_idx]
            top_colors = [cat_to_color.get(c, (0.5, 0.5, 0.5, 1.0)) for c in top_cats]

            tag = f"k{k}_s{s}" if K > 1 else f"s{s}"

            fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.7)))
            # Draw highest-prob bar at the top
            y_pos = list(range(top_n - 1, -1, -1))
            ax.barh(y_pos, top_probs[::-1], color=top_colors[::-1],
                    edgecolor="black", linewidth=0.4, height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_cats[::-1], fontsize=14)
            ax.set_xlabel("Probability", fontsize=16)
            ax.tick_params(axis="x", labelsize=14)

            x_max = max(top_probs) * 1.25
            ax.set_xlim(0, x_max)
            for y, prob in zip(y_pos, top_probs[::-1]):
                ax.text(prob + x_max * 0.01, y, f"{prob:.4f}",
                        va="center", ha="left", fontsize=13)

            plt.tight_layout()
            out_path = cat_out / f"state_{tag}_cat_top{top_n}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        sfx = f"(k={k})" if K > 1 else ""
        print(f"  Saved {S} cat bar charts {sfx} → {cat_out}/")


# =========================================================
# 5. Product 2 — per-state top-N nearest POIs on geo map
# =========================================================
def make_state_geo_maps(model, pid_coords, pid_to_macro,
                        cluster_names, macro_to_color, K, S, top_n, out_dir):
    print(f"=== Product 2: per-state geo maps (top-{top_n} nearest POIs) ===")

    if not pid_coords:
        print("  WARNING: pid_coords is empty — skipping geo maps.")
        return

    all_loc_emb = F.normalize(
        model.loc_emb.weight[1:], p=2, dim=-1).detach().cpu()
    mu_base_cpu = F.normalize(
        model.mu_base.detach(), p=2, dim=-1).cpu()                      # [K, S, D]

    geo_out = out_dir / "state_geo_maps"
    geo_out.mkdir(exist_ok=True)

    for k in range(K):
        for s in range(S):
            mu_ks   = mu_base_cpu[k, s]
            cos_sim = all_loc_emb @ mu_ks                               # [L]
            top_idx = cos_sim.topk(top_n).indices.numpy()

            lats, lons, macros = [], [], []
            for pid0 in top_idx:
                pid   = int(pid0) + 1
                coord = pid_coords.get(pid)
                if coord is None:
                    continue
                lats.append(coord[0])
                lons.append(coord[1])
                macros.append(int(pid_to_macro[pid0]))

            if not lats:
                continue

            colors = [macro_to_color[cluster_names[m]] for m in macros]
            tag = f"k{k}_s{s}" if K > 1 else f"s{s}"

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(lons, lats, c=colors, s=60, alpha=0.8,
                       edgecolor="black", linewidths=0.4, zorder=2)

            seen_macros = sorted(set(macros))
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=macro_to_color[cluster_names[m]],
                           markeredgecolor="black", markersize=8,
                           label=cluster_names[m])
                for m in seen_macros
            ]
            ax.legend(handles=handles, loc="lower left", fontsize=7,
                      frameon=True, title="Macro category", title_fontsize=7)
            ax.set_xlabel("Longitude", fontsize=9)
            ax.set_ylabel("Latitude",  fontsize=9)
            ax.set_title(
                f"State {tag} — top-{len(lats)} nearest POIs by μ_base cosine sim",
                fontsize=10)
            plt.tight_layout()
            out_path = geo_out / f"state_{tag}_geo_top{top_n}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        sfx = f"k={k}" if K > 1 else ""
        print(f"  Saved {S} geo maps {sfx} → {geo_out}/")


# =========================================================
# 6. Transition matrix heatmap
# =========================================================
def make_transition_heatmap(model, K, S, out_dir):
    print("=== Transition matrix heatmap ===")
    _, trans_prob = model.get_hmm_params()
    trans_np = trans_prob.detach().cpu().numpy()                        # [K, S, S]

    for k in range(K):
        A = trans_np[k]

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=A.max(), aspect="auto")
        ax.set_xticks(range(S))
        ax.set_xticklabels([f"s{s}" for s in range(S)],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(S))
        ax.set_yticklabels([f"s{s}" for s in range(S)], fontsize=8)
        ax.set_xlabel("Next state", fontsize=10)
        ax.set_ylabel("Current state", fontsize=10)
        tag = f"k{k}" if K > 1 else ""
        ax.set_title(
            f"HMM transition matrix A{tag}  (row=current, col=next)", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = max(0.05, A.max() * 0.2)
        for i in range(S):
            for j in range(S):
                v = A[i, j]
                if v >= thresh:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > 0.5 * A.max() else "black",
                            fontsize=7)

        plt.tight_layout()
        fname = f"transition_matrix_k{k}.png" if K > 1 else "transition_matrix.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        print(f"  Top-3 transitions per state (k={k}):")
        for s in range(S):
            top3 = A[s].argsort()[::-1][:3]
            parts = [f"→s{j}({A[s,j]:.3f})" for j in top3]
            print(f"    s{s:2d}: " + "  ".join(parts))


# =========================================================
# 7. HMM forward helper (single trajectory)
# =========================================================
@t.no_grad()
def get_last_step_state(model, traj, device):
    """
    Run HMM forward pass on one trajectory.
    Returns (k_star, s_star, hist_seq [1,T,H], T).
    k_star, s_star are the jointly most-likely group and state at the last step.
    """
    uid_id  = traj["user_id"]
    poi_seq = traj["poi_seq"]
    cat_seq = [c + 1 for c in traj["cat_seq"]]
    th_seq  = traj["hour_bin_seq"]
    T       = len(poi_seq)
    S       = model.mu_base.shape[1]

    loc   = t.LongTensor(poi_seq).unsqueeze(0).to(device)
    cat   = t.LongTensor(cat_seq).unsqueeze(0).to(device)
    th    = t.LongTensor(th_seq).unsqueeze(0).to(device)
    uid_t = t.LongTensor([uid_id]).to(device)

    seq_in           = model._build_seq_in(loc, cat, th)
    hist_seq, _      = model.encoder2(seq_in, [T])
    log_emiss        = model.compute_emission_logprob(loc, hist_seq, uid_t)  # [1,K,T,S]

    init_prob, trans_prob = model.get_hmm_params()
    log_init  = t.log(init_prob.clamp_min(1e-12))
    log_trans = t.log(trans_prob.clamp_min(1e-12))

    log_alpha = log_init.unsqueeze(0) + log_emiss[:, :, 0, :]          # [1, K, S]
    for tt in range(1, T):
        log_pred  = t.logsumexp(
            log_alpha.unsqueeze(-1) + log_trans.unsqueeze(0), dim=2)
        log_alpha = log_pred + log_emiss[:, :, tt, :]

    # [1, K, S] → pick argmax over K*S jointly
    flat_idx = log_alpha[0].reshape(-1).argmax().item()
    k_star   = flat_idx // S
    s_star   = flat_idx % S
    return k_star, s_star, hist_seq.cpu(), T


@t.no_grad()
def compute_mu_final_for_traj(model, uid_id, hist_seq, T, target_k, target_state, device):
    """
    Compute μ_final[target_k, target_state] for this trajectory's context.
    Returns [D] unit vector on CPU.
    """
    H = hist_seq.shape[-1]
    h_ctx = hist_seq[:, T - 2, :].to(device) if T > 1 else t.zeros(1, H, device=device)

    u_idx = min(uid_id, model.user_emb.num_embeddings - 1)
    u_vec = model.user_emb(t.LongTensor([u_idx]).to(device))
    z_t   = t.cat([u_vec, h_ctx], dim=-1)
    delta_mu, _ = model.hybrid_enc(z_t)                                # [1, K, S, D]
    mu_final = model.bounded_angular_update(model.mu_base, delta_mu)  # [1, K, S, D]
    return mu_final[0, target_k, target_state, :].cpu()


# =========================================================
# 7. Product 3 — residual displacement for target (k, state)
# =========================================================
def make_residual_displacement(model, test_trajs, pid_to_macro,
                               cluster_names, macro_to_color, K, S, args, out_dir):
    print("=== Product 3: residual displacement t-SNE ===")
    device = next(model.parameters()).device

    print("  Scanning test trajectories for state assignment …")
    state_records = {}
    for traj in tqdm(test_trajs, desc="  HMM forward", leave=False):
        uid = traj["user_id"]
        if uid == -1 or len(traj["poi_seq"]) < 2:
            continue
        k_star, s_star, hist_seq_cpu, T = get_last_step_state(model, traj, device)
        key = (k_star, s_star)
        if key not in state_records:
            state_records[key] = []
        state_records[key].append(
            {"traj": traj, "uid": uid, "hist_seq": hist_seq_cpu, "T": T,
             "k_star": k_star, "s_star": s_star})

    counts = {key: len(v) for key, v in state_records.items()}
    print("  (k,s) counts:", {f"k{k}s{s}": counts[(k, s)]
                               for k, s in sorted(counts)})

    # ---- Select target (k, s) ----
    user_k = args.target_k
    user_s = args.target_state

    if user_k is not None and user_s is not None:
        target_key = (user_k, user_s)
        if target_key not in state_records:
            print(f"  WARNING: (k={user_k}, s={user_s}) has 0 trajectories.")
            state_records[target_key] = []
    elif user_s is not None:
        # pick the k that has the most trajectories for this s
        candidates = {k: len(state_records.get((k, user_s), []))
                      for k in range(K)}
        best_k = max(candidates, key=candidates.get)
        target_key = (best_k, user_s)
        print(f"  target_state={user_s} → auto-selected k={best_k}")
    else:
        # pick (k, s) with most trajectories that meets n_traj threshold
        eligible = [(key, len(v)) for key, v in state_records.items()
                    if len(v) >= args.n_traj]
        if eligible:
            target_key = max(eligible, key=lambda x: x[1])[0]
        else:
            target_key = max(state_records, key=lambda key: len(state_records[key]))
            print(f"  No (k,s) has >= {args.n_traj} trajectories; "
                  f"using {target_key} ({counts.get(target_key, 0)} trajs).")
        print(f"  Auto-selected target = {target_key} "
              f"({counts.get(target_key, 0)} trajectories)")

    target_k, target_s = target_key
    selected = state_records.get(target_key, [])[:args.n_traj]
    print(f"  Using {len(selected)} trajectories for k={target_k} s={target_s}")

    if not selected:
        print("  No trajectories — skipping residual displacement.")
        return

    # ---- Compute μ_final ----
    mu_final_list, uid_list = [], []
    for rec in selected:
        mf = compute_mu_final_for_traj(
            model, rec["uid"], rec["hist_seq"], rec["T"], target_k, target_s, device)
        mu_final_list.append(mf)
        uid_list.append(rec["uid"])

    mu_final_10 = t.stack(mu_final_list, dim=0)                        # [n, D]
    n_sel       = mu_final_10.shape[0]

    mu_base_flat_k = F.normalize(
        model.mu_base[target_k].detach(), p=2, dim=-1).cpu()           # [S, D]
    mu_base_s = mu_base_flat_k[target_s].unsqueeze(0)                  # [1, D]

    # ---- t-SNE ----
    all_loc_emb_np = F.normalize(
        model.loc_emb.weight[1:], p=2, dim=-1).detach().cpu().numpy()
    L = all_loc_emb_np.shape[0]

    X3 = np.concatenate([
        all_loc_emb_np,
        mu_base_s.numpy(),
        mu_final_10.numpy(),
    ], axis=0)

    print(f"  Running t-SNE on {X3.shape[0]} points …")
    tsne3   = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                   init="pca", random_state=args.tsne_seed,
                   n_iter=1000, learning_rate="auto")
    coords3  = tsne3.fit_transform(X3.astype(np.float64))
    poi_xy3  = coords3[:L]
    base_xy3 = coords3[L]
    fin_xy3  = coords3[L + 1:]

    tag_ks = f"k{target_k}_s{target_s}" if K > 1 else f"s{target_s}"

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(poi_xy3[:, 0], poi_xy3[:, 1],
               s=4, c="lightgray", alpha=0.3, zorder=1, edgecolor="none")
    ax.scatter(base_xy3[0], base_xy3[1],
               s=600, marker="*", c="gold", edgecolor="black", linewidth=1.5,
               zorder=5, label=f"μ_base[{tag_ks}]")

    for i in range(n_sel):
        ax.annotate("",
                    xy=(fin_xy3[i, 0], fin_xy3[i, 1]),
                    xytext=(base_xy3[0], base_xy3[1]),
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.4, lw=1),
                    zorder=3)

    ax.scatter(fin_xy3[:, 0], fin_xy3[:, 1],
               s=80, c="red", edgecolor="black", linewidth=0.8,
               zorder=4, label=f"μ_final (n={n_sel} trajectories, last step)")

    for i in range(n_sel):
        ax.annotate(f"{i + 1}", (fin_xy3[i, 0], fin_xy3[i, 1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="darkred", zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"Residual displacement: μ_base[{tag_ks}] → μ_final\n"
        f"({n_sel} test trajectories with last-step state = {tag_ks})", fontsize=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    out_path = out_dir / f"residual_displacement_{tag_ks}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ---- Stats ----
    mu_b_vec = mu_base_s[0].numpy()
    rows = []
    print(f"\n  {'Traj':>5}  {'user_id':>8}  "
          f"{'cos(base,final)':>16}  {'||Δμ||/||μ_base||':>18}")
    print("  " + "-" * 55)
    for i in range(n_sel):
        mf_vec  = mu_final_10[i].numpy()
        cos_sim = float(np.dot(mu_b_vec, mf_vec) /
                        (np.linalg.norm(mu_b_vec) * np.linalg.norm(mf_vec) + 1e-12))
        delta   = mf_vec - mu_b_vec
        ratio   = float(np.linalg.norm(delta) /
                        (np.linalg.norm(mu_b_vec) + 1e-12))
        print(f"  {i+1:>5}  {uid_list[i]:>8}  {cos_sim:>16.4f}  {ratio:>18.4f}")
        rows.append({"traj_idx": i + 1, "user_id": uid_list[i],
                     "cos_base_final": cos_sim, "delta_mu_ratio": ratio})

    df = pd.DataFrame(rows)
    csv_path = out_dir / "trajectory_displacement_stats.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved stats: {csv_path}")


# =========================================================
# 8. All-POI geo map colored by nearest μ_base state
#    One map per k for K>1
# =========================================================
def make_poi_nearest_state_geo(model, pid_coords, K, S, out_dir):
    print("=== POI nearest-state geo map ===")

    if not pid_coords:
        print("  WARNING: no pid_coords — skipping.")
        return

    all_loc_emb = F.normalize(
        model.loc_emb.weight[1:], p=2, dim=-1).detach().cpu()

    n_colors   = max(S, 4)
    state_cmap = cm.get_cmap("tab10" if n_colors <= 10 else "tab20", n_colors)
    state_colors = [state_cmap(s) for s in range(S)]

    for k in range(K):
        mu_states    = F.normalize(
            model.mu_base[k].detach(), p=2, dim=-1).cpu()              # [S, D]
        cos_sims     = all_loc_emb @ mu_states.T                       # [L, S]
        state_assign = cos_sims.argmax(dim=-1).numpy()                 # [L]

        lats, lons, states = [], [], []
        for pid0 in range(len(state_assign)):
            coord = pid_coords.get(pid0 + 1)
            if coord is None:
                continue
            lats.append(coord[0])
            lons.append(coord[1])
            states.append(int(state_assign[pid0]))

        if not lats:
            print(f"  k={k}: no POI coordinates found — skipped.")
            continue

        colors = [state_colors[s] for s in states]
        tag    = f"_k{k}" if K > 1 else ""

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.scatter(lons, lats, c=colors, s=18, alpha=0.75,
                   edgecolor="none", zorder=2)

        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=state_colors[s], markeredgecolor="black",
                       markersize=9, label=f"s{s}")
            for s in range(S)
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=9,
                  frameon=True, title="Nearest state")
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude",  fontsize=10)
        title = f"All POIs — colour = nearest μ_base state  (S={S})"
        if K > 1:
            title += f"  [k={k}]"
        ax.set_title(title, fontsize=11)
        plt.tight_layout()

        out_path = out_dir / f"poi_nearest_state_geo{tag}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        from collections import Counter
        cnts = Counter(states)
        print("  POI count per state (k={k}): " +
              "  ".join(f"s{s}={cnts.get(s, 0)}" for s in range(S)))


# =========================================================
# 9. Geo grid heatmap — mean P(l|state) per spatial cell
# =========================================================
def make_geo_grid_heatmap(model, pid_coords, K, S, out_dir, n_grid=16):
    """
    Divide the bounding box into n_grid×n_grid cells.
    For each state (k, s), compute P(l|k,s) = softmax(κ·⟨μ,e_l⟩) using
    mu_base and log_kappa_base (static, no context), then average over POIs
    within each cell.  Saves one PNG per (k, s) under geo_grid_heatmaps/.
    """
    print(f"=== Geo grid heatmap ({n_grid}×{n_grid}) ===")

    if not pid_coords:
        print("  WARNING: no pid_coords — skipping.")
        return

    all_loc_emb  = F.normalize(
        model.loc_emb.weight[1:], p=2, dim=-1).detach().cpu()          # [L, D]
    mu_base_norm = F.normalize(
        model.mu_base.detach(), p=2, dim=-1).cpu()                     # [K, S, D]
    lk_base      = model.log_kappa_base.detach().cpu()                 # [K, S]
    L = all_loc_emb.shape[0]

    # Build lat/lon arrays aligned with POI index (0-indexed)
    lats = np.array([pid_coords.get(p, (np.nan, np.nan))[0]
                     for p in range(1, L + 1)], dtype=np.float64)
    lons = np.array([pid_coords.get(p, (np.nan, np.nan))[1]
                     for p in range(1, L + 1)], dtype=np.float64)

    valid     = ~(np.isnan(lats) | np.isnan(lons))
    valid_idx = np.where(valid)[0]
    print(f"  Valid POIs with lat/lon: {valid_idx.size} / {L}")

    if valid_idx.size == 0:
        print("  No valid coords — skipping.")
        return

    lat_min, lat_max = lats[valid].min(), lats[valid].max()
    lon_min, lon_max = lons[valid].min(), lons[valid].max()

    # Bin valid POIs into grid cells
    lat_edges = np.linspace(lat_min, lat_max, n_grid + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_grid + 1)
    row_idx = np.clip(np.digitize(lats[valid], lat_edges) - 1, 0, n_grid - 1)
    col_idx = np.clip(np.digitize(lons[valid], lon_edges) - 1, 0, n_grid - 1)

    # Count POIs per cell (for averaging)
    cell_count = np.zeros((n_grid, n_grid), dtype=np.float64)
    np.add.at(cell_count, (row_idx, col_idx), 1.0)

    geo_out = out_dir / "geo_grid_heatmaps"
    geo_out.mkdir(exist_ok=True)

    import torch as t_local
    for k in range(K):
        for s in range(S):
            kappa_s = lk_base[k, s].exp().item()
            mu_s    = mu_base_norm[k, s]                               # [D]
            scaled  = kappa_s * (all_loc_emb @ mu_s)                  # [L]
            p_l     = t_local.softmax(scaled, dim=0).numpy()          # [L]

            p_valid = p_l[valid_idx]                                   # POIs with coords

            # Accumulate sum per cell
            cell_sum = np.zeros((n_grid, n_grid), dtype=np.float64)
            np.add.at(cell_sum, (row_idx, col_idx), p_valid)

            # Mean (NaN for empty cells)
            grid_mean = np.where(cell_count > 0, cell_sum / cell_count, np.nan)

            tag = f"k{k}_s{s}" if K > 1 else f"s{s}"

            fig, ax = plt.subplots(figsize=(7, 6))
            # flip row axis: row 0 = smallest lat → bottom of image
            im = ax.imshow(
                np.flipud(grid_mean),
                cmap="YlOrRd", aspect="auto",
                extent=[lon_min, lon_max, lat_min, lat_max],
                interpolation="nearest",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Probability", fontsize=16)
            cbar.ax.tick_params(labelsize=14)
            ax.set_xlabel("Longitude", fontsize=18)
            ax.set_ylabel("Latitude",  fontsize=18)
            ax.tick_params(labelsize=14)
            plt.tight_layout()
            out_path = geo_out / f"geo_grid_{tag}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        sfx = f"(k={k})" if K > 1 else ""
        print(f"  Saved {S} geo grid heatmaps {sfx} → {geo_out}/")


# =========================================================
# 10. Temporal heatmap — 7 weekdays × 12 two-hour slots per state
# =========================================================
@t.no_grad()
def _get_per_step_states(model, traj, device):
    """
    Forward-max per-step state assignment.
    At each step t, takes argmax of log_alpha[t] over K×S jointly.
    Returns list[(k, s)] of length T.
    """
    poi_seq = traj["poi_seq"]
    cat_seq = [c + 1 for c in traj["cat_seq"]]
    th_seq  = traj["hour_bin_seq"]
    uid_id  = traj["user_id"]
    T       = len(poi_seq)
    K_      = model.mu_base.shape[0]
    S_      = model.mu_base.shape[1]

    loc   = t.LongTensor(poi_seq).unsqueeze(0).to(device)
    cat   = t.LongTensor(cat_seq).unsqueeze(0).to(device)
    th    = t.LongTensor(th_seq).unsqueeze(0).to(device)
    uid_t = t.LongTensor([uid_id]).to(device)

    seq_in      = model._build_seq_in(loc, cat, th)
    hist_seq, _ = model.encoder2(seq_in, [T])
    log_emiss   = model.compute_emission_logprob(loc, hist_seq, uid_t)  # [1,K,T,S]

    init_prob, trans_prob = model.get_hmm_params()
    log_init  = t.log(init_prob.clamp_min(1e-12))
    log_trans = t.log(trans_prob.clamp_min(1e-12))

    log_alpha = log_init.unsqueeze(0) + log_emiss[:, :, 0, :]           # [1, K, S]
    per_step  = [log_alpha[0].reshape(-1).argmax().item()]

    for tt in range(1, T):
        log_pred  = t.logsumexp(
            log_alpha.unsqueeze(-1) + log_trans.unsqueeze(0), dim=2)
        log_alpha = log_pred + log_emiss[:, :, tt, :]
        per_step.append(log_alpha[0].reshape(-1).argmax().item())

    return [(idx // S_, idx % S_) for idx in per_step]


def make_temporal_heatmap(model, test_trajs, K, S, th_size, out_dir):
    """
    One 7×12 heatmap per (k, s) state.
    Rows = weekday (0=Mon … 6=Sun), columns = 2-hour slots (0-2 … 22-24).
    Cell value = P(weekday=d, slot=h | state assigned at this step).
    State assignments come from per-step forward-max on test trajectories.
    """
    print("=== Temporal heatmap (7 weekdays × 12 two-hour slots) ===")
    device  = next(model.parameters()).device
    N_SLOTS = 12

    counts = [[np.zeros((7, N_SLOTS), dtype=np.float64)
               for _ in range(S)] for _ in range(K)]

    n_skipped = 0
    for traj in tqdm(test_trajs, desc="  Temporal: HMM forward", leave=False):
        uid       = traj["user_id"]
        poi_seq   = traj["poi_seq"]
        weekdays  = traj.get("weekday_seq")
        hour_bins = traj.get("hour_bin_seq")

        if uid == -1 or len(poi_seq) < 2 or weekdays is None:
            n_skipped += 1
            continue

        per_step_ks = _get_per_step_states(model, traj, device)

        for step_i, (k, s) in enumerate(per_step_ks):
            wd   = int(weekdays[step_i]) % 7
            hb   = int(hour_bins[step_i])
            slot = min(int(hb * N_SLOTS / th_size), N_SLOTS - 1)
            counts[k][s][wd, slot] += 1.0

    if n_skipped:
        print(f"  Skipped {n_skipped} trajectories (missing weekday_seq or uid=-1)")

    time_out = out_dir / "temporal_heatmaps"
    time_out.mkdir(exist_ok=True)

    day_labels  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    slot_labels = [f"{h*2}-{h*2+2}" for h in range(N_SLOTS)]

    for k in range(K):
        for s in range(S):
            mat   = counts[k][s]
            total = mat.sum()
            mat   = mat / total if total > 0 else mat
            vmax  = float(mat.max()) if mat.max() > 0 else 1.0
            tag   = f"k{k}_s{s}" if K > 1 else f"s{s}"
            _draw_heatmap(mat, slot_labels, day_labels,
                          "YlOrRd", 0.0, vmax,
                          "", "Hour of day", "Day of week",
                          "{:.4f}", time_out / f"temporal_{tag}.png",
                          cbar_label="Probability", font_scale=1.6, cell_fontsize=16)

        sfx = f"(k={k})" if K > 1 else ""
        print(f"  Saved {S} temporal heatmaps {sfx} → {time_out}/")


# =========================================================
# 11. Main
# =========================================================
def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "analysis_hybrid_vmf"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    device = t.device(args.device)
    print(f"Device: {device}")

    model, sizes, K, S, emb_dim = load_model(str(ckpt_path), device)
    loc_size = sizes["loc_size"]

    print(f"Loading pkl: {args.pkl_path}")
    pkl_data   = load_pickle(args.pkl_path)
    test_trajs = pkl_data["test_trajectories"]
    print(f"  test trajectories: {len(test_trajs)}")

    print(f"Building macro mapping from: {args.macro_csv_path}")
    pid_to_macro, cluster_names, macro_to_color, n_macro = build_macro_mapping(
        pkl_data, args.macro_csv_path, loc_size)
    print(f"  Macro categories ({n_macro}): {cluster_names}")

    pid_to_cat_idx, cat_names, cat_to_color = build_cat_mapping(
        pkl_data, args.macro_csv_path, loc_size)
    print(f"  Fine-grained categories: {len(cat_names)}")

    # Product 1: heatmaps (all K)
    p_macro_dict = make_state_macro_heatmap(
        model, pid_to_macro, cluster_names, n_macro, K, S, out_dir)
    make_state_macro_excess_heatmap(
        p_macro_dict, pid_to_macro, cluster_names, n_macro, K, S, out_dir)
    make_state_macro_contrast_heatmap(
        p_macro_dict, cluster_names, n_macro, K, S, out_dir)
    make_state_cat_barcharts(
        model, pid_to_cat_idx, cat_names, cat_to_color, K, S, args.cat_top_n, out_dir)

    # Product 2: geo maps (all K)
    pid_coords = build_pid_coords(pkl_data)
    print(f"  pid_coords: {len(pid_coords)} POIs with lat/lon")
    make_state_geo_maps(model, pid_coords, pid_to_macro,
                        cluster_names, macro_to_color, K, S, args.geo_top_n, out_dir)
    make_poi_nearest_state_geo(model, pid_coords, K, S, out_dir)
    make_geo_grid_heatmap(model, pid_coords, K, S, out_dir, n_grid=args.n_grid)
    make_temporal_heatmap(model, test_trajs, K, S, sizes["th_size"], out_dir)

    # Transition matrix (all K)
    make_transition_heatmap(model, K, S, out_dir)

    # Product 3: residual displacement
    make_residual_displacement(model, test_trajs, pid_to_macro,
                               cluster_names, macro_to_color, K, S, args, out_dir)

    print(f"\nAll done.  Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
