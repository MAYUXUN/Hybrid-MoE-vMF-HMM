# -*- coding: utf-8 -*-
"""
mixture_hybrid_vmf_hmm.py
=========================
Mixture-of-Experts Hybrid vMF-HMM for interpretable next-POI prediction.

This script implements a hybrid Hidden Markov Model in which the emission
probability is defined over LINE-based POI embeddings using a discrete
von Mises--Fisher-style formulation. The base emission parameters provide
interpretable latent mobility states, while a trajectory-history encoder
generates residual emission parameters to improve prediction accuracy.

Architecture
------------
Frozen LINE embeddings
    user_emb [U+1, D]  loc_emb [L+1, D]  cat_emb [C+1, D]  time_emb [12+1, D]
    loaded from line_ckpt_dir/embeddings.pt or individual *.npy files.
    These embeddings are fixed during vMF-HMM training.

User-specific mixture gate
    user_gate_logits : nn.Embedding(U+1, K), zero-initialized, padding_idx=0
    pi_logits = user_gate_logits[uid]                    -> [B, K]
    pi_{u,k}  = softmax(pi_logits / pi_temperature)
    The gate assigns each user to a mixture of expert HMMs.

Encoder2History
    input  : concat(loc_emb[l], cat_emb[c], time_emb[th]) at each step
             -> [B, T, 3D]
    LSTM   : 3D -> hist_hidden
    output : hist_seq [B, T, H], final_h [B, H]
    The previous hidden output h_{t-1} is used as sequential context.

HybridEncoder
    z = concat(user_emb[uid], h_{t-1})                    -> [B, T, D+H]
    MLP_mu    : D+H -> 256 -> K*S*D                       -> delta_mu [B, T, K, S, D]
    MLP_kappa : D+H -> 64  -> K*S                         -> delta_log_kappa [B, T, K, S]
    These residuals make the emission distribution history-dependent.

Base emission parameters
    mu_base          [K, S, D]   initialized from K-means centroids of POI embeddings
    log_kappa_base   [K, S]      initialized to log(30)
    Each base direction represents an interpretable latent mobility state.

Angular-bounded residual update
    mu_final = normalize(mu_base + delta_mu), with the angular deviation from
    mu_base clipped by max_angle.
    This constraint allows history-dependent adaptation while preserving the
    interpretability of the base latent states.

Discrete vMF emission
    log P(l | s, k, h, u)
        = kappa * <mu_final, loc_emb_norm[l]>
          - logsumexp_{l'} kappa * <mu_final, loc_emb_norm[l']>

    The emission probability is normalized over the finite POI set rather than
    over the continuous unit sphere.

HMM forward
    Log-space forward algorithm for each expert HMM, followed by user-specific
    mixture aggregation over experts.

Loss
    Negative log-likelihood plus state diversity regularization.

Metrics
    Recall@1/5/10 and NDCG@5/10.

Usage
-----
python mixture_hybrid_vmf_hmm.py \\
    --pkl_path data/NYC_getnext_ready.pkl \\
    --line_ckpt_dir embeddings/multimodal_line_runs \\
    --save_dir outputs/mhvmf_K4_S4_NYC \\
    --num_classes 4 \\
    --num_states 4 \\
    --max_angle 0.15 \\
    --beta_div 10 \\
    --epochs 100
"""
import os
import math
import time
import json
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# =========================================================
# 0. Seed
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


# =========================================================
# 1. IO helpers
# =========================================================
def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"pkl not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# 2. PKL helpers
# =========================================================
def infer_sizes_from_pkl(data_obj):
    stats    = data_obj["stats"]
    loc_size = stats["num_pois"]
    cat_size = stats["num_cats"]
    return dict(loc_size=loc_size, cat_size=cat_size, tw_size=7, th_size=12)


def build_train_sequential_histories(train_trajectories, max_hist_trajs=20):
    user_trajs = defaultdict(list)
    result = []
    for traj in train_trajectories:
        uid = traj["user_id"]
        if uid == -1:
            result.append(None)
            continue
        result.append(list(user_trajs[uid][-max_hist_trajs:]))
        user_trajs[uid].append({
            "loc": traj["poi_seq"],
            "cat": [c + 1 for c in traj["cat_seq"]],
            "tw":  traj["weekday_seq"],
            "th":  traj["hour_bin_seq"],
        })
    return result


def build_full_train_histories(train_trajectories, max_hist_trajs=20):
    user_trajs = defaultdict(list)
    for traj in train_trajectories:
        uid = traj["user_id"]
        if uid == -1:
            continue
        user_trajs[uid].append({
            "loc": traj["poi_seq"],
            "cat": [c + 1 for c in traj["cat_seq"]],
            "tw":  traj["weekday_seq"],
            "th":  traj["hour_bin_seq"],
        })
    return {uid: trajs[-max_hist_trajs:] for uid, trajs in user_trajs.items()}


# =========================================================
# 3. Dataset / Collate
# =========================================================
class TrajectoryDataset(t.utils.data.Dataset):
    def __init__(self, trajectories, hist_list):
        self.items = []
        for traj, hist_trajs in zip(trajectories, hist_list):
            uid = traj["user_id"]
            if uid == -1 or hist_trajs is None or len(hist_trajs) == 0:
                continue
            poi_seq = traj["poi_seq"]
            if len(poi_seq) < 2:
                continue
            self.items.append({
                "uid":       uid,
                "loc_seq":   t.LongTensor(poi_seq),
                "cat_seq":   t.LongTensor([c + 1 for c in traj["cat_seq"]]),
                "tw_seq":    t.LongTensor(traj["weekday_seq"]),
                "th_seq":    t.LongTensor(traj["hour_bin_seq"]),
                "oov_mask":  traj.get("oov_mask", [False] * len(poi_seq)),
                "hist_trajs": [
                    {"loc": t.LongTensor(h["loc"]),
                     "cat": t.LongTensor(h["cat"]),
                     "tw":  t.LongTensor(h["tw"]),
                     "th":  t.LongTensor(h["th"])}
                    for h in hist_trajs
                ],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_traj_batch(batch):
    batch = sorted(batch, key=lambda x: len(x["loc_seq"]), reverse=True)
    lengths = [len(x["loc_seq"]) for x in batch]

    loc = pad_sequence([x["loc_seq"] for x in batch], batch_first=True, padding_value=0)
    cat = pad_sequence([x["cat_seq"] for x in batch], batch_first=True, padding_value=0)
    tw  = pad_sequence([x["tw_seq"]  for x in batch], batch_first=True, padding_value=0)
    th  = pad_sequence([x["th_seq"]  for x in batch], batch_first=True, padding_value=0)
    uid      = t.LongTensor([x["uid"] for x in batch])
    oov_mask = [x["oov_mask"] for x in batch]

    flat_loc, flat_cat, flat_tw, flat_th = [], [], [], []
    flat_len, sample_idx = [], []
    hist_n_trajs = []
    for b, item in enumerate(batch):
        hist_n_trajs.append(len(item["hist_trajs"]))
        for ht in item["hist_trajs"]:
            flat_loc.append(ht["loc"])
            flat_cat.append(ht["cat"])
            flat_tw.append(ht["tw"])
            flat_th.append(ht["th"])
            flat_len.append(len(ht["loc"]))
            sample_idx.append(b)

    order = sorted(range(len(flat_len)), key=lambda i: flat_len[i], reverse=True)
    hist_flat_loc = pad_sequence([flat_loc[i] for i in order], batch_first=True, padding_value=0)
    hist_flat_cat = pad_sequence([flat_cat[i] for i in order], batch_first=True, padding_value=0)
    hist_flat_tw  = pad_sequence([flat_tw[i]  for i in order], batch_first=True, padding_value=0)
    hist_flat_th  = pad_sequence([flat_th[i]  for i in order], batch_first=True, padding_value=0)
    hist_flat_lengths = [flat_len[i]    for i in order]
    hist_sample_idx   = t.LongTensor([sample_idx[i] for i in order])

    return dict(uid=uid, loc=loc, cat=cat, tw=tw, th=th,
                lengths=lengths, oov_mask=oov_mask,
                hist_flat_loc=hist_flat_loc, hist_flat_cat=hist_flat_cat,
                hist_flat_tw=hist_flat_tw,   hist_flat_th=hist_flat_th,
                hist_flat_lengths=hist_flat_lengths,
                hist_sample_idx=hist_sample_idx,
                hist_n_trajs=hist_n_trajs)


def move_batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, t.Tensor) else v
            for k, v in batch.items()}


# =========================================================
# 4. Metrics
# =========================================================
TOPK_LIST = (1, 5, 10)


def init_metrics():
    return {k: {"recall_sum": 0.0, "ndcg_sum": 0.0, "total": 0} for k in TOPK_LIST}


def finalize_metrics(metrics, prefix):
    out = {}
    for k in TOPK_LIST:
        total = metrics[k]["total"]
        out[f"{prefix}_recall@{k}"] = metrics[k]["recall_sum"] / total if total > 0 else 0.0
        out[f"{prefix}_ndcg@{k}"]   = metrics[k]["ndcg_sum"]   / total if total > 0 else 0.0
    return out


def update_metrics_gpu(metrics, logits, targets):
    maxk = max(TOPK_LIST)
    _, topk_idx = t.topk(logits, k=maxk, dim=-1)
    y    = (targets - 1).unsqueeze(1)
    hits = (topk_idx == y)

    any_hit   = hits.any(dim=1)
    first_hit = hits.float().argmax(dim=1) + 1
    rank      = t.where(any_hit, first_hit, t.full_like(first_hit, maxk + 1))
    ndcg_full = math.log(2.0) / t.log(rank.float() + 1.0)

    N = targets.size(0)
    for k in TOPK_LIST:
        hit_k = (rank <= k)
        metrics[k]["recall_sum"] += hit_k.float().sum().item()
        metrics[k]["ndcg_sum"]   += (hit_k.float() * ndcg_full).sum().item()
        metrics[k]["total"]      += N


# =========================================================
# 5. (discrete vMF — no analytic norm constant needed)
# =========================================================
# Emission is normalised by logsumexp over the L POIs, not the continuous
# sphere integral.  log P(ℓ|s,k,h,u) = κ·<μ,e_ℓ> - logsumexp_{ℓ'}(κ·<μ,e_ℓ'>)


# =========================================================
# 6. (User-specific gate is now an nn.Embedding inside the core model;
#     Encoder1Pi has been removed.)
# =========================================================


# =========================================================
# 7. Encoder2History  (sequential context LSTM)
# =========================================================
class Encoder2History(nn.Module):
    """LSTM that reads frozen LINE (loc, cat, time) embeddings to build h_t."""

    def __init__(self, in_dim: int, hist_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=in_dim, hidden_size=hist_hidden,
                               num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_in: t.Tensor, lengths) -> tuple:
        """
        seq_in  [B, T, in_dim]  (already built from frozen embeddings)
        lengths  list[int]
        returns: hist_seq [B, T, H],  final_h [B, H]
        """
        packed           = pack_padded_sequence(self.dropout(seq_in), lengths,
                                                batch_first=True, enforce_sorted=True)
        output, (h_n, _) = self.lstm(packed)
        output_pad, _    = pad_packed_sequence(output, batch_first=True)
        return self.dropout(output_pad), self.dropout(h_n[-1])


# =========================================================
# 8. HybridEncoder  (residual emission parameters)
# =========================================================
class HybridEncoder(nn.Module):
    """
    Given z = concat(user_emb[uid], h_{t-1}) → delta_mu and delta_log_kappa.
    Operates over all (b, t) at once via matmul on [B, T, z_dim].
    """

    def __init__(self, z_dim: int, K: int, S: int, emb_dim: int, dropout: float = 0.2):
        super().__init__()
        self.K = K
        self.S = S
        self.D = emb_dim

        KSD = K * S * emb_dim
        KS  = K * S

        self.mlp_mu = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, KSD),
        )
        self.mlp_kappa = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, KS),
        )
        # Small init so residuals start near zero
        nn.init.zeros_(self.mlp_mu[-1].weight)
        nn.init.zeros_(self.mlp_mu[-1].bias)
        nn.init.zeros_(self.mlp_kappa[-1].weight)
        nn.init.zeros_(self.mlp_kappa[-1].bias)

    def forward(self, z: t.Tensor):
        """
        z : [B, T, z_dim]  OR  [B, z_dim]
        returns delta_mu [..., K, S, D], delta_log_kappa [..., K, S]
        """
        shape   = z.shape[:-1]
        delta_mu       = self.mlp_mu(z).view(*shape, self.K, self.S, self.D)
        delta_log_kappa = self.mlp_kappa(z).view(*shape, self.K, self.S)
        return delta_mu, delta_log_kappa


# =========================================================
# 9. Core model — MixtureHybridVMFHMM
# =========================================================
class MixtureHybridVMFHMM(nn.Module):

    def __init__(self, loc_size, cat_size, th_size,
                 user_size, emb_dim=128,
                 num_classes=4, num_states=16,
                 seq_hidden=128, hist_hidden=128,
                 dropout=0.2,
                 max_angle=0.15,
                 log_kappa_max: float = 5.0,
                 log_kappa_min: float = None,
                 line_user_emb_np=None,
                 line_loc_emb_np=None,
                 line_cat_emb_np=None,
                 line_time_emb_np=None):
        super().__init__()
        self.K = num_classes
        self.S = num_states
        self.D = emb_dim
        self.loc_size = loc_size
        self.max_angle = max_angle
        self.log_kappa_max = log_kappa_max
        self.log_kappa_min = math.log(20.0) if log_kappa_min is None else log_kappa_min

        # -- Frozen LINE embeddings ------------------------------------------------
        def _make_frozen(np_arr):
            w = t.from_numpy(np_arr).float()
            emb = nn.Embedding(w.shape[0], w.shape[1], padding_idx=0)
            emb.weight.data.copy_(w)
            emb.weight.requires_grad = False
            return emb

        self.user_emb = _make_frozen(line_user_emb_np)   # [U+1, D]
        self.loc_emb  = _make_frozen(line_loc_emb_np)    # [L+1, D]
        self.cat_emb  = _make_frozen(line_cat_emb_np)    # [C+1, D]
        self.time_emb = _make_frozen(line_time_emb_np)   # [12+1, D]

        # -- Sub-modules ----------------------------------------------------------
        # User-specific learnable gate logits α_{u,k} → π_{u,k} = softmax(α_{u,k}/T).
        # Replaces Encoder1Pi: each user has its own raw K-dim logits.
        self.user_gate_logits = nn.Embedding(user_size + 1, num_classes, padding_idx=0)
        nn.init.zeros_(self.user_gate_logits.weight)         # uniform prior at init
        self.user_gate_logits.weight.requires_grad = True    # explicit: trainable

        enc2_in = 3 * emb_dim                            # loc + cat + time
        self.encoder2 = Encoder2History(enc2_in, hist_hidden, dropout)

        z_dim = emb_dim + hist_hidden                    # user_emb + h_t
        self.hybrid_enc = HybridEncoder(z_dim, num_classes, num_states, emb_dim, dropout)

        # -- Base vMF parameters --------------------------------------------------
        mu_init = self._kmeans_init_mu(
            line_loc_emb_np, num_classes, num_states, emb_dim)
        self.mu_base       = nn.Parameter(mu_init)

        log_k_init = t.full((num_classes, num_states), math.log(30.0))
        self.log_kappa_base = nn.Parameter(log_k_init)

        # -- HMM parameters -------------------------------------------------------
        self.init_logits  = nn.Parameter(self._orth_init(num_classes, num_states, scale=3.0))
        self.trans_logits = nn.Parameter(self._orth_trans(num_classes, num_states, scale=2.0))

        # Gating temperature for π sharpening (T<1 makes routing more decisive)
        self.pi_temperature = 0.1

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _kmeans_init_mu(loc_emb_np, K, S, D):
        """
        Run MiniBatchKMeans with K*S clusters on L2-normalised POI embeddings.
        Returns unit-normalised centroids reshaped to [K, S, D].
        Falls back to random init if sklearn is unavailable or clustering fails.
        """
        n_clusters = K * S
        try:
            from sklearn.cluster import MiniBatchKMeans
            arr = loc_emb_np[1:].astype(np.float32)           # drop padding row
            norms = np.linalg.norm(arr, axis=-1, keepdims=True).clip(1e-8)
            arr = arr / norms                                   # L2-normalise
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                n_init=5,
                max_iter=200,
                random_state=42,
                batch_size=min(4096, arr.shape[0]),
                verbose=0,
            )
            km.fit(arr)
            centroids = km.cluster_centers_.astype(np.float32) # [K*S, D]
            c_norms = np.linalg.norm(centroids, axis=-1, keepdims=True).clip(1e-8)
            centroids = centroids / c_norms
            mu_init = t.from_numpy(centroids).view(K, S, D)
            print(f"  [mu_base init] K-means  n_clusters={n_clusters}  "
                  f"inertia={km.inertia_:.2f}")
        except Exception as e:
            print(f"  [mu_base init] K-means failed ({e}), falling back to random")
            mu_init = F.normalize(t.randn(K, S, D), dim=-1)
        return mu_init

    @staticmethod
    def _orth_init(K, S, scale=3.0):
        dim = max(K, S)
        tmp = t.empty(dim, dim)
        nn.init.orthogonal_(tmp)
        return tmp[:K, :S] * scale

    @staticmethod
    def _orth_trans(K, S, scale=2.0):
        SS  = S * S
        dim = max(K, SS)
        tmp = t.empty(dim, dim)
        nn.init.orthogonal_(tmp)
        return tmp[:K, :SS].reshape(K, S, S) * scale

    # ------------------------------------------------------------------
    # HMM parameters (π and A)
    # ------------------------------------------------------------------
    def get_hmm_params(self):
        return (F.softmax(self.init_logits,  dim=-1),
                F.softmax(self.trans_logits, dim=-1))

    # ------------------------------------------------------------------
    # Angular-bounded residual update
    # ------------------------------------------------------------------
    def bounded_angular_update(self, mu_base, delta_mu, max_angle=None):
        """
        Compute mu_final = normalize(mu_base + delta_mu), but clip the
        angular deviation to max_angle (radians) when it would exceed it.

        mu_base  : [K, S, D]
        delta_mu : [B, K, S, D]  or  [B, T, K, S, D]
        returns  : same leading shape as delta_mu, unit vectors
        """
        if max_angle is None:
            max_angle = self.max_angle

        mu_b = F.normalize(mu_base, p=2, dim=-1)
        while mu_b.dim() < delta_mu.dim():
            mu_b = mu_b.unsqueeze(0)                    # broadcast to delta_mu dims

        mu_raw  = F.normalize(mu_b + delta_mu, p=2, dim=-1)

        cos_sim = (mu_raw * mu_b).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle   = t.acos(cos_sim)                       # [..., K, S, 1]

        max_angle_t = t.as_tensor(max_angle, device=delta_mu.device, dtype=delta_mu.dtype)

        # Tangential unit vector from mu_base toward mu_raw
        u      = mu_raw - cos_sim * mu_b
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u      = u / u_norm

        # Version clamped to exactly max_angle
        mu_clip = t.cos(max_angle_t) * mu_b + t.sin(max_angle_t) * u
        mu_clip = F.normalize(mu_clip, p=2, dim=-1)

        mu_final = t.where(angle <= max_angle_t, mu_raw, mu_clip)
        return F.normalize(mu_final, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Build Encoder2 input tensor from current batch
    # ------------------------------------------------------------------
    def _build_seq_in(self, loc, cat, th):
        """
        loc, cat, th : [B, T]  long tensors
        returns [B, T, 3D]  float
        """
        l_emb = self.loc_emb(loc)
        c_emb = self.cat_emb(cat)
        th_clamped = th.clamp(0, self.time_emb.num_embeddings - 1)
        t_emb = self.time_emb(th_clamped)
        return t.cat([l_emb, c_emb, t_emb], dim=-1)

    # ------------------------------------------------------------------
    # Compute discrete vMF emission log-probs at OBSERVED locations  [B, K, T, S]
    # ------------------------------------------------------------------
    def compute_emission_logprob(self, loc, hist_seq, uid):
        """
        Discrete vMF: log P(ℓ|s,k,h,u) = κ·<μ,e_ℓ> - logsumexp_ℓ'(κ·<μ,e_ℓ'>)

        Iterates over T steps.  In training mode each step is gradient-
        checkpointed: only [B, D+H] inputs are retained during forward;
        backward re-runs the step, peaking at K×S×[B,L] ~ 320 MB for one
        step at a time (vs K×S×T×[B,L] ≈ 16 GB without checkpointing).

        loc      : [B, T]  observed POI indices (1-indexed, 0=pad)
        hist_seq : [B, T, H]  LSTM output
        uid      : [B]  user ids
        returns  : log_emiss [B, K, T, S]
        """
        from torch.utils.checkpoint import checkpoint as _ckpt
        B, T = loc.shape
        K, S = self.K, self.S

        zero_h = t.zeros(B, 1, hist_seq.shape[-1], device=hist_seq.device)
        h_prev = t.cat([zero_h, hist_seq[:, :-1, :]], dim=1)           # [B, T, H]

        u_idx       = uid.clamp(0, self.user_emb.num_embeddings - 1)
        u_vec       = self.user_emb(u_idx)                             # [B, D]
        loc_emb_all = F.normalize(self.loc_emb.weight[1:], p=2, dim=-1)  # [L, D]
        loc_0       = (loc - 1).clamp(min=0)                           # [B, T]

        mu_base    = self.mu_base
        lk_base    = self.log_kappa_base
        lk_max     = self.log_kappa_max
        hybrid_enc = self.hybrid_enc
        _bau       = self.bounded_angular_update

        def _step(u_v, h_t, l0_t, mu_b, lk_b):
            # u_v [B,D]  h_t [B,H]  l0_t [B] 0-indexed  mu_b [K,S,D]  lk_b [K,S]
            # Returns log_e [B, K, S]  — only [B,L] per (k,s) allocated at a time
            z_t = t.cat([u_v, h_t], dim=-1)
            delta_mu, delta_lk = hybrid_enc(z_t)
            mu_t    = _bau(mu_b, delta_mu)                                    # [B,K,S,D]
            kappa_t = (lk_b.unsqueeze(0) + delta_lk).clamp(self.log_kappa_min, lk_max).exp()  # [B,K,S]

            rows = []
            for k in range(K):
                cols = []
                for s in range(S):
                    mu_ks    = mu_t[:, k, s, :]                               # [B, D]
                    kappa_ks = kappa_t[:, k, s]                               # [B]
                    sc = kappa_ks.unsqueeze(-1) * (mu_ks @ loc_emb_all.T)    # [B, L]
                    log_Z = t.logsumexp(sc, dim=-1)                           # [B]
                    tgt   = sc.gather(1, l0_t.view(B, 1)).squeeze(1)         # [B]
                    cols.append(tgt - log_Z)
                rows.append(t.stack(cols, dim=1))                             # [B, S]
            return t.stack(rows, dim=1)                                       # [B, K, S]

        steps = []
        for tt in range(T):
            h_t  = h_prev[:, tt, :].contiguous()
            l0_t = loc_0[:, tt]
            if self.training:
                log_e = _ckpt(_step, u_vec, h_t, l0_t, mu_base, lk_base,
                               use_reentrant=False)
            else:
                log_e = _step(u_vec, h_t, l0_t, mu_base, lk_base)
            steps.append(log_e)                                               # [B, K, S]

        log_emiss = t.stack(steps, dim=2)                                     # [B, K, T, S]

        # Mask padded positions
        valid = (loc > 0).unsqueeze(1).unsqueeze(-1)                          # [B, 1, T, 1]
        return t.where(valid, log_emiss, t.zeros_like(log_emiss))

    # ------------------------------------------------------------------
    # HMM forward  →  log_marginal [B, K],  next_state [B, K, S]
    # ------------------------------------------------------------------
    def hmm_forward(self, loc, hist_seq, uid, lengths):
        B, T  = loc.shape
        K, S  = self.K, self.S

        init_prob, trans_prob = self.get_hmm_params()
        log_emiss = self.compute_emission_logprob(loc, hist_seq, uid)   # [B, K, T, S]

        log_init  = t.log(init_prob.clamp_min(1e-12))                   # [K, S]
        log_trans = t.log(trans_prob.clamp_min(1e-12))                  # [K, S, S]

        lengths_t = t.tensor(lengths, device=loc.device)
        valid_bt  = t.arange(T, device=loc.device).unsqueeze(0) < lengths_t.unsqueeze(1)

        log_alpha = log_init.unsqueeze(0) + log_emiss[:, :, 0, :]       # [B, K, S]

        for tt in range(1, T):
            log_predict   = t.logsumexp(
                log_alpha.unsqueeze(-1) + log_trans.unsqueeze(0), dim=2)  # [B, K, S]
            log_alpha_new = log_predict + log_emiss[:, :, tt, :]
            mask_t        = valid_bt[:, tt].view(B, 1, 1)
            log_alpha     = t.where(mask_t, log_alpha_new, log_alpha)

        log_marginal_k = t.logsumexp(log_alpha, dim=-1)                  # [B, K]

        # Next-step state distribution (used in step-predict for eval)
        alpha_T     = t.exp(log_alpha - log_marginal_k.unsqueeze(-1))    # [B, K, S]
        next_state  = t.einsum("bks,kst->bkt", alpha_T, trans_prob)
        next_state  = next_state / next_state.sum(-1, keepdim=True).clamp_min(1e-12)

        return log_marginal_k, next_state

    # ------------------------------------------------------------------
    # Training objective  (NLL + state diversity penalty)
    # ------------------------------------------------------------------
    def compute_loglik(self, batch, beta_div=1.0):
        uid      = batch["uid"]
        loc, th  = batch["loc"], batch["th"]
        lengths  = batch["lengths"]
        B, T     = loc.shape

        # User-specific gate: π_{u,k} = softmax(α_{u,k} / T)
        u_vec     = self.user_emb(uid.clamp(0, self.user_emb.num_embeddings - 1))
        pi_logits = self.user_gate_logits(
            uid.clamp(0, self.user_gate_logits.num_embeddings - 1))       # [B, K]
        log_pi    = F.log_softmax(pi_logits / self.pi_temperature, dim=-1)  # [B, K]

        # Encoder2: sequential LSTM
        seq_in           = self._build_seq_in(loc, batch["cat"], th)     # [B, T, 3D]
        hist_seq, _      = self.encoder2(seq_in, lengths)

        log_marginal_k, _ = self.hmm_forward(loc, hist_seq, uid, lengths)

        log_p = t.logsumexp(log_pi + log_marginal_k, dim=-1)            # [B]

        mu_n = F.normalize(self.mu_base, p=2, dim=-1)                   # [K, S, D]
        sim  = t.einsum("ksd,kpd->ksp", mu_n, mu_n)                     # [K, S, S]
        eye  = t.eye(self.S, device=sim.device).unsqueeze(0)
        diversity_loss = ((sim * (1 - eye)) ** 2).sum() \
            / (self.K * self.S * max(self.S - 1, 1))

        return -log_p.mean() + beta_div * diversity_loss

    # ------------------------------------------------------------------
    # Step-predict for evaluation  →  list of (loc_prob [B, L], valid [B])
    # ------------------------------------------------------------------
    @t.no_grad()
    def forward_step_predict(self, batch):
        uid      = batch["uid"]
        loc, th  = batch["loc"], batch["th"]
        lengths  = batch["lengths"]
        B, T     = loc.shape
        K, S     = self.K, self.S

        u_idx = uid.clamp(0, self.user_emb.num_embeddings - 1)
        u_vec = self.user_emb(u_idx)                                     # [B, D]

        # User-specific gate: π_{u,k} = softmax(α_{u,k} / T)
        pi_logits = self.user_gate_logits(
            uid.clamp(0, self.user_gate_logits.num_embeddings - 1))      # [B, K]
        log_pi    = F.log_softmax(pi_logits / self.pi_temperature, dim=-1)  # [B, K]

        seq_in           = self._build_seq_in(loc, batch["cat"], th)
        hist_seq, _      = self.encoder2(seq_in, lengths)

        # h_prev for per-step hybrid encoder calls (no [B,T,K,S,D] prealloc)
        zero_h = t.zeros(B, 1, hist_seq.shape[-1], device=hist_seq.device)
        h_prev = t.cat([zero_h, hist_seq[:, :-1, :]], dim=1)            # [B, T, H]

        loc_emb_all = F.normalize(self.loc_emb.weight[1:], p=2, dim=-1)  # [L, D]
        L = loc_emb_all.shape[0]

        init_prob, trans_prob = self.get_hmm_params()
        log_init  = t.log(init_prob.clamp_min(1e-12))
        log_trans = t.log(trans_prob.clamp_min(1e-12))

        log_emiss_full = self.compute_emission_logprob(loc, hist_seq, uid)  # [B,K,T,S]

        log_alpha = log_init.unsqueeze(0) + log_emiss_full[:, :, 0, :]   # [B, K, S]

        lengths_t = t.tensor(lengths, device=loc.device)
        step_preds = []

        for tt in range(T):
            log_marg_t = t.logsumexp(log_alpha, dim=-1)                  # [B, K]
            post_k     = F.softmax(log_pi + log_marg_t, dim=-1)          # [B, K]

            alpha_norm = t.exp(log_alpha - log_marg_t.unsqueeze(-1))     # [B, K, S]
            next_state = t.einsum("bks,kst->bkt", alpha_norm, trans_prob)
            next_state = next_state / next_state.sum(-1, keepdim=True).clamp_min(1e-12)

            if tt < T - 1:
                # Compute emission params for this step only — no [B,T,K,S,D] prealloc
                z_t        = t.cat([u_vec, h_prev[:, tt, :]], dim=-1)    # [B, z_dim]
                delta_mu, delta_lk = self.hybrid_enc(z_t)
                mu_t    = self.bounded_angular_update(self.mu_base, delta_mu)  # [B, K, S, D]
                kappa_t = (self.log_kappa_base.unsqueeze(0) + delta_lk
                           ).clamp(self.log_kappa_min, self.log_kappa_max).exp()  # [B, K, S]

                # Accumulate P(l|k) = Σ_s p(s|k) P(l|k,s) directly into [B,K,L]
                # Avoids allocating [B,K,S,L] = 328 MB; each (k,s) uses [B,L] = 5 MB
                log_next_s = t.log(next_state.clamp_min(1e-12))          # [B, K, S]
                log_p_l_k  = t.full((B, K, L), float('-inf'), device=loc.device)
                for ki in range(K):
                    for si in range(S):
                        mu_ks    = mu_t[:, ki, si, :]                     # [B, D]
                        kappa_ks = kappa_t[:, ki, si]                     # [B]
                        sc = kappa_ks.unsqueeze(-1) * (mu_ks @ loc_emb_all.T)  # [B, L]
                        log_e_ks = sc - t.logsumexp(sc, dim=-1, keepdim=True)  # [B, L]
                        w_ks     = log_next_s[:, ki, si].unsqueeze(-1)   # [B, 1]
                        log_p_l_k[:, ki, :] = t.logaddexp(
                            log_p_l_k[:, ki, :], w_ks + log_e_ks)

                # Mix over K
                log_pk  = t.log(post_k.clamp_min(1e-12)).unsqueeze(-1)   # [B, K, 1]
                log_p_l = t.logsumexp(log_pk + log_p_l_k, dim=1)         # [B, L]

                valid = lengths_t > (tt + 1)
                step_preds.append((log_p_l.exp(), valid))

                # Advance alpha
                log_predict   = t.logsumexp(
                    log_alpha.unsqueeze(-1) + log_trans.unsqueeze(0), dim=2)
                log_alpha_new = log_predict + log_emiss_full[:, :, tt + 1, :]
                mask_t        = (lengths_t > (tt + 1)).view(B, 1, 1)
                log_alpha     = t.where(mask_t, log_alpha_new, log_alpha)

        return step_preds


# =========================================================
# 10. Training epoch
# =========================================================
def run_train_epoch(model, loader, optimizer, device, beta_div=1.0):
    model.train()
    sum_loss = 0.0
    total_n  = 0
    last_batch = None

    for batch in tqdm(loader, desc="  [Train]", leave=False):
        batch_dev = move_batch_to_device(batch, device)

        loss = model.compute_loglik(batch_dev, beta_div=beta_div)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs        = len(batch["lengths"])
        sum_loss += loss.item() * bs
        total_n  += bs
        last_batch = batch_dev

    # -- Diagnostics from last batch ------------------------------------------
    with t.no_grad():
        uid     = last_batch["uid"]
        loc     = last_batch["loc"]
        th      = last_batch["th"]
        lengths = last_batch["lengths"]

        u_idx   = uid.clamp(0, model.user_emb.num_embeddings - 1)
        u_vec   = model.user_emb(u_idx)
        gate_idx  = uid.clamp(0, model.user_gate_logits.num_embeddings - 1)
        pi_logits = model.user_gate_logits(gate_idx)
        pi_prob   = F.softmax(pi_logits / model.pi_temperature, dim=-1).cpu()
        print(f"  [π] " + "  ".join(f"k{k+1}={pi_prob[:,k].mean():.3f}"
                                     for k in range(model.K)))

        # Per-sample sharpness diagnostics (batch-mean π hides sharp per-sample routing)
        pi_max_per_sample = pi_prob.max(dim=-1).values                    # [B]
        pi_entropy = -(pi_prob * pi_prob.clamp_min(1e-12).log()).sum(-1)  # [B]
        uniform_entropy = math.log(model.K)
        print(f"  [π sharp] max_mean={pi_max_per_sample.mean():.3f}  "
              f"max_std={pi_max_per_sample.std():.3f}  "
              f"entropy={pi_entropy.mean():.3f} (uniform={uniform_entropy:.3f}, "
              f"T={model.pi_temperature})")

        seq_in   = model._build_seq_in(loc, last_batch["cat"], th)
        hist_seq, _ = model.encoder2(seq_in, lengths)
        B2, T2   = loc.shape

        zero_h = t.zeros(B2, 1, hist_seq.shape[-1], device=device)
        h_prev = t.cat([zero_h, hist_seq[:, :-1, :]], dim=1)
        u_exp  = u_vec.unsqueeze(1).expand(-1, T2, -1)
        z_all  = t.cat([u_exp, h_prev], dim=-1)
        delta_mu, delta_lk = model.hybrid_enc(z_all)

        base_norm_ks = model.mu_base.norm(dim=-1)                        # [K, S]
        delta_norm   = delta_mu.norm(dim=-1)                             # [B2, T2, K, S]
        ratio_per    = delta_norm / (base_norm_ks.unsqueeze(0).unsqueeze(0) + 1e-8)
        ratio_std    = ratio_per.std().item()
        print(f"  [Δμ ratio] mean={ratio_per.mean():.4f}  std={ratio_std:.4f}")

        mu_n_d   = F.normalize(model.mu_base, p=2, dim=-1)
        sim_d    = t.einsum("ksd,kpd->ksp", mu_n_d, mu_n_d)
        eye_d    = t.eye(model.S, device=sim_d.device).unsqueeze(0)
        div_loss = ((sim_d * (1 - eye_d)) ** 2).sum() / (model.K * model.S * max(model.S - 1, 1))
        print(f"  [reg] diversity_loss={div_loss:.4f}  beta_div={beta_div}")

        # True angular deviation after bounded_angular_update
        mu_final_diag = model.bounded_angular_update(model.mu_base, delta_mu)  # [B2,T2,K,S,D]
        mu_base_n     = F.normalize(model.mu_base, p=2, dim=-1).unsqueeze(0).unsqueeze(0)
        cos_diag      = (mu_final_diag * mu_base_n).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle_diag    = t.acos(cos_diag)
        a_mean = angle_diag.mean().item()
        a_max  = angle_diag.max().item()
        print(f"  [angle] mean={a_mean:.4f} rad ({a_mean * 180 / math.pi:.2f}°)  "
              f"max={a_max:.4f} rad ({a_max * 180 / math.pi:.2f}°)  "
              f"max_allowed={model.max_angle:.4f} rad ({model.max_angle * 180 / math.pi:.2f}°)")

        lk_base_exp = model.log_kappa_base.unsqueeze(0).unsqueeze(0)
        kappa_final = (lk_base_exp + delta_lk).clamp(model.log_kappa_min, model.log_kappa_max).exp()
        print(f"  [κ] mean={kappa_final.mean():.1f}  "
              f"min={kappa_final.min():.1f}  max={kappa_final.max():.1f}  "
              f"median={kappa_final.median():.1f}")

        # State diversity: average pairwise cosine similarity between mu_base states
        mu_n = F.normalize(model.mu_base, p=2, dim=-1)   # [K, S, D]
        # Average over K groups
        cos_div = 0.0
        for k in range(model.K):
            sim  = t.mm(mu_n[k], mu_n[k].T)              # [S, S]
            eye  = t.eye(model.S, device=sim.device)
            off  = (sim * (1 - eye)).sum() / max(model.S * (model.S - 1), 1)
            cos_div += off.item()
        print(f"  [state div] avg pairwise cosine={cos_div / model.K:.4f} (↓ better)")

    return {"loss": sum_loss / max(total_n, 1)}


# =========================================================
# 11. Evaluation
# =========================================================
@t.no_grad()
def run_eval(model, loader, device):
    model.eval()
    loc_metrics = init_metrics()

    for batch in tqdm(loader, desc="  [Eval]", leave=False):
        oov_mask  = batch.get("oov_mask", None)
        batch_dev = move_batch_to_device(batch, device)

        step_preds = model.forward_step_predict(batch_dev)

        loc_gpu = batch_dev["loc"]
        B, T    = loc_gpu.shape

        if oov_mask is not None:
            oov_t = t.zeros(B, T, dtype=t.bool, device=device)
            for b, row in enumerate(oov_mask):
                L_b = min(len(row), T)
                if L_b > 0:
                    oov_t[b, :L_b] = t.tensor(row[:L_b], dtype=t.bool, device=device)
        else:
            oov_t = t.zeros(B, T, dtype=t.bool, device=device)

        for tt, (loc_prob, valid) in enumerate(step_preds):
            target_l = loc_gpu[:, tt + 1]
            keep = valid & (target_l > 0) & (~oov_t[:, tt + 1])
            if not keep.any():
                continue
            loc_logits = t.log(loc_prob.clamp_min(1e-12))
            update_metrics_gpu(loc_metrics, loc_logits[keep], target_l[keep])

    return finalize_metrics(loc_metrics, "loc")


# =========================================================
# 12. Load LINE embeddings
# =========================================================
def load_line_embeddings(line_ckpt_dir, user_size, loc_size, cat_size, th_size):
    """
    Load LINE embeddings from line_ckpt_dir.  Expects either:
      - embeddings.pt  (dict with keys user_emb, loc_emb, cat_emb, time_emb)
      - or individual user_emb.npy, loc_emb.npy, cat_emb.npy, time_emb.npy

    Pads or trims each array to match (vocab_size+1, D) with row 0 = zeros.
    """
    pt_path = os.path.join(line_ckpt_dir, "embeddings.pt")
    if os.path.exists(pt_path):
        ckpt = t.load(pt_path, map_location="cpu", weights_only=False)
        arrays = {k: ckpt[k].numpy() if isinstance(ckpt[k], t.Tensor) else ckpt[k]
                  for k in ("user_emb", "loc_emb", "cat_emb", "time_emb")}
    else:
        arrays = {
            "user_emb": np.load(os.path.join(line_ckpt_dir, "user_emb.npy")),
            "loc_emb":  np.load(os.path.join(line_ckpt_dir, "loc_emb.npy")),
            "cat_emb":  np.load(os.path.join(line_ckpt_dir, "cat_emb.npy")),
            "time_emb": np.load(os.path.join(line_ckpt_dir, "time_emb.npy")),
        }

    def _pad_to(arr, target_rows):
        """Ensure array has exactly target_rows rows (pad/trim). Row 0 kept as-is."""
        D = arr.shape[1]
        if arr.shape[0] < target_rows:
            pad = np.zeros((target_rows - arr.shape[0], D), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > target_rows:
            arr = arr[:target_rows]
        return arr

    user_np = _pad_to(arrays["user_emb"], user_size + 1)
    loc_np  = _pad_to(arrays["loc_emb"],  loc_size  + 1)
    cat_np  = _pad_to(arrays["cat_emb"],  cat_size  + 1)
    time_np = _pad_to(arrays["time_emb"], th_size   + 1)

    # Zero out row 0 (padding token)
    for arr in (user_np, loc_np, cat_np, time_np):
        arr[0] = 0.0

    return user_np, loc_np, cat_np, time_np


# =========================================================
# 13. Main
# =========================================================
def main():
    BASE = "."

    parser = argparse.ArgumentParser(description="Mixture Hybrid vMF HMM")

    parser.add_argument("--pkl_path", type=str,
        default=os.path.join(BASE, "NYC_data.pkl"))
    parser.add_argument("--line_ckpt_dir", type=str,
        default=os.path.join(BASE, "multimodal_line_runs"),
        help="Directory containing LINE embeddings (embeddings.pt or *.npy files)")
    parser.add_argument("--save_dir", type=str,
        default=os.path.join(BASE, "mhvmf_K1_S4_angle_010_t01_10penalty_NYC"))

    parser.add_argument("--num_classes", type=int, default=1,
        help="K: number of mixture groups / expert HMMs")
    parser.add_argument("--num_states",  type=int, default=4,
        help="S: HMM states per group")

    parser.add_argument("--seq_hidden",  type=int,   default=128)
    parser.add_argument("--hist_hidden", type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.2)

    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-5)
    parser.add_argument("--beta_div",    type=float, default=10,
                        help="Coefficient for state diversity penalty")
    parser.add_argument("--pi_temperature", type=float, default=0.1,
        help="Temperature for π softmax. T<1 sharpens routing (T=0.1 strongly sharpens, "
             "T=1.0 is standard softmax).")
    parser.add_argument("--log_kappa_max", type=float, default=4.0,
                        help="Upper bound for log-kappa (max concentration = exp(log_kappa_max))")
    parser.add_argument("--log_kappa_min", type=float, default=1.609,
                        help="Lower bound for log-kappa (min kappa = exp(log_kappa_min)). "
                             "Default: log(20)")
    parser.add_argument("--max_angle",   type=float, default=0.10,
                        help="Maximum angular deviation of mu_final from mu_base (radians, "
                             "0.15 rad ≈ 8.59°). Acts as a hard safety bound: residuals within "
                             "the limit pass through unchanged; larger ones are clipped back.")
    parser.add_argument("--device",      type=str,
                        default="cuda" if t.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Auto-name save_dir by hyperparams if still default
    if args.save_dir.endswith("mhvmf_K4_S16_NYC"):
        tag = os.path.basename(args.pkl_path).split("_")[0]   # e.g. "NYC"
        args.save_dir = os.path.join(
            BASE, f"mhvmf_K{args.num_classes}_S{args.num_states}_{tag}")

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = t.device(args.device)
    print("DEVICE:", device)

    # 1) Load pkl
    data_obj = load_pickle(args.pkl_path)
    for key in ["train_trajectories", "val_trajectories", "test_trajectories",
                "pid_dict", "cid_dict", "pid_cid_dict", "stats"]:
        if key not in data_obj:
            raise KeyError(f"pkl missing key: {key}")

    sizes    = infer_sizes_from_pkl(data_obj)
    loc_size = sizes["loc_size"]
    cat_size = sizes["cat_size"]
    th_size  = sizes["th_size"]

    # Infer user_size from trajectories
    all_uids = set()
    for split in ("train_trajectories", "val_trajectories", "test_trajectories"):
        for traj in data_obj[split]:
            if traj["user_id"] != -1:
                all_uids.add(traj["user_id"])
    user_size = max(all_uids) if all_uids else 1
    print(f"loc={loc_size}  cat={cat_size}  th={th_size}  users={user_size}")

    # 2) Load LINE embeddings
    print("Loading LINE embeddings from:", args.line_ckpt_dir)
    user_np, loc_np, cat_np, time_np = load_line_embeddings(
        args.line_ckpt_dir, user_size, loc_size, cat_size, th_size)
    emb_dim = loc_np.shape[1]
    print(f"  LINE emb_dim={emb_dim}  user={user_np.shape}  loc={loc_np.shape}  "
          f"cat={cat_np.shape}  time={time_np.shape}")

    # 3) Histories
    train_trajs     = data_obj["train_trajectories"]
    train_hist_list = build_train_sequential_histories(train_trajs)
    full_train_hist = build_full_train_histories(train_trajs)

    val_hist_list  = [full_train_hist.get(traj["user_id"])
                      for traj in data_obj["val_trajectories"]]
    test_hist_list = [full_train_hist.get(traj["user_id"])
                      for traj in data_obj["test_trajectories"]]

    train_ds = TrajectoryDataset(train_trajs,                       train_hist_list)
    val_ds   = TrajectoryDataset(data_obj["val_trajectories"],  val_hist_list)
    test_ds  = TrajectoryDataset(data_obj["test_trajectories"], test_hist_list)

    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # 4) DataLoaders
    train_loader = t.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_traj_batch)
    val_loader = t.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_traj_batch)
    test_loader = t.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_traj_batch)

    # 5) Model & optimizer
    model = MixtureHybridVMFHMM(
        loc_size=loc_size, cat_size=cat_size, th_size=th_size,
        user_size=user_size, emb_dim=emb_dim,
        num_classes=args.num_classes, num_states=args.num_states,
        seq_hidden=args.seq_hidden, hist_hidden=args.hist_hidden,
        dropout=args.dropout,
        max_angle=args.max_angle,
        log_kappa_max=args.log_kappa_max,
        log_kappa_min=args.log_kappa_min,
        line_user_emb_np=user_np,
        line_loc_emb_np=loc_np,
        line_cat_emb_np=cat_np,
        line_time_emb_np=time_np,
    ).to(device)

    model.pi_temperature = args.pi_temperature

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters: trainable={n_trainable:,}  frozen={n_frozen:,}")

    optimizer = t.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    best_recall10 = -1.0
    best_path     = os.path.join(args.save_dir, "best_model.pt")
    history       = []

    # 6) Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch:03d}")

        train_m = run_train_epoch(model, train_loader, optimizer, device,
                                   beta_div=args.beta_div)
        val_m   = run_eval(model, val_loader,  device)
        test_m  = run_eval(model, test_loader, device)
        dt      = time.time() - t0

        print(
            f"  time={dt:.1f}s  loss={train_m['loss']:.4f}\n"
            f"  [Val  ] R@1={val_m['loc_recall@1']:.4f}  "
            f"R@5={val_m['loc_recall@5']:.4f}  "
            f"R@10={val_m['loc_recall@10']:.4f}  "
            f"N@5={val_m['loc_ndcg@5']:.4f}  "
            f"N@10={val_m['loc_ndcg@10']:.4f}\n"
            f"  [Test ] R@1={test_m['loc_recall@1']:.4f}  "
            f"R@5={test_m['loc_recall@5']:.4f}  "
            f"R@10={test_m['loc_recall@10']:.4f}  "
            f"N@5={test_m['loc_ndcg@5']:.4f}  "
            f"N@10={test_m['loc_ndcg@10']:.4f}"
        )

        val_r10 = val_m["loc_recall@10"]
        if val_r10 > best_recall10:
            best_recall10 = val_r10
            t.save(dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                best_val_recall10=best_recall10,
                test_metrics_at_best=test_m,
                args=vars(args),
                sizes=sizes,
                user_size=user_size,
                emb_dim=emb_dim,
            ), best_path)
            print(f"  ★ Best saved  (val R@10={best_recall10:.4f}  "
                  f"test R@10={test_m['loc_recall@10']:.4f})")

        row = dict(epoch=epoch, time_sec=dt,
                   **train_m,
                   **{f"val_{k}":  v for k, v in val_m.items()},
                   **{f"test_{k}": v for k, v in test_m.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(args.save_dir, "training_log.csv"),
            index=False, encoding="utf-8-sig")

    save_json(dict(
        pkl_path=args.pkl_path,
        line_ckpt_dir=args.line_ckpt_dir,
        loc_size=loc_size, cat_size=cat_size,
        user_size=user_size, emb_dim=emb_dim,
        train_instances=len(train_ds),
        val_instances=len(val_ds),
        test_instances=len(test_ds),
        best_val_recall10=float(best_recall10),
        best_model_path=best_path,
    ), os.path.join(args.save_dir, "run_meta.json"))

    print(f"\n{'='*60}")
    print("Done. save_dir:", args.save_dir)
    print("Best val Recall@10:", best_recall10)


if __name__ == "__main__":
    main()
