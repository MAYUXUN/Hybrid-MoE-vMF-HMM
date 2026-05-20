# -*- coding: utf-8 -*-
"""
train.py
========
Training script for the Mixture-of-Experts Hybrid vMF-HMM model.

Example
-------
python train.py \
    --pkl_path data/NYC_getnext_ready.pkl \
    --line_ckpt_dir embeddings/multimodal_line_runs \
    --save_dir outputs/mhvmf_K4_S4_angle015_NYC \
    --num_classes 4 \
    --num_states 4 \
    --max_angle 0.15 \
    --beta_div 10 \
    --epochs 100
"""

import os
import time
import argparse

import pandas as pd
import torch as t

from mixture_hybrid_vmf_hmm import (
    set_seed,
    load_pickle,
    save_json,
    infer_sizes_from_pkl,
    build_train_sequential_histories,
    build_full_train_histories,
    TrajectoryDataset,
    collate_traj_batch,
    MixtureHybridVMFHMM,
    run_train_epoch,
    run_eval,
    load_line_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mixture-of-Experts Hybrid vMF-HMM for next-POI prediction"
    )

    # Paths
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to the processed next-POI dataset pickle file.")
    parser.add_argument("--line_ckpt_dir", type=str, required=True,
                        help="Directory containing pretrained LINE embeddings.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory for saving checkpoints and logs.")

    # Model
    parser.add_argument("--num_classes", type=int, default=4,
                        help="K: number of mixture experts.")
    parser.add_argument("--num_states", type=int, default=4,
                        help="S: number of hidden states per expert.")
    parser.add_argument("--seq_hidden", type=int, default=128)
    parser.add_argument("--hist_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_angle", type=float, default=0.15,
                        help="Maximum angular deviation of mu_final from mu_base in radians.")
    parser.add_argument("--log_kappa_max", type=float, default=4.0,
                        help="Upper bound for log-kappa.")
    parser.add_argument("--log_kappa_min", type=float, default=1.609,
                        help="Lower bound for log-kappa. Default is approximately log(5).")
    parser.add_argument("--pi_temperature", type=float, default=0.1,
                        help="Temperature for user-specific mixture gate.")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--beta_div", type=float, default=10.0,
                        help="Coefficient for state diversity regularization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if t.cuda.is_available() else "cpu")

    return parser.parse_args()


def infer_user_size(data_obj):
    all_uids = set()
    for split in ("train_trajectories", "val_trajectories", "test_trajectories"):
        for traj in data_obj[split]:
            if traj["user_id"] != -1:
                all_uids.add(traj["user_id"])
    return max(all_uids) if all_uids else 1


def build_dataloaders(data_obj, batch_size):
    train_trajs = data_obj["train_trajectories"]

    train_hist_list = build_train_sequential_histories(train_trajs)
    full_train_hist = build_full_train_histories(train_trajs)

    val_hist_list = [
        full_train_hist.get(traj["user_id"])
        for traj in data_obj["val_trajectories"]
    ]
    test_hist_list = [
        full_train_hist.get(traj["user_id"])
        for traj in data_obj["test_trajectories"]
    ]

    train_ds = TrajectoryDataset(train_trajs, train_hist_list)
    val_ds = TrajectoryDataset(data_obj["val_trajectories"], val_hist_list)
    test_ds = TrajectoryDataset(data_obj["test_trajectories"], test_hist_list)

    train_loader = t.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_traj_batch,
    )

    val_loader = t.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_traj_batch,
    )

    test_loader = t.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_traj_batch,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = t.device(args.device)
    print("DEVICE:", device)

    # Load processed dataset
    data_obj = load_pickle(args.pkl_path)
    required_keys = [
        "train_trajectories",
        "val_trajectories",
        "test_trajectories",
        "pid_dict",
        "cid_dict",
        "pid_cid_dict",
        "stats",
    ]
    for key in required_keys:
        if key not in data_obj:
            raise KeyError(f"pkl missing key: {key}")

    sizes = infer_sizes_from_pkl(data_obj)
    loc_size = sizes["loc_size"]
    cat_size = sizes["cat_size"]
    th_size = sizes["th_size"]
    user_size = infer_user_size(data_obj)

    print(f"loc={loc_size}  cat={cat_size}  th={th_size}  users={user_size}")

    # Load frozen LINE embeddings
    print("Loading LINE embeddings from:", args.line_ckpt_dir)
    user_np, loc_np, cat_np, time_np = load_line_embeddings(
        args.line_ckpt_dir,
        user_size,
        loc_size,
        cat_size,
        th_size,
    )
    emb_dim = loc_np.shape[1]

    print(
        f"LINE emb_dim={emb_dim}  "
        f"user={user_np.shape}  loc={loc_np.shape}  "
        f"cat={cat_np.shape}  time={time_np.shape}"
    )

    # Build datasets and dataloaders
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        data_obj,
        args.batch_size,
    )

    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # Build model
    model = MixtureHybridVMFHMM(
        loc_size=loc_size,
        cat_size=cat_size,
        th_size=th_size,
        user_size=user_size,
        emb_dim=emb_dim,
        num_classes=args.num_classes,
        num_states=args.num_states,
        seq_hidden=args.seq_hidden,
        hist_hidden=args.hist_hidden,
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
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters: trainable={n_trainable:,}  frozen={n_frozen:,}")

    optimizer = t.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_recall10 = -1.0
    best_path = os.path.join(args.save_dir, "best_model.pt")
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch:03d}")

        train_m = run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta_div=args.beta_div,
        )

        val_m = run_eval(model, val_loader, device)
        test_m = run_eval(model, test_loader, device)

        dt = time.time() - t0

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

        val_recall10 = val_m["loc_recall@10"]

        if val_recall10 > best_val_recall10:
            best_val_recall10 = val_recall10

            t.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_recall10": best_val_recall10,
                    "test_metrics_at_best": test_m,
                    "args": vars(args),
                    "sizes": sizes,
                    "user_size": user_size,
                    "emb_dim": emb_dim,
                },
                best_path,
            )

            print(
                f"  ★ Best saved "
                f"(val R@10={best_val_recall10:.4f}, "
                f"test R@10={test_m['loc_recall@10']:.4f})"
            )

        row = {
            "epoch": epoch,
            "time_sec": dt,
            **train_m,
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }
        history.append(row)

        pd.DataFrame(history).to_csv(
            os.path.join(args.save_dir, "training_log.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    save_json(
        {
            "pkl_path": args.pkl_path,
            "line_ckpt_dir": args.line_ckpt_dir,
            "loc_size": loc_size,
            "cat_size": cat_size,
            "user_size": user_size,
            "emb_dim": emb_dim,
            "train_instances": len(train_ds),
            "val_instances": len(val_ds),
            "test_instances": len(test_ds),
            "best_val_recall10": float(best_val_recall10),
            "best_model_path": best_path,
            "args": vars(args),
        },
        os.path.join(args.save_dir, "run_meta.json"),
    )

    print(f"\n{'=' * 60}")
    print("Done. save_dir:", args.save_dir)
    print("Best val Recall@10:", best_val_recall10)


if __name__ == "__main__":
    main()