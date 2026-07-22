"""Train AIA on DrugOOD IC50 assay/scaffold/size splits."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

try:
    from .data import discover_data_root, load_splits
    from .model import AIADrugOOD, initialize_aia
except ImportError:  # Direct execution: python3 train_ic50.py
    from data import discover_data_root, load_splits
    from model import AIADrugOOD, initialize_aia


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("assay", "scaffold", "size"), default="assay")
    parser.add_argument("--subset", choices=("core", "general", "refined"), default="core")
    parser.add_argument("--endpoint", choices=("ic50", "ec50"), default="ic50")
    parser.add_argument("--data-root", type=Path, default=discover_data_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50, help="Maximum AIA epochs after pretraining.")
    parser.add_argument("--erm-pretrain-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--causaler-lr", type=float, default=1e-3)
    parser.add_argument("--attacker-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--stable-feature-ratio",
        type=float,
        default=0.5,
        help="AIA cau_gamma; sweep over 0.1, 0.3, 0.5, 0.7, 0.9.",
    )
    parser.add_argument(
        "--adversarial-penalty-weight",
        type=float,
        default=0.5,
        help="AIA adv_reg; sweep over 0.01, 0.1, 0.2, 0.5, 1, 3, 5.",
    )
    parser.add_argument("--causal-penalty-weight", type=float, default=0.5)
    parser.add_argument("--adversarial-distance-weight", type=float, default=0.5)
    parser.add_argument("--adversarial-node-ratio", type=float, default=1.0)
    parser.add_argument("--adversarial-edge-ratio", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("accuracy", "roc_auc"), default="accuracy")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    for name in ("stable_feature_ratio", "adversarial_node_ratio", "adversarial_edge_ratio"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({name}) but is unavailable")
    return device


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return math.nan
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def evaluate(model: AIADrugOOD, loader: DataLoader, device: torch.device, mode: str = "causal") -> dict:
    model.eval()
    labels_all, scores_all, predictions_all = [], [], []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        logits = model.forward_erm(batch) if mode == "erm" else model.forward_causal(batch)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        total_count += labels.numel()
        labels_all.append(labels.cpu())
        scores_all.append(logits.softmax(dim=-1)[:, 1].cpu())
        predictions_all.append(logits.argmax(dim=-1).cpu())
    labels = torch.cat(labels_all).numpy()
    scores = torch.cat(scores_all).numpy()
    predictions = torch.cat(predictions_all).numpy()
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float((predictions == labels).mean()),
        "roc_auc": binary_auc(labels, scores),
        "count": int(total_count),
    }


def make_loader(dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def set_phase(model: AIADrugOOD, phase: str) -> None:
    causaler_modules = (model.front, model.back, model.causaler, model.predictor)
    if phase == "causaler":
        for module in causaler_modules:
            module.train()
        model.attacker.eval()
    elif phase == "attacker":
        for module in causaler_modules:
            module.eval()
        model.attacker.train()
    else:
        raise ValueError(phase)


def pretrain_erm(
    model: AIADrugOOD,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    history: list,
) -> None:
    parameters = list(model.front.parameters()) + list(model.back.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.causaler_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.erm_pretrain_epochs + 1):
        model.front.train()
        model.back.train()
        model.predictor.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.view(-1).long()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.forward_erm(batch), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_count += labels.numel()
        val_metrics = evaluate(model, val_loader, device, mode="erm")
        record = {
            "phase": "erm_pretrain",
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "ood_val": val_metrics,
        }
        history.append(record)
        if epoch % args.log_every == 0:
            print(
                f"phase=erm_pretrain epoch={epoch:03d} train={record['train_loss']:.4f} "
                f"ood_val_acc={val_metrics['accuracy']:.4f} ood_val_auc={val_metrics['roc_auc']:.4f}"
            )


def train_aia_epoch(
    model: AIADrugOOD,
    loader: DataLoader,
    device: torch.device,
    causaler_optimizer: torch.optim.Optimizer,
    attacker_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> dict:
    totals = {"causaler": 0.0, "attacker": 0.0}
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()

        set_phase(model, "causaler")
        causaler_optimizer.zero_grad(set_to_none=True)
        attacker_optimizer.zero_grad(set_to_none=True)
        output = model.forward_advcausal(batch)
        causaler_loss = (
            F.cross_entropy(output["pred_causal"], labels)
            + F.cross_entropy(output["pred_combined"], labels)
            + args.causal_penalty_weight * output["causal_regularizer"]
        )
        causaler_loss.backward()
        causaler_optimizer.step()

        set_phase(model, "attacker")
        causaler_optimizer.zero_grad(set_to_none=True)
        attacker_optimizer.zero_grad(set_to_none=True)
        attack = model.forward_attack(batch)
        attacker_objective = (
            -F.cross_entropy(attack["pred_adversarial"], labels)
            + args.adversarial_distance_weight * attack["distance"]
            + args.adversarial_penalty_weight * attack["adversarial_regularizer"]
        )
        attacker_objective.backward()
        attacker_optimizer.step()

        totals["causaler"] += float(causaler_loss.detach()) * labels.numel()
        totals["attacker"] += float(attacker_objective.detach()) * labels.numel()
        total_count += labels.numel()
    return {name: value / max(total_count, 1) for name, value in totals.items()}


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    stem, splits = load_splits(args.data_root, args.subset, args.domain, args.endpoint)
    train_loader = make_loader(splits["train"], args, shuffle=True)
    eval_loaders = {
        name: make_loader(dataset, args, shuffle=False)
        for name, dataset in splits.items()
        if name != "train"
    }
    sample = splits["train"][0]
    node_dim = int(sample.x.shape[-1]) if sample.x.ndim > 1 else 1
    edge_dim = int(sample.edge_attr.shape[-1]) if sample.edge_attr.ndim > 1 else 1
    model = AIADrugOOD(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        stable_feature_ratio=args.stable_feature_ratio,
        adv_node_ratio=args.adversarial_node_ratio,
        adv_edge_ratio=args.adversarial_edge_ratio,
    ).to(device)
    model.apply(initialize_aia)

    output_dir = args.output_dir or (
        Path(__file__).resolve().parent
        / "outputs"
        / f"aia_{stem}_seed{args.seed}_{int(time.time())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history = []
    print(f"method=AIA dataset={stem} device={device} output={output_dir}")

    if args.erm_pretrain_epochs > 0:
        pretrain_erm(model, train_loader, eval_loaders["ood_val"], device, args, history)

    causaler_parameters = (
        list(model.front.parameters())
        + list(model.back.parameters())
        + list(model.causaler.parameters())
        + list(model.predictor.parameters())
    )
    causaler_optimizer = torch.optim.Adam(
        causaler_parameters, lr=args.causaler_lr, weight_decay=args.weight_decay
    )
    attacker_optimizer = torch.optim.Adam(
        model.attacker.parameters(), lr=args.attacker_lr, weight_decay=args.weight_decay
    )

    best_value = -math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_losses = train_aia_epoch(
            model, train_loader, device, causaler_optimizer, attacker_optimizer, args
        )
        val_metrics = evaluate(model, eval_loaders["ood_val"], device)
        selection_value = val_metrics[args.selection_metric]
        history.append(
            {
                "phase": "main",
                "epoch": epoch,
                "train": train_losses,
                "ood_val": val_metrics,
            }
        )
        if epoch % args.log_every == 0:
            print(
                f"phase=main epoch={epoch:03d} causaler={train_losses['causaler']:.4f} "
                f"attacker={train_losses['attacker']:.4f} "
                f"ood_val_acc={val_metrics['accuracy']:.4f} ood_val_auc={val_metrics['roc_auc']:.4f}"
            )
        if selection_value > best_value:
            best_value = selection_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                best_path,
            )
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = {name: evaluate(model, loader, device) for name, loader in eval_loaders.items()}
    summary = {
        "method": "AIA",
        "dataset": stem,
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_ood_val": best_value,
        "metrics": metrics,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train(parse_args())
