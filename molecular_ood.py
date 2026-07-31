#!/usr/bin/env python3
"""AIA runner for the shared GOODHIV/OGBG-Molbbbp OOD protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


AIA_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = AIA_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from molecular_ood.data import load_molecular_ood


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_branch_mode(causaler_branch, attacker_branch, train: str) -> None:
    for module in causaler_branch:
        module.train(train == "causaler")
        for parameter in module.parameters():
            parameter.requires_grad_(train == "causaler")
    for module in attacker_branch:
        module.train(train == "attacker")
        for parameter in module.parameters():
            parameter.requires_grad_(train == "attacker")


@torch.no_grad()
def evaluate(model, loader, evaluator, device) -> float:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        labeled = batch.y == batch.y
        if not labeled.any():
            continue
        logits = model.forward_causal(batch)
        y_true.append(batch.y[labeled].view(-1, 1).cpu())
        y_pred.append(logits[labeled].view(-1, 1).cpu())
    result = evaluator.eval({
        "y_true": torch.cat(y_true).numpy(),
        "y_pred": torch.cat(y_pred).numpy(),
    })
    return float(result["rocauc"])


def load_model_class(dataset: str):
    module_dir = AIA_ROOT / ("Molhiv" if dataset == "goodhiv" else "Molbbbp")
    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module("models")
        return module.CausalAdvGNNMol
    finally:
        sys.path.pop(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["goodhiv", "molbbbp"], required=True)
    parser.add_argument("--domain", choices=["scaffold", "size"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-cache-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--stable-feature-ratio", type=float, required=True)
    parser.add_argument("--adversarial-penalty", type=float, required=True)
    parser.add_argument("--adv-distance", type=float, default=0.5)
    parser.add_argument("--causal-regularization", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    loaders, meta = load_molecular_ood(
        args.dataset,
        args.domain,
        args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
        split_cache_root=args.split_cache_root or None,
    )
    model_cls = load_model_class(args.dataset)
    model = model_cls(
        num_class=1,
        emb_dim=args.hidden,
        fro_layer=2,
        bac_layer=2,
        cau_layer=2,
        att_layer=2,
        cau_gamma=args.stable_feature_ratio,
        adv_gamma_node=1.0,
        adv_gamma_edge=1.0,
    ).to(device)

    causaler_branch = [
        model.graph_front, model.graph_backs, model.causaler, model.predictor]
    attacker_branch = [model.attacker]
    opt_causaler = torch.optim.Adam(
        [p for module in causaler_branch for p in module.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    opt_attacker = torch.optim.Adam(
        model.attacker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val = float("-inf")
    best_test = float("nan")
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        causal_sum = attacker_sum = 0.0
        batches = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            labeled = batch.y == batch.y
            if not labeled.any():
                continue
            target = batch.y.float()[labeled]

            set_branch_mode(causaler_branch, attacker_branch, "causaler")
            opt_causaler.zero_grad(set_to_none=True)
            output = model.forward_advcausal(batch)
            causal_loss = (
                criterion(output["pred_cau"].float()[labeled], target)
                + criterion(output["pred_com"].float()[labeled], target)
                + args.causal_regularization * output["cau_loss_reg"]
            )
            causal_loss.backward()
            opt_causaler.step()

            set_branch_mode(causaler_branch, attacker_branch, "attacker")
            opt_attacker.zero_grad(set_to_none=True)
            output = model.forward_attack(batch)
            attack_objective = (
                criterion(output["pred_adv"].float()[labeled], target)
                - args.adv_distance * output["loss_dis"]
                - args.adversarial_penalty * output["adv_loss_reg"]
            )
            (-attack_objective).backward()
            opt_attacker.step()
            causal_sum += float(causal_loss)
            attacker_sum += float(attack_objective)
            batches += 1

        set_branch_mode(causaler_branch, attacker_branch, "causaler")
        val_auc = evaluate(model, loaders["val"], meta["evaluator"], device)
        test_auc = evaluate(model, loaders["test"], meta["evaluator"], device)
        improved = val_auc > best_val
        if improved:
            best_val, best_test, best_epoch = val_auc, test_auc, epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} causal_loss={causal_sum/max(batches,1):.6f} "
            f"attacker_objective={attacker_sum/max(batches,1):.6f} "
            f"ood_val_rocauc={val_auc:.6f} ood_test_rocauc={test_auc:.6f} "
            f"best_epoch={best_epoch} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early_stopping epoch={epoch} patience={args.patience}")
            break

    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
    summary = {
        "method": "AIA",
        "dataset": args.dataset,
        "domain": args.domain,
        "seed": args.seed,
        "selection_metric": "ood_val_rocauc",
        "best_epoch": best_epoch,
        "best_ood_val": best_val,
        "ood_test_at_best_val": best_test,
        "runtime_seconds": time.time() - started,
        "protocol": {
            "optimizer": "Adam",
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "classifier_layers": 4,
            "hidden": args.hidden,
        },
        "hyperparameters": {
            "stable_feature_ratio": args.stable_feature_ratio,
            "adversarial_penalty": args.adversarial_penalty,
            "adversarial_distance": args.adv_distance,
            "causal_regularization": args.causal_regularization,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
