#!/usr/bin/env python3
"""AIA on GALA CMNIST-sp and SentiGraph degree-OOD datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parent
BASELINES = REPO.parent
sys.path.insert(0, str(REPO / "CMNIST"))
sys.path.insert(0, str(BASELINES))

from models import CausalAdvGNNSyn  # noqa: E402
from structural_ood.data import load_structural_ood  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", required=True,
        choices=(
            "cmnist-sp", "mnist75sp",
            "graph-sst2", "graph-sst5", "graph-twitter"))
    p.add_argument("--baselines-root", type=Path, default=BASELINES)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--selection-start-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-6)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--stable-feature-ratio", type=float, default=0.5)
    p.add_argument("--adversarial-penalty-weight", type=float, default=0.1)
    p.add_argument("--adv-distance-weight", type=float, default=0.5)
    p.add_argument("--causal-penalty-weight", type=float, default=0.05)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for graph in loader:
        graph = graph.to(device)
        pred = model.forward_causal(graph).argmax(-1)
        target = graph.y.view(-1).long()
        correct += int((pred == target).sum())
        total += target.numel()
    return correct / total


def set_trainable(modules, enabled):
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
        module.train(enabled)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loaders, meta = load_structural_ood(
        args.baselines_root, args.dataset, args.batch_size, args.num_workers)
    default_hidden = {
        "cmnist-sp": 32,
        "mnist75sp": 64,
    }.get(args.dataset, 128)
    hidden = args.hidden_dim or default_hidden
    model = CausalAdvGNNSyn(
        num_class=meta["num_classes"], in_dim=meta["input_dim"],
        emb_dim=hidden, fro_layer=2, bac_layer=1,
        cau_layer=2, att_layer=2, dropout_rate=args.dropout,
        cau_gamma=args.stable_feature_ratio,
        adv_gamma_node=1.0, adv_gamma_edge=0.8).to(device)

    causaler = [model.graph_front, model.graph_backs,
                model.causaler, model.predictor]
    attacker = [model.attacker]
    opt_c = torch.optim.Adam(
        [p for m in causaler for p in m.parameters()],
        lr=args.lr, weight_decay=args.weight_decay)
    opt_a = torch.optim.Adam(
        model.attacker.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)

    best_val = -math.inf
    best_test = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        causal_loss = attacker_loss = 0.0
        batches = 0
        for graph in loaders["train"]:
            graph = graph.to(device)
            set_trainable(causaler, True)
            set_trainable(attacker, False)
            opt_c.zero_grad()
            out = model.forward_advcausal(graph)
            loss_c = (
                F.cross_entropy(out["pred_cau"], graph.y.view(-1).long())
                + F.cross_entropy(out["pred_com"], graph.y.view(-1).long())
                + args.causal_penalty_weight * out["cau_loss_reg"])
            loss_c.backward()
            opt_c.step()

            set_trainable(causaler, False)
            set_trainable(attacker, True)
            opt_a.zero_grad()
            out = model.forward_attack(graph)
            objective = (
                out["loss_dis"] * args.adv_distance_weight
                + out["adv_loss_reg"] * args.adversarial_penalty_weight
                - F.cross_entropy(out["pred_adv"], graph.y.view(-1).long()))
            objective.backward()
            opt_a.step()
            causal_loss += float(loss_c.detach())
            attacker_loss += float(objective.detach())
            batches += 1

        set_trainable(causaler, True)
        set_trainable(attacker, True)
        val = accuracy(model, loaders["val"], device)
        test = accuracy(model, loaders["test"], device)
        history.append({"epoch": epoch, "val_accuracy": val,
                        "test_accuracy": test})
        print(
            f"epoch={epoch} causal_loss={causal_loss/batches:.6f} "
            f"attacker_objective={attacker_loss/batches:.6f} "
            f"val_acc={val:.8f} test_acc={test:.8f}", flush=True)
        if epoch < args.selection_start_epoch:
            continue
        if val > best_val:
            best_val, best_test, best_epoch = val, test, epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, args.output_dir / "best.pt")
    # vars(args)는 Namespace 내부 dict 자체를 반환한다. 이를 summary에 그대로
    # 넣고 Path를 문자열로 바꾸면 args.output_dir까지 str로 변해 아래의
    # summary.json 경로 결합이 실패하므로 독립 복사본을 직렬화한다.
    serialized_args = vars(args).copy()
    serialized_args["baselines_root"] = str(args.baselines_root)
    serialized_args["output_dir"] = str(args.output_dir)
    summary = {
        "method": "AIA", "dataset": meta["dataset"], "seed": args.seed,
        "selection_metric": "accuracy", "best_epoch": best_epoch,
        "best_ood_val": best_val,
        "metrics": {"ood_test": {"accuracy": best_test}},
        "split_sizes": meta["split_sizes"], "history": history,
        "elapsed_seconds": time.time() - started, "args": serialized_args,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best valid perf:{best_val}, Test perf accordingly:{best_test}")


if __name__ == "__main__":
    main()
