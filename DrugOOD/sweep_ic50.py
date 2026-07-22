"""Grid-search AIA stable-feature ratio and adversarial penalty on DrugOOD IC50."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


STABLE_FEATURE_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
ADVERSARIAL_PENALTY_WEIGHTS = (0.01, 0.1, 0.2, 0.5, 1.0, 3.0, 5.0)


def value_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def metric_stats(values) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "std": None}
    return {"mean": statistics.fmean(finite), "std": statistics.pstdev(finite)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", choices=("assay", "scaffold", "size"), default=["assay"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--subset", choices=("core", "general", "refined"), default="core")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "sweeps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stable-feature-ratios", nargs="+", type=float, default=STABLE_FEATURE_RATIOS)
    parser.add_argument(
        "--adversarial-penalty-weights",
        nargs="+",
        type=float,
        default=ADVERSARIAL_PENALTY_WEIGHTS,
    )
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")

    train_script = Path(__file__).resolve().parent / "train_ic50.py"
    jobs = []
    combinations = itertools.product(
        args.domains,
        args.seeds,
        args.stable_feature_ratios,
        args.adversarial_penalty_weights,
    )
    for domain, seed, stable_ratio, penalty_weight in combinations:
        output_dir = (
            args.output_root
            / domain
            / f"stable_{value_name(stable_ratio)}_advpen_{value_name(penalty_weight)}"
            / f"seed_{seed}"
        )
        command = [
            sys.executable,
            str(train_script),
            "--domain",
            domain,
            "--subset",
            args.subset,
            "--seed",
            str(seed),
            "--stable-feature-ratio",
            str(stable_ratio),
            "--adversarial-penalty-weight",
            str(penalty_weight),
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
        ]
        if args.data_root is not None:
            command.extend(("--data-root", str(args.data_root)))
        command.extend(extra)
        jobs.append((command, output_dir, domain, stable_ratio, penalty_weight))

    print(f"method=AIA jobs={len(jobs)} max_parallel={args.max_parallel}")
    for command, _, _, _, _ in jobs:
        print(" ".join(command))
    if args.dry_run:
        return

    def launch(job):
        completed = subprocess.run(job[0], check=False)
        return job, completed.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            job, returncode = future.result()
            if returncode:
                failures.append((job[0], returncode))
    if failures:
        details = "\n".join(f"exit={code}: {' '.join(command)}" for command, code in failures)
        raise SystemExit(f"{len(failures)}/{len(jobs)} jobs failed:\n{details}")

    grouped = {}
    for _, output_dir, domain, stable_ratio, penalty_weight in jobs:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        key = (domain, stable_ratio, penalty_weight)
        grouped.setdefault(key, []).append(summary)

    aggregate = {"method": "AIA", "seeds": args.seeds, "groups": {}}
    best_by_domain = {}
    for (domain, stable_ratio, penalty_weight), summaries in sorted(grouped.items()):
        key = f"{domain}/stable={stable_ratio:g}/adversarial_penalty={penalty_weight:g}"
        entry = {
            "runs": len(summaries),
            "ood_val_selection": metric_stats(summary["best_ood_val"] for summary in summaries),
            "ood_test_accuracy": metric_stats(
                summary["metrics"]["ood_test"]["accuracy"] for summary in summaries
            ),
            "ood_test_roc_auc": metric_stats(
                summary["metrics"]["ood_test"]["roc_auc"] for summary in summaries
            ),
        }
        aggregate["groups"][key] = entry
        validation_mean = entry["ood_val_selection"]["mean"]
        previous = best_by_domain.get(domain)
        if previous is None or validation_mean > previous[0]:
            best_by_domain[domain] = (validation_mean, key)
    aggregate["best_by_domain"] = {
        domain: key for domain, (_, key) in sorted(best_by_domain.items())
    }
    aggregate_path = args.output_root / "aggregate.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"aggregate={aggregate_path}")


if __name__ == "__main__":
    main()

