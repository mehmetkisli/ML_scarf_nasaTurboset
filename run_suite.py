import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from experiment_window_size import ExperimentConfig, Method, run_tag


def parse_csv_strings(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


@dataclass(frozen=True)
class SuiteRun:
    subset: str
    method: Method
    train_dir: Path
    eval_dir: Path


def run_cmd(args: Sequence[str], cwd: Path) -> None:
    print("$", " ".join(args))
    subprocess.run(list(args), cwd=str(cwd), check=True)


def iter_runs(
    subsets: Iterable[str],
    methods: Iterable[Method],
    train_root: Path,
    eval_root: Path,
    cfg: ExperimentConfig,
) -> List[SuiteRun]:
    runs: List[SuiteRun] = []
    for subset in subsets:
        for method in methods:
            tag = run_tag(cfg, method=method)
            train_dir = train_root / subset / tag
            eval_dir = eval_root / subset / method
            runs.append(SuiteRun(subset=subset, method=method, train_dir=train_dir, eval_dir=eval_dir))
    return runs


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baselines/SCARF and evaluate on CMAPSS (FD001-004) producing a summary CSV."
    )

    parser.add_argument("--dataset-root", type=str, default="Datasets")
    parser.add_argument("--subsets", type=str, default="FD001", help="Comma-separated: FD001,FD002,FD003,FD004")
    parser.add_argument(
        "--methods",
        type=str,
        default="supervised_only,scarf_full",
        help="Comma-separated: supervised_only,scarf_head_only,scarf_full",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--rul-cap", type=float, default=125)
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=1.0,
        help="Fraction of labeled training windows used for fine-tuning (1.0=all).",
    )
    parser.add_argument(
        "--label-fractions",
        type=str,
        default="",
        help="Optional comma-separated sweep of label fractions (overrides --label-fraction). Example: 0.01,0.05,0.1,0.25,1.0",
    )

    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--ft-head-epochs", type=int, default=5)
    parser.add_argument("--ft-full-epochs", type=int, default=10)

    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)

    parser.add_argument("--train-root", type=str, default="artifacts_suite")
    parser.add_argument("--eval-root", type=str, default="evaluation_suite")
    parser.add_argument("--summary-csv", type=str, default="evaluation_suite/summary.csv")

    args = parser.parse_args()

    cwd = Path.cwd()
    dataset_root = Path(args.dataset_root)

    subsets = parse_csv_strings(args.subsets)
    methods_raw = parse_csv_strings(args.methods)
    methods: List[Method] = []
    for m in methods_raw:
        if m not in ("supervised_only", "scarf_head_only", "scarf_full"):
            raise ValueError(f"Unknown method: {m}")
        methods.append(m)  # type: ignore[arg-type]

    if args.label_fractions.strip():
        label_fracs = parse_csv_floats(args.label_fractions)
    else:
        label_fracs = [float(args.label_fraction)]

    for lf in label_fracs:
        if lf <= 0.0 or lf > 1.0:
            raise ValueError(f"label fraction must be in (0,1], got: {lf}")

    train_root = Path(args.train_root)
    eval_root = Path(args.eval_root)
    summary_csv = Path(args.summary_csv)

    rows = []
    for lf in label_fracs:
        cfg = ExperimentConfig(
            data_path="",
            seed=args.seed,
            test_size=args.test_size,
            window_size=args.window_size,
            pretrain_epochs=args.pretrain_epochs,
            ft_head_epochs=args.ft_head_epochs,
            ft_full_epochs=args.ft_full_epochs,
            lr_head=args.lr_head,
            lr_encoder=args.lr_encoder,
            out_dir="",
            rul_cap=args.rul_cap,
            label_fraction=float(lf),
        )

        all_runs = iter_runs(subsets, methods, train_root=train_root, eval_root=eval_root, cfg=cfg)

        for run in all_runs:
            # --- Train ---
            run.train_dir.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(
                [
                    sys.executable,
                    "experiment_window_size.py",
                    "--dataset-root",
                    str(dataset_root),
                    "--subset",
                    run.subset,
                    "--seed",
                    str(args.seed),
                    "--test-size",
                    str(args.test_size),
                    "--window-size",
                    str(args.window_size),
                    "--rul-cap",
                    str(args.rul_cap),
                    "--label-fraction",
                    str(lf),
                    "--pretrain-epochs",
                    str(args.pretrain_epochs),
                    "--ft-head-epochs",
                    str(args.ft_head_epochs),
                    "--ft-full-epochs",
                    str(args.ft_full_epochs),
                    "--lr-head",
                    str(args.lr_head),
                    "--lr-encoder",
                    str(args.lr_encoder),
                    "--method",
                    run.method,
                    "--out-dir",
                    str((train_root / run.subset).as_posix()),
                ],
                cwd=cwd,
            )

            if not run.train_dir.exists():
                raise FileNotFoundError(f"Expected train_dir not found: {run.train_dir}")

            # --- Evaluate (official test protocol) ---
            run.eval_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(
                [
                    sys.executable,
                    "evaluate_best.py",
                    "--dataset-root",
                    str(dataset_root),
                    "--subset",
                    run.subset,
                    "--eval",
                    "test",
                    "--window-size",
                    str(args.window_size),
                    "--rul-cap",
                    str(args.rul_cap),
                    "--best-dir",
                    str(run.train_dir.as_posix()),
                    "--out-dir",
                    str(run.eval_dir.as_posix()),
                ],
                cwd=cwd,
            )

            metrics_path = run.eval_dir / "metrics.json"
            metrics = load_metrics(metrics_path)

            rows.append(
                {
                    "subset": run.subset,
                    "method": run.method,
                    "label_fraction": float(lf),
                    **metrics,
                    "train_dir": str(run.train_dir),
                    "eval_dir": str(run.eval_dir),
                }
            )

    new_df = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    if summary_csv.exists():
        old_df = pd.read_csv(summary_csv)
        df = pd.concat([old_df, new_df], ignore_index=True)
        if all(c in df.columns for c in ["subset", "method", "label_fraction"]):
            df = df.drop_duplicates(subset=["subset", "method", "label_fraction"], keep="last")
    else:
        df = new_df

    df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
