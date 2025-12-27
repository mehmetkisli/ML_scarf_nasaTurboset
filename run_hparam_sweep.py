import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def run_cmd(args: Sequence[str], cwd: Path) -> None:
    print("$", " ".join(args))
    subprocess.run(list(args), cwd=str(cwd), check=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter sweep helper for SCARF on a single CMAPSS subset and label fraction. "
            "Runs supervised_only baseline once, runs SCARF grid (corruption/temperature/lr_encoder), "
            "selects best by val RMSE, then evaluates both on official test."
        )
    )

    parser.add_argument("--dataset-root", type=str, default="Datasets")
    parser.add_argument("--subset", type=str, default="FD004", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--rul-cap", type=float, default=125)
    parser.add_argument("--label-fraction", type=float, default=0.1)

    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--ft-head-epochs", type=int, default=5)
    parser.add_argument("--ft-full-epochs", type=int, default=10)

    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--grid-lr-encoder", type=str, default="1e-4")
    parser.add_argument("--grid-corruption", type=str, default="0.3,0.6")
    parser.add_argument("--grid-temperature", type=str, default="0.05,0.1,0.2")

    parser.add_argument("--train-root", type=str, default="artifacts_hparam_sweep")
    parser.add_argument("--eval-root", type=str, default="evaluation_hparam_sweep")
    parser.add_argument("--summary-csv", type=str, default="evaluation_hparam_sweep/summary.csv")

    args = parser.parse_args()

    if args.label_fraction <= 0.0 or args.label_fraction > 1.0:
        raise ValueError("--label-fraction must be in (0,1]")

    cwd = Path.cwd()
    dataset_root = Path(args.dataset_root)

    train_root = Path(args.train_root) / args.subset / f"lf{args.label_fraction}"
    eval_root = Path(args.eval_root) / args.subset / f"lf{args.label_fraction}"
    summary_csv = Path(args.summary_csv)

    # 1) Supervised-only baseline
    sup_train_base = train_root / "supervised_only"
    sup_eval_dir = eval_root / "supervised_only"
    sup_train_base.mkdir(parents=True, exist_ok=True)
    sup_eval_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            "experiment_window_size.py",
            "--dataset-root",
            str(dataset_root),
            "--subset",
            args.subset,
            "--seed",
            str(args.seed),
            "--test-size",
            str(args.test_size),
            "--window-size",
            str(args.window_size),
            "--rul-cap",
            str(args.rul_cap),
            "--label-fraction",
            str(args.label_fraction),
            "--pretrain-epochs",
            str(args.pretrain_epochs),
            "--ft-head-epochs",
            str(args.ft_head_epochs),
            "--ft-full-epochs",
            str(args.ft_full_epochs),
            "--lr-head",
            str(args.lr_head),
            "--lr-encoder",
            str(parse_csv_floats(args.grid_lr_encoder)[0]),
            "--method",
            "supervised_only",
            "--out-dir",
            str(sup_train_base.as_posix()),
        ],
        cwd=cwd,
    )

    # Resolve supervised run folder (newest ws* folder under sup_train_base)
    sup_candidates = sorted([p for p in sup_train_base.glob("ws*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not sup_candidates:
        raise FileNotFoundError(f"No supervised run folder found under: {sup_train_base}")
    sup_train_dir = sup_candidates[-1]

    run_cmd(
        [
            sys.executable,
            "evaluate_best.py",
            "--dataset-root",
            str(dataset_root),
            "--subset",
            args.subset,
            "--eval",
            "test",
            "--window-size",
            str(args.window_size),
            "--rul-cap",
            str(args.rul_cap),
            "--best-dir",
            str(sup_train_dir.as_posix()),
            "--out-dir",
            str(sup_eval_dir.as_posix()),
        ],
        cwd=cwd,
    )
    sup_metrics = load_json(sup_eval_dir / "metrics.json")

    # 2) SCARF grid
    grid_train_base = train_root / "scarf_full_grid"
    grid_eval_dir = eval_root / "scarf_full_best"
    grid_train_base.mkdir(parents=True, exist_ok=True)
    grid_eval_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            "experiment_window_size.py",
            "--dataset-root",
            str(dataset_root),
            "--subset",
            args.subset,
            "--seed",
            str(args.seed),
            "--test-size",
            str(args.test_size),
            "--window-size",
            str(args.window_size),
            "--rul-cap",
            str(args.rul_cap),
            "--label-fraction",
            str(args.label_fraction),
            "--pretrain-epochs",
            str(args.pretrain_epochs),
            "--ft-head-epochs",
            str(args.ft_head_epochs),
            "--ft-full-epochs",
            str(args.ft_full_epochs),
            "--lr-head",
            str(args.lr_head),
            "--lr-encoder",
            str(parse_csv_floats(args.grid_lr_encoder)[0]),
            "--method",
            "scarf_full",
            "--grid",
            "--grid-corruption",
            args.grid_corruption,
            "--grid-temperature",
            args.grid_temperature,
            "--grid-lr-encoder",
            args.grid_lr_encoder,
            "--out-dir",
            str(grid_train_base.as_posix()),
        ],
        cwd=cwd,
    )

    grid_results_path = grid_train_base / "grid_results.csv"
    if not grid_results_path.exists():
        raise FileNotFoundError(f"Missing grid results: {grid_results_path}")

    grid_df = pd.read_csv(grid_results_path).sort_values("best_val_rmse")
    best_row = grid_df.iloc[0].to_dict()
    best_train_dir = Path(str(best_row["out_dir"]))

    run_cmd(
        [
            sys.executable,
            "evaluate_best.py",
            "--dataset-root",
            str(dataset_root),
            "--subset",
            args.subset,
            "--eval",
            "test",
            "--window-size",
            str(args.window_size),
            "--rul-cap",
            str(args.rul_cap),
            "--best-dir",
            str(best_train_dir.as_posix()),
            "--out-dir",
            str(grid_eval_dir.as_posix()),
        ],
        cwd=cwd,
    )
    scarf_metrics = load_json(grid_eval_dir / "metrics.json")

    rows = [
        {
            "subset": args.subset,
            "label_fraction": float(args.label_fraction),
            "method": "supervised_only",
            **sup_metrics,
            "train_dir": str(sup_train_dir),
            "eval_dir": str(sup_eval_dir),
        },
        {
            "subset": args.subset,
            "label_fraction": float(args.label_fraction),
            "method": "scarf_full_best",
            **scarf_metrics,
            "best_tag": best_row.get("tag"),
            "corruption_rate": best_row.get("corruption_rate"),
            "temperature": best_row.get("temperature"),
            "lr_encoder": best_row.get("lr_encoder"),
            "best_val_rmse": best_row.get("best_val_rmse"),
            "train_dir": str(best_train_dir),
            "eval_dir": str(grid_eval_dir),
        },
    ]

    new_df = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    if summary_csv.exists():
        old_df = pd.read_csv(summary_csv)
        df = pd.concat([old_df, new_df], ignore_index=True)
        if all(c in df.columns for c in ["subset", "label_fraction", "method"]):
            df = df.drop_duplicates(subset=["subset", "label_fraction", "method"], keep="last")
    else:
        df = new_df

    df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
