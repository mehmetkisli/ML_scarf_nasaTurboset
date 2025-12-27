import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot label-fraction sweep results from run_suite summary CSV.")
    parser.add_argument("--csv", type=str, default="evaluation_label_sweep/summary.csv")
    parser.add_argument("--subset", type=str, default="FD001")
    parser.add_argument("--metric", type=str, default="rmse_test", choices=["rmse_test", "mae_test", "rmse_val", "mae_val"])
    parser.add_argument("--out", type=str, default="evaluation_label_sweep/label_sweep_FD001.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    df = df[df["subset"] == args.subset].copy()
    if df.empty:
        raise SystemExit(f"No rows for subset={args.subset} in {csv_path}")

    if "label_fraction" not in df.columns:
        raise SystemExit("CSV missing 'label_fraction' column. Re-run with updated run_suite.py.")

    methods = sorted(df["method"].unique().tolist())

    plt.figure(figsize=(7, 4.5))
    for method in methods:
        d = df[df["method"] == method].copy()
        d = d.sort_values("label_fraction")
        plt.plot(d["label_fraction"], d[args.metric], marker="o", label=method)

    plt.xscale("log")
    plt.xlabel("Label fraction (log scale)")
    plt.ylabel(args.metric)
    plt.title(f"Label-efficiency sweep ({args.subset})")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
