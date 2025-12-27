import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from experiment_window_size import (
    ExperimentConfig,
    set_seed,
    load_cmapss_train,
    prepare_dataset_for_window,
    MLPEncoder,
    RULRegressor,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@torch.no_grad()
def predict(reg_model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    reg_model.eval()
    preds = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        yhat = reg_model(xb).detach().cpu().numpy()
        preds.append(yhat)
    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best saved SCARF+RUL model on the validation split")
    parser.add_argument("--data-path", type=str, default="train_FD001.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--rul-cap", type=float, default=125)
    parser.add_argument(
        "--best-dir",
        type=str,
        default="",
        help=(
            "Directory containing 'scarf_pretrained.pt' and 'rul_regressor.pt'. "
            "If omitted, tries (in order): artifacts_best_notebook/, newest artifacts_best_run/ws*/ , then legacy best/."
        ),
    )
    parser.add_argument("--out-dir", type=str, default="evaluation")
    args = parser.parse_args()


    def has_ckpts(p: Path) -> bool:
        return (p / "scarf_pretrained.pt").exists() and (p / "rul_regressor.pt").exists()


    def resolve_best_dir(best_dir_arg: str) -> Path:
        if best_dir_arg:
            p = Path(best_dir_arg)
            if has_ckpts(p):
                return p
            raise FileNotFoundError(
                f"best-dir provided but checkpoints not found: {p} "
                f"(expected scarf_pretrained.pt and rul_regressor.pt)"
            )

        # 1) Notebook output (often checkpoints are written directly here)
        p1 = Path("artifacts_best_notebook")
        if has_ckpts(p1):
            return p1

        # 2) Newest run under artifacts_best_run/ws*
        p2 = Path("artifacts_best_run")
        if p2.exists():
            candidates = [p for p in p2.glob("ws*") if p.is_dir() and has_ckpts(p)]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)

        # 3) Legacy default (grid 'best' folder)
        legacy = Path("artifacts_grid_search_cap125_mse") / "best"
        if has_ckpts(legacy):
            return legacy

        raise FileNotFoundError(
            "Could not auto-resolve best checkpoints. "
            "Provide --best-dir explicitly (e.g., --best-dir artifacts_best_notebook or a ws* run folder)."
        )

    cfg = ExperimentConfig(
        data_path=args.data_path,
        seed=args.seed,
        test_size=args.test_size,
        window_size=args.window_size,
        rul_cap=args.rul_cap,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df_raw = load_cmapss_train(cfg.data_path)
    ds = prepare_dataset_for_window(df_raw, window_size=cfg.window_size, cfg=cfg)

    X_tr = ds["X_tr"]
    y_tr = ds["y_tr"]
    X_va = ds["X_va"]
    y_va = ds["y_va"]
    input_dim = int(ds["input_dim"][0])

    best_dir = resolve_best_dir(args.best_dir)
    scarf_path = best_dir / "scarf_pretrained.pt"
    reg_path = best_dir / "rul_regressor.pt"

    print("best_dir:", str(best_dir))

    encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
    ).to(device)
    encoder.load_state_dict(torch.load(str(scarf_path), map_location=device))

    reg_model = RULRegressor(encoder, emb_dim=cfg.emb_dim).to(device)
    reg_model.load_state_dict(torch.load(str(reg_path), map_location=device))

    yhat_tr = predict(reg_model, X_tr, device=device)
    yhat_va = predict(reg_model, X_va, device=device)

    metrics = {
        "rmse_train": rmse(y_tr, yhat_tr),
        "mae_train": mae(y_tr, yhat_tr),
        "rmse_val": rmse(y_va, yhat_va),
        "mae_val": mae(y_va, yhat_va),
    }

    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Save predictions
    pred_df = pd.DataFrame({"y_true": y_va, "y_pred": yhat_va})
    pred_path = os.path.join(args.out_dir, "val_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Save metrics
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json

        json.dump(metrics, f, indent=2)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_va, yhat_va, s=8)
    lim_min = float(min(y_va.min(), yhat_va.min()))
    lim_max = float(max(y_va.max(), yhat_va.max()))
    plt.plot([lim_min, lim_max], [lim_min, lim_max])
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Validation: True vs Predicted")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "val_true_vs_pred.png")
    plt.savefig(fig_path, dpi=160)

    print(f"saved: {pred_path}")
    print(f"saved: {metrics_path}")
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
