import os
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Amaç
# =========================
# - NASA CMAPSS FD001 train verisinde RUL regresyonunu geliştirmek.
# - En etkili ilk adım: window_size (sliding window) ayarı.
# - Deney protokolü: split UNIT bazında, scaler sadece TRAIN unit'lerinde fit edilir.
#   (window bazlı split + tüm data scaling -> leakage yaratır)


@dataclass
class ExperimentConfig:
    data_path: str = "train_FD001.txt"
    seed: int = 42
    test_size: float = 0.2

    # Window
    window_size: int = 50

    # SCARF pretrain
    pretrain_epochs: int = 5
    pretrain_batch_size: int = 256
    corruption_rate: float = 0.6
    temperature: float = 0.1
    lr_pretrain: float = 1e-3
    weight_decay: float = 1e-5

    # Encoder
    hidden_dim: int = 256
    emb_dim: int = 64
    proj_dim: int = 64
    dropout: float = 0.1

    # Fine-tune
    ft_head_epochs: int = 5
    ft_full_epochs: int = 10
    ft_batch_size: int = 256
    lr_head: float = 1e-3
    lr_encoder: float = 1e-4

    # RUL
    rul_cap: Optional[float] = 125

    # Label-efficiency experiments
    # 1.0 means use all labeled training windows. Smaller values subsample labeled windows for fine-tuning.
    label_fraction: float = 1.0

    # Fine-tune loss
    loss: str = "mse"  # mse | huber
    huber_beta: float = 10.0

    # Saving
    out_dir: str = "artifacts_best"


Method = Literal["scarf_full", "scarf_head_only", "supervised_only"]


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def run_tag(cfg: ExperimentConfig, method: Method) -> str:
    def fmt(x: float) -> str:
        s = f"{x:.6g}"
        return s.replace(".", "p")

    base = (
        f"ws{cfg.window_size}"
        f"_pre{cfg.pretrain_epochs}"
        f"_h{cfg.ft_head_epochs}"
        f"_f{cfg.ft_full_epochs}"
        f"_cr{fmt(cfg.corruption_rate)}"
        f"_t{fmt(cfg.temperature)}"
        f"_lre{fmt(cfg.lr_encoder)}"
        f"_lrh{fmt(cfg.lr_head)}"
    )

    if cfg.label_fraction < 1.0:
        base = f"{base}_lf{fmt(cfg.label_fraction)}"

    if method == "scarf_full":
        return base
    return f"{base}_m{method}"


def subsample_labeled(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    label_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if label_fraction >= 1.0:
        return X_tr, y_tr
    if label_fraction <= 0.0:
        raise ValueError("label_fraction must be in (0, 1]")

    n = int(X_tr.shape[0])
    k = max(1, int(round(n * float(label_fraction))))

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx = np.sort(idx)
    return X_tr[idx], y_tr[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cmapss_train(data_path: str) -> pd.DataFrame:
    index_names = ["unit_nr", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = [f"s_{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    df_raw = pd.read_csv(data_path, sep=r"\s+", header=None, names=col_names)
    return df_raw


def split_units(df_raw: pd.DataFrame, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    units = df_raw["unit_nr"].unique()
    train_units, val_units = train_test_split(units, test_size=test_size, random_state=seed)
    return train_units, val_units


def compute_feature_cols_train_only(df_train_raw: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Train subset'te varyansı 0 olan kolonlar bilgi taşımaz.
    # unit_nr ve time_cycles asla drop edilmez.
    variances = df_train_raw.var(numeric_only=True)
    constant_cols = variances[variances == 0].index.tolist()
    cols_to_drop = [c for c in constant_cols if c not in ["unit_nr", "time_cycles"]]

    base_cols = [c for c in df_train_raw.columns if c not in cols_to_drop]
    feature_cols = [c for c in base_cols if c not in ["unit_nr", "time_cycles"]]
    return feature_cols, cols_to_drop


def fit_transform_scaler(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()

    df_train = df_train.copy()
    df_val = df_val.copy()

    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])

    return df_train, df_val, scaler


def create_windows(data: pd.DataFrame, feature_cols: List[str], window_size: int) -> Tuple[np.ndarray, pd.DataFrame]:
    X: List[np.ndarray] = []
    meta: List[Tuple[int, int, int]] = []

    for unit_id in data["unit_nr"].unique():
        unit_df = data[data["unit_nr"] == unit_id].sort_values("time_cycles")
        values = unit_df[feature_cols].values
        cycles = unit_df["time_cycles"].values

        if len(values) < window_size:
            continue

        for i in range(len(values) - window_size + 1):
            window_matrix = values[i : i + window_size]
            window_vec = window_matrix.flatten()
            X.append(window_vec.astype(np.float32))
            meta.append((int(unit_id), int(cycles[i]), int(cycles[i + window_size - 1])))

    X_np = np.array(X, dtype=np.float32)
    meta_df = pd.DataFrame(meta, columns=["unit_nr", "start_cycle", "end_cycle"])
    return X_np, meta_df


def add_rul_labels(meta: pd.DataFrame, df_subset: pd.DataFrame, rul_cap: Optional[float]) -> pd.DataFrame:
    max_cycle_per_unit = df_subset.groupby("unit_nr")["time_cycles"].max()

    meta = meta.copy()
    meta["max_cycle"] = meta["unit_nr"].map(max_cycle_per_unit)
    meta["RUL"] = meta["max_cycle"] - meta["end_cycle"]

    if rul_cap is not None:
        meta["RUL"] = meta["RUL"].clip(upper=float(rul_cap))

    return meta


def scarf_corrupt(x: torch.Tensor, corruption_rate: float) -> torch.Tensor:
    batch_size, feature_dim = x.shape
    mask = torch.rand(batch_size, feature_dim, device=x.device) < corruption_rate
    x_perm = x[torch.randperm(batch_size, device=x.device)]
    x_tilde = x.clone()
    x_tilde[mask] = x_perm[mask]
    return x_tilde


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, emb_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


def info_nce_loss(z: torch.Tensor, z_tilde: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (z @ z_tilde.T) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, labels)


class RULRegressor(nn.Module):
    def __init__(self, encoder: MLPEncoder, emb_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.head(h).squeeze(1)


def pretrain_scarf(
    X_unlabeled: np.ndarray,
    input_dim: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> MLPEncoder:
    model = MLPEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
    ).to(device)

    X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=cfg.pretrain_batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(1, cfg.pretrain_epochs + 1):
        loss_sum = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            xb_tilde = scarf_corrupt(xb, corruption_rate=cfg.corruption_rate)

            _, z = model(xb)
            _, z_tilde = model(xb_tilde)

            loss = info_nce_loss(z, z_tilde, temperature=cfg.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(f"[pretrain] epoch {epoch:02d}/{cfg.pretrain_epochs} loss={loss_sum / len(loader):.4f}")

    return model


@torch.no_grad()
def eval_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse_sum = 0.0
    n_batches = 0
    criterion = nn.MSELoss()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        mse_sum += criterion(yhat, yb).item()
        n_batches += 1

    return math.sqrt(mse_sum / max(1, n_batches))


def make_criterion(cfg: ExperimentConfig) -> nn.Module:
    if cfg.loss.lower() == "mse":
        return nn.MSELoss()
    if cfg.loss.lower() == "huber":
        # SmoothL1Loss is Huber with beta parameter
        return nn.SmoothL1Loss(beta=float(cfg.huber_beta))
    raise ValueError(f"Unknown loss: {cfg.loss}")


def finetune_regressor(
    encoder: MLPEncoder,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    do_full_phase: bool = True,
) -> float:
    reg_model = RULRegressor(encoder, emb_dim=cfg.emb_dim).to(device)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xva_t = torch.tensor(X_va, dtype=torch.float32)
    yva_t = torch.tensor(y_va, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.ft_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=cfg.ft_batch_size, shuffle=False)

    criterion = make_criterion(cfg)
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    # -----------------
    # Phase 1: train head only (encoder frozen)
    # -----------------
    for p in reg_model.encoder.parameters():
        p.requires_grad = False
    for p in reg_model.head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(reg_model.head.parameters(), lr=cfg.lr_head)

    for epoch in range(1, cfg.ft_head_epochs + 1):
        reg_model.train()
        mse_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = reg_model(xb)
            loss = criterion(yhat, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse_sum += loss.item()

        train_rmse = math.sqrt(mse_sum / len(train_loader))
        val_rmse = eval_rmse(reg_model, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in reg_model.state_dict().items()}

        print(
            f"[ft-head] epoch {epoch:02d}/{cfg.ft_head_epochs} "
            f"train_rmse={train_rmse:.2f} val_rmse={val_rmse:.2f} best={best_val:.2f}"
        )

    # -----------------
    # Phase 2: unfreeze encoder with smaller LR
    # -----------------
    if do_full_phase:
        for p in reg_model.encoder.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            [
                {"params": reg_model.encoder.parameters(), "lr": cfg.lr_encoder},
                {"params": reg_model.head.parameters(), "lr": cfg.lr_head},
            ]
        )

        for epoch in range(1, cfg.ft_full_epochs + 1):
            reg_model.train()
            mse_sum = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = reg_model(xb)
                loss = criterion(yhat, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mse_sum += loss.item()

            train_rmse = math.sqrt(mse_sum / len(train_loader))
            val_rmse = eval_rmse(reg_model, val_loader, device)
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in reg_model.state_dict().items()}

            print(
                f"[ft-full] epoch {epoch:02d}/{cfg.ft_full_epochs} "
                f"train_rmse={train_rmse:.2f} val_rmse={val_rmse:.2f} best={best_val:.2f}"
            )

    # Restore best weights (for saving / further eval)
    if best_state is not None:
        reg_model.load_state_dict(best_state)

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(reg_model.encoder.state_dict(), os.path.join(cfg.out_dir, "scarf_pretrained.pt"))
    torch.save(reg_model.state_dict(), os.path.join(cfg.out_dir, "rul_regressor.pt"))
    return best_val


def train_supervised_only(
    input_dim: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
    ).to(device)

    reg_model = RULRegressor(encoder, emb_dim=cfg.emb_dim).to(device)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xva_t = torch.tensor(X_va, dtype=torch.float32)
    yva_t = torch.tensor(y_va, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.ft_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=cfg.ft_batch_size, shuffle=False)

    criterion = make_criterion(cfg)
    optimizer = torch.optim.Adam(
        [
            {"params": reg_model.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": reg_model.head.parameters(), "lr": cfg.lr_head},
        ]
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    total_epochs = int(cfg.ft_head_epochs + cfg.ft_full_epochs)
    for epoch in range(1, total_epochs + 1):
        reg_model.train()
        mse_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = reg_model(xb)
            loss = criterion(yhat, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse_sum += loss.item()

        train_rmse = math.sqrt(mse_sum / len(train_loader))
        val_rmse = eval_rmse(reg_model, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in reg_model.state_dict().items()}

        print(
            f"[supervised] epoch {epoch:02d}/{total_epochs} "
            f"train_rmse={train_rmse:.2f} val_rmse={val_rmse:.2f} best={best_val:.2f}"
        )

    if best_state is not None:
        reg_model.load_state_dict(best_state)

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(reg_model.encoder.state_dict(), os.path.join(cfg.out_dir, "scarf_pretrained.pt"))
    torch.save(reg_model.state_dict(), os.path.join(cfg.out_dir, "rul_regressor.pt"))
    return best_val


def resolve_cmapss_train_path(dataset_root: str, subset: str) -> str:
    root = Path(dataset_root)
    train_path = root / f"train_{subset}.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    return str(train_path)


def prepare_dataset_for_window(
    df_raw: pd.DataFrame,
    window_size: int,
    cfg: ExperimentConfig,
) -> Dict[str, np.ndarray]:
    train_units, val_units = split_units(df_raw, test_size=cfg.test_size, seed=cfg.seed)

    df_train_raw = df_raw[df_raw["unit_nr"].isin(train_units)].copy()
    df_val_raw = df_raw[df_raw["unit_nr"] .isin(val_units)].copy()

    feature_cols, cols_to_drop = compute_feature_cols_train_only(df_train_raw)
    if cols_to_drop:
        df_train_raw = df_train_raw.drop(columns=cols_to_drop)
        df_val_raw = df_val_raw.drop(columns=cols_to_drop)

    df_train_scaled, df_val_scaled, _ = fit_transform_scaler(df_train_raw, df_val_raw, feature_cols)

    X_tr, meta_tr = create_windows(df_train_scaled, feature_cols, window_size=window_size)
    X_va, meta_va = create_windows(df_val_scaled, feature_cols, window_size=window_size)

    meta_tr = add_rul_labels(meta_tr, df_train_scaled, cfg.rul_cap)
    meta_va = add_rul_labels(meta_va, df_val_scaled, cfg.rul_cap)

    y_tr = meta_tr["RUL"].values.astype(np.float32)
    y_va = meta_va["RUL"].values.astype(np.float32)

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_va": X_va,
        "y_va": y_va,
        "input_dim": np.array([X_tr.shape[1]], dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SCARF + RUL: window_size=50, longer pretrain, staged fine-tuning")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="If set, uses CMAPSS train_<subset>.txt under this folder.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="CMAPSS subset name when using --dataset-root",
    )
    parser.add_argument("--data-path", type=str, default="train_FD001.txt", help="Used if --dataset-root not set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--ft-head-epochs", type=int, default=5)
    parser.add_argument("--ft-full-epochs", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default="artifacts_best")
    parser.add_argument("--rul-cap", type=float, default=125)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber-beta", type=float, default=10.0)

    parser.add_argument(
        "--method",
        type=str,
        default="scarf_full",
        choices=["scarf_full", "scarf_head_only", "supervised_only"],
        help="Training method: SCARF full finetune | SCARF head-only | supervised-only (no pretrain).",
    )

    parser.add_argument(
        "--label-fraction",
        type=float,
        default=1.0,
        help="Fraction of labeled training windows to use for fine-tuning (1.0=all). Pretraining remains unlabeled.",
    )

    # Optional small grid search (comma-separated lists)
    parser.add_argument("--grid", action="store_true", help="Run a small grid over corruption_rate/temperature/lr_encoder")
    parser.add_argument("--grid-corruption", type=str, default="0.6")
    parser.add_argument("--grid-temperature", type=str, default="0.1")
    parser.add_argument("--grid-lr-encoder", type=str, default="1e-4")
    args = parser.parse_args()

    if args.dataset_root:
        data_path = resolve_cmapss_train_path(args.dataset_root, args.subset)
    else:
        data_path = args.data_path

    base_cfg = ExperimentConfig(
        data_path=data_path,
        seed=args.seed,
        test_size=args.test_size,
        window_size=args.window_size,
        pretrain_epochs=args.pretrain_epochs,
        ft_head_epochs=args.ft_head_epochs,
        ft_full_epochs=args.ft_full_epochs,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        out_dir=args.out_dir,
        rul_cap=args.rul_cap,
        loss=args.loss,
        huber_beta=args.huber_beta,
        label_fraction=float(args.label_fraction),
    )
    set_seed(base_cfg.seed)

    if not os.path.exists(base_cfg.data_path):
        raise FileNotFoundError(f"Data file not found: {base_cfg.data_path} (cwd={os.getcwd()})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df_raw = load_cmapss_train(base_cfg.data_path)
    print("df_raw:", df_raw.shape)

    ds = prepare_dataset_for_window(df_raw, window_size=base_cfg.window_size, cfg=base_cfg)
    X_tr_full = ds["X_tr"]
    y_tr_full = ds["y_tr"]
    X_va = ds["X_va"]
    y_va = ds["y_va"]
    input_dim = int(ds["input_dim"][0])

    # Label-efficiency: subsample only the labeled set used for supervised training.
    # Unlabeled pool for SCARF pretraining stays as the full training windows.
    X_tr, y_tr = subsample_labeled(
        X_tr_full,
        y_tr_full,
        label_fraction=base_cfg.label_fraction,
        seed=base_cfg.seed,
    )

    print("\n" + "=" * 70)
    print(f"WINDOW_SIZE = {base_cfg.window_size}")
    print("X_tr_full:", X_tr_full.shape, "y_tr_full:", y_tr_full.shape)
    if base_cfg.label_fraction < 1.0:
        print(f"label_fraction: {base_cfg.label_fraction} -> labeled X_tr:", X_tr.shape, "y_tr:", y_tr.shape)
    else:
        print("X_tr:", X_tr.shape, "y_tr:", y_tr.shape)
    print("X_va:", X_va.shape, "y_va:", y_va.shape)
    print("input_dim:", input_dim)

    method: Method = args.method

    if args.grid and method != "scarf_full":
        raise ValueError("--grid is supported only with --method scarf_full (for now)")

    if args.grid:
        corruption_list = parse_csv_floats(args.grid_corruption)
        temperature_list = parse_csv_floats(args.grid_temperature)
        lr_encoder_list = parse_csv_floats(args.grid_lr_encoder)

        rows = []
        best_overall = float("inf")
        best_dir: Optional[str] = None

        base_out = base_cfg.out_dir
        os.makedirs(base_out, exist_ok=True)

        for cr in corruption_list:
            for temp in temperature_list:
                for lre in lr_encoder_list:
                    # Fair comparison: reset RNG for each run
                    set_seed(base_cfg.seed)
                    cfg = ExperimentConfig(**{**base_cfg.__dict__})
                    cfg.corruption_rate = float(cr)
                    cfg.temperature = float(temp)
                    cfg.lr_encoder = float(lre)

                    tag = run_tag(cfg, method="scarf_full")
                    cfg.out_dir = os.path.join(base_out, tag)

                    print("\n" + "-" * 70)
                    print(f"RUN: {tag}")

                    encoder = pretrain_scarf(X_tr_full, input_dim=input_dim, cfg=cfg, device=device)
                    best_val_rmse = finetune_regressor(
                        encoder,
                        X_tr=X_tr,
                        y_tr=y_tr,
                        X_va=X_va,
                        y_va=y_va,
                        cfg=cfg,
                        device=device,
                    )

                    rows.append(
                        {
                            "tag": tag,
                            "window_size": cfg.window_size,
                            "pretrain_epochs": cfg.pretrain_epochs,
                            "ft_head_epochs": cfg.ft_head_epochs,
                            "ft_full_epochs": cfg.ft_full_epochs,
                            "corruption_rate": cfg.corruption_rate,
                            "temperature": cfg.temperature,
                            "lr_encoder": cfg.lr_encoder,
                            "lr_head": cfg.lr_head,
                            "best_val_rmse": float(best_val_rmse),
                            "out_dir": cfg.out_dir,
                        }
                    )

                    if best_val_rmse < best_overall:
                        best_overall = best_val_rmse
                        best_dir = cfg.out_dir

        results_df = pd.DataFrame(rows).sort_values("best_val_rmse")
        results_path = os.path.join(base_out, "grid_results.csv")
        results_df.to_csv(results_path, index=False)

        print("\nGRID RESULTS (sorted by best_val_rmse)")
        print(results_df[["tag", "best_val_rmse"]].to_string(index=False))
        print(f"saved results to: {results_path}")

        if best_dir is not None:
            # Copy best artifacts to a stable location
            best_copy_dir = os.path.join(base_out, "best")
            os.makedirs(best_copy_dir, exist_ok=True)
            torch.save(
                torch.load(os.path.join(best_dir, "scarf_pretrained.pt"), map_location="cpu"),
                os.path.join(best_copy_dir, "scarf_pretrained.pt"),
            )
            torch.save(
                torch.load(os.path.join(best_dir, "rul_regressor.pt"), map_location="cpu"),
                os.path.join(best_copy_dir, "rul_regressor.pt"),
            )
            with open(os.path.join(best_copy_dir, "best_run.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_dir={best_dir}\n")
                f.write(f"best_val_rmse={best_overall}\n")

            print(f"best_val_rmse={best_overall:.4f}")
            print(f"best artifacts copied to: {best_copy_dir}")

    else:
        cfg = base_cfg
        cfg.out_dir = os.path.join(cfg.out_dir, run_tag(cfg, method=method))
        set_seed(cfg.seed)
        print(
            f"pretrain_epochs={cfg.pretrain_epochs} | ft_head_epochs={cfg.ft_head_epochs} | "
            f"ft_full_epochs={cfg.ft_full_epochs} | lr_head={cfg.lr_head} | lr_encoder={cfg.lr_encoder}"
        )

        if method == "supervised_only":
            best_val_rmse = train_supervised_only(
                input_dim=input_dim,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                cfg=cfg,
                device=device,
            )
        else:
            encoder = pretrain_scarf(X_tr_full, input_dim=input_dim, cfg=cfg, device=device)
            best_val_rmse = finetune_regressor(
                encoder,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                cfg=cfg,
                device=device,
                do_full_phase=(method == "scarf_full"),
            )

        print("\nDONE")
        print(f"best_val_rmse={best_val_rmse:.4f}")
        print(f"saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
