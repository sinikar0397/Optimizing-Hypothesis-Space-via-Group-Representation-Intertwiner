from __future__ import annotations

import copy
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "sweep"
LOG_DIR = ROOT / "logs" / "sweep"
RESULT_PATH = ROOT / "total_result.pkl"
RESULT_CSV_PATH = ROOT / "total_result.csv"
RESULT_JSON_PATH = ROOT / "total_result.json"

SEED = 42
DT = 0.01
K = 1.0
EPSILON = 0.5
MASSES = np.array([10.0, 10.0, 20.0], dtype=np.float32)
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.0

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_acceleration(x: np.ndarray, m: np.ndarray, k: float, epsilon: float) -> np.ndarray:
    n = len(m)
    accel = np.zeros_like(x)
    for i in range(n):
        diff = x - x[i]
        dist_sq = np.sum(diff**2, axis=1) + epsilon**2
        dist_cubed = (dist_sq**1.5).reshape(-1, 1)
        force_i = k * (m.reshape(-1, 1) * diff / dist_cubed)
        accel[i] = force_i.sum(axis=0)
    return accel


def generate_dataset(num_trajectories: int = 10, steps_per_traj: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    all_x, all_y = [], []
    for _ in tqdm(range(num_trajectories), desc="Generate trajectories"):
        x = np.random.randn(3, 2).astype(np.float32)
        v = np.random.randn(3, 2).astype(np.float32)
        v -= v.mean(axis=0, keepdims=True)

        traj = []
        for _ in range(steps_per_traj):
            current_state = np.concatenate([x.flatten(), v.flatten()]).astype(np.float32)
            traj.append(current_state)
            a = get_acceleration(x, MASSES, K, EPSILON)
            v += a * DT
            x += v * DT

        traj = np.asarray(traj, dtype=np.float32)
        all_x.append(traj[:-10:10])
        all_y.append(traj[10::10])

    return torch.from_numpy(np.concatenate(all_x)), torch.from_numpy(np.concatenate(all_y))


def ensure_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    x_train_path = DATA_DIR / "X_train.pt"
    y_train_path = DATA_DIR / "Y_train.pt"
    x_test_path = DATA_DIR / "X_test.pt"
    y_test_path = DATA_DIR / "Y_test.pt"

    if all(path.exists() for path in [x_train_path, y_train_path, x_test_path, y_test_path]):
        x_train = torch.load(x_train_path, map_location="cpu")
        y_train = torch.load(y_train_path, map_location="cpu")
        x_test = torch.load(x_test_path, map_location="cpu")
        y_test = torch.load(y_test_path, map_location="cpu")
        return (x_train, y_train), (x_test, y_test)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train = generate_dataset(num_trajectories=400, steps_per_traj=5000)
    x_test, y_test = generate_dataset(num_trajectories=50, steps_per_traj=5000)

    torch.save(x_train, x_train_path)
    torch.save(y_train, y_train_path)
    torch.save(x_test, x_test_path)
    torch.save(y_test, y_test_path)
    return (x_train, y_train), (x_test, y_test)


def swap_first_two_particles(x: torch.Tensor) -> torch.Tensor:
    swapped = x.clone()
    swapped[..., 0:4] = x[..., 4:8]
    swapped[..., 4:8] = x[..., 0:4]
    return swapped


class EquivariantLinear2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        scale = 1.0 / np.sqrt(in_channels * 3)
        self.A = nn.Parameter(torch.randn(out_channels, in_channels) * scale)
        self.B = nn.Parameter(torch.randn(out_channels, in_channels) * scale)
        self.C = nn.Parameter(torch.randn(out_channels, in_channels) * scale)
        self.D = nn.Parameter(torch.randn(out_channels, in_channels) * scale)
        self.E = nn.Parameter(torch.randn(out_channels, in_channels) * scale)
        self.bias_sym = nn.Parameter(torch.zeros(out_channels))
        self.bias_third = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row1 = torch.cat([self.A, self.B, self.C], dim=1)
        row2 = torch.cat([self.B, self.A, self.C], dim=1)
        row3 = torch.cat([self.D, self.D, self.E], dim=1)
        weight = torch.cat([row1, row2, row3], dim=0)
        bias = torch.cat([self.bias_sym, self.bias_sym, self.bias_third], dim=0)
        return x @ weight.t() + bias


def build_vanilla(layer_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
        if idx < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_shared(channel_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(channel_dims) - 1):
        layers.append(EquivariantLinear2D(channel_dims[idx], channel_dims[idx + 1]))
        if idx < len(channel_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ModelConfig:
    vanilla_dims: list[int]
    shared_dims: list[int]


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig(
        vanilla_dims=[12, 48, 48, 12],
        shared_dims=[4, 16, 16, 4],
    ),
    "medium": ModelConfig(
        vanilla_dims=[12, 48, 96, 96, 48, 12],
        shared_dims=[4, 16, 24, 24, 16, 4],
    ),
    "large": ModelConfig(
        vanilla_dims=[12, 96, 192, 48, 12],
        shared_dims=[4, 32, 64, 16, 4],
    ),
}

DEFAULT_DATA_SIZE_CANDIDATES = [1000, 5000, 20000, 40000]


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def architecture_signature(model_name: str, size_tag: str) -> str:
    config = MODEL_CONFIGS[size_tag]
    dims = config.shared_dims if "shared" in model_name else config.vanilla_dims
    prefix = "E" if "shared" in model_name else "L"
    return " - ".join(f"{prefix}({dims[i]}->{dims[i + 1]})" for i in range(len(dims) - 1))


def build_data_size_schedule(total_size: int) -> list[int]:
    sizes = [size for size in DEFAULT_DATA_SIZE_CANDIDATES if size < total_size]
    # sizes.append(total_size)
    return sorted(set(sizes))


def pick_batch_size(train_size: int) -> int:
    if train_size <= 2000:
        return 128
    if train_size <= 10000:
        return 256
    if train_size <= 40000:
        return 512
    return 1024


def make_subset(
    x: torch.Tensor,
    y: torch.Tensor,
    num_samples: int,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator)[:num_samples]
    return x[indices], y[indices]


def build_loaders_for_size(
    x_train_all: torch.Tensor,
    y_train_all: torch.Tensor,
    data_size: int,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    sub_x, sub_y = make_subset(x_train_all, y_train_all, num_samples=data_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        sub_x,
        sub_y,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    batch_size = pick_batch_size(len(x_train))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=max(batch_size, 512), shuffle=False)
    meta = {
        "requested_data_size": int(data_size),
        "train_size": int(len(x_train)),
        "valid_size": int(len(x_valid)),
        "batch_size": int(batch_size),
    }
    return train_loader, valid_loader, meta


def maybe_augment_batch(batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    aug_x = swap_first_two_particles(batch_x)
    aug_y = swap_first_two_particles(batch_y)
    return torch.cat([batch_x, aug_x], dim=0), torch.cat([batch_y, aug_y], dim=0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = MAX_EPOCHS,
    lr: float = 1e-3,
    patience: int = EARLY_STOPPING_PATIENCE,
    min_delta: float = EARLY_STOPPING_MIN_DELTA,
    augment: bool = False,
) -> dict[str, list[float] | int]:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_state = None
    best_valid = float("inf")
    patience_counter = 0
    train_log: list[float] = []
    valid_log: list[float] = []

    bar = tqdm(range(epochs), desc="Training", leave=False)
    for epoch in bar:
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            if augment:
                batch_x, batch_y = maybe_augment_batch(batch_x, batch_y)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)
            total_train_count += batch_x.size(0)

        avg_train = total_train_loss / total_train_count
        train_log.append(avg_train)

        model.eval()
        total_valid_loss = 0.0
        total_valid_count = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                total_valid_loss += loss.item() * batch_x.size(0)
                total_valid_count += batch_x.size(0)

        avg_valid = total_valid_loss / total_valid_count
        valid_log.append(avg_valid)
        bar.set_description(f"train {avg_train:.6f} | valid {avg_valid:.6f}")

        if avg_valid < (best_valid - min_delta):
            best_valid = avg_valid
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(best valid loss: {best_valid:.6f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_loss": train_log,
        "valid_loss": valid_log,
        "best_epoch": int(np.argmin(valid_log) + 1) if valid_log else len(train_log),
        "stopped_epoch": len(train_log),
    }


@torch.no_grad()
def symmetry_error_3body(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    values = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(DEVICE)
        out1 = model(batch)
        out2 = model(swap_first_two_particles(batch))
        out2_swapped = swap_first_two_particles(out2)
        err = ((out1 - out2_swapped) ** 2).mean(dim=-1)
        values.append(err.cpu())
    return torch.cat(values).mean().item()


@torch.no_grad()
def evaluate_model(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor) -> tuple[float, float]:
    model.eval()
    preds = []
    batch_size = 1024
    for start in range(0, len(x_test), batch_size):
        batch_x = x_test[start : start + batch_size].to(DEVICE)
        preds.append(model(batch_x).cpu())
    pred = torch.cat(preds, dim=0)
    loss = nn.MSELoss()(pred, y_test).item()
    sym_error = symmetry_error_3body(model, x_test)
    return loss, sym_error


def run_experiments() -> dict[str, dict[str, dict[str, dict[str, float | int | str]]]]:
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    (x_train_all, y_train_all), (x_test, y_test) = ensure_dataset()
    data_sizes = build_data_size_schedule(len(x_train_all))
    x_test = x_test.cpu()
    y_test = y_test.cpu()

    builders = {
        "vanilla": lambda size_tag: build_vanilla(MODEL_CONFIGS[size_tag].vanilla_dims),
        "vanilla_aug": lambda size_tag: build_vanilla(MODEL_CONFIGS[size_tag].vanilla_dims),
        "shared": lambda size_tag: build_shared(MODEL_CONFIGS[size_tag].shared_dims),
        "shared_aug": lambda size_tag: build_shared(MODEL_CONFIGS[size_tag].shared_dims),
    }
    aug_flags = {
        "vanilla": False,
        "vanilla_aug": True,
        "shared": False,
        "shared_aug": True,
    }

    results: dict[str, dict[str, dict[str, dict[str, float | int | str]]]] = {}
    rows = []

    for size_tag in ["small", "medium", "large"]:
        results[size_tag] = {}
        for data_size in data_sizes:
            results[size_tag][str(data_size)] = {}
            train_loader, valid_loader, loader_meta = build_loaders_for_size(
                x_train_all=x_train_all,
                y_train_all=y_train_all,
                data_size=data_size,
            )
            for model_name in ["vanilla", "vanilla_aug", "shared", "shared_aug"]:
                model = builders[model_name](size_tag).to(DEVICE)
                log = train_model(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=MAX_EPOCHS,
                    lr=1e-3,
                    patience=EARLY_STOPPING_PATIENCE,
                    min_delta=EARLY_STOPPING_MIN_DELTA,
                    augment=aug_flags[model_name],
                )
                test_loss, sym_error = evaluate_model(model, x_test, y_test)
                param_count = count_parameters(model)

                save_stem = f"{model_name}_{size_tag}_{data_size}"
                torch.save(model.state_dict(), MODEL_DIR / f"{save_stem}.pt")
                with open(LOG_DIR / f"{save_stem}.pkl", "wb") as f:
                    pickle.dump(log, f)

                metrics = {
                    "loss": float(test_loss),
                    "sym_error": float(sym_error),
                    "params": int(param_count),
                    "best_epoch": int(log["best_epoch"]),
                    "stopped_epoch": int(log["stopped_epoch"]),
                    "architecture": architecture_signature(model_name, size_tag),
                    "augment": bool(aug_flags[model_name]),
                    **loader_meta,
                    "test_size": int(len(x_test)),
                    "max_epochs": int(MAX_EPOCHS),
                    "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
                    "early_stopping_min_delta": float(EARLY_STOPPING_MIN_DELTA),
                }
                results[size_tag][str(data_size)][model_name] = metrics
                rows.append(
                    {
                        "size_tag": size_tag,
                        "data_size": int(data_size),
                        "model": model_name,
                        **metrics,
                    }
                )
                print(
                    f"[{size_tag} | n={data_size} | {model_name}] "
                    f"loss={test_loss:.6f}, sym_error={sym_error:.6e}, params={param_count}"
                )

    with open(RESULT_PATH, "wb") as f:
        pickle.dump(results, f)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV_PATH, index=False)
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    run_experiments()
