from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
TABLE_DIR = FIG_DIR / "summary_tables"
DEEPSET_DATA_DIR = ROOT / "DeepSetProblem" / "data" / "final"
DEEPSET_MODEL_DIR = ROOT / "DeepSetProblem" / "models" / "final"

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def save_figure(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=220)
    print(f"Saved figure -> {path}")


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved table -> {path}")


class SharedLinear(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 32, is_last: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_last = is_last

        self.A = nn.Parameter(torch.empty(out_dim, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[:-1] + (x.shape[-1] // self.in_dim, self.in_dim))
        s = x.mean(dim=-2, keepdim=True)
        local = F.linear(x, self.A)
        global_term = F.linear(s, self.B)
        out = local + global_term + self.bias
        out = out.view(out.shape[:-2] + (-1,))

        if self.is_last:
            return out.mean(dim=-1)
        return out


class VanillaMLP(nn.Module):
    def __init__(self, n_blocks: int, hidden_size: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = 1
        for h_size in hidden_size:
            layers.append(nn.Linear(n_blocks * current_dim, n_blocks * h_size))
            layers.append(nn.LeakyReLU(0.01))
            current_dim = h_size
        layers.append(nn.Linear(n_blocks * current_dim, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SharedMLP(nn.Module):
    def __init__(self, hidden_size: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = 1
        for h_size in hidden_size:
            layers.append(SharedLinear(in_dim=current_dim, out_dim=h_size))
            layers.append(nn.LeakyReLU(0.01))
            current_dim = h_size
        layers.append(SharedLinear(in_dim=current_dim, out_dim=1, is_last=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def symmetry_error(model: nn.Module, x: torch.Tensor, n_blocks: int, repeat: int = 5) -> float:
    model.eval()
    batch_size = x.shape[0]
    y_base = model(x)
    error = 0.0
    for _ in range(repeat):
        x_stacked = x.view(batch_size, n_blocks, 1)
        p_indices = torch.randperm(n_blocks)
        x_permuted = x_stacked[:, p_indices, :].reshape(batch_size, -1)
        y_perm = model(x_permuted)
        error += torch.mean((y_base - y_perm) ** 2).item()
    return error / repeat


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_tensor(name: str) -> torch.Tensor:
    return torch.load(DEEPSET_DATA_DIR / name, map_location="cpu")


def build_deepset_summary() -> pd.DataFrame:
    hidden_shape = [8, 16, 8, 4, 2]
    rows = []

    for n_block in [1, 2, 5, 10, 20]:
        for requested_size in [50, 100, 500, 1000, 5000, 10000]:
            x_train = load_tensor(f"X_train_{n_block}_{requested_size}.pt")
            x_valid = load_tensor(f"X_valid_{n_block}_{requested_size}.pt")
            x_test = load_tensor(f"X_test_{n_block}_{requested_size}.pt")
            y_test = load_tensor(f"Y_test_{n_block}_{requested_size}.pt")

            vanilla = VanillaMLP(n_block, hidden_shape)
            shared = SharedMLP(hidden_shape)
            vanilla.load_state_dict(
                torch.load(
                    DEEPSET_MODEL_DIR / f"vanilla_model_{n_block}_{requested_size}.pt",
                    map_location="cpu",
                )
            )
            shared.load_state_dict(
                torch.load(
                    DEEPSET_MODEL_DIR / f"shared_model_{n_block}_{requested_size}.pt",
                    map_location="cpu",
                )
            )
            vanilla.eval()
            shared.eval()

            with torch.no_grad():
                vanilla_pred = vanilla(x_test).view(y_test.shape)
                shared_pred = shared(x_test).view(y_test.shape)
                vanilla_loss = nn.MSELoss()(vanilla_pred, y_test).item()
                shared_loss = nn.MSELoss()(shared_pred, y_test).item()

            vanilla_sym = symmetry_error(vanilla, x_test, n_blocks=n_block)
            shared_sym = symmetry_error(shared, x_test, n_blocks=n_block)

            rows.append(
                {
                    "task": "DeepSet",
                    "n_block": n_block,
                    "requested_size": requested_size,
                    "train_size": int(len(x_train)),
                    "valid_size": int(len(x_valid)),
                    "test_size": int(len(x_test)),
                    "vanilla_loss": float(vanilla_loss),
                    "shared_loss": float(shared_loss),
                    "vanilla_symmetry": float(vanilla_sym),
                    "shared_symmetry": float(shared_sym),
                    "vanilla_size": int(count_parameters(vanilla)),
                    "shared_size": int(count_parameters(shared)),
                }
            )

    df = pd.DataFrame(rows).sort_values(["n_block", "train_size"]).reset_index(drop=True)
    df["loss_gain"] = df["vanilla_loss"] - df["shared_loss"]
    df["symmetry_gain"] = df["vanilla_symmetry"] - df["shared_symmetry"]
    df["size_ratio"] = df["vanilla_size"] / df["shared_size"]
    return df


def plot_deepset(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for n_block in sorted(df["n_block"].unique()):
        sub = df[df["n_block"] == n_block]
        axes[0].plot(
            sub["train_size"],
            sub["vanilla_loss"],
            marker="o",
            linestyle="--",
            alpha=0.7,
            label=f"vanilla, n={n_block}",
        )
        axes[0].plot(
            sub["train_size"],
            sub["shared_loss"],
            marker="o",
            linewidth=2.5,
            label=f"shared, n={n_block}",
        )
        axes[1].plot(
            sub["train_size"],
            sub["vanilla_symmetry"],
            marker="o",
            linestyle="--",
            alpha=0.7,
            label=f"vanilla, n={n_block}",
        )
        axes[1].plot(
            sub["train_size"],
            sub["shared_symmetry"],
            marker="o",
            linewidth=2.5,
            label=f"shared, n={n_block}",
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("DeepSet: Test Loss vs Train Size")
    axes[0].set_xlabel("Training data size")
    axes[0].set_ylabel("Test loss")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("DeepSet: Symmetry Error vs Train Size")
    axes[1].set_xlabel("Training data size")
    axes[1].set_ylabel("Symmetry error")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    save_figure("deepset_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    loss_heat = df.pivot(index="n_block", columns="train_size", values="loss_gain")
    sym_heat = df.pivot(index="n_block", columns="train_size", values="symmetry_gain")
    sns.heatmap(loss_heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[0])
    axes[0].set_title("DeepSet: (Vanilla loss - Shared loss)")
    axes[0].set_xlabel("Training data size")
    axes[0].set_ylabel("Number of blocks")
    sns.heatmap(sym_heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[1])
    axes[1].set_title("DeepSet: (Vanilla symmetry error - Shared symmetry error)")
    axes[1].set_xlabel("Training data size")
    axes[1].set_ylabel("Number of blocks")
    plt.tight_layout()
    save_figure("deepset_heatmaps.png")
    plt.close(fig)

    scatter_rows = []
    for _, row in df.iterrows():
        scatter_rows.append(
            {
                "model": "vanilla",
                "n_block": row["n_block"],
                "train_size": row["train_size"],
                "params": row["vanilla_size"],
                "test_loss": row["vanilla_loss"],
            }
        )
        scatter_rows.append(
            {
                "model": "shared",
                "n_block": row["n_block"],
                "train_size": row["train_size"],
                "params": row["shared_size"],
                "test_loss": row["shared_loss"],
            }
        )
    scatter_df = pd.DataFrame(scatter_rows)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=scatter_df,
        x="params",
        y="test_loss",
        hue="model",
        size="n_block",
        sizes=(60, 260),
        palette={"vanilla": "#d62728", "shared": "#1f77b4"},
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("DeepSet: Parameter Efficiency Frontier")
    plt.xlabel("Trainable parameters")
    plt.ylabel("Test loss")
    plt.tight_layout()
    save_figure("deepset_parameter_efficiency.png")
    plt.close()


def main() -> None:
    df = build_deepset_summary()
    save_table(df, "deepset_summary")
    plot_deepset(df)


if __name__ == "__main__":
    main()
