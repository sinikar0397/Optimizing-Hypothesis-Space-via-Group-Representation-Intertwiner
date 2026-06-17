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
THREE_BODY_DIR = ROOT / "3BodyProblem"
THREE_BODY_DATA_DIR = THREE_BODY_DIR / "data"
THREE_BODY_MODEL_DIR = THREE_BODY_DIR / "models" / "sweep"
THREE_BODY_RESULT_PATH = THREE_BODY_DIR / "total_result.csv"

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


def swap_first_two_particles(x: torch.Tensor) -> torch.Tensor:
    swapped = x.clone()
    swapped[..., 0:4] = x[..., 4:8]
    swapped[..., 4:8] = x[..., 0:4]
    return swapped


def build_vanilla_3body(layer_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
        if idx < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_shared_3body(channel_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(channel_dims) - 1):
        layers.append(EquivariantLinear2D(channel_dims[idx], channel_dims[idx + 1]))
        if idx < len(channel_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


THREE_BODY_MODEL_CONFIGS = {
    "small": {
        "vanilla_dims": [12, 48, 48, 12],
        "shared_dims": [4, 16, 16, 4],
    },
    "medium": {
        "vanilla_dims": [12, 48, 96, 96, 48, 12],
        "shared_dims": [4, 16, 24, 24, 16, 4],
    },
    "large": {
        "vanilla_dims": [12, 96, 192, 48, 12],
        "shared_dims": [4, 32, 64, 16, 4],
    },
}

THREE_BODY_MODEL_LABELS = {
    "vanilla": "Vanilla",
    "vanilla_aug": "Vanilla + Aug",
    "shared": "Share",
    "shared_aug": "Share + Aug",
}


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


def build_three_body_summary() -> pd.DataFrame:
    df = pd.read_csv(THREE_BODY_RESULT_PATH).copy()
    df["task"] = "3Body"
    df["model_label"] = df["model"].map(THREE_BODY_MODEL_LABELS)
    df["model_family"] = np.where(df["model"].str.contains("shared"), "shared", "vanilla")
    df["augment_label"] = np.where(df["augment"], "Aug", "No Aug")
    df["size_tag"] = pd.Categorical(df["size_tag"], ["small", "medium", "large"], ordered=True)
    df = df.sort_values(["size_tag", "data_size", "model"]).reset_index(drop=True)
    return df


def plot_three_body_summary(df: pd.DataFrame) -> None:
    palette = {
        "Vanilla": "#d62728",
        "Vanilla + Aug": "#ff9896",
        "Share": "#1f77b4",
        "Share + Aug": "#17becf",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    size_tags = ["small", "medium", "large"]
    model_order = ["Vanilla", "Vanilla + Aug", "Share", "Share + Aug"]

    for col, size_tag in enumerate(size_tags):
        sub = df[df["size_tag"] == size_tag]
        ax_loss = axes[0, col]
        ax_sym = axes[1, col]

        for model_label in model_order:
            model_sub = sub[sub["model_label"] == model_label].sort_values("data_size")
            ax_loss.plot(
                model_sub["data_size"],
                model_sub["loss"],
                marker="o",
                linewidth=2.2,
                color=palette[model_label],
                label=model_label,
            )
            ax_sym.plot(
                model_sub["data_size"],
                model_sub["sym_error"],
                marker="o",
                linewidth=2.2,
                color=palette[model_label],
                label=model_label,
            )

        ax_loss.set_xscale("log")
        ax_loss.set_yscale("log")
        ax_loss.set_title(f"{size_tag.capitalize()} model")
        ax_loss.set_ylabel("Test loss")

        ax_sym.set_xscale("log")
        ax_sym.set_yscale("log")
        ax_sym.set_xlabel("Data size")
        ax_sym.set_ylabel("Symmetry error")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Task 3: Test Loss and Symmetry Error by Data Size", y=1.02)
    plt.tight_layout()
    save_figure("3body_summary.png")
    plt.close(fig)


def load_three_body_model(model_name: str, size_tag: str, data_size: int) -> nn.Module:
    config = THREE_BODY_MODEL_CONFIGS[size_tag]
    if "shared" in model_name:
        model = build_shared_3body(config["shared_dims"])
    else:
        model = build_vanilla_3body(config["vanilla_dims"])
    state_dict = torch.load(
        THREE_BODY_MODEL_DIR / f"{model_name}_{size_tag}_{data_size}.pt",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_rollout_start(x_test: torch.Tensor, y_test: torch.Tensor, segment_length: int = 15) -> int:
    max_start = max(0, len(x_test) - segment_length)
    for start in range(max_start):
        contiguous = True
        for offset in range(segment_length - 1):
            if not torch.allclose(y_test[start + offset], x_test[start + offset + 1], atol=1e-5):
                contiguous = False
                break
        if contiguous:
            return start
    return 0


def rollout_model(model: nn.Module, init_state: torch.Tensor, steps: int) -> list[np.ndarray]:
    states = [init_state.squeeze(0).detach().cpu().numpy()]
    current = init_state.clone()
    with torch.no_grad():
        for _ in range(steps):
            current = model(current)
            states.append(current.squeeze(0).detach().cpu().numpy())
    return states


def plot_three_body_rollout(df: pd.DataFrame) -> None:
    x_test = torch.load(THREE_BODY_DATA_DIR / "X_test.pt", map_location="cpu")
    y_test = torch.load(THREE_BODY_DATA_DIR / "Y_test.pt", map_location="cpu")

    best_shared = (
        df[df["model_family"] == "shared"]
        .sort_values(["loss", "sym_error", "params"], ascending=[True, True, True])
        .iloc[0]
    )
    best_vanilla = (
        df[df["model_family"] == "vanilla"]
        .sort_values(["loss", "sym_error", "params"], ascending=[True, True, True])
        .iloc[0]
    )

    shared_model = load_three_body_model(
        model_name=best_shared["model"],
        size_tag=str(best_shared["size_tag"]),
        data_size=int(best_shared["data_size"]),
    )
    vanilla_model = load_three_body_model(
        model_name=best_vanilla["model"],
        size_tag=str(best_vanilla["size_tag"]),
        data_size=int(best_vanilla["data_size"]),
    )

    start = find_rollout_start(x_test, y_test, segment_length=15)
    gt_states = [x_test[start].detach().cpu().numpy()]
    for idx in range(start, start + 15):
        gt_states.append(y_test[idx].detach().cpu().numpy())

    init_state = x_test[start : start + 1]
    shared_states = rollout_model(shared_model, init_state, steps=15)
    vanilla_states = rollout_model(vanilla_model, init_state, steps=15)

    state_sets = [
        ("Ground truth", gt_states),
        ("Best Vanilla", vanilla_states),
        ("Best Share", shared_states),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (title, states) in zip(axes, state_sets):
        arr = np.asarray(states)
        positions = arr[:, :6].reshape(len(states), 3, 2)
        for particle_idx in range(3):
            ax.plot(
                positions[:, particle_idx, 0],
                positions[:, particle_idx, 1],
                marker=markers[particle_idx],
                markersize=4,
                linewidth=1.8,
                color=colors[particle_idx],
                label=f"Particle {particle_idx + 1}",
            )
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Task 3: Representative Rollout Comparison", y=1.03)
    plt.tight_layout()
    save_figure("3body_rollout.png")
    plt.close(fig)


def main() -> None:
    deepset_df = build_deepset_summary()
    save_table(deepset_df, "deepset_summary")
    plot_deepset(deepset_df)

    three_body_df = build_three_body_summary()
    save_table(three_body_df, "three_body_summary")
    plot_three_body_summary(three_body_df)
    plot_three_body_rollout(three_body_df)


if __name__ == "__main__":
    main()
