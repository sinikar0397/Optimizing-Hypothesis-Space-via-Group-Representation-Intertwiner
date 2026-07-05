"""
전체 실험 결과 시각화 스크립트 (multi-seed 재실행 + 추가 실험 반영 버전)

세 태스크(Task1: DeepSet, Task2: Graph/NCI1, Task3: 3Body)의 multi-seed raw 결과를 읽어
논문에 쓸 그림/표를 생성한다. 각 main.py 가 이미 만들어 둔 long-format raw csv를 그대로
사용하며, seaborn에 seed 단위 raw 값을 그대로 넘겨 신뢰구간(표준편차 band)을 직접
계산하게 한다 (미리 평균으로 뭉개서 넘기지 않음).

입력 파일:
    - total_result_multiseed_raw.csv
        Task 1 (DeepSet). 컬럼: n_block, data_size, seed, model, loss, symmetry_error, n_params

    - GraphProblem/total_result_graph_multiseed_raw_with_params.csv
        Task 2 (Graph). compute_graph_params.py 를 먼저 돌려서 만든, n_params가 포함된 버전.
        없으면 GraphProblem/total_result_graph_multiseed_raw.csv 로 대체하고
        파라미터 효율성 그림은 건너뜀.
        컬럼: n_block, data_size, seed, model, loss, accuracy, symmetry_error, (n_params)

    - 3BodyProblem/total_result_multiseed_raw.csv
        Task 3 (3Body), in-distribution 결과.
        컬럼: size_tag, data_size, seed, model, loss, symmetry_error, n_params

    - 3BodyProblem/total_result_ood_raw.csv
        Task 3, evaluate_ood.py 결과 (OOD 검증). 없으면 관련 그림은 건너뜀.
        컬럼: size_tag, data_size, seed, model, ood_loss, ood_symmetry_error

세 태스크는 실험 축(n_block vs size_tag)과 지표(accuracy 유무 등)가 서로 다르므로
공용 유틸(model_palette, save_figure, save_table)만 공유하고 태스크별로 별도 함수를 둔다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
TABLE_DIR = FIG_DIR / "summary_tables"

DEEPSET_RESULT_PATH = ROOT / "total_result_multiseed_raw.csv"

GRAPH_RESULT_WITH_PARAMS_PATH = ROOT / "GraphProblem" / "total_result_graph_multiseed_raw_with_params.csv"
GRAPH_RESULT_PATH = ROOT / "GraphProblem" / "total_result_graph_multiseed_raw.csv"

THREE_BODY_DIR = ROOT / "3BodyProblem"
THREE_BODY_RESULT_PATH = THREE_BODY_DIR / "total_result_multiseed_raw.csv"
THREE_BODY_OOD_PATH = THREE_BODY_DIR / "total_result_ood_raw.csv"
THREE_BODY_DATA_DIR = THREE_BODY_DIR / "data"
THREE_BODY_MODEL_DIR = THREE_BODY_DIR / "models"

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# ------------------------------------------------------------------
# 공용 유틸
# ------------------------------------------------------------------

def save_figure(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=220)
    plt.close("all")
    print(f"[figure] {path}")


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[table]  {path}")


def model_palette(models: list[str]) -> dict[str, str]:
    """
    태스크마다 모델 이름은 다르지만(vanilla/shared vs vanilla/symmetric),
    "대칭 구조 여부"와 "augmentation 여부"라는 같은 두 축을 가지므로
    이름에 포함된 문자열을 보고 색을 일관되게 배정한다.
        - 대칭 구조, no aug   -> 진한 파랑
        - 대칭 구조, aug      -> 청록
        - vanilla, no aug     -> 진한 빨강
        - vanilla, aug        -> 연한 빨강
    """
    palette = {}
    for m in models:
        is_sym = ("shared" in m) or ("symmetric" in m)
        is_aug = "aug" in m
        if is_sym and not is_aug:
            palette[m] = "#1f77b4"
        elif is_sym and is_aug:
            palette[m] = "#17becf"
        elif (not is_sym) and not is_aug:
            palette[m] = "#d62728"
        else:
            palette[m] = "#ff9896"
    return palette


# ------------------------------------------------------------------
# Task 1: DeepSet (S_n symmetry)
# ------------------------------------------------------------------

def load_deepset() -> pd.DataFrame:
    return pd.read_csv(DEEPSET_RESULT_PATH)


def plot_deepset(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())
    palette = model_palette(models)

    # (1) Loss vs data_size, n_block별 facet (mean ± sd, 5 seeds)
    g = sns.relplot(
        data=df, x="data_size", y="loss", hue="model", col="n_block",
        kind="line", marker="o", palette=palette, errorbar="sd",
        col_wrap=3, height=4, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Test loss (MSE)")
    g.set_titles("n_block = {col_name}")
    g.fig.suptitle("Task 1 (DeepSet): Test Loss vs Data Size (mean \u00b1 sd, 5 seeds)", y=1.03)
    save_figure("task1_deepset_loss.png")

    # (2) Symmetry error vs data_size. n_block=1은 정의상 항상 0이므로 제외.
    #     shared/shared_aug는 구조적으로 정확히 0에 가까운 값을 내므로
    #     로그축 표시를 위해 아주 작은 값으로만 클리핑한다 (표 값 자체는 원본 유지).
    sym_df = df[df["n_block"] > 1].copy()
    sym_df["symmetry_error_plot"] = sym_df["symmetry_error"].clip(lower=1e-16)
    g = sns.relplot(
        data=sym_df, x="data_size", y="symmetry_error_plot", hue="model", col="n_block",
        kind="line", marker="o", palette=palette, errorbar="sd",
        col_wrap=2, height=4, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Symmetry error (MSE under permutation)")
    g.set_titles("n_block = {col_name}")
    g.fig.suptitle(
        "Task 1 (DeepSet): Symmetry Error vs Data Size (n_block=1 제외, mean \u00b1 sd)\n"
        "shared 계열은 구조적으로 0에 가까워 로그축 표시를 위해 클리핑됨",
        y=1.05,
    )
    save_figure("task1_deepset_symmetry_error.png")

    # (3) 파라미터 효율성: 가장 큰 data_size에서 (n_params, mean loss)
    max_data_size = df["data_size"].max()
    largest = df[df["data_size"] == max_data_size].copy()
    agg = largest.groupby(["model", "n_block"], as_index=False).agg(
        mean_loss=("loss", "mean"), n_params=("n_params", "first")
    )
    agg["n_block"] = agg["n_block"].astype(str)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=agg, x="n_params", y="mean_loss", hue="model", style="n_block",
        s=200, palette=palette,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Trainable parameters")
    plt.ylabel(f"Mean test loss (data_size={int(max_data_size)})")
    plt.title("Task 1 (DeepSet): Parameter Efficiency Frontier")
    save_figure("task1_deepset_parameter_efficiency.png")

    # (4) Heatmap: mean(vanilla loss) - mean(shared loss), n_block x data_size
    #     augmentation 유무 각각에 대해 그림
    for suffix, label in [("", "no augmentation"), ("_aug", "with augmentation")]:
        pivot_gap = (
            df[df["model"] == f"vanilla{suffix}"].groupby(["n_block", "data_size"])["loss"].mean()
            - df[df["model"] == f"shared{suffix}"].groupby(["n_block", "data_size"])["loss"].mean()
        ).unstack("data_size")
        plt.figure(figsize=(9, 5))
        sns.heatmap(pivot_gap, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title(f"Task 1 (DeepSet): Mean(Vanilla loss) \u2212 Mean(Shared loss), {label}")
        plt.xlabel("Training data size")
        plt.ylabel("Number of blocks (n_block)")
        save_figure(f"task1_deepset_loss_gap_heatmap{suffix}.png")


def summarize_deepset(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["n_block", "data_size", "model"], as_index=False).agg(
        mean_loss=("loss", "mean"), std_loss=("loss", "std"),
        mean_symmetry_error=("symmetry_error", "mean"), std_symmetry_error=("symmetry_error", "std"),
        n_params=("n_params", "first"), n_seeds=("seed", "nunique"),
    )


# ------------------------------------------------------------------
# Task 2: Graph / NCI1 (S_n symmetry)
# ------------------------------------------------------------------

def load_graph() -> tuple[pd.DataFrame, bool]:
    if GRAPH_RESULT_WITH_PARAMS_PATH.exists():
        return pd.read_csv(GRAPH_RESULT_WITH_PARAMS_PATH), True
    print(
        "[warn] n_params가 포함된 결과 파일이 없습니다. "
        "GraphProblem/compute_graph_params.py 를 먼저 실행하면 파라미터 효율성 그림도 생성됩니다."
    )
    return pd.read_csv(GRAPH_RESULT_PATH), False


def plot_graph(df: pd.DataFrame, has_params: bool) -> None:
    models = sorted(df["model"].unique())
    palette = model_palette(models)

    # (1) Accuracy vs data_size, n_block별 facet
    g = sns.relplot(
        data=df, x="data_size", y="accuracy", hue="model", col="n_block",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.15,
    )
    g.set(xscale="log")
    g.set_axis_labels("Training data size", "Test accuracy")
    g.set_titles("n_block = {col_name}")
    g.fig.suptitle("Task 2 (Graph/NCI1): Accuracy vs Data Size (mean \u00b1 sd, 5 seeds)", y=1.03)
    save_figure("task2_graph_accuracy.png")

    # (2) Symmetry error vs data_size (log scale)
    #     symmetric 계열은 ~1e-15 수준으로 사실상 0, vanilla는 최대 0.4~0.5까지 벌어짐
    g = sns.relplot(
        data=df, x="data_size", y="symmetry_error", hue="model", col="n_block",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.15,
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Symmetry error (logit MSE under node permutation)")
    g.set_titles("n_block = {col_name}")
    g.fig.suptitle("Task 2 (Graph/NCI1): Symmetry Error vs Data Size (mean \u00b1 sd, 5 seeds)", y=1.03)
    save_figure("task2_graph_symmetry_error.png")

    if not has_params:
        return

    # (3) 파라미터 효율성: n_block별 최대 data_size에서 (n_params, mean accuracy)
    largest = df[df["data_size"] == df.groupby("n_block")["data_size"].transform("max")].copy()
    agg = largest.groupby(["model", "n_block"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), n_params=("n_params", "first")
    )
    agg["n_block"] = agg["n_block"].astype(str)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=agg, x="n_params", y="mean_accuracy", hue="model", style="n_block",
        s=200, palette=palette,
    )
    plt.xscale("log")
    plt.xlabel("Trainable parameters")
    plt.ylabel("Mean test accuracy (n_block별 최대 data_size)")
    plt.title("Task 2 (Graph/NCI1): Parameter Efficiency Frontier")
    save_figure("task2_graph_parameter_efficiency.png")


def summarize_graph(df: pd.DataFrame, has_params: bool) -> pd.DataFrame:
    agg_kwargs = dict(
        mean_loss=("loss", "mean"), std_loss=("loss", "std"),
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_symmetry_error=("symmetry_error", "mean"), std_symmetry_error=("symmetry_error", "std"),
        n_seeds=("seed", "nunique"),
    )
    if has_params:
        agg_kwargs["n_params"] = ("n_params", "first")
    return df.groupby(["n_block", "data_size", "model"], as_index=False).agg(**agg_kwargs)


# ------------------------------------------------------------------
# Task 3: 3Body (S_2 symmetry)
# ------------------------------------------------------------------

def load_three_body() -> pd.DataFrame:
    df = pd.read_csv(THREE_BODY_RESULT_PATH)
    df["size_tag"] = pd.Categorical(df["size_tag"], ["small", "medium", "large"], ordered=True)
    return df


def load_three_body_ood() -> pd.DataFrame | None:
    if not THREE_BODY_OOD_PATH.exists():
        print(
            "[warn] OOD 결과 파일이 없습니다. "
            "3BodyProblem/evaluate_ood.py 를 먼저 실행하면 OOD 비교 그림도 생성됩니다."
        )
        return None
    df = pd.read_csv(THREE_BODY_OOD_PATH)
    df["size_tag"] = pd.Categorical(df["size_tag"], ["small", "medium", "large"], ordered=True)
    return df


def plot_three_body(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())
    palette = model_palette(models)

    # (1) Loss vs data_size, size_tag별 facet
    g = sns.relplot(
        data=df, x="data_size", y="loss", hue="model", col="size_tag",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.1,
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Test loss (MSE)")
    g.set_titles("size_tag = {col_name}")
    g.fig.suptitle("Task 3 (3Body): Test Loss vs Data Size (mean \u00b1 sd, 5 seeds)", y=1.03)
    save_figure("task3_3body_loss.png")

    # (2) In-distribution symmetry error vs data_size (log scale)
    g = sns.relplot(
        data=df, x="data_size", y="symmetry_error", hue="model", col="size_tag",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.1,
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Symmetry error (in-distribution)")
    g.set_titles("size_tag = {col_name}")
    g.fig.suptitle("Task 3 (3Body): In-Distribution Symmetry Error vs Data Size (mean \u00b1 sd)", y=1.03)
    save_figure("task3_3body_symmetry_error_id.png")

    # (3) 파라미터 효율성
    max_data_size = df["data_size"].max()
    largest = df[df["data_size"] == max_data_size].copy()
    agg = largest.groupby(["model", "size_tag"], as_index=False, observed=True).agg(
        mean_loss=("loss", "mean"), n_params=("n_params", "first")
    )
    agg["size_tag"] = agg["size_tag"].astype(str)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=agg, x="n_params", y="mean_loss", hue="model", style="size_tag",
        s=200, palette=palette,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Trainable parameters")
    plt.ylabel(f"Mean test loss (data_size={int(max_data_size)})")
    plt.title("Task 3 (3Body): Parameter Efficiency Frontier")
    save_figure("task3_3body_parameter_efficiency.png")


def summarize_three_body(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["size_tag", "data_size", "model"], as_index=False, observed=True).agg(
        mean_loss=("loss", "mean"), std_loss=("loss", "std"),
        mean_symmetry_error=("symmetry_error", "mean"), std_symmetry_error=("symmetry_error", "std"),
        n_params=("n_params", "first"), n_seeds=("seed", "nunique"),
    )


def plot_three_body_ood(id_df: pd.DataFrame, ood_df: pd.DataFrame) -> pd.DataFrame:
    """
    추가 실험(OOD 검증)의 핵심 그림.
    augmentation은 훈련 분포 "안"에서만 대칭성을 근사적으로 학습하지만,
    구조적 제약(shared)은 입력 분포와 무관하게 항상 정확히 대칭성을 만족해야 한다는
    가설을 직접 검증한다.

    Returns:
        merged (pd.DataFrame): id/ood 결과를 (size_tag, data_size, seed, model) 기준으로
            합친 long-format 데이터프레임 (요약 테이블 저장용으로 재사용).
    """
    models = sorted(ood_df["model"].unique())
    palette = model_palette(models)

    merged = pd.merge(
        id_df[["size_tag", "data_size", "seed", "model", "symmetry_error"]],
        ood_df[["size_tag", "data_size", "seed", "model", "ood_symmetry_error", "ood_loss"]],
        on=["size_tag", "data_size", "seed", "model"],
        how="inner",
    )

    # (4) ID vs OOD symmetry error, 모델별 평균 막대 그래프 (log scale)
    long_df = merged.melt(
        id_vars=["model"], value_vars=["symmetry_error", "ood_symmetry_error"],
        var_name="regime", value_name="symmetry_error_value",
    )
    long_df["regime"] = long_df["regime"].map({
        "symmetry_error": "In-distribution",
        "ood_symmetry_error": "OOD (\u00d73 wider init.)",
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=long_df, x="model", y="symmetry_error_value", hue="regime",
        estimator="mean", errorbar="sd",
    )
    plt.yscale("log")
    plt.ylabel("Symmetry error (log scale)")
    plt.title("Task 3 (3Body): Symmetry Error, In-Distribution vs OOD")
    plt.xticks(rotation=15)
    save_figure("task3_3body_ood_symmetry_error_bar.png")

    # (5) OOD loss vs data_size, size_tag별 facet
    g = sns.relplot(
        data=ood_df, x="data_size", y="ood_loss", hue="model", col="size_tag",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.1,
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "OOD test loss (MSE)")
    g.set_titles("size_tag = {col_name}")
    g.fig.suptitle("Task 3 (3Body): OOD Test Loss vs Data Size (mean \u00b1 sd)", y=1.03)
    save_figure("task3_3body_ood_loss.png")

    return merged


def summarize_three_body_ood(merged: pd.DataFrame) -> pd.DataFrame:
    return merged.groupby(["size_tag", "data_size", "model"], as_index=False, observed=True).agg(
        mean_id_symmetry_error=("symmetry_error", "mean"),
        mean_ood_symmetry_error=("ood_symmetry_error", "mean"),
        std_ood_symmetry_error=("ood_symmetry_error", "std"),
        mean_ood_loss=("ood_loss", "mean"),
        std_ood_loss=("ood_loss", "std"),
        n_seeds=("seed", "nunique"),
    )


# ------------------------------------------------------------------
# Task 3 부가: 궤적 rollout 정성적 비교 (선택적 그림, 있으면 생성)
# ------------------------------------------------------------------

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


def build_shared_3body(channel_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(channel_dims) - 1):
        layers.append(EquivariantLinear2D(channel_dims[idx], channel_dims[idx + 1]))
        if idx < len(channel_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_vanilla_3body(layer_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
        if idx < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


THREE_BODY_MODEL_CONFIGS = {
    "small": {"vanilla_dims": [12, 48, 48, 12], "shared_dims": [4, 16, 16, 4]},
    "medium": {"vanilla_dims": [12, 48, 96, 96, 48, 12], "shared_dims": [4, 16, 24, 24, 16, 4]},
    "large": {"vanilla_dims": [12, 96, 192, 48, 12], "shared_dims": [4, 32, 64, 16, 4]},
}


def load_three_body_checkpoint(model_name: str, size_tag: str, data_size: int, seed: int) -> nn.Module:
    config = THREE_BODY_MODEL_CONFIGS[size_tag]
    model = (
        build_shared_3body(config["shared_dims"])
        if "shared" in model_name
        else build_vanilla_3body(config["vanilla_dims"])
    )
    suffix = f"{size_tag}_{data_size}_seed{seed}"
    ckpt_path = THREE_BODY_MODEL_DIR / f"{model_name}_model_{suffix}.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def rollout_model(model: nn.Module, init_state: torch.Tensor, steps: int) -> list[np.ndarray]:
    states = [init_state.squeeze(0).detach().numpy()]
    current = init_state.clone()
    with torch.no_grad():
        for _ in range(steps):
            current = model(current)
            states.append(current.squeeze(0).detach().numpy())
    return states


def plot_three_body_rollout(size_tag: str = "large", data_size: int = 40000,
                             seed: int = 0, steps: int = 15) -> None:
    """
    한 대표 설정(size_tag/data_size/seed)에 대해 vanilla vs shared 모델의
    자기회귀적(autoregressive) rollout 궤적을 실제 정답 궤적과 정성적으로 비교한다.
    (테스트셋은 shuffle 없이 생성되므로 앞부분 몇 개 샘플은 같은 궤적 안에서 연속적임)
    """
    suffix = f"{data_size}_seed{seed}"
    x_test_path = THREE_BODY_DATA_DIR / f"X_test_{suffix}.pt"
    y_test_path = THREE_BODY_DATA_DIR / f"Y_test_{suffix}.pt"
    if not x_test_path.exists() or not y_test_path.exists():
        print(f"[warn] {x_test_path.name} 이 없어 rollout 그림을 건너뜁니다.")
        return

    x_test = torch.load(x_test_path, map_location="cpu")
    y_test = torch.load(y_test_path, map_location="cpu")

    # 연속성 확인: X[i+1] == Y[i] 이어야 같은 궤적 안의 연속 구간
    n_check = min(steps, len(y_test) - 1)
    is_contiguous = all(
        torch.allclose(y_test[i], x_test[i + 1], atol=1e-4) for i in range(n_check)
    )
    if not is_contiguous:
        print("[warn] 테스트셋 앞부분이 연속적이지 않아 rollout 그림을 건너뜁니다.")
        return

    gt_states = [x_test[0].numpy()] + [y_test[i].numpy() for i in range(steps)]

    init_state = x_test[0:1]
    try:
        vanilla_model = load_three_body_checkpoint("vanilla", size_tag, data_size, seed)
        shared_model = load_three_body_checkpoint("shared", size_tag, data_size, seed)
    except FileNotFoundError as exc:
        print(f"[warn] 체크포인트를 찾을 수 없어 rollout 그림을 건너뜁니다: {exc}")
        return

    vanilla_states = rollout_model(vanilla_model, init_state, steps=steps)
    shared_states = rollout_model(shared_model, init_state, steps=steps)

    state_sets = [
        ("Ground truth", gt_states),
        ("Vanilla (rollout)", vanilla_states),
        ("Shared (rollout)", shared_states),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (title, states) in zip(axes, state_sets):
        arr = np.asarray(states)
        positions = arr[:, :6].reshape(len(states), 3, 2)
        for particle_idx in range(3):
            ax.plot(
                positions[:, particle_idx, 0], positions[:, particle_idx, 1],
                marker=markers[particle_idx], markersize=4, linewidth=1.8,
                color=colors[particle_idx], label=f"Particle {particle_idx + 1}",
            )
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(
        f"Task 3 (3Body): Rollout Comparison (size_tag={size_tag}, data_size={data_size}, seed={seed})",
        y=1.03,
    )
    plt.tight_layout()
    save_figure("task3_3body_rollout.png")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    # Task 1: DeepSet
    print("\n=== Task 1: DeepSet ===")
    deepset_df = load_deepset()
    save_table(summarize_deepset(deepset_df), "task1_deepset_summary")
    plot_deepset(deepset_df)

    # Task 2: Graph / NCI1
    print("\n=== Task 2: Graph (NCI1) ===")
    graph_df, has_params = load_graph()
    save_table(summarize_graph(graph_df, has_params), "task2_graph_summary")
    plot_graph(graph_df, has_params)

    # Task 3: 3Body (in-distribution)
    print("\n=== Task 3: 3Body ===")
    three_body_df = load_three_body()
    save_table(summarize_three_body(three_body_df), "task3_3body_summary")
    plot_three_body(three_body_df)

    # Task 3: 3Body OOD (있으면)
    ood_df = load_three_body_ood()
    if ood_df is not None:
        merged = plot_three_body_ood(three_body_df, ood_df)
        save_table(summarize_three_body_ood(merged), "task3_3body_ood_summary")

    # Task 3: rollout 정성적 비교 (체크포인트/데이터 있으면)
    plot_three_body_rollout()

    print("\n모든 그림/표 생성 완료.")
    print(f"그림 저장 위치: {FIG_DIR}")
    print(f"표 저장 위치:   {TABLE_DIR}")


if __name__ == "__main__":
    main()
