"""
전체 실험 결과 시각화 스크립트 (이상치 제외 버전: Exclude Min/Max)

`visualize.py`와 동일한 세 태스크(Task1: DeepSet, Task2: Graph/NCI1, Task3: NBody)의
multi-seed raw 결과를 읽어 논문에 쓸 그림/표를 생성하되, 각 (조건, model) 그룹 안에서
seed 축으로 존재하는 5개의 값 중 "최댓값 1개, 최솟값 1개"를 제외한 나머지 값들만으로
평균 및 표준편차(오차 막대/band)를 계산한다.

배경: 5-seed 실험 결과를 그대로 시각화했을 때, 일부 조건에서 소수의 이상치(outlier) seed로
인해 분산(표준편차 band)이 과도하게 크게 나타나는 문제가 있었다. 이를 완화하기 위해
"trimmed statistics" 방식(양 끝 극단값 제외 후 평균/표준편차)을 도입한다.

주의:
    - 각 그룹의 seed 수가 5개이므로, 최댓값/최솟값을 각각 1개씩 제외하면 남는 값은 3개이다.
    - 그룹 내 값 개수가 3개 미만이면(예: 일부 실험이 누락된 경우) trimming을 적용하지 않고
      원본 값을 그대로 사용한다.
    - 지표별(loss, symmetry_error, accuracy, ood_loss, ood_symmetry_error)로 "이상치인 seed"가
      서로 다를 수 있으므로, trimming은 그림/표에 사용되는 지표마다 독립적으로 수행한다.
      (즉, loss 그림에서 제외되는 seed와 symmetry_error 그림에서 제외되는 seed는 다를 수 있음)
    - 원본 raw csv나 `visualize.py`의 결과는 전혀 수정하지 않으며, 이 스크립트 내부에서만
      trimming된 사본을 만들어 사용한다.

입력 파일 (visualize.py와 동일):
    - total_result_multiseed_raw.csv
        Task 1 (DeepSet). 컬럼: n_block, data_size, seed, model, loss, symmetry_error, n_params

    - GraphProblem/total_result_graph_multiseed_raw_with_params.csv
        Task 2 (Graph). compute_graph_params.py 를 먼저 돌려서 만든, n_params가 포함된 버전.
        없으면 GraphProblem/total_result_graph_multiseed_raw.csv 로 대체하고
        파라미터 효율성 그림은 건너뜀.
        컬럼: n_block, data_size, seed, model, loss, accuracy, symmetry_error, (n_params)

    - 3BodyProblem/total_result_multiseed_raw.csv
        Task 3 (NBody), in-distribution 결과.
        [2026-07 갱신] symmetry의 영향력이 너무 작다는 관측(Task3의 test loss가 vanilla/shared
        간 유의미한 차이를 보이지 않음)에 따라, N=3/S_2 고정 실험을 N_total=4 고정 + k_sym(대칭
        입자 수) in {2,3,4} 스윕으로 일반화함. 이제 "대칭 비중(k_sym/n_total)"이 커질수록 shared의
        이점이 커지는지를 직접 확인할 수 있음.
        컬럼: k_sym, n_total, size_tag, data_size, seed, model, loss, symmetry_error, n_params
        (구 스키마 호환: k_sym/n_total 컬럼이 없는 옛 결과 파일이 들어오면 k_sym=2, n_total=3으로
        간주하여 기존 방식대로 처리함)

    - 3BodyProblem/total_result_ood_raw.csv
        Task 3, evaluate_ood.py 결과 (OOD 검증). 없으면 관련 그림은 건너뜀.
        [주의] evaluate_ood.py가 아직 k_sym 스윕에 맞춰 갱신되지 않았다면 이 파일에는 k_sym
        컬럼이 없을 수 있음. 그 경우 OOD 비교 그림은 스키마 불일치로 건너뛰고 경고를 출력함.
        컬럼: size_tag, data_size, seed, model, ood_loss, ood_symmetry_error, (k_sym, n_total)

출력:
    - 그림: ./figures/Exclude_MinMax/*.png
    - 표:   ./figures/Exclude_MinMax/summary_tables/*.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import koreanize_matplotlib
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
# THREE_BODY_OOD_PATH = THREE_BODY_DIR / "total_result_ood_raw.csv"
THREE_BODY_DATA_DIR = THREE_BODY_DIR / "data"
THREE_BODY_MODEL_DIR = THREE_BODY_DIR / "models"

# 새 main.py(NBody 일반화 버전)의 설정과 반드시 일치해야 rollout 체크포인트를 올바르게 복원할 수 있음.
THREE_BODY_N_TOTAL = 4
THREE_BODY_K_SYM_LIST = [2, 3, 4]
THREE_BODY_STATE_DIM_PER_PARTICLE = 4  # (x, y, vx, vy)
THREE_BODY_FLAT_DIM = THREE_BODY_STATE_DIM_PER_PARTICLE * THREE_BODY_N_TOTAL

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

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


def trim_minmax_rows(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """
    group_cols로 정의되는 각 그룹(보통 seed를 제외한 실험 조건 + model) 안에서
    value_col 기준 최댓값 1개, 최솟값 1개에 해당하는 행을 제외한 나머지 행만 남긴다.

    - 그룹 내 행 수가 3개 미만이면 trimming 없이 그대로 둔다 (제외할 만큼 여유가 없음).
    - 최댓값과 최솟값이 같은 행(그룹 내 값이 모두 동일)이면 한 행만 제외된다.
    """
    drop_idx: set = set()
    for _, g in df.groupby(group_cols, observed=True):
        if len(g) < 3:
            continue
        vals = g[value_col]
        drop_idx.add(vals.idxmax())
        drop_idx.add(vals.idxmin())
    return df.drop(index=drop_idx).reset_index(drop=True)


def summarize_metric(
    df: pd.DataFrame, group_cols: list[str], value_col: str, out_prefix: str
) -> pd.DataFrame:
    """
    group_cols 기준으로 value_col에 대해 이상치(최댓값/최솟값) 1개씩을 제외한 뒤
    평균/표준편차/실제 사용된 seed 수를 계산한다.
    """
    trimmed = trim_minmax_rows(df, group_cols, value_col)
    out = trimmed.groupby(group_cols, as_index=False, observed=True).agg(
        **{
            f"{out_prefix}_mean": (value_col, "mean"),
            f"{out_prefix}_std": (value_col, "std"),
            f"{out_prefix}_n_seeds_used": ("seed", "nunique"),
        }
    )
    return out


# ------------------------------------------------------------------
# Task 1: DeepSet (S_n symmetry)
# ------------------------------------------------------------------

def load_deepset() -> pd.DataFrame:
    return pd.read_csv(DEEPSET_RESULT_PATH)


def plot_deepset(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())
    palette = model_palette(models)
    group_cols = ["n_block", "data_size", "model"]

    # (1) Loss vs data_size, n_block별 facet (mean ± sd, seed 축 최대/최소 제외)
    loss_trimmed = trim_minmax_rows(df, group_cols, "loss")
    g = sns.relplot(
        data=loss_trimmed, x="data_size", y="loss", hue="model", col="n_block",
        kind="line", marker="o", palette=palette, errorbar="sd",
        col_wrap=3, height=4, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Test loss (MSE)")
    g.set_titles("n_block = {col_name}")
    g.fig.suptitle(
        "Task 1 (DeepSet): Test Loss vs Data Size\n"
        "(mean \u00b1 sd)",
        y=1.05,
    )
    save_figure("task1_deepset_loss.png")

    # (2) Symmetry error vs data_size. n_block=1은 정의상 항상 0이므로 제외.
    #     shared/shared_aug는 구조적으로 정확히 0에 가까운 값을 내므로
    #     로그축 표시를 위해 아주 작은 값으로만 클리핑한다 (표 값 자체는 원본 유지).
    sym_df = df[df["n_block"] > 1].copy()
    sym_trimmed = trim_minmax_rows(sym_df, group_cols, "symmetry_error")
    sym_trimmed["symmetry_error_plot"] = sym_trimmed["symmetry_error"].clip(lower=1e-16)
    g = sns.relplot(
        data=sym_trimmed, x="data_size", y="symmetry_error_plot", hue="model", col="n_block",
        kind="line", marker="o", palette=palette, errorbar="sd",
        col_wrap=2, height=4, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Symmetry error")
    g.set_titles("n_block = {col_name}")
    g.fig.suptitle(
        "Task 1: Symmetry Error vs Data Size\n"
        "mean \u00b1 sd",
        y=1.08,
    )
    save_figure("task1_deepset_symmetry_error.png")

    # (3) 파라미터 효율성: 가장 큰 data_size에서 (n_params, mean loss), 이상치 seed 제외
    max_data_size = df["data_size"].max()
    largest = df[df["data_size"] == max_data_size].copy()
    largest_trimmed = trim_minmax_rows(largest, ["model", "n_block"], "loss")
    agg = largest_trimmed.groupby(["model", "n_block"], as_index=False).agg(
        mean_loss=("loss", "mean"), n_params=("n_params", "first")
    )
    agg["n_block"] = agg["n_block"].astype(str)
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=agg, x="n_params", y="mean_loss", hue="model", style="n_block",
        s=200, palette=palette,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Trainable parameters")
    plt.ylabel(f"Mean test loss (data_size={int(max_data_size)})")
    plt.title("Task 1: Parameter Efficiency Frontier")
    save_figure("task1_deepset_parameter_efficiency.png")

    # (4) Heatmap: mean(vanilla loss) - mean(shared loss), n_block x data_size
    #     augmentation 유무 각각에 대해, 이상치 seed 제외 후 평균으로 계산
    for suffix, label in [("", "no augmentation"), ("_aug", "with augmentation")]:
        vanilla_sub = df[df["model"] == f"vanilla{suffix}"]
        shared_sub = df[df["model"] == f"shared{suffix}"]
        vanilla_trimmed = trim_minmax_rows(vanilla_sub, ["n_block", "data_size"], "loss")
        shared_trimmed = trim_minmax_rows(shared_sub, ["n_block", "data_size"], "loss")
        pivot_gap = (
            vanilla_trimmed.groupby(["n_block", "data_size"])["loss"].mean()
            - shared_trimmed.groupby(["n_block", "data_size"])["loss"].mean()
        ).unstack("data_size")
        plt.figure(figsize=(9, 5))
        sns.heatmap(pivot_gap, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title(
            f"Task 1: Mean(Vanilla loss) - Mean(Shared loss)\n({label})\n"
        )
        plt.xlabel("Training data size")
        plt.ylabel("Number of blocks (n_block)")
        save_figure(f"task1_deepset_loss_gap_heatmap{suffix}.png")


def summarize_deepset(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["n_block", "data_size", "model"]
    loss_summary = summarize_metric(df, group_cols, "loss", "loss")
    sym_summary = summarize_metric(df, group_cols, "symmetry_error", "symmetry_error")
    params = df.groupby(group_cols, as_index=False)["n_params"].first()
    out = loss_summary.merge(sym_summary, on=group_cols).merge(params, on=group_cols)
    return out


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
    group_cols = ["model depth", "data_size", "model"]

    # (1) Accuracy vs data_size, n_block별 facet (이상치 seed 제외)
    acc_trimmed = trim_minmax_rows(df, group_cols, "accuracy")
    g = sns.relplot(
        data=acc_trimmed, x="data_size", y="accuracy", hue="model", col="model depth",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.15,
    )
    g.set(xscale="log")
    g.set_axis_labels("Training data size", "Test accuracy")
    g.set_titles("model depth = {col_name}")
    g.fig.suptitle(
        "Task 2: Accuracy vs Data Size    "
        "(mean \u00b1 sd)",
        y=1.06,
    )
    save_figure("task2_graph_accuracy.png")

    # (2) Symmetry error vs data_size (log scale), 이상치 seed 제외
    sym_trimmed = trim_minmax_rows(df, group_cols, "symmetry_error")
    g = sns.relplot(
        data=sym_trimmed, x="data_size", y="symmetry_error", hue="model", col="model depth",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=4.2, aspect=1.15,
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Symmetry error")
    g.set_titles("model depth = {col_name}")
    g.fig.suptitle(
        "Task 2: Symmetry Error vs Data Size    "
        "(mean \u00b1 sd)",
        y=1.06,
    )
    save_figure("task2_graph_symmetry_error.png")

    if not has_params:
        return

    # (3) 파라미터 효율성: n_block별 최대 data_size에서 (n_params, mean accuracy), 이상치 제외
    largest = df[df["data_size"] == df.groupby("model depth")["data_size"].transform("max")].copy()
    largest_trimmed = trim_minmax_rows(largest, ["model", "model depth"], "accuracy")
    agg = largest_trimmed.groupby(["model", "model depth"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), n_params=("n_params", "first")
    )
    agg["model depth"] = agg["model depth"].astype(str)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=agg, x="n_params", y="mean_accuracy", hue="model", style="model depth",
        s=200, palette=palette,
    )
    plt.xscale("log")
    plt.xlabel("Trainable parameters")
    plt.ylabel("Mean test accuracy (data_size=4110)")
    plt.title("Task 2: Parameter Efficiency Frontier")
    save_figure("task2_graph_parameter_efficiency.png")


def summarize_graph(df: pd.DataFrame, has_params: bool) -> pd.DataFrame:
    group_cols = ["model depth", "data_size", "model"]
    loss_summary = summarize_metric(df, group_cols, "loss", "loss")
    acc_summary = summarize_metric(df, group_cols, "accuracy", "accuracy")
    sym_summary = summarize_metric(df, group_cols, "symmetry_error", "symmetry_error")
    out = loss_summary.merge(acc_summary, on=group_cols).merge(sym_summary, on=group_cols)
    if has_params:
        params = df.groupby(group_cols, as_index=False)["n_params"].first()
        out = out.merge(params, on=group_cols)
    return out


# ------------------------------------------------------------------
# Task 3: NBody (N_total 고정, k_sym in {2,3,4} 스윕 -> S_k symmetry)
#
# [2026-07 갱신] 기존 N=3/S_2 고정 실험에서는 vanilla와 shared의 test loss 차이가
# 유의미하지 않았음(symmetry error만 낮고 예측 정확도는 그대로). 이를 "대칭성이 시스템에서
# 차지하는 비중이 너무 작다"는 가설로 보고, N_total=4를 고정한 채 동일 질량 입자 수
# k_sym in {2,3,4}를 스윕하도록 실험을 재설계함. k_sym=2는 기존과 동일 비중(baseline),
# k_sym=4는 완전 대칭(S_4)인 극단 케이스에 해당함.
# ------------------------------------------------------------------

def load_three_body() -> pd.DataFrame:
    df = pd.read_csv(THREE_BODY_RESULT_PATH)
    df["size_tag"] = pd.Categorical(df["size_tag"], ["small", "medium", "large"], ordered=True)
    if "k_sym" not in df.columns:
        # 구 스키마(N=3, k=2 고정) 호환: k_sym/n_total이 없으면 기존 실험값으로 간주
        print("[warn] 3Body 결과에 k_sym 컬럼이 없어 구 스키마(k_sym=2, n_total=3)로 간주합니다.")
        df["k_sym"] = 2
        df["n_total"] = 3
    return df


# def load_three_body_ood() -> pd.DataFrame | None:
#     if not THREE_BODY_OOD_PATH.exists():
#         print(
#             "[warn] OOD 결과 파일이 없습니다. "
#             "3BodyProblem/evaluate_ood.py 를 먼저 실행하면 OOD 비교 그림도 생성됩니다."
#         )
#         return None
#     df = pd.read_csv(THREE_BODY_OOD_PATH)
#     df["size_tag"] = pd.Categorical(df["size_tag"], ["small", "medium", "large"], ordered=True)
#     if "k_sym" not in df.columns:
#         print(
#             "[warn] OOD 결과에 k_sym 컬럼이 없습니다 (evaluate_ood.py가 아직 k_sym 스윕에 맞춰 "
#             "갱신되지 않은 것으로 보입니다). k_sym=2(기존 baseline)로 간주하여 처리하되, "
#             "다른 k_sym 조건과의 비교는 생략합니다."
#         )
#         df["k_sym"] = 2
#         df["n_total"] = 3
#     return df


def plot_three_body(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())
    palette = model_palette(models)
    group_cols = ["k_sym", "size_tag", "data_size", "model"]

    # (1) Loss vs data_size, row=k_sym x col=size_tag facet (이상치 seed 제외)
    loss_trimmed = trim_minmax_rows(df, group_cols, "loss")
    g = sns.relplot(
        data=loss_trimmed, x="data_size", y="loss", hue="model",
        col="size_tag", row="k_sym",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=3.8, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Test loss (MSE)")
    g.set_titles(row_template="k_sym = {row_name}", col_template="size_tag = {col_name}")
    g.fig.suptitle(
        "Task 3: Test Loss vs Data Size (row: k_sym, col: size_tag)    "
        "(mean \u00b1 sd)",
        y=1.02,
    )
    save_figure("task3_3body_loss.png")

    # (2) In-distribution symmetry error vs data_size (log scale), 이상치 seed 제외
    sym_trimmed = trim_minmax_rows(df, group_cols, "symmetry_error")
    g = sns.relplot(
        data=sym_trimmed, x="data_size", y="symmetry_error", hue="model",
        col="size_tag", row="k_sym",
        kind="line", marker="o", palette=palette, errorbar="sd",
        height=3.8, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Training data size", "Symmetry error (in-distribution)")
    g.set_titles(row_template="k_sym = {row_name}", col_template="size_tag = {col_name}")
    g.fig.suptitle(
        "Task 3: In-Distribution Symmetry Error vs Data Size (row: k_sym, col: size_tag)    "
        "(mean \u00b1 sd)",
        y=1.02,
    )
    save_figure("task3_3body_symmetry_error_id.png")

    # (3) 파라미터 효율성: k_sym별 facet (shared의 파라미터 수는 k_sym에 따라 달라짐)
    max_data_size = df["data_size"].max()
    largest = df[df["data_size"] == max_data_size].copy()
    largest_trimmed = trim_minmax_rows(largest, ["k_sym", "model", "size_tag"], "loss")
    agg = largest_trimmed.groupby(["k_sym", "model", "size_tag"], as_index=False, observed=True).agg(
        mean_loss=("loss", "mean"), n_params=("n_params", "first")
    )
    agg["size_tag"] = agg["size_tag"].astype(str)
    g = sns.relplot(
        data=agg, x="n_params", y="mean_loss", hue="model", style="size_tag",
        col="k_sym", kind="scatter", palette=palette, s=180, height=5, aspect=1,
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Trainable parameters", f"Mean test loss (data_size={int(max_data_size)})")
    g.set_titles("k_sym = {col_name}")
    g.fig.suptitle(
        "Task 3: Parameter Efficiency Frontier by k_sym", y=1.06,
    )
    save_figure("task3_3body_parameter_efficiency.png")

    # (4) 핵심 가설 검증: 대칭 비중(k_sym / n_total)에 따른 vanilla-shared loss 격차
    plot_three_body_k_sweep_gap(df)


def plot_three_body_k_sweep_gap(df: pd.DataFrame) -> None:
    """
    핵심 가설 검증 그림: 대칭 비중(k_sym / n_total)이 커질수록 vanilla와 shared(비augmentation)
    사이의 test loss 격차(양수 = shared 유리)가 커지는지를 직접 시각화한다.
    각 (k_sym, size_tag, data_size) 그룹에서 이상치 seed를 제외한 평균으로 격차를 계산한 뒤,
    data_size 전반에 걸쳐 다시 평균(및 seed 제외 이후의 표준편차)을 낸다.
    """
    n_total = df["n_total"].iloc[0]
    sub = df[df["model"].isin(["vanilla", "shared"])].copy()
    group_cols = ["k_sym", "size_tag", "data_size", "model"]
    trimmed = trim_minmax_rows(sub, group_cols, "loss")
    means = trimmed.groupby(group_cols, as_index=False, observed=True)["loss"].mean()

    pivot = means.pivot_table(
        index=["k_sym", "size_tag", "data_size"], columns="model", values="loss"
    ).reset_index()
    if "vanilla" not in pivot.columns or "shared" not in pivot.columns:
        print("[warn] vanilla 또는 shared 결과가 없어 k_sym-loss gap 그림을 건너뜁니다.")
        return
    pivot["gap"] = pivot["vanilla"] - pivot["shared"]
    pivot["k_ratio"] = pivot["k_sym"] / n_total

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=pivot, x="k_ratio", y="gap", hue="size_tag", style="size_tag",
        marker="o", markersize=9, errorbar="sd",
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Symmetry ratio (k_sym / n_total)")
    plt.ylabel("Mean(Vanilla loss) \u2212 Mean(Shared loss)")
    plt.title(
        "Task 3: Symmetry Ratio vs Vanilla\u2212Shared Loss Gap"
    )
    save_figure("task3_3body_k_ratio_loss_gap.png")


def summarize_three_body(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["k_sym", "size_tag", "data_size", "model"]
    loss_summary = summarize_metric(df, group_cols, "loss", "loss")
    sym_summary = summarize_metric(df, group_cols, "symmetry_error", "symmetry_error")
    params = df.groupby(group_cols, as_index=False, observed=True)["n_params"].first()
    out = loss_summary.merge(sym_summary, on=group_cols).merge(params, on=group_cols)
    return out


# def plot_three_body_ood(id_df: pd.DataFrame, ood_df: pd.DataFrame) -> pd.DataFrame | None:
#     """
#     추가 실험(OOD 검증)의 핵심 그림. 이상치 seed를 제외한 버전.

#     augmentation은 훈련 분포 "안"에서만 대칭성을 근사적으로 학습하지만,
#     구조적 제약(shared)은 입력 분포와 무관하게 항상 정확히 대칭성을 만족해야 한다는
#     가설을 직접 검증한다.

#     [주의] id_df는 k_sym in {2,3,4} 스윕 결과를 담고 있으나, ood_df가 아직 k_sym 스윕에 맞춰
#     갱신되지 않았다면(evaluate_ood.py 미개정) 여러 k_sym에 걸쳐 (size_tag, data_size, seed, model)
#     조합이 중복되어 merge 시 예기치 않게 행이 늘어날 수 있다. 이를 방지하기 위해, ood_df에
#     k_sym이 여러 값을 갖지 않는 한(즉 단일 k_sym에 대해서만 OOD를 계산한 경우) id_df를
#     ood_df와 동일한 k_sym으로 먼저 필터링한 뒤 merge한다.

#     Returns:
#         merged (pd.DataFrame): id/ood 결과를 (k_sym, size_tag, data_size, seed, model) 기준으로
#             합친 long-format 데이터프레임 (요약 테이블 저장용으로 재사용, trimming 이전 원본).
#             merge할 수 없는 경우 None을 반환한다.
#     """
#     ood_k_syms = sorted(ood_df["k_sym"].unique())
#     if len(ood_k_syms) > 1:
#         print(
#             f"[warn] OOD 결과에 여러 k_sym 값({ood_k_syms})이 섞여 있습니다. "
#             "evaluate_ood.py의 k_sym 스윕 지원 여부를 확인하세요. OOD 비교 그림을 건너뜁니다."
#         )
#         return None

#     ood_k_sym = ood_k_syms[0]
#     id_df_filtered = id_df[id_df["k_sym"] == ood_k_sym].copy()
#     if id_df_filtered.empty:
#         print(
#             f"[warn] ID 결과에 OOD와 동일한 k_sym={ood_k_sym} 조건이 없어 OOD 비교 그림을 건너뜁니다."
#         )
#         return None

#     group_cols = ["size_tag", "data_size", "model"]
#     models = sorted(ood_df["model"].unique())
#     palette = model_palette(models)

#     merged = pd.merge(
#         id_df_filtered[["k_sym", "size_tag", "data_size", "seed", "model", "symmetry_error"]],
#         ood_df[["size_tag", "data_size", "seed", "model", "ood_symmetry_error", "ood_loss"]],
#         on=["size_tag", "data_size", "seed", "model"],
#         how="inner",
#     )

#     # (5) ID vs OOD symmetry error, 모델별 평균 막대 그래프 (log scale)
#     #     symmetry_error / ood_symmetry_error 각각에 대해 (size_tag, data_size, model) 그룹
#     #     안에서 이상치 seed를 독립적으로 제외한 뒤, model 기준으로 다시 합쳐 막대그래프를 그림.
#     id_trimmed = trim_minmax_rows(merged, group_cols, "symmetry_error")
#     ood_trimmed = trim_minmax_rows(merged, group_cols, "ood_symmetry_error")

#     id_long = id_trimmed[["model", "symmetry_error"]].rename(
#         columns={"symmetry_error": "symmetry_error_value"}
#     )
#     id_long["regime"] = "In-distribution"
#     ood_long = ood_trimmed[["model", "ood_symmetry_error"]].rename(
#         columns={"ood_symmetry_error": "symmetry_error_value"}
#     )
#     ood_long["regime"] = "OOD (\u00d73 wider init.)"
#     long_df = pd.concat([id_long, ood_long], ignore_index=True)

#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         data=long_df, x="model", y="symmetry_error_value", hue="regime",
#         estimator="mean", errorbar="sd",
#     )
#     plt.yscale("log")
#     plt.ylabel("Symmetry error (log scale)")
#     plt.title(
#         f"Task 3 (NBody, k_sym={ood_k_sym}): Symmetry Error, In-Distribution vs OOD\n"
#         "(각 조건 그룹 내 이상치 seed 제외)"
#     )
#     plt.xticks(rotation=15)
#     save_figure("task3_3body_ood_symmetry_error_bar.png")

#     # (6) OOD loss vs data_size, size_tag별 facet (이상치 seed 제외)
#     ood_loss_trimmed = trim_minmax_rows(ood_df, group_cols, "ood_loss")
#     g = sns.relplot(
#         data=ood_loss_trimmed, x="data_size", y="ood_loss", hue="model", col="size_tag",
#         kind="line", marker="o", palette=palette, errorbar="sd",
#         height=4.2, aspect=1.1,
#     )
#     g.set(xscale="log", yscale="log")
#     g.set_axis_labels("Training data size", "OOD test loss (MSE)")
#     g.set_titles("size_tag = {col_name}")
#     g.fig.suptitle(
#         f"Task 3 (NBody, k_sym={ood_k_sym}): OOD Test Loss vs Data Size\n"
#         "(mean \u00b1 sd, 그룹별 최댓값/최솟값 seed 제외)",
#         y=1.06,
#     )
#     save_figure("task3_3body_ood_loss.png")

#     return merged


# def summarize_three_body_ood(merged: pd.DataFrame) -> pd.DataFrame:
#     group_cols = ["k_sym", "size_tag", "data_size", "model"]
#     id_sym_summary = summarize_metric(merged, group_cols, "symmetry_error", "id_symmetry_error")
#     ood_sym_summary = summarize_metric(merged, group_cols, "ood_symmetry_error", "ood_symmetry_error")
#     ood_loss_summary = summarize_metric(merged, group_cols, "ood_loss", "ood_loss")
#     out = id_sym_summary.merge(ood_sym_summary, on=group_cols).merge(ood_loss_summary, on=group_cols)
#     return out


# ------------------------------------------------------------------
# Task 3 부가: 궤적 rollout 정성적 비교
#   (단일 대표 조건에 대한 정성적 비교이며 seed 집계가 없으므로 이상치 제외 대상이 아님.)
#
#   [2026-07 갱신] N=3/k=2 고정 EquivariantLinear2D 대신, main.py와 동일한
#   EquivariantLinearGeneral(N_total=4, k_sym 스윕)을 그대로 사용하도록 일반화함.
# ------------------------------------------------------------------

class EquivariantLinearGeneral(nn.Module):
    '''
        main.py의 EquivariantLinearGeneral과 동일한 구조 (rollout 체크포인트 복원용으로 복제).
        N개 입자 중 앞의 k개가 S_k 대칭(동일 질량), 나머지 (N-k)개는 서로 구별되는 입자.
    '''

    def __init__(self, in_channels: int, out_channels: int, k_sym: int, n_total: int):
        super().__init__()
        self.k = k_sym
        self.n = n_total
        self.n_asym = n_total - k_sym
        scale = 1.0 / np.sqrt(in_channels * n_total)

        def p():
            return nn.Parameter(torch.randn(out_channels, in_channels) * scale)

        self.W_self_sym = p()
        self.W_cross_sym = p() if self.k > 1 else None

        self.W_sym_to_asym = nn.ParameterList([p() for _ in range(self.n_asym)])
        self.W_asym_to_sym = nn.ParameterList([p() for _ in range(self.n_asym)])
        self.W_asym_diag = nn.ParameterList([p() for _ in range(self.n_asym)])

        self.asym_pair_index = {}
        cross_params = []
        for j in range(self.n_asym):
            for l in range(self.n_asym):
                if j == l:
                    continue
                self.asym_pair_index[(j, l)] = len(cross_params)
                cross_params.append(p())
        self.W_asym_cross = nn.ParameterList(cross_params)

        self.bias_sym = nn.Parameter(torch.zeros(out_channels))
        self.bias_asym = nn.ParameterList([nn.Parameter(torch.zeros(out_channels)) for _ in range(self.n_asym)])

    def _block(self, i: int, j: int) -> torch.Tensor:
        k = self.k
        if i < k and j < k:
            return self.W_self_sym if i == j else self.W_cross_sym
        if i < k and j >= k:
            return self.W_asym_to_sym[j - k]
        if i >= k and j < k:
            return self.W_sym_to_asym[i - k]
        if i == j:
            return self.W_asym_diag[i - k]
        return self.W_asym_cross[self.asym_pair_index[(i - k, j - k)]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        rows = []
        for i in range(n):
            row_blocks = [self._block(i, j) for j in range(n)]
            rows.append(torch.cat(row_blocks, dim=1))
        weight = torch.cat(rows, dim=0)

        bias_blocks = [self.bias_sym if i < self.k else self.bias_asym[i - self.k] for i in range(n)]
        bias = torch.cat(bias_blocks, dim=0)
        return x @ weight.t() + bias


def build_shared_3body(channel_dims: list[int], k_sym: int, n_total: int = THREE_BODY_N_TOTAL) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(channel_dims) - 1):
        layers.append(
            EquivariantLinearGeneral(channel_dims[idx], channel_dims[idx + 1], k_sym=k_sym, n_total=n_total)
        )
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
    "small": {"vanilla_dims": [THREE_BODY_FLAT_DIM, 48, 48, THREE_BODY_FLAT_DIM], "shared_dims": [4, 16, 16, 4]},
    "medium": {
        "vanilla_dims": [THREE_BODY_FLAT_DIM, 48, 96, 96, 48, THREE_BODY_FLAT_DIM],
        "shared_dims": [4, 16, 24, 24, 16, 4],
    },
    "large": {"vanilla_dims": [THREE_BODY_FLAT_DIM, 96, 192, 48, THREE_BODY_FLAT_DIM], "shared_dims": [4, 32, 64, 16, 4]},
}


def load_three_body_checkpoint(model_name: str, size_tag: str, data_size: int, seed: int, k_sym: int) -> nn.Module:
    config = THREE_BODY_MODEL_CONFIGS[size_tag]
    model = (
        build_shared_3body(config["shared_dims"], k_sym=k_sym)
        if "shared" in model_name
        else build_vanilla_3body(config["vanilla_dims"])
    )
    suffix = f"k{k_sym}_{size_tag}_{data_size}_seed{seed}"
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


def plot_three_body_rollout(k_sym: int, size_tag: str = "large", data_size: int = 8000,
                             seed: int = 0, steps: int = 15) -> None:
    """
    한 대표 설정(k_sym/size_tag/data_size/seed)에 대해 vanilla vs shared 모델의
    자기회귀적(autoregressive) rollout 궤적을 실제 정답 궤적과 정성적으로 비교한다.
    (테스트셋은 shuffle 없이 생성되므로 앞부분 몇 개 샘플은 같은 궤적 안에서 연속적임)
    """
    suffix = f"k{k_sym}_{data_size}_seed{seed}"
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
        print(f"[warn] (k_sym={k_sym}) 테스트셋 앞부분이 연속적이지 않아 rollout 그림을 건너뜁니다.")
        return

    gt_states = [x_test[0].numpy()] + [y_test[i].numpy() for i in range(steps)]

    init_state = x_test[0:1]
    try:
        vanilla_model = load_three_body_checkpoint("vanilla", size_tag, data_size, seed, k_sym=k_sym)
        shared_model = load_three_body_checkpoint("shared", size_tag, data_size, seed, k_sym=k_sym)
    except FileNotFoundError as exc:
        print(f"[warn] (k_sym={k_sym}) 체크포인트를 찾을 수 없어 rollout 그림을 건너뜁니다: {exc}")
        return

    vanilla_states = rollout_model(vanilla_model, init_state, steps=steps)
    shared_states = rollout_model(shared_model, init_state, steps=steps)

    state_sets = [
        ("Ground truth", gt_states),
        ("Vanilla (rollout)", vanilla_states),
        (f"Shared, k_sym={k_sym} (rollout)", shared_states),
    ]
    n_particles = THREE_BODY_N_TOTAL
    colors = plt.cm.tab10(np.linspace(0, 1, n_particles))
    markers = ["o", "s", "^", "D", "v", "P"][:n_particles]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (title, states) in zip(axes, state_sets):
        arr = np.asarray(states)
        pos_dim = n_particles * 2
        positions = arr[:, :pos_dim].reshape(len(states), n_particles, 2)
        for particle_idx in range(n_particles):
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
        f"Task 3: Rollout Comparison "
        f"(k_sym={k_sym}, size_tag={size_tag}, data_size={data_size}, seed={seed})",
        y=1.03,
    )
    plt.tight_layout()
    save_figure(f"task3_3body_rollout_k{k_sym}.png")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    # Task 1: DeepSet
    print("\n=== Task 1: DeepSet (Exclude Min/Max) ===")
    deepset_df = load_deepset()
    save_table(summarize_deepset(deepset_df), "task1_deepset_summary")
    plot_deepset(deepset_df)

    # Task 2: Graph / NCI1
    print("\n=== Task 2: Graph (NCI1) (Exclude Min/Max) ===")
    graph_df, has_params = load_graph()
    save_table(summarize_graph(graph_df, has_params), "task2_graph_summary")
    plot_graph(graph_df, has_params)

    # Task 3: NBody (in-distribution, k_sym 스윕)
    print("\n=== Task 3: NBody (Exclude Min/Max) ===")
    three_body_df = load_three_body()
    save_table(summarize_three_body(three_body_df), "task3_3body_summary")
    plot_three_body(three_body_df)

    # # Task 3: NBody OOD (있으면; k_sym 스윕에 아직 대응하지 않았을 수 있음)
    # ood_df = load_three_body_ood()
    # if ood_df is not None:
    #     merged = plot_three_body_ood(three_body_df, ood_df)
    #     if merged is not None:
    #         save_table(summarize_three_body_ood(merged), "task3_3body_ood_summary")

    # Task 3: rollout 정성적 비교 (체크포인트/데이터 있으면, seed 집계 없음)
    # k_sym 스윕 전체(2, 3, 4)에 대해 각각 생성하여 대칭 비중에 따른 정성적 차이도 함께 보여줌
    available_k_syms = sorted(three_body_df["k_sym"].unique())
    for k_sym in available_k_syms:
        plot_three_body_rollout(k_sym=int(k_sym))

    print("\n모든 그림/표 생성 완료 (이상치 최댓값/최솟값 seed 제외 버전).")
    print(f"그림 저장 위치: {FIG_DIR}")
    print(f"표 저장 위치:   {TABLE_DIR}")


if __name__ == "__main__":
    main()
