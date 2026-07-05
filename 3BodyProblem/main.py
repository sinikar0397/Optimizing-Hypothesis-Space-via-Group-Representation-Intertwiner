import copy
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


#
# defining basic part
#
device = torch.device(
    "mps"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#
# HyperParameters
#
# 이 스크립트는 3BodyProblem/main.py의 일반화 버전임.
# 기존 실험은 N=3, k=2(S_2)로 고정되어 있었으나, symmetry의 영향력이 너무 작다는 관측(test loss 차이 미미)에
# 따라 "대칭 비중(k/N)"을 독립변수로 스윕하기 위해 N, k를 파라미터화함.
#
# 질량 배치 규칙: 앞의 k개 입자는 동일 질량(=10.0)을 가져 S_k 대칭을 구성하고,
# 나머지 (N-k)개 입자는 서로 다른 질량(ASYM_MASS_POOL에서 순서대로 채택)을 가져 완전히 비대칭으로 만듦.
# 이렇게 해야 실제 대칭군이 정확히 S_k로 고정되고, 우연히 추가 대칭이 생기는 것을 방지함.

DT = 0.01
K = 1.0
EPSILON = 0.5

N_TOTAL = 4  # 전체 입자 수 (고정: 시뮬레이션 비용을 억제하기 위해 N은 늘리지 않고 k/N 비율만 스윕)
K_SYM_LIST = [2, 3, 4]  # 대칭 입자 수 스윕 (k=2: 기존과 동일 비중, k=4: 완전 대칭 S_4)
SYM_MASS = 10.0
ASYM_MASS_POOL = [20.0, 25.0, 30.0]  # N_TOTAL - k_sym 개만큼 앞에서부터 사용


def get_masses(k_sym: int, n_total: int = N_TOTAL) -> np.ndarray:
    '''
        k_sym개의 동일 질량 입자(S_k 대칭) + (n_total - k_sym)개의 서로 다른 질량 입자로 구성된
        질량 배열을 반환함. 비대칭 입자들은 서로 다른 질량 값을 가지므로 실제 대칭군은 정확히 S_k가 됨.
    '''
    n_asym = n_total - k_sym
    assert n_asym <= len(ASYM_MASS_POOL), "ASYM_MASS_POOL이 부족합니다. 값을 더 추가하세요."
    masses = [SYM_MASS] * k_sym + ASYM_MASS_POOL[:n_asym]
    return np.array(masses, dtype=np.float32)


# shared_dims는 입자당 채널 폭(block 내부 in/out channel)이라 N에 의존하지 않음.
# vanilla_dims만 입력/출력 차원이 4 * N_TOTAL이 되도록 조정함.
STATE_DIM_PER_PARTICLE = 4  # (x, y, vx, vy)
FLAT_DIM = STATE_DIM_PER_PARTICLE * N_TOTAL

MODEL_CONFIGS = {
    "small": {
        "vanilla_dims": [FLAT_DIM, 48, 48, FLAT_DIM],
        "shared_dims": [4, 16, 16, 4],
    },
    "medium": {
        "vanilla_dims": [FLAT_DIM, 48, 96, 96, 48, FLAT_DIM],
        "shared_dims": [4, 16, 24, 24, 16, 4],
    },
    "large": {
        "vanilla_dims": [FLAT_DIM, 96, 192, 48, FLAT_DIM],
        "shared_dims": [4, 32, 64, 16, 4],
    },
}

size_tag_lst = ["small", "medium", "large"]
data_size_lst = [500, 2000, 8000]  # 학습 비용 축소를 위해 최대값을 40000 -> 8000으로 축소

# --- multi-seed 설정 (타 Task와 동일하게 유지) ---
seed_lst = [0, 1, 2, 3, 4]

MAX_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0

G_SAMPLES = 5  # symmetry error 계산 시 군 작용 샘플 수 (다른 Task와 동일한 규약)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


#
# Dataset
#

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


def generate_dataset(num_trajectories: int, masses: np.ndarray, steps_per_traj: int = 1000):
    '''
        N-body trajectory 데이터를 생성함 (N = len(masses)).

        Args:
            num_trajectories (int) : 생성할 궤적 개수
            masses (np.ndarray) : 각 입자의 질량 (get_masses로 생성)
            steps_per_traj (int) : 궤적당 시뮬레이션 스텝 수

        Returns:
            (X, Y) (tuple of tensor) : 입력/타겟 state 쌍
    '''
    n = len(masses)
    all_x, all_y = [], []
    for _ in tqdm(range(num_trajectories), desc="Generate trajectories", leave=False):
        x = np.random.randn(n, 2).astype(np.float32) * 10
        v = np.random.randn(n, 2).astype(np.float32) * 5
        v -= v.mean(axis=0, keepdims=True)

        traj = []
        for _ in range(steps_per_traj):
            current_state = np.concatenate([x.flatten(), v.flatten()]).astype(np.float32)
            traj.append(current_state)
            a = get_acceleration(x, masses, K, EPSILON)
            v += a * DT
            x += v * DT

        traj = np.asarray(traj, dtype=np.float32)
        all_x.append(traj[:-10:10])
        all_y.append(traj[10::10])

    X = torch.from_numpy(np.concatenate(all_x)).to(device)
    Y = torch.from_numpy(np.concatenate(all_y)).to(device)
    return X, Y


def generate_and_save_data(data_path: Path, k_sym: int, data_size: int, seed: int) -> None:
    '''
        N-body Task에 맞는 Dataset을 저장함 (k_sym에 따라 질량 배치가 달라짐).
        (k_sym, data_size, seed) 조합별로 독립적인 파일을 생성함.
        이미 해당 조합의 파일이 존재하면 재생성하지 않음.
    '''
    suffix = f"k{k_sym}_{data_size}_seed{seed}"
    x_train_path = data_path / f"X_train_{suffix}.pt"
    if x_train_path.exists():
        return  # 이미 생성된 데이터 재사용

    masses = get_masses(k_sym)

    rng_state = np.random.get_state()
    np.random.seed(seed)

    num_train_traj = data_size // 1000 + 1
    num_test_traj = (data_size // 5) // 1000 + 1

    X_train, Y_train = generate_dataset(num_trajectories=num_train_traj, masses=masses, steps_per_traj=10000)
    X_test, Y_test = generate_dataset(num_trajectories=num_test_traj, masses=masses, steps_per_traj=10000)

    X_train, Y_train = X_train[:data_size], Y_train[:data_size]
    X_test, Y_test = X_test[: data_size // 5], Y_test[: data_size // 5]

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train.cpu(), Y_train.cpu(), test_size=0.2, random_state=seed
    )

    np.random.set_state(rng_state)  # 전역 random state 오염 방지

    torch.save(X_train, data_path / f"X_train_{suffix}.pt")
    torch.save(X_test.cpu(), data_path / f"X_test_{suffix}.pt")
    torch.save(X_valid, data_path / f"X_valid_{suffix}.pt")
    torch.save(Y_train, data_path / f"Y_train_{suffix}.pt")
    torch.save(Y_test.cpu(), data_path / f"Y_test_{suffix}.pt")
    torch.save(Y_valid, data_path / f"Y_valid_{suffix}.pt")


def load_data(data_path: Path, k_sym: int, data_size: int, seed: int) -> dict:
    '''
        N-body Task에 맞는 Dataset을 불러옴 (k_sym, seed별 파일)
    '''
    suffix = f"k{k_sym}_{data_size}_seed{seed}"
    return {
        "X_train": torch.load(data_path / f"X_train_{suffix}.pt"),
        "X_test": torch.load(data_path / f"X_test_{suffix}.pt"),
        "X_valid": torch.load(data_path / f"X_valid_{suffix}.pt"),
        "Y_train": torch.load(data_path / f"Y_train_{suffix}.pt"),
        "Y_test": torch.load(data_path / f"Y_test_{suffix}.pt"),
        "Y_valid": torch.load(data_path / f"Y_valid_{suffix}.pt"),
    }


#
# defining model
#

def sample_permutation(k_sym: int, exclude_identity: bool = False) -> list:
    '''
        S_{k_sym}에서 permutation 하나를 랜덤 샘플링함.
        exclude_identity=True면 항등원이 아닌 permutation만 샘플링 (symmetry error 계산용).
        k_sym=2일 때는 항등원을 제외하면 swap 하나뿐이므로 결정론적으로 swap을 반환함.
    '''
    perm = list(range(k_sym))
    if not exclude_identity:
        random.shuffle(perm)
        return perm

    while True:
        random.shuffle(perm)
        if perm != list(range(k_sym)):
            return perm


def apply_particle_permutation(x: torch.Tensor, perm: list, per_particle_dim: int = STATE_DIM_PER_PARTICLE) -> torch.Tensor:
    '''
        첫 k_sym개 입자(대칭군 대상)의 state 블록을 perm에 따라 재배치함.
        나머지 (n_total - k_sym)개 입자는 대칭군 대상이 아니므로 그대로 유지됨.

        Args:
            x (Tensor) : (..., n_total * per_particle_dim) 형태의 state
            perm (list) : len(perm) = k_sym인 permutation (perm[new_pos] = old_idx)
    '''
    original = x.clone()
    out = x.clone()
    for new_pos, old_idx in enumerate(perm):
        out[..., new_pos * per_particle_dim:(new_pos + 1) * per_particle_dim] = \
            original[..., old_idx * per_particle_dim:(old_idx + 1) * per_particle_dim]
    return out


class EquivariantLinearGeneral(nn.Module):
    '''
        N개 입자 중 앞의 k개가 S_k 대칭(동일 질량), 나머지 (N-k)개는 서로 구별되는 입자일 때의
        equivariant linear layer. k=2, n_total=3인 경우 기존 EquivariantLinear2D(A,B,C,D,E)와 정확히 일치함.

        파라미터 공유 규칙:
        - W_self_sym       : 대칭 입자의 self-block (모든 대칭 입자가 공유)
        - W_cross_sym      : 서로 다른 두 대칭 입자 사이의 off-diagonal block (공유)
        - W_sym_to_asym[j] : 임의의 대칭 입자 -> 비대칭 입자 j로의 block (대칭 입자 쪽으로 공유, j마다 별도)
        - W_asym_to_sym[j] : 비대칭 입자 j -> 임의의 대칭 입자로의 block (대칭 입자 쪽으로 공유, j마다 별도)
        - W_asym_diag[j]   : 비대칭 입자 j의 self-block (공유 없음)
        - W_asym_cross[j,l]: 서로 다른 두 비대칭 입자 j,l 사이의 block (공유 없음, vanilla와 동일)
        - bias_sym         : 대칭 입자들이 공유하는 bias
        - bias_asym[j]     : 비대칭 입자 j 고유의 bias
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

        # 서로 다른 비대칭 입자 쌍 (j, l), j != l 마다 별도 파라미터 (공유 없음)
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
        # i >= k and j >= k (둘 다 비대칭)
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

        bias_blocks = []
        for i in range(n):
            bias_blocks.append(self.bias_sym if i < self.k else self.bias_asym[i - self.k])
        bias = torch.cat(bias_blocks, dim=0)

        return x @ weight.t() + bias


def build_shared(channel_dims, k_sym: int, n_total: int = N_TOTAL):
    layers = []
    for idx in range(len(channel_dims) - 1):
        layers.append(EquivariantLinearGeneral(channel_dims[idx], channel_dims[idx + 1], k_sym=k_sym, n_total=n_total))
        if idx < len(channel_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# vanilla model

def build_vanilla(layer_dims):
    layers = []
    for idx in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
        if idx < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_batch_size(train_size: int) -> int:
    if train_size <= 2000:
        return 128
    if train_size <= 10000:
        return 256
    if train_size <= 40000:
        return 512
    return 1024


def build_loaders(x_train, y_train, x_valid, y_valid):
    batch_size = pick_batch_size(len(x_train))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=max(batch_size, 512), shuffle=False)
    return train_loader, valid_loader, batch_size


def augment_batch(batch_x: torch.Tensor, batch_y: torch.Tensor, k_sym: int) -> tuple:
    '''
        매 batch마다 50% 확률로 첫 k_sym개 입자에 대해 랜덤 permutation을 적용해 "대체"함
        (Task1의 augmentation과 동일한 철학: 확률적 치환, batch size 유지).
        실제 대칭군 S_{k_sym}만 반영하며, 나머지 비대칭 입자는 건드리지 않음.
    '''
    if torch.rand(1).item() < 0.5:
        perm = sample_permutation(k_sym)
        return apply_particle_permutation(batch_x, perm), apply_particle_permutation(batch_y, perm)
    return batch_x, batch_y


#
# Training Function
#

def train(model, train_loader, valid_loader, epochs=MAX_EPOCHS, lr=1e-3,
          patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA,
          augment=False, k_sym=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_log = []
    valid_log = []
    best_valid = float("inf")
    best_state = None
    patience_counter = 0

    bar = tqdm(range(epochs), desc="training", leave=False)
    for epoch in bar:
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if augment:
                batch_x, batch_y = augment_batch(batch_x, batch_y, k_sym=k_sym)

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
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
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
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"train_loss": train_log, "valid_loss": valid_log}


@torch.no_grad()
def symmetry_error(model, x, k_sym: int, batch_size=1024, g_samples=G_SAMPLES):
    '''
        E_g[L(g . f(x), f(g . x))]를 g_samples개의 랜덤 permutation(항등원 제외)에 대해 평균냄.
        k_sym=2인 경우 항등원 제외 permutation은 swap 하나뿐이므로 기존 구현과 동치임.
    '''
    model.eval()
    sample_errs = []
    for _ in range(g_samples):
        perm = sample_permutation(k_sym, exclude_identity=True)
        values = []
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device)
            out1 = model(batch)
            out2 = model(apply_particle_permutation(batch, perm))
            out1_permuted_back = apply_particle_permutation(out1, perm)
            err = ((out2 - out1_permuted_back) ** 2).mean(dim=-1)
            values.append(err.cpu())
        sample_errs.append(torch.cat(values).mean().item())
    return float(np.mean(sample_errs))


@torch.no_grad()
def evaluate(model, x_test, y_test, k_sym: int, batch_size=1024):
    model.eval()
    preds = []
    for start in range(0, len(x_test), batch_size):
        batch_x = x_test[start:start + batch_size].to(device)
        preds.append(model(batch_x).cpu())
    pred = torch.cat(preds, dim=0)
    loss = nn.MSELoss()(pred, y_test).item()
    sym_err = symmetry_error(model, x_test, k_sym=k_sym)
    return loss, sym_err


#
# training iteration
#

# raw 결과를 long-format으로 보존 (mean/CI로 바로 뭉개지 않음)
records = []

builders = {
    "vanilla": lambda size_tag, k_sym: build_vanilla(MODEL_CONFIGS[size_tag]["vanilla_dims"]),
    "vanilla_aug": lambda size_tag, k_sym: build_vanilla(MODEL_CONFIGS[size_tag]["vanilla_dims"]),
    "shared": lambda size_tag, k_sym: build_shared(MODEL_CONFIGS[size_tag]["shared_dims"], k_sym=k_sym),
    "shared_aug": lambda size_tag, k_sym: build_shared(MODEL_CONFIGS[size_tag]["shared_dims"], k_sym=k_sym),
}
aug_flags = {"vanilla": False, "vanilla_aug": True, "shared": False, "shared_aug": True}

generate_data = True

k_sym_bar = tqdm(K_SYM_LIST, desc="k_sym iterating...")
for k_sym in k_sym_bar:
    size_tag_bar = tqdm(size_tag_lst, desc="size tag iterating...", leave=False)
    for size_tag in size_tag_bar:
        data_size_bar = tqdm(data_size_lst, desc="data size iterating...", leave=False)
        for data_size in data_size_bar:

            seed_bar = tqdm(seed_lst, desc="seed iterating...", leave=False)
            for seed in seed_bar:
                k_sym_bar.set_description(f"k_sym={k_sym}")
                size_tag_bar.set_description(f"size_tag={size_tag}")
                data_size_bar.set_description(f"data_size={data_size}")
                seed_bar.set_description(f"seed={seed}")

                # --- seed마다 모델 초기화를 모두 다시 함 ---
                set_seed(seed)

                # --- 데이터는 (k_sym, data_size, seed)에만 의존 -> 이미 있으면 재생성하지 않음 ---
                if generate_data:
                    generate_and_save_data(data_path=DATA_DIR, k_sym=k_sym, data_size=data_size, seed=seed)
                dataset = load_data(data_path=DATA_DIR, k_sym=k_sym, data_size=data_size, seed=seed)

                x_train, x_valid, x_test = dataset["X_train"], dataset["X_valid"], dataset["X_test"]
                y_train, y_valid, y_test = dataset["Y_train"], dataset["Y_valid"], dataset["Y_test"]

                train_loader, valid_loader, batch_size = build_loaders(x_train, y_train, x_valid, y_valid)

                for model_name in ["vanilla", "vanilla_aug", "shared", "shared_aug"]:
                    set_seed(seed)  # 모델별로도 동일 seed에서 초기화 시작하도록 재고정
                    model = builders[model_name](size_tag, k_sym).to(device)
                    n_params = count_parameters(model)

                    train(
                        model=model,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        epochs=MAX_EPOCHS,
                        lr=1e-3,
                        patience=EARLY_STOPPING_PATIENCE,
                        min_delta=EARLY_STOPPING_MIN_DELTA,
                        augment=aug_flags[model_name],
                        k_sym=k_sym,
                    )

                    test_loss, sym_err = evaluate(model, x_test, y_test, k_sym=k_sym)

                    records.append({
                        "k_sym": k_sym,
                        "n_total": N_TOTAL,
                        "size_tag": size_tag,
                        "data_size": data_size,
                        "seed": seed,
                        "model": model_name,
                        "loss": test_loss,
                        "symmetry_error": sym_err,
                        "n_params": n_params,
                    })

                    suffix = f"k{k_sym}_{size_tag}_{data_size}_seed{seed}"
                    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_model_{suffix}.pt")

                # 매 seed 루프마다 중간 저장 -> 중간에 죽어도 데이터 유실 최소화
                pd.DataFrame(records).to_pickle(LOG_DIR / "total_result_multiseed_raw.pkl")

#
# 최종 raw 결과 저장 (long-format DataFrame)
#
df = pd.DataFrame(records)
df.to_pickle(ROOT / "total_result_multiseed_raw.pkl")
df.to_csv(ROOT / "total_result_multiseed_raw.csv", index=False)

print(f"총 {len(df)}개 row 저장 완료 (k_sym x size_tag x data_size x seed x model 조합)")