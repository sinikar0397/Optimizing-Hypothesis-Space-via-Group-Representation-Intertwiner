"""
3BodyProblem symmetry_error 재계산 스크립트 (재학습 불필요)

배경:
    기존 symmetry_error()는 E_g[L(g . f(x), f(g . x))]를 계산하려 했으나,
    perm으로 permute한 출력을 되돌릴 때 perm의 역순열이 아니라 perm 자체를
    다시 적용하고 있었음. k_sym=2일 때는 항등원 아닌 permutation이 swap
    (involution, g == g^-1) 하나뿐이라 우연히 문제가 없었지만,
    k_sym=3, 4에서는 S_3/S_4의 non-involutive 원소(3-cycle, 4-cycle 등)가
    섞여 샘플링되어 완전히 잘못된 비교가 이루어졌고, 그 결과
    실제 equivariance 위반과 무관하게 symmetry_error가 크게(10^2~10^5) 부풀려짐.

    apply_particle_permutation(x, perm)은 perm[new_pos] = old_idx 컨벤션으로
    out[new_pos] = x[old_idx] 재배치를 수행하므로, 역순열은
    inv[old_idx] = new_pos 로 계산해야 함 (perm이 involution일 때만 inv == perm).

이 스크립트는:
    1. main.py와 동일한 EquivariantLinearGeneral / build_shared / build_vanilla /
       load_data 로직을 재사용해 models/*.pt 체크포인트를 정확히 복원하고,
    2. 수정된(역순열 적용) symmetry_error로 재평가하며,
    3. loss는 참고용으로 동일하게 재계산하고,
    4. 결과를 기존 raw와 동일한 long-format으로 저장한다
       (컬럼명은 symmetry_error_fixed로 구분하여 기존 raw와 merge 가능하게 함).

사용법:
    3BodyProblem/ 디렉토리에서 실행:
        python evaluate_symmetry_fix.py

    결과는 total_result_symmetry_fixed_raw.csv 로 저장됨.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ------------------------------------------------------------------
# main.py 에서 그대로 가져온 상수 / 구조 (동일한 물리·모델 설정 유지)
# ------------------------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

N_TOTAL = 4
K_SYM_LIST = [2, 3, 4]
STATE_DIM_PER_PARTICLE = 4
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
data_size_lst = [500, 2000, 8000]
seed_lst = [0, 1, 2, 3, 4]
model_names = ["vanilla", "vanilla_aug", "shared", "shared_aug"]

G_SAMPLES = 5  # main.py와 동일한 규약

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_data(data_path: Path, k_sym: int, data_size: int, seed: int) -> dict:
    '''main.py의 load_data와 동일 - 재평가에는 X_test/Y_test만 필요'''
    suffix = f"k{k_sym}_{data_size}_seed{seed}"
    return {
        "X_test": torch.load(data_path / f"X_test_{suffix}.pt"),
        "Y_test": torch.load(data_path / f"Y_test_{suffix}.pt"),
    }


# --- permutation 유틸 (수정된 부분: invert_permutation 추가) ---

def sample_permutation(k_sym: int, exclude_identity: bool = False) -> list:
    perm = list(range(k_sym))
    if not exclude_identity:
        random.shuffle(perm)
        return perm
    while True:
        random.shuffle(perm)
        if perm != list(range(k_sym)):
            return perm


def invert_permutation(perm: list) -> list:
    '''
        perm[new_pos] = old_idx 컨벤션의 역순열을 구함.
        inv[old_idx] = new_pos 이며, apply_particle_permutation(out, inv)가
        perm 적용 이전 순서를 정확히 복원하도록 함.
        (perm이 involution인 경우에만 inv == perm)
    '''
    inv = [0] * len(perm)
    for new_pos, old_idx in enumerate(perm):
        inv[old_idx] = new_pos
    return inv


def apply_particle_permutation(x: torch.Tensor, perm: list, per_particle_dim: int = STATE_DIM_PER_PARTICLE) -> torch.Tensor:
    original = x.clone()
    out = x.clone()
    for new_pos, old_idx in enumerate(perm):
        out[..., new_pos * per_particle_dim:(new_pos + 1) * per_particle_dim] = \
            original[..., old_idx * per_particle_dim:(old_idx + 1) * per_particle_dim]
    return out


# --- 모델 구조 (main.py와 동일, 체크포인트 복원용) ---

class EquivariantLinearGeneral(nn.Module):
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


def build_vanilla(layer_dims):
    layers = []
    for idx in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
        if idx < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_model(model_name: str, size_tag: str, k_sym: int) -> nn.Module:
    if model_name.startswith("shared"):
        return build_shared(MODEL_CONFIGS[size_tag]["shared_dims"], k_sym=k_sym)
    return build_vanilla(MODEL_CONFIGS[size_tag]["vanilla_dims"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------
# 수정된 symmetry_error (역순열 적용)
# ------------------------------------------------------------------

@torch.no_grad()
def symmetry_error_fixed(model, x, k_sym: int, batch_size=1024, g_samples=G_SAMPLES):
    model.eval()
    sample_errs = []
    for _ in range(g_samples):
        perm = sample_permutation(k_sym, exclude_identity=True)
        inv_perm = invert_permutation(perm)  # 수정 지점: perm이 아니라 역순열로 되돌림
        values = []
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device)
            out1 = model(batch)
            out2 = model(apply_particle_permutation(batch, perm))
            out2_permuted_back = apply_particle_permutation(out2, inv_perm)
            err = ((out1 - out2_permuted_back) ** 2).mean(dim=-1)
            values.append(err.cpu())
        sample_errs.append(torch.cat(values).mean().item())
    return float(np.mean(sample_errs))


@torch.no_grad()
def evaluate_fixed(model, x_test, y_test, k_sym: int, batch_size=1024):
    model.eval()
    preds = []
    for start in range(0, len(x_test), batch_size):
        batch_x = x_test[start:start + batch_size].to(device)
        preds.append(model(batch_x).cpu())
    pred = torch.cat(preds, dim=0)
    loss = nn.MSELoss()(pred, y_test).item()
    sym_err = symmetry_error_fixed(model, x_test, k_sym=k_sym)
    return loss, sym_err


# ------------------------------------------------------------------
# 재평가 iteration (재학습 없음, 체크포인트만 로드)
# ------------------------------------------------------------------

def main():
    records = []

    total = len(K_SYM_LIST) * len(size_tag_lst) * len(data_size_lst) * len(seed_lst) * len(model_names)
    bar = tqdm(total=total, desc="symmetry_error 재평가 중")

    for k_sym in K_SYM_LIST:
        for size_tag in size_tag_lst:
            for data_size in data_size_lst:
                for seed in seed_lst:
                    # 체크포인트가 seed 의존적 랜덤 연산(perm 샘플링) 결과와 재현성을 갖도록 고정
                    set_seed(seed)

                    data_suffix = f"k{k_sym}_{data_size}_seed{seed}"
                    x_test_path = DATA_DIR / f"X_test_{data_suffix}.pt"
                    if not x_test_path.exists():
                        bar.update(len(model_names))
                        continue

                    test_data = load_test_data(DATA_DIR, k_sym=k_sym, data_size=data_size, seed=seed)
                    x_test, y_test = test_data["X_test"], test_data["Y_test"]

                    for model_name in model_names:
                        ckpt_suffix = f"k{k_sym}_{size_tag}_{data_size}_seed{seed}"
                        ckpt_path = MODEL_DIR / f"{model_name}_model_{ckpt_suffix}.pt"
                        if not ckpt_path.exists():
                            bar.update(1)
                            continue

                        model = build_model(model_name, size_tag, k_sym).to(device)
                        state_dict = torch.load(ckpt_path, map_location=device)
                        model.load_state_dict(state_dict)
                        n_params = count_parameters(model)

                        loss, sym_err_fixed = evaluate_fixed(model, x_test, y_test, k_sym=k_sym)

                        records.append({
                            "k_sym": k_sym,
                            "n_total": N_TOTAL,
                            "size_tag": size_tag,
                            "data_size": data_size,
                            "seed": seed,
                            "model": model_name,
                            "loss": loss,
                            "symmetry_error_fixed": sym_err_fixed,
                            "n_params": n_params,
                        })
                        bar.update(1)

    bar.close()

    df = pd.DataFrame(records)
    out_csv = ROOT / "total_result_symmetry_fixed_raw.csv"
    out_pkl = ROOT / "total_result_symmetry_fixed_raw.pkl"
    df.to_csv(out_csv, index=False)
    df.to_pickle(out_pkl)

    print(f"\n총 {len(df)}개 row 저장 완료: {out_csv}")
    print("\n=== k_sym x model 별 평균 symmetry_error_fixed ===")
    print(df.groupby(["k_sym", "model"])["symmetry_error_fixed"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
