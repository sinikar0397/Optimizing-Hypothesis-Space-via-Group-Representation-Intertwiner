import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

#
# defining device
#
device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')

#
# Dataset
#
def generate_invariant_data(num_of_data, n_blocks, phi=lambda x: np.sin(x) + np.exp(x) + np.log(np.abs(x)), device=device):
    '''
        논문에서 정의한 첫번째 Task에 맞는 Dataset을 반환함
        (이 함수 쓰지 말고 generate_and_save_data를 쓸 것.)

        Args:
            num_of_data (int) : 데이터셋 크기
            phi (function | int -> int) : 각 입력에 취할 함수
            n_blocks (int) : 한 데이터에서 X 크기
            device (str) : 데이터를 올릴 device

        Returns:
            dataset (tuple of tensor) :  (X data (num_of_data, n_blocks), Y data (num_of_data))
    '''
    phi = np.vectorize(phi)
    X = np.random.uniform(-2, 2, size=(num_of_data, n_blocks))
    Y = phi(X).sum(axis=1)

    return torch.tensor(X, dtype=torch.float32, device=device), \
           torch.tensor(Y, dtype=torch.float32, device=device)


def generate_and_save_data(data_path, n_block=5, data_size=10000, seed=0):
    '''
        논문에서 정의한 첫번째 Task에 맞는 Dataset을 저장함.
        seed별로 독립적인 파일을 생성함 (multi-seed 실험을 위해 파일명에 seed 포함).

        Args:
            data_path (str) : 데이터를 저장할 위치
            n_block (int) : 한 데이터에서 X 크기
            data_size (int) : 데이터셋 크기
            seed (int) : 이 데이터 생성에 사용할 seed

        Returns:
            None
    '''
    # 데이터 생성 자체도 seed에 종속되도록 명시적으로 고정
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X_train, Y_train = generate_invariant_data(num_of_data=data_size, n_blocks=n_block, device=device)
    X_test, Y_test = generate_invariant_data(num_of_data=data_size // 5, n_blocks=n_block, device=device)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed)

    np.random.set_state(rng_state)  # 전역 random state 오염 방지

    suffix = f"{n_block}_{data_size}_seed{seed}"
    torch.save(X_train, os.path.join(data_path, f"X_train_{suffix}.pt"))
    torch.save(X_test, os.path.join(data_path, f"X_test_{suffix}.pt"))
    torch.save(X_valid, os.path.join(data_path, f"X_valid_{suffix}.pt"))
    torch.save(Y_train, os.path.join(data_path, f"Y_train_{suffix}.pt"))
    torch.save(Y_test, os.path.join(data_path, f"Y_test_{suffix}.pt"))
    torch.save(Y_valid, os.path.join(data_path, f"Y_valid_{suffix}.pt"))


def load_data(data_path, n_block, data_size, seed):
    '''
        논문에서 정의한 첫번째 Task에 맞는 Dataset을 불러옴 (seed별 파일)

        Args:
            data_path (str) : 데이터가 저장된 위치
            n_block (int) : 한 데이터에서 X 크기
            data_size (int) : 데이터셋 크기
            seed (int) : 불러올 데이터의 seed

        Returns:
            dataset (dictionary of tuple) : key = X_train, X_test, X_valid, Y_train, Y_test, Y_valid
    '''
    suffix = f"{n_block}_{data_size}_seed{seed}"
    X_train = torch.load(os.path.join(data_path, f"X_train_{suffix}.pt"))
    X_test = torch.load(os.path.join(data_path, f"X_test_{suffix}.pt"))
    X_valid = torch.load(os.path.join(data_path, f"X_valid_{suffix}.pt"))
    Y_train = torch.load(os.path.join(data_path, f"Y_train_{suffix}.pt"))
    Y_test = torch.load(os.path.join(data_path, f"Y_test_{suffix}.pt"))
    Y_valid = torch.load(os.path.join(data_path, f"Y_valid_{suffix}.pt"))
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_valid': X_valid,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'Y_valid': Y_valid
    }


#
# defining model
#

# shared model
class SharedLinear(nn.Module):
    '''
        Weight - Shared Single Layer Perceptron

        Args:
            in_dim (int) : 입력층 크기 (실제로는 in_dim x n_blocks가 입력층 크기)
            out_dim (int) : 출력층 크기 (실제로는 out_dim x n_blocks가 출력층 크기)
            is_last (bool) : 출력층과 연결된 레이어인지 (True면 내부에서 평균냄)
    '''
    def __init__(self, in_dim=1, out_dim=32, is_last=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_last = is_last

        self.A = nn.Parameter(torch.empty(out_dim, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        nn.init.kaiming_uniform_(self.A, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.B, a=0.01, nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.view(x.shape[:-1] + (x.shape[-1] // self.in_dim, self.in_dim))
        s = x.mean(dim=-2, keepdim=True)
        local = F.linear(x, self.A)
        global_term = F.linear(s, self.B)
        out = local + global_term
        out = out + self.bias
        out = out.view(out.shape[:-2] + (-1,))

        if self.is_last:
            return out.mean(dim=-1)
        else:
            return out


class shared_mlp(nn.Module):
    '''
        Weight - Shared Multi Layer Perceptron
    '''
    def __init__(self, n_blocks, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()

        current_dim = 1
        for h_size in hidden_size:
            self.layers.append(SharedLinear(in_dim=current_dim, out_dim=h_size))
            self.layers.append(nn.LeakyReLU(0.01))
            current_dim = h_size
        self.layers.append(SharedLinear(in_dim=current_dim, out_dim=1, is_last=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# vanilla model
class vanilla_mlp(nn.Module):
    '''
        Multi Layer Perceptron
    '''
    def __init__(self, n_blocks, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()

        current_dim = 1
        for h_size in hidden_size:
            self.layers.append(nn.Linear(n_blocks * current_dim, n_blocks * h_size))
            self.layers.append(nn.LeakyReLU(0.01))
            current_dim = h_size
        self.layers.append(nn.Linear(n_blocks * current_dim, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#
# Training Function
#

def train(model, X, Y, epochs=1000, batch_size=256,
          early_stop=False, X_valid=None, Y_valid=None, patience=-1, augment=False):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_log = []
    valid_log = []
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_valid = TensorDataset(X_valid, Y_valid)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)

    bar = tqdm(range(epochs), desc='training', leave=False)
    for epoch in bar:

        model.train()
        epoch_loss = 0
        for batch_X, batch_Y in loader:

            if augment:
                N = batch_X.shape[1]
                p_indices = torch.randperm(N, device=batch_X.device)
                batch_X_permuted = batch_X[:, p_indices]
                batch_Y_permuted = batch_Y
            else:
                batch_X_permuted = batch_X
                batch_Y_permuted = batch_Y
            optimizer.zero_grad()
            pred = model(batch_X_permuted)
            loss = criterion(pred.view(batch_Y_permuted.shape), batch_Y_permuted)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if early_stop:
            model.eval()
            with torch.no_grad():
                epoch_loss_valid = 0
                for batch_X_permuted, batch_Y_permuted in loader_valid:
                    pred = model(batch_X_permuted)
                    loss = criterion(pred.view(batch_Y_permuted.shape), batch_Y_permuted)
                    epoch_loss_valid += loss.item()
                avg_loss_valid = epoch_loss_valid / len(loader_valid)

            if avg_loss_valid < best_loss:
                best_loss = avg_loss_valid
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                bar.close()
                break

        if early_stop:
            bar.set_description(f'loss : {avg_loss:.6f}, valid loss : {avg_loss_valid:.6f}')
        else:
            bar.set_description(f'loss : {avg_loss:.6f}')
        train_log.append(avg_loss)
        if early_stop:
            valid_log.append(avg_loss_valid)

    if early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
    if early_stop:
        return {'train_loss': train_log, 'valid_loss': valid_log}
    return {'train_loss': train_log}


@torch.no_grad()
def evaluate(name, model, X_test, Y_test, n_blocks):
    model.eval()
    pred = model(X_test)
    loss = nn.MSELoss()(pred.view(Y_test.shape), Y_test).item()
    sym = symmetry_error(model, X_test, n_blocks=n_blocks)

    if name is not None:
        print(f"{name}")
        print(f"  Loss: {loss:.6f}")
        print(f"  Symmetry error: {sym:.6e}")
    return loss, sym


@torch.no_grad()
def symmetry_error(model, X, repeat=5, n_blocks=5):
    model.eval()
    batch_size = X.shape[0]

    y_base = model(X)

    error = 0
    for _ in range(repeat):
        x_stacked = X.view(batch_size, n_blocks, 1)
        p_indices = torch.randperm(n_blocks)
        x_permuted = x_stacked[:, p_indices, :].reshape(batch_size, -1)

        y2 = model(x_permuted)

        error += torch.mean((y_base - y2) ** 2)
    error /= repeat

    return error.item()


#
# HyperParameters
#

n_block_lst = [1, 2, 5, 10, 20]
data_size_lst = [50, 100, 500, 1000, 5000, 10000]
batch_size_dict = {
    50: 4,
    100: 8,
    500: 16,
    1000: 32,
    5000: 128,
    10000: 256
}
model_shape = [8, 16, 8, 4, 2]

# --- multi-seed 설정 ---
# 최소 5개 권장. 여유가 있으면 10개로 늘려서 CI를 더 안정적으로.
seed_lst = [0, 1, 2, 3, 4]

data_path = "./data/final_multiseed"
logs_path = "./logs/final_multiseed"
models_path = "./models/final_multiseed"
os.makedirs(data_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

generate_data = True
device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')

# --- 결과를 저장할 long-format 레코드 리스트 ---
# raw 값을 그대로 보존 -> 이후 groupby, paired test, 시각화 등에 재사용 가능
records = []

#
# training iteration
#

n_block_bar = tqdm(n_block_lst, desc='n block iterating...')
for n_block in n_block_bar:
    data_size_bar = tqdm(data_size_lst, desc='data size iterating...', leave=False)
    for data_size in data_size_bar:
        batch_size = batch_size_dict[data_size]

        seed_bar = tqdm(seed_lst, desc='seed iterating...', leave=False)
        for seed in seed_bar:
            # --- seed마다 데이터 생성 및 모델 초기화를 모두 다시 함 ---
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if generate_data:
                generate_and_save_data(data_path=data_path, n_block=n_block, data_size=data_size, seed=seed)
            dataset = load_data(data_path=data_path, n_block=n_block, data_size=data_size, seed=seed)

            (X_train, X_valid, X_test, Y_train, Y_valid, Y_test) = \
                (dataset['X_train'], dataset['X_valid'], dataset['X_test'],
                 dataset['Y_train'], dataset['Y_valid'], dataset['Y_test'])
            X_train = X_train.to(device)
            X_test = X_test.to(device)
            X_valid = X_valid.to(device)
            Y_train = Y_train.to(device)
            Y_test = Y_test.to(device)
            Y_valid = Y_valid.to(device)

            # ---- vanilla / shared (no augmentation) ----
            vanilla_model = vanilla_mlp(n_block, hidden_size=model_shape).to(device)
            shared_model = shared_mlp(n_block, hidden_size=model_shape).to(device)

            vanilla_size = sum(p.numel() for p in vanilla_model.parameters() if p.requires_grad)
            shared_size = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)

            train(vanilla_model, X=X_train, Y=Y_train,
                  epochs=250, batch_size=batch_size, early_stop=True,
                  X_valid=X_valid, Y_valid=Y_valid, patience=15)
            train(shared_model, X=X_train, Y=Y_train,
                  epochs=250, batch_size=batch_size, early_stop=True,
                  X_valid=X_valid, Y_valid=Y_valid, patience=15)

            vanilla_loss, vanilla_symmetry = evaluate(None, vanilla_model, X_test, Y_test, n_block)
            shared_loss, shared_symmetry = evaluate(None, shared_model, X_test, Y_test, n_block)

            # ---- vanilla / shared (augmentation) ----
            vanilla_model_aug = vanilla_mlp(n_block, hidden_size=model_shape).to(device)
            shared_model_aug = shared_mlp(n_block, hidden_size=model_shape).to(device)

            vanilla_aug_size = sum(p.numel() for p in vanilla_model_aug.parameters() if p.requires_grad)
            shared_aug_size = sum(p.numel() for p in shared_model_aug.parameters() if p.requires_grad)

            train(vanilla_model_aug, X=X_train, Y=Y_train,
                  epochs=250, batch_size=batch_size, early_stop=True,
                  X_valid=X_valid, Y_valid=Y_valid, patience=15, augment=True)
            train(shared_model_aug, X=X_train, Y=Y_train,
                  epochs=250, batch_size=batch_size, early_stop=True,
                  X_valid=X_valid, Y_valid=Y_valid, patience=15, augment=True)

            vanilla_aug_loss, vanilla_aug_symmetry = evaluate(None, vanilla_model_aug, X_test, Y_test, n_block)
            shared_aug_loss, shared_aug_symmetry = evaluate(None, shared_model_aug, X_test, Y_test, n_block)

            # ---- long-format record 추가 (모델 4종 x 1 row씩) ----
            for model_name, loss_v, sym_v, size_v in [
                ('vanilla', vanilla_loss, vanilla_symmetry, vanilla_size),
                ('shared', shared_loss, shared_symmetry, shared_size),
                ('vanilla_aug', vanilla_aug_loss, vanilla_aug_symmetry, vanilla_aug_size),
                ('shared_aug', shared_aug_loss, shared_aug_symmetry, shared_aug_size),
            ]:
                records.append({
                    'n_block': n_block,
                    'data_size': data_size,
                    'seed': seed,
                    'model': model_name,
                    'loss': loss_v,
                    'symmetry_error': sym_v,
                    'n_params': size_v,
                })

            # ---- 모델 저장 (파일명에 seed 포함, 필요 없으면 주석 처리해서 용량 절약 가능) ----
            suffix = f"{n_block}_{data_size}_seed{seed}"
            torch.save(vanilla_model.state_dict(), os.path.join(models_path, f'vanilla_model_{suffix}.pt'))
            torch.save(shared_model.state_dict(), os.path.join(models_path, f'shared_model_{suffix}.pt'))
            torch.save(vanilla_model_aug.state_dict(), os.path.join(models_path, f'vanilla_model_aug_{suffix}.pt'))
            torch.save(shared_model_aug.state_dict(), os.path.join(models_path, f'shared_model_aug_{suffix}.pt'))

            # 매 seed 루프마다 중간 저장 -> 중간에 죽어도 데이터 유실 최소화
            pd.DataFrame(records).to_pickle(os.path.join(logs_path, 'total_result_multiseed_raw.pkl'))

#
# 최종 raw 결과 저장 (long-format DataFrame)
#
df = pd.DataFrame(records)
df.to_pickle('total_result_multiseed_raw.pkl')
df.to_csv('total_result_multiseed_raw.csv', index=False)

print(f"총 {len(df)}개 row 저장 완료 (n_block x data_size x seed x model 조합)")
