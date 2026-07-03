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
import copy
import os

#
# defining device
#
device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')

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
                best_model_state = copy.deepcopy(model.state_dict())
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
        p_indices = torch.randperm(n_blocks, device = X.device)
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

#
# training iteration
#

#
# 최종 raw 결과 저장 (long-format DataFrame)
#