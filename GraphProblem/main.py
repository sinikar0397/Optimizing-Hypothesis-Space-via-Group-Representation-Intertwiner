import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm.notebook import tqdm
import pytorch_model_summary
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader
from torch.utils.data import TensorDataset
import pickle
import os
import random
import copy


from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch.utils.data import random_split, Subset
from torchinfo import summary



#
# defining device
#
device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')

#
# Dataset
#

class PermutedDataset(Dataset):
    '''
    NCI1 dataset의 wrapper class
    기존 데이터셋에서 순서정보가 전처리되어 자장되어 잇음, 이를 제거하고자 실험 전에 permutation을 가한 dataset
    permutation이 매번 바뀌는 것이 아니기에, augment와 같은 효과를 내지 못함
    기타 전처리 (노드 최대 50개로 조절)도 함께 시행

    Args:
        base_datset (Dataset) : 기존 Datset (probably NCI1 Dataset)
        max_nodes (int) : NCI1 dataset에서 저장할 최대 노드 개수
        permutations (array of tensor) :  
    '''

    def __init__(self, base_dataset, max_nodes):

        self.base_dataset = base_dataset
        self.max_nodes = max_nodes

        self.permutations = []

        for data in base_dataset:

            num_nodes = min(data.x.size(0), max_nodes)
            perm_real = torch.randperm(num_nodes)
            perm_full = torch.cat([
                perm_real,
                torch.arange(num_nodes, max_nodes)
            ])

            self.permutations.append(perm_full)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        perm = self.permutations[idx]

        adj = to_dense_adj(
            data.edge_index,
            max_num_nodes=self.max_nodes
        )[0]

        x = torch.zeros(
            (self.max_nodes, node_features)
        )

        num_nodes = min(data.x.size(0), self.max_nodes)

        x[:num_nodes] = data.x[:num_nodes]

        adj = adj[perm][:, perm]
        x = x[perm]
        return adj, x, data.y

def collate_fn(batch):

    adjs = []
    xs = []
    ys = []

    for adj, x, y in batch:

        adjs.append(adj)
        xs.append(x)
        ys.append(y)

    return (
        torch.stack(adjs).to(device),
        torch.stack(xs).to(device),
        torch.stack(ys).to(device)
    )



dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
max_nodes = 50
node_features = dataset.num_node_features
num_classes = dataset.num_classes

permuted_dataset = PermutedDataset(dataset, max_nodes)
















#
# defining model
#

# shared model


class symmetricGraphLayer(nn.Module):
    """
    general linear S_n-equivariant layer
    for

        (A, X) -> Y

    where

        A : (B, N, N)
        X : (B, N, Fin)
        Y : (B, N, Fout)

    satisfying

        f(PAP^T, PX) = P f(A, X)

    for every permutation matrix P.

    --------------------------------------------------------
    Basis construction
    --------------------------------------------------------

    We classify all admissible linear operators using
    equality patterns of indices (i,j,k).

    Bell number B_3 = 5
    => exactly 5 basis operators.

    Basis:

        B1 : i=j=k
        B2 : i=j!=k
        B3 : i=k!=j
        B4 : j=k!=i
        B5 : all distinct

    Any linear S_n-equivariant operator is a linear
    combination of these 5 basis operators.
    """

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.W3 = nn.Linear(in_dim, out_dim, bias=False)
        self.W4 = nn.Linear(in_dim, out_dim, bias=False)
        self.W5 = nn.Linear(in_dim, out_dim, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)   

    def forward(self, A, X):

        B, N, _ = A.shape

        ############################################
        # Basic quantities
        ############################################

        # diagonal(A)
        diagA = torch.diagonal(A, dim1=-2, dim2=-1)
        # (B,N)

        diagA_col = diagA.unsqueeze(-1)
        # (B,N,1)

        # degree
        deg = A.sum(dim=-1, keepdim=True)
        # (B,N,1)

        ############################################
        # Basis 1
        #
        # i = j = k
        #
        # diag(A)_i * X_i
        ############################################

        B1 = diagA_col * X

        ############################################
        # Basis 2
        #
        # i = j != k
        #
        # sum_{k!=i} A_ik X_k
        #
        # = AX - diag(A)X
        ############################################

        AX = torch.matmul(A, X)

        B2 = AX - B1

        ############################################
        # Basis 3
        #
        # i = k != j
        #
        # sum_{j!=i} A_ji X_i
        #
        # undirected:
        # (deg_i - A_ii) X_i
        ############################################

        B3 = (deg - diagA_col) * X

        ############################################
        # Basis 4
        #
        # j = k != i
        #
        # sum_{j!=i} A_jj X_j
        ############################################

        global_diag = (diagA_col * X).sum(dim=1, keepdim=True)
        # (B,1,F)

        B4 = global_diag.expand(-1, N, -1) - B1

        ############################################
        # Basis 5
        #
        # i,j,k all distinct
        #
        # remaining interaction
        ############################################

        global_AX = AX.sum(dim=1, keepdim=True)
        # (B,1,F)

        total_expand = global_AX.expand(-1, N, -1)

        B5 = total_expand - B1 - B2 - B3 - B4

        ############################################
        # Linear combination of basis operators
        ############################################

        Y = (
            self.W1(B1)
            + self.W2(B2)
            + self.W3(B3)
            + self.W4(B4)
            + self.W5(B5)
        )

        if self.bias is not None:
            Y = Y + self.bias

        return Y
    

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 각 노드의 중요도를 스칼라(1차원) 점수로 변환하는 신경망
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
        
    def forward(self, X):
        # X shape: [batch_size, 50, current_dim]
        
        # 1. 각 노드별 중요도 점수 계산 -> [batch_size, 50, 1]
        scores = self.attn_net(X)
        
        # 2. 노드 방향(dim=1)으로 Softmax를 취해 가중치(확률 분포) 생성 -> [batch_size, 50, 1]
        # 제로 패딩된 노드가 있을 경우 가중치가 분산되는 것을 막기 위해 Softmax를 취합니다.
        weights = F.softmax(scores, dim=1)
        
        # 3. 가중치를 각 노드 피처에 곱하고 합산 (Weighted Sum) -> [batch_size, current_dim]
        graph_feat = torch.sum(weights * X, dim=1)
        
        return graph_feat, weights
    


class symmetricGraphMLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()        # [추가] 각 레이어별 LayerNorm을 담을 리스트
        
        current_dim = in_dim
        for h_size in hidden_size:
            # 1. 대칭성 그래프 레이어 추가
            self.layers.append(symmetricGraphLayer(in_dim=current_dim, out_dim=h_size))
            self.norms.append(nn.LayerNorm(h_size))
                
            current_dim = h_size

        # 정보 보존하기 위한 가중치 합 풀링 적용
        self.pool = AttentionPooling(in_dim=current_dim)

        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(current_dim, out_dim)
        )

    def forward(self, A, X):
        for layer, norm in zip(self.layers, self.norms):
            out = layer(A, X)
            out = norm(out)
            X = F.leaky_relu(out, negative_slope=0.01)
        graph_feat, attn_weights = self.pool(X)
        
        # 3. 분류기 통과 -> [batch_size, out_dim]
        return self.classifier(graph_feat)





# vanilla model
class vanillaGraphMLP(nn.Module):
    def __init__(self, max_nodes, in_dim, hidden_size, out_dim):
        super().__init__()
        self.max_nodes = max_nodes
        
        # 1. 입력 데이터를 일렬로 펼쳤을 때의 총 차원 계산
        # 인접 행렬(max_nodes * max_nodes) + 노드 특징(max_nodes * in_dim)
        # NCI1의 예시 (max_nodes=50, in_dim=37) 일 때: 2500 + 1850 = 4350 차원
        input_flat_dim = max_nodes * max_nodes + max_nodes * in_dim
        
        self.layers = nn.ModuleList()
        current_dim = input_flat_dim
        
        # 2. 대칭 모델과 동일한 hidden_size 깊이와 너비로 일반 Linear 레이어 구축
        for h_size in hidden_size:
            self.layers.append(nn.Linear(current_dim, h_size))
            current_dim = h_size
            
        # 3. 최종 분류기 (대칭 모델과 동일한 구조 적용)
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(current_dim, out_dim)
        )

    def forward(self, A, X):
        # A shape: [batch_size, 50, 50] -> [batch_size, 2500]
        # X shape: [batch_size, 50, in_dim] -> [batch_size, 50 * in_dim]
        
        # 1. 그래프 구조를 완전히 무시하고 일렬로 평탄화(Flatten)
        A_flat = A.view(A.size(0), -1)
        X_flat = X.view(X.size(0), -1)
        
        # 2. 두 텐서를 결합하여 하나의 거대한 벡터로 변환
        # feat shape: [batch_size, input_flat_dim]
        feat = torch.cat([A_flat, X_flat], dim=-1)
        
        # 3. 일반 MLP 은닉층 통과
        for layer in self.layers:
            feat = layer(feat)
            feat = F.leaky_relu(feat, negative_slope=0.01)
            
        # 4. 최종 분류기 통과 -> [batch_size, out_dim]
        return self.classifier(feat)







#
# Training Function
#


def train(model, train_loader, val_loader=None, epochs=100, lr=1e-3, early_stop=False, patience=-1, augment = False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # NCI1 데이터셋은 그래프 분류 태스크이므로 CrossEntropyLoss를 사용합니다.
    criterion = nn.CrossEntropyLoss()

    train_log = []
    valid_log = []
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    bar = tqdm(range(epochs), desc='Training')
    for epoch in bar:
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # collate_fn을 통해 생성된 batch_A, batch_X, batch_Y 사용
        for batch_A, batch_X, batch_Y in train_loader:
            if augment:
                N = batch_A.shape[1] # max_nodes = 50  
                p_indices = torch.randperm(N, device=batch_A.device)
                
                batch_X_permuted = batch_X[:, p_indices, :]
                batch_A_permuted = batch_A[:, p_indices, :][:, :, p_indices]
            else:
                batch_X_permuted = batch_X
                batch_A_permuted = batch_A
            optimizer.zero_grad()
            
            # 모델 순전파 (A와 X를 입력으로 받음)
            pred = model(batch_A_permuted, batch_X_permuted)
            loss = criterion(pred, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_A_permuted.size(0)
            _, predicted = pred.max(1)
            total += batch_Y.size(0)
            correct += predicted.eq(batch_Y).sum().item()
        
        avg_train_loss = epoch_loss / total
        train_acc = correct / total
        train_log.append(avg_train_loss)

        if early_stop and val_loader is not None:
            model.eval()
            epoch_loss_valid = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_A, batch_X, batch_Y in val_loader:
                    N = batch_A.shape[1] # max_nodes = 50  
                    p_indices = torch.randperm(N, device=batch_A.device)
                    
                    batch_X_permuted = batch_X[:, p_indices, :]
                    batch_A_permuted = batch_A[:, p_indices, :][:, :, p_indices]

                    pred = model(batch_A_permuted, batch_X_permuted)
                    loss = criterion(pred, batch_Y)
                    epoch_loss_valid += loss.item() * batch_A_permuted.size(0)
                    
                    _, predicted = pred.max(1)
                    val_total += batch_Y.size(0)
                    val_correct += predicted.eq(batch_Y).sum().item()
                    
            avg_loss_valid = epoch_loss_valid / val_total
            val_acc = val_correct / val_total
            valid_log.append(avg_loss_valid)

            bar.set_description(f'Loss: {avg_train_loss:.4f} | Acc: {train_acc*100:.1f}% | Val Loss: {avg_loss_valid:.4f} | Val Acc: {val_acc*100:.1f}%')

            # Early Stopping 조건 체크
            if avg_loss_valid < best_loss:
                best_loss = avg_loss_valid
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            bar.set_description(f'Loss: {avg_train_loss:.4f} | Acc: {train_acc*100:.1f}%')

    if early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return {
        'train_loss': train_log,
        'valid_loss': valid_log if early_stop else []
    }

@torch.no_grad()
def symmetry_error_graph(model, loader, repeat=5):
    """
    노드의 순서를 무작위로 섞었을 때(Permutation) 모델의 출력(Logit) 변화량이 얼마나 발생하는지 측정합니다.
    Symmetric 모델은 이 오차가 0에 수렴해야 하며, 단순 MLP는 큰 오차가 발생합니다.
    """
    model.eval()
    error_list = []
    
    for batch_A, batch_X, _ in loader:
        N = batch_A.shape[1] # max_nodes = 50
        
        # 1. 원본 그래프 데이터 예측 결과 (Logits)
        y_base = model(batch_A, batch_X)
        
        for _ in range(repeat):
            # 무작위 노드 순서 인덱스 생성
            p_indices = torch.randperm(N, device=batch_A.device)
            
            # 2. 노드 피처 치환 (P * X)
            X_permuted = batch_X[:, p_indices, :]
            
            # 3. 인접 행렬 치환 (P * A * P^T) -> 행과 열을 모두 동일한 인덱스로 셔플
            A_permuted = batch_A[:, p_indices, :][:, :, p_indices]
            
            # 4. 변형된 그래프 데이터 예측 결과
            y_permuted = model(A_permuted, X_permuted)
            
            # 원본 결과와 변형 결과 사이의 Mean Squared Error 계산
            mse = torch.mean((y_base - y_permuted) ** 2)
            error_list.append(mse.item())
            
    return np.mean(error_list)

@torch.no_grad()
def evaluate_graph(name, model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_A, batch_X, batch_Y in loader:
        N = batch_A.shape[1] # max_nodes = 50
        p_indices = torch.randperm(N, device=batch_A.device)
        
        batch_X_permuted = batch_X[:, p_indices, :]
        batch_A_permuted = batch_A[:, p_indices, :][:, :, p_indices]
        
        pred = model(batch_A_permuted, batch_X_permuted)
        loss = criterion(pred, batch_Y)
        
        total_loss += loss.item() * batch_A.size(0)
        _, predicted = pred.max(1)
        total += batch_Y.size(0)
        correct += predicted.eq(batch_Y).sum().item()
        
    avg_loss = total_loss / total
    accuracy = correct / total
    
    sym_err = symmetry_error_graph(model, loader, repeat=5)
    
    print(f'[{name} Evaluation Result - Permuted Data]')
    print(f'  Loss: {avg_loss:.6f}')
    print(f'  Accuracy: {accuracy * 100:.2f}%')
    print(f'  Symmetry Error: {sym_err:.6e}\n')
    
    return avg_loss, accuracy, sym_err




#
# HyperParameters
#


# 1. 결과 및 체크포인트를 저장할 디렉토리 생성
models_path = './models'
logs_path = './logs'
os.makedirs(models_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)

# ====================================================================
# [수정] 전체 데이터셋의 인덱스를 미리 무작위로 섞어둡니다.
# 이렇게 해야 특정 data_size를 뽑았을 때 클래스가 한쪽으로 몰리지 않습니다.
# ====================================================================
all_indices = list(range(len(dataset)))
random.seed(42)      # 실험의 재현성을 위해 파이썬 내장 random 시드 고정
random.shuffle(all_indices)
# ====================================================================

# 2. Grid Search를 수행할 하이퍼파라미터 조건 정의
n_blocks_list = [3, 5, 7]
data_sizes_list = [500, 1500, len(dataset)]
hidden_dims_lst = [
    [64,  64,  16],
    [256, 256, 128, 64,  16],
    [256, 512, 256, 128, 128, 64, 16]
]







#
# training iteration
#

result = {}

for i in range(len(n_blocks_list)):
    n_block = n_blocks_list[i]
    curr_result = {}
    hidden_dims = hidden_dims_lst[i] 
    
    for data_size in data_sizes_list:
        print(f"\n" + "="*60)
        print(f"▶ [실험 시작] Blocks(깊이): {n_block} | Dataset Size: {data_size}")
        print("="*60)
        
        indices = all_indices[:data_size]
        subset_dataset = Subset(permuted_dataset, indices)
        
        train_size = int(0.8 * data_size)
        val_size = int(0.1 * data_size)
        test_size = data_size - train_size - val_size
        
        if train_size == 0 or val_size == 0 or test_size == 0:
            print(f"데이터 크기가 너무 작아 실험을 건너뜁니다. (Size: {data_size})")
            continue
            
        train_dataset, val_dataset, test_dataset = random_split(
            subset_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = PyTorchDataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = PyTorchDataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = PyTorchDataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


        symmetric_model = symmetricGraphMLP(
            in_dim=node_features, 
            hidden_size=hidden_dims, 
            out_dim=num_classes
        ).to(device)
        
        shared_log = train(
            model=symmetric_model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=100, # Grid Search 속도를 위해 에포크 조정 가능
            lr=1e-3, 
            early_stop=True, 
            patience=5
        )

        
        vanilla_model = vanillaGraphMLP(
            max_nodes=max_nodes, 
            in_dim=node_features, 
            hidden_size=hidden_dims, 
            out_dim=num_classes
        ).to(device)
        
        vanilla_log = train(
            model=vanilla_model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=100, 
            lr=1e-3, 
            early_stop=True, 
            patience=5
        )

        symmetric_model_augmented = symmetricGraphMLP(
            in_dim=node_features, 
            hidden_size=hidden_dims, 
            out_dim=num_classes
        ).to(device)
        
        shared_log = train(
            model=symmetric_model_augmented, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=100, # Grid Search 속도를 위해 에포크 조정 가능
            lr=1e-3, 
            early_stop=True, 
            patience=5,
            augment = True
        )
        
        vanilla_model_augmented = vanillaGraphMLP(
            max_nodes=max_nodes, 
            in_dim=node_features, 
            hidden_size=hidden_dims, 
            out_dim=num_classes
        ).to(device)
        
        vanilla_log = train(
            model=vanilla_model_augmented, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=100, 
            lr=1e-3, 
            early_stop=True, 
            patience=5,
            augment = True
        )
        
        sym_loss, sym_acc, sym_err = evaluate_graph("Symmetric Model", symmetric_model, test_loader)
        van_loss, van_acc, van_err = evaluate_graph("Vanilla Model", vanilla_model, test_loader)
        sym_aug_loss, sym_aug_acc, sym_aug_err = evaluate_graph("Symmetric Model(augmented)", symmetric_model_augmented, test_loader)
        van_aug_loss, van_aug_acc, van_aug_err = evaluate_graph("Vanilla Model(augmented)", vanilla_model_augmented, test_loader)
        
        curr_result[data_size] = {
            'symmetric': {'loss': sym_loss, 'acc': sym_acc, 'sym_error': sym_err},
            'vanilla': {'loss': van_loss, 'acc': van_acc, 'sym_error': van_err},
            'symmetric (data augmented)': {'loss': sym_aug_loss, 'acc': sym_aug_acc, 'sym_error': sym_aug_err},
            'vanilla (data augmented)': {'loss': van_aug_loss, 'acc': van_aug_acc, 'sym_error': van_aug_err}
        }
        
        torch.save(symmetric_model.state_dict(), os.path.join(models_path, f'shared_model_{n_block}_{data_size}.pt'))
        torch.save(vanilla_model.state_dict(), os.path.join(models_path, f'vanilla_model_{n_block}_{data_size}.pt'))
        
        with open(os.path.join(logs_path, f'shared_log_{n_block}_{data_size}.pkl'), 'wb') as f:
            pickle.dump(shared_log, f)
        with open(os.path.join(logs_path, f'vanilla_log_{n_block}_{data_size}.pkl'), 'wb') as f:
            pickle.dump(vanilla_log, f)

    result[n_block] = curr_result


with open('./total_result.pkl', 'wb') as f:
    pickle.dump(result, f)
print("\n모든 Grid Search 실험 조건 완수 및 파일 저장 완료")