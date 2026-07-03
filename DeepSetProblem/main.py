import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm.notebook import tqdm
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

    [multi-seed 수정] seed 인자를 받아 permutation 생성을 해당 seed에 종속시킴.
    -> seed별로 독립된 "고정 permutation"을 갖게 되어, 진짜 독립 반복(replicate)이 됨.

    Args:
        base_datset (Dataset) : 기존 Datset (probably NCI1 Dataset)
        max_nodes (int) : NCI1 dataset에서 저장할 최대 노드 개수
        seed (int) : permutation 생성에 사용할 seed
        permutations (array of tensor) :  사전 저장한 각 데이터별 permutation
    '''

    def __init__(self, base_dataset, max_nodes, seed=0):

        self.base_dataset = base_dataset
        self.max_nodes = max_nodes

        # permutation 생성을 이 함수 안에서만 seed에 종속되도록 격리
        # (torch 전역 random state 오염 방지)
        g = torch.Generator()
        g.manual_seed(seed)

        self.permutations = []

        for data in base_dataset:

            num_nodes = min(data.x.size(0), max_nodes)
            perm_real = torch.randperm(num_nodes, generator=g)
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
    '''
        DataLoader 제작을 위한 collate function

        Args:
            batch (Iterational data) : 데이터 배치
    '''

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

# [수정] permuted_dataset은 이제 seed별로 loop 안에서 생성 (아래 training iteration 참고)
# 전역에서 한번만 만들면 모든 seed가 동일한 permutation을 공유하게 되어
# "독립 반복"이라는 전제가 깨지기 때문.


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
    '''
    pooling layer for graph.
    Shared MLP를 통과시킨 후 Attention pooling을 통해 flatten시킴

        Args:
            attn_net (nn.Sequential) : in_dimension -> in_dimension/2 -> 1 순서의 Linear Dense Layer
    '''
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
    '''
    Weight-shared Multi Layer perceptron

    Args:
        layers (nn.ModuleList) : 각각의 레이어들 저장하는 ModuleList
        norms (nn.ModuleList) : 각각의 LayerNorm을 저장하는 ModuleList
        pool (AttentionPooling) : Shared layer를 통과한 후 이용하는 Pooling Layer
        classifier (nn.Sequential) : 최종 classifier
    '''
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
    '''
    Vanilla Multi Layer perceptron

    Args:
        max_nodes (int) : NCI1 dataset에 이용하는 노드의 개수 최댓값
        layers (nn.ModuleList) : 내부 Layer들을 저장해둔 ModuelList
        classifier (nn.Sequential) : 최종 classifier

    '''
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
    
    if name is not None:
        print(f'[{name} Evaluation Result - Permuted Data]')
        print(f'  Loss: {avg_loss:.6f}')
        print(f'  Accuracy: {accuracy * 100:.2f}%')
        print(f'  Symmetry Error: {sym_err:.6e}\n')
    
    return avg_loss, accuracy, sym_err




#
# HyperParameters
#


# 1. 결과 및 체크포인트를 저장할 디렉토리 생성 (multi-seed 버전은 별도 폴더에 저장)
models_path = './models/'
logs_path = './logs/'
os.makedirs(models_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)

# 2. Grid Search를 수행할 하이퍼파라미터 조건 정의
n_blocks_list = [3, 5, 7]
data_sizes_list = [500, 1500, len(dataset)]
hidden_dims_lst = [
    [64,  64,  16],
    [256, 256, 128, 64,  16],
    [256, 512, 256, 128, 128, 64, 16]
]

# --- multi-seed 설정 ---
# 최소 5개 권장. NCI1 학습이 DeepSet 예제보다 무거우니 먼저 seed 2개로 파일럿 실행 권장.
seed_lst = [0, 1, 2, 3, 4]


#
# training iteration
#

# raw 결과를 long-format으로 보존 (mean/CI로 바로 뭉개지 않음)
records = []

for i in range(len(n_blocks_list)):
    n_block = n_blocks_list[i]
    hidden_dims = hidden_dims_lst[i]

    for data_size in data_sizes_list:

        if data_size < 10:
            print(f"데이터 크기가 너무 작아 실험을 건너뜁니다. (Size: {data_size})")
            continue

        for seed in seed_lst:
            print(f"\n" + "="*60)
            print(f"▶ [실험 시작] Blocks(깊이): {n_block} | Dataset Size: {data_size} | Seed: {seed}")
            print("="*60)

            # --- seed마다 전역 random state를 전부 고정 ---
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # --- 데이터 셔플/분할도 seed에 종속 ---
            all_indices = list(range(len(dataset)))
            random.shuffle(all_indices)  # 위에서 random.seed(seed)로 고정했으므로 seed별로 다른 셔플

            # --- permutation 자체도 seed별로 새로 생성 (독립 반복을 위해 재구성) ---
            permuted_dataset = PermutedDataset(dataset, max_nodes, seed=seed)

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
                generator=torch.Generator().manual_seed(seed)
            )

            train_loader = PyTorchDataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
            val_loader = PyTorchDataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
            test_loader = PyTorchDataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

            # ---- symmetric / vanilla (no augmentation) ----
            symmetric_model = symmetricGraphMLP(
                in_dim=node_features,
                hidden_size=hidden_dims,
                out_dim=num_classes
            ).to(device)

            train(
                model=symmetric_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
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

            train(
                model=vanilla_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
                lr=1e-3,
                early_stop=True,
                patience=5
            )

            # ---- symmetric / vanilla (augmentation) ----
            symmetric_model_augmented = symmetricGraphMLP(
                in_dim=node_features,
                hidden_size=hidden_dims,
                out_dim=num_classes
            ).to(device)

            train(
                model=symmetric_model_augmented,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
                lr=1e-3,
                early_stop=True,
                patience=5,
                augment=True
            )

            vanilla_model_augmented = vanillaGraphMLP(
                max_nodes=max_nodes,
                in_dim=node_features,
                hidden_size=hidden_dims,
                out_dim=num_classes
            ).to(device)

            train(
                model=vanilla_model_augmented,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
                lr=1e-3,
                early_stop=True,
                patience=5,
                augment=True
            )

            sym_loss, sym_acc, sym_err = evaluate_graph(None, symmetric_model, test_loader)
            van_loss, van_acc, van_err = evaluate_graph(None, vanilla_model, test_loader)
            sym_aug_loss, sym_aug_acc, sym_aug_err = evaluate_graph(None, symmetric_model_augmented, test_loader)
            van_aug_loss, van_aug_acc, van_aug_err = evaluate_graph(None, vanilla_model_augmented, test_loader)

            # ---- long-format record 추가 ----
            for model_name, loss_v, acc_v, err_v in [
                ('symmetric', sym_loss, sym_acc, sym_err),
                ('vanilla', van_loss, van_acc, van_err),
                ('symmetric_aug', sym_aug_loss, sym_aug_acc, sym_aug_err),
                ('vanilla_aug', van_aug_loss, van_aug_acc, van_aug_err),
            ]:
                records.append({
                    'n_block': n_block,
                    'data_size': data_size,
                    'seed': seed,
                    'model': model_name,
                    'loss': loss_v,
                    'accuracy': acc_v,
                    'symmetry_error': err_v,
                })

            # ---- 모델 저장 (파일명에 seed 포함) ----
            suffix = f"{n_block}_{data_size}_seed{seed}"
            torch.save(symmetric_model.state_dict(), os.path.join(models_path, f'shared_model_{suffix}.pt'))
            torch.save(vanilla_model.state_dict(), os.path.join(models_path, f'vanilla_model_{suffix}.pt'))
            torch.save(symmetric_model_augmented.state_dict(), os.path.join(models_path, f'shared_model_aug_{suffix}.pt'))
            torch.save(vanilla_model_augmented.state_dict(), os.path.join(models_path, f'vanilla_model_aug_{suffix}.pt'))

            # 매 seed 루프마다 중간 저장 -> 중간에 죽어도 데이터 유실 최소화
            pd.DataFrame(records).to_pickle(os.path.join(logs_path, 'total_result_multiseed_raw.pkl'))

#
# 최종 raw 결과 저장 (long-format DataFrame)
#
df = pd.DataFrame(records)
df.to_pickle('total_result_graph_multiseed_raw.pkl')
df.to_csv('total_result_graph_multiseed_raw.csv', index=False)

print(f"\n총 {len(df)}개 row 저장 완료 (n_block x data_size x seed x model 조합)")
print("모든 Grid Search + Multi-seed 실험 조건 완수 및 파일 저장 완료")
