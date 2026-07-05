"""
GraphProblem 파라미터 수 계산 스크립트 (재학습 불필요)

목적:
    total_result_graph_multiseed_raw.csv 에는 n_params 컬럼이 없어서
    symmetric vs vanilla 모델의 파라미터 효율성을 정량적으로 보여줄 수 없었음.
    파라미터 수는 모델 "구조"(n_block에 대응하는 hidden_dims, node_features, num_classes)에만
    의존하고 실제 학습된 가중치 값과는 무관하므로, 체크포인트를 다시 학습할 필요 없이
    모델을 구조만 맞춰 새로 생성한 뒤 파라미터 개수만 세면 된다.

    aug 버전(symmetric_aug, vanilla_aug)은 학습 방식만 다르고 구조는 동일하므로
    symmetric/vanilla와 같은 파라미터 수를 갖는다.

사용법:
    GraphProblem/ 디렉토리에서 실행:
        python compute_graph_params.py

    total_result_graph_multiseed_raw.csv 를 읽어 n_params 컬럼을 추가한 뒤
    total_result_graph_multiseed_raw_with_params.csv 로 저장한다.
    (원본 파일은 보존됨)
"""

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset


# ------------------------------------------------------------------
# main.py 에서 그대로 가져온 모델 정의
# (파라미터 개수만 필요하므로 forward 로직은 생략하거나 최소화함)
# ------------------------------------------------------------------

class symmetricGraphLayer(nn.Module):
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
        raise NotImplementedError("이 스크립트는 파라미터 수 계산 전용입니다 (forward 미구현).")


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )


class symmetricGraphMLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = in_dim
        for h_size in hidden_size:
            self.layers.append(symmetricGraphLayer(in_dim=current_dim, out_dim=h_size))
            self.norms.append(nn.LayerNorm(h_size))
            current_dim = h_size

        self.pool = AttentionPooling(in_dim=current_dim)
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(current_dim, out_dim),
        )


class vanillaGraphMLP(nn.Module):
    def __init__(self, max_nodes, in_dim, hidden_size, out_dim):
        super().__init__()
        self.max_nodes = max_nodes
        input_flat_dim = max_nodes * max_nodes + max_nodes * in_dim

        self.layers = nn.ModuleList()
        current_dim = input_flat_dim
        for h_size in hidden_size:
            self.layers.append(nn.Linear(current_dim, h_size))
            current_dim = h_size

        self.classifier = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(current_dim, out_dim),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # main.py 와 동일한 방식으로 NCI1의 메타데이터만 가져옴 (학습은 하지 않음)
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    max_nodes = 50
    node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # main.py 의 하이퍼파라미터와 정확히 동일하게 맞춤 (순서 중요)
    n_blocks_list = [3, 5, 7]
    hidden_dims_lst = [
        [64, 64, 16],
        [256, 256, 128, 64, 16],
        [256, 512, 256, 128, 128, 64, 16],
    ]

    n_params_by_nblock = {}
    for n_block, hidden_dims in zip(n_blocks_list, hidden_dims_lst):
        symmetric_model = symmetricGraphMLP(
            in_dim=node_features, hidden_size=hidden_dims, out_dim=num_classes
        )
        vanilla_model = vanillaGraphMLP(
            max_nodes=max_nodes, in_dim=node_features, hidden_size=hidden_dims, out_dim=num_classes
        )
        n_params_by_nblock[n_block] = {
            'symmetric': count_parameters(symmetric_model),
            'vanilla': count_parameters(vanilla_model),
        }
        print(
            f"n_block={n_block}: symmetric={n_params_by_nblock[n_block]['symmetric']:,}, "
            f"vanilla={n_params_by_nblock[n_block]['vanilla']:,}"
        )

    def lookup_params(row):
        # symmetric_aug -> symmetric, vanilla_aug -> vanilla (구조 동일)
        base_model = row['model'].replace('_aug', '')
        return n_params_by_nblock[row['n_block']][base_model]

    df = pd.read_csv('total_result_graph_multiseed_raw.csv')
    df['n_params'] = df.apply(lookup_params, axis=1)

    out_path = 'total_result_graph_multiseed_raw_with_params.csv'
    df.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")
    print(df.groupby(['n_block', 'model'])['n_params'].first())


if __name__ == "__main__":
    main()
