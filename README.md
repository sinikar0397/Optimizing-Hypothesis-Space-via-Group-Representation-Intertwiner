# Group Representation Intertwiner를 이용한 MLP의 가중치 기저 분해 및 함수 공간 최적화
서울과학고등학교 24031 김주환 졸업연구

I. 연구의 필요성과 목적

인공지능 모델을 구성할 때 대칭성은 중요한 요소이다. 데이터에 대칭성이 있음에도, 이를 고려하지 않고 모델을 구성한다면 탐색할 필요가 없는 함수에 대해서도 탐색이 이루어지고, 파라미터의 낭비가 발생하게 된다. 이에 따라 지금까지 대칭성에 대응하고자 여러 가지 모델이 개발되어 왔다. 이미지에서 병진 운동에 대한 대칭성에 대응하고자 동일한 가중치를 평행이동시키면서 적용하는 모델로 CNN이 있고, 모든 점의 대칭성을 고려하고자 GNN이 제시되었다.

 다만, 위 방법으로는 일부 노드에 대해서만 대칭성이 있거나 윤환적 대칭 구조를 가지는 경우에 대한 대응이 불가능하다. 그래서 대칭 구조를 군(Group)을 이용해 표기하고 Group Representation을 이용해 가능한 함수(가중치 행렬)이 Group Representation에 대한 Intertwiner가 됨을 제시할 수 있다. 가중치 행렬을 Intertwiner로 제한해 가중치 공간의 차원을 줄이고자 한다. 즉, 대칭 구조를 만족하는 가중치 행렬의 기저를 구해 함수 공간을 최적화하고자 한다.
이런 방식으로 함수 공간을 최적화할 수 있다면 물리 시뮬레이션 등의 대칭 구조를 가지는  모델을 최적화할 수 있을 것이다.
 단, 범용적인 DL의 가중치를 제한하기 전에 단순 MLP에서만 성능을 확인해보고자 한다.

II. 연구의 설계

(1) 가설

MLP에서 가중치 행렬을 Group Representation Intertwiner로 구성함으로써 MLP의 성능을 향상할 수 있다.

(2) 내용

특정 계의 상태 $x\ \in V$에 대해 대칭군(Symmetric Group) G이 작용할 때, $f∶\ V\ \rightarrow\ V$는 아래 수식을 만족해야 한다.

$$g\cdot\left(f\left(x\right)\right)=f\left(g\cdot x\right) \quad\forall g\in G$$

위 식에서 군을 Group Representation $\rho$으로, f를 가중치 행렬 W을 곱하는 함수로 고려할 때 아래 수식으로 표기할 수 있다.[1]

$$\rho W=W\rho$$

 따라서 V를 feature로, V를 label로 하는 mlp의 가중치 행렬은 위 식을 만족한다. 따라서 가중치 함수의 탐색 범위를 식 (2)의 해공간으로 한정시킴으로써 탐색 성능을 유지하며 탐색 범위를 줄일 수 있다.

(3) 방법

입자 3개가 운동하는 물리 계를 정의하고, 질량이 동일한 두 입자 사이의 교환 대칭성을 \operatorname{S}_2 군으로 설정한다. 해당 대칭성을 만족하는 가중치 W를 기저 분해하여 파라미터 수를 획기적으로 줄인 모델을 구현한다. 이후 단순 MLP 및 GNN과 비교하여 예측 오차(Loss)와 대칭성 유지 성능을 검증한다.
세부적인 가중치 기저 분해 과정은 Equivariant Flow.ipynb를 참고하라


III. 기대효과

Intertwiner를 사용하는 방식을 통해 가중치 파라미터를 절반 이하로 줄일 수 있으며, 불필요한 함수 공간 탐색을 배제함으로써 예측 오차를 낮추고 학습의 안정성을 높일 수 있을 것으로 기대된다.. 특히 물리 법칙을 준수해야 하는 시뮬레이션 분야에서 높은 정합성을 제공할 것이다.

IV. 참고문헌

[1] Finzi, M., et al. (2021). "A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups." ICML.

[2] Ravanbakhsh, S., et al. (2017). "Equivariance through parameter-sharing." ICML.
