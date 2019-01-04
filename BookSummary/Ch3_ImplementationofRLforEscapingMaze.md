# Ch2. 미로찾기를 위한 강화학습 구현 (2.5부터 다시 복습)

코드 : https://github.com/SunandBean/Pytorch_RLstudy.git 의 Notebook/Ch2_3_Maze~ 참고

---

## 2.1 주피터 노트붝 체험 페이지 사용법

- https://jupyter.org/try

---

## 2.2 미로와 에이전트 구현

### 미로 구현

![1546517552075](C:\Users\Sunbin\AppData\Roaming\Typora\typora-user-images\1546517552075.png)

녹색 동그라미 : 에이전트

- 에이전트가 어떻게 행동할지를 결정하는 규칙 : 정책

  - $$
    \pi_\theta(s,a)
    $$

  - 상태가 s일 때 행동 a를 취할 확률은 파라미터 theta가 결정하는 정책 pi를 따른다.

    - 상태 s : s0 ~ s8
    - 행동 a : 상, 하, 좌, 우

  - 파라미터 theta는 정책 pi가 함수인 경우 함수의 파라미터, 정책이 신경망인 경우 유닛 간의 결합 가중치에 해당

    - 파라미터 theta로부터 단순하게 값의 비율을 계산하여 정책 pi를 구함

붉은색 벽 : 통과 불가능한 벽

----

## 2.3 정책반복 구현

### 정책반복과 가치반복

에이전트가 목표로 곧장 향하도록 정책을 학습!

1. **정책반복** : 정책에 따라 목표에 빠르게 도달했던 경우에 수행했던 **행동을 중요한 것**으로 보고, 이 떄의 행동을 앞으로도 취할 수 있도록 **정책을 수정**

2. **가치반복** : 목표 지점부터 거슬러 올라가며 목표 지점과 가까운 상태로 **에이전트를 유도**해 오는 방법
   - 목표 지점 외의 지점(상태)에도 **가치(우선도)를 부여**하는 것

#### 정책반복 알고리즘 중 하나인 정책경사(policy gradient) 알고리즘 구현!

정책 파라미터 theta로부터 정책 pi를 구하는 방법 변경

- 단순한 비율 -> softmax 함수 사용

- $$
  P(\theta_i) = exp(\beta\theta_i)/(exp(\beta\theta_1)+exp(\beta\theta_2)+...) = exp(\beta\theta_i)/\Sigma_{j=1}^{N_a}exp(\beta\theta_j)
  $$




### 정책경사 알고리즘으로 정책 수정

정책경사 알고리즘은 아래 식과 같이 파라미터 theta를 수정
$$
\theta_{s_i,a_j}=\theta_{s_i,a_j} + \eta.\Delta\theta_{s,a_j}
$$

$$
\Delta\theta_{s,a_j} = \{N(s_i,a_j) + P(s_i,a_j)N(s_i,a)\}/T
$$


$$
\theta_{s_i,a_j} = 상태(위치)\ s_i에서\ 행동\ a_j를\ 취할\ 확률을\ 결정하는\ 파라미터\\
\eta=학습률,\ \theta_{s_i,a_j}가\ 1번\ 학습에\ 수정되는\ 정도를\ 제어\\
N(s_i,a_j) =상태\ s_i에서\ 행동\ a_j를\ 취했던\ 횟수\\
P(s_i,a_j) = 현재\ 정책하에서\ 상태\ s_i일\ 때\ 행동\ a_j를\ 취할\ 확률\\
N(s_i,a)=상태\ s_i에서\ 행동을\ 취한\ 횟수의\ 합계\\
T=목표\ 지점에\ 이르기까지\ 걸린\ 모든\ 단계의\ 수
$$


### 정책경사 알고리즘에 대한 이론

#### 왜 정책을 계산할 때 소프트맥스를 사용했는가?

- 파라미터 theta가 음수가 돼도 정책을 계산할 수 있기 때문
  - exp(지수함수)의 함수값은 항상 양수이기 때문

#### theta를 수정하는 식은 왜 그런 형태여야 하는가?

- 정책경사 정리(policy gradient theorem)이 있고, 이를 근사적으로 구현한 알고리즘이 REINFORCE
  - 소프트맥스 함수로 확률을 변환하고 REINFORCE 알고리즘을 사용하면 위의 식이 유도
  - 유도 과정 참고 문헌 : Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." 

---

## 2.4 가치반복 알고리즘 관련 용어 정리

**보상, 행동가치, 상태가치, 벨만 방정식, 마르코프 결정 프로세스**

### 보상

가치의 척도 - **보상**! -> 학습을 위해서 보상을 태스크에 맞게 적절히 결정해야함!

- 즉각보상 (immediate reward) : 어떤 시각 t에 받을 수 있는 보상 R_t

- 총보상 : 앞으로 받을 수 있으리라 예상되는 보상의 합계 G_t
  $$
  G_t = R_{t+1}+R_{t+2}+R_{t+3}+...
  $$

- 할인총보상 (discounted total reward) : 시간할인율을 계산에 넣은 앞으로의 보상 합계
  $$
  G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...
  $$




### 행동가치와 상태가치

![img](data:image/png;base64,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)

상황 : 에이전트가 S7에 위치할 때 

#### 행동가치(action value)

- 오른쪽으로 이동하면 목표에 도달 가능 -> R_{t+1} = 1

  - 상태 s = s7이고 행동 a = 오른쪽 => R_{t+1} = 1

  $$
  Q^\pi(s=7,a=1(오른쪽))=R_{t+1}=1\\
  Q^\pi=정책\ \pi를\ 기반으로\ 하는\ 행동가치\ 함수
  $$


- 위쪽으로 이동하는 경우,

  - 목표로 가기 위해서는 s7->s4->s7->s8 => 2단계 증가
    $$
    Q^\pi(s=7,a=0(위쪽))=\gamma^2*1
    $$


#### 상태가치(state value)

- 상태 s에서 정책 pi를 따라 행동할 때 얻으리라 기대할 수 있는 할인총보상 G_t
  $$
  V^\pi(s) = 상태\ s의\ 상태가치\ 함수
  $$




## 벨만 방정식과 마르코프 결정 프로세스

#### 벨만 방정식 

일반화한 상태가치 함수의 수식
$$
V^\pi(s)=max_aE[R_{s,a}+\gamma*V^\pi(s(s,a))]\\
V^\pi = 상태\ s에서의\ 상태가치\ V\\
상태가치\ V =우변의\ 값이\ 가장\ 커지는\ 행동을\ 취했을\ 때\ 기대할\ 수\ 있는\ 값\\
R_{s,a}=상태\ s에서\ 행동\ a를\ 취했을\ 떼\ 얻을\ 수\ 있는\ 즉각보상\ R_{t+1}\\
s(s,a)=상태\ s에서\ 행동\ a를\ 취해서\ 이동한\ 다음\ 단계의\ 새로운\ 상태\ s_{t+1}
$$


#### 마르코프 결정 프로세스(Markov decision process, MDP)

벨만 방정식이 성립하기 위한 **전제조건**

MDP란

- 다음 단계의 상태 s_{t+1}이 현재 상태 s_t에서 취한 행동 a_t에 의해 결정되는 시스템
  $$
  s(s,a)=상태\ s에서\ 행동\ a를\ 취해서\ 이동한\ 다음\ 단계의\ 새로운\ 상태\ s_{t+1}\\
  위의 \ 설명이\ 성립하기\ 위해서는\ 이\ 시스템이\ MDP여야한다.
  $$

- 현재 상태 s_t 외의 과거 상태들, s_{t-1}등에도 영향을 받으면 MDP가 아니게 된다.



**가치반복 알고리즘의 대표적인 예**

- Sarsa, Q러닝

---

## 2.5 Sarsa 알고리즘 구현

### e-greedy 알고리즘으로 정책 구현





---

## 2.6 Q러닝 구현

### Q러닝의 알고리즘