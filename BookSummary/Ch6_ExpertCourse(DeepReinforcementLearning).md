# Ch6. 딥러닝을 적용한 강화학습 - 심화 과정

## 6.1 심층강화학습 알고리즘 지도

### Ch2

정책반복 알고리즘 (REINFORCE), Sarsa, Q러닝

### Ch5

DQN

### Ch6

DQN 이후에 나온 알고리즘

- DDQN
  - Double Q-러닝 + DQN
- Dueling Network
  - 행동가치 함수의 출력층 앞에 상태가치 V(s)와 어드밴티지 함수를 배치
- Prioritized Experience Replay
  - Replay memory에서 데이터를 샘플링할 때 우선순위를 매겨서 추출
- A3C -> A2C
  - A3C (Asynchronous Advantage Actor-Critic)
    - 비동기 분산 학습 시스템 / Q 함수를 수정할 때 두 단계 이상 고려 / 정책반복과 가치반복을 조합
  - A2C (Advangate Actor-Critic)
    - 동기 분산 학습 시스템 / Q 함수를 수정할 때 두 단계 이상 고려 / 정책반복과 가치반복을 조합
  - 장점 
    - 실세계에 강화학습을 적용하기 쉽다
    - Experience Replay를 사용하지 않아도 된다. => RNN, LSTM을 사용할 수 있다

---

## 6.2 DDQN(Double-DQN) 구현

### DDQN

2015년 네이처에 발표된 DQN의 수정식
$$
Q_m(s_t,a_t)=Q_m(s_t,a_t)+\eta*(R_{t+1}+\gamma max_a Q_t(s_{t+1},a)-Q_m(s_t,a_t))\\
Q_m = Main\ Q-Network\\
Q_t = Target\ Q-Network
$$
DDQN은 이 수정식을 더욱 안정화시킨 것
$$
a_m=argmax_a Q_m(s_{t+1},a)\\
Q_m(s_t,a_t)=Q_m(s_t,a_t)+\eta*(R_{t+1}+\gamma Q_t(s_{t+1},a_m)-Q_m(s_t,a_t))
$$

- Main Q-Network의 수정값을 구하는 데 2개의 신경망을 사용

  1. 다음 상태 s_{t+1}에서 Q값이 최대가 되는 행동 a(m)을 Main Q-Network에서 구하고,  

  2. 그 때의 Q값은 Target Q-Network에서 구하는 것

### DDQN 구현

코드 참고

---

## 6.3 Dueling Network 구현

### Dueling Network

기존의 행동가치 함수를 출력하는 부분의 앞 단에 새로운 층을 둠

- Advantage 함수를 이용!
  - A(s, 오른쪽) = Q(s,오른쪽) - V(s)
- 도입 이유
  - 행동가치 함수 : 어떤 행동을 취하든 받게되는 할인 총보상이 상태에 의해서만 결정
    - Q 함수가 가지고 있는 정보를 상태만으로 결정되는 부분과 행동에 따라 결정되는 부분으로 나누어 학습한 다음 마지막 출력층에서 둘을 더해 행동가치 함수를 계산

- 장점
  - V(s)로 이어지는 결합 가중치를 행동 a에 상관없이 매 단계마다 학습할 수 있음
    - => 적은 수의 에피소드만으로도 학습을 마칠 수 있다.
    - => 선택 가능한 행동의 가짓수가 늘어날수록 큰 이점이 된다.

### Dueling Network 구현

코드 참고

---

## 6.4 Prioritized Experience Replay 구현

### Prioritized Experience Replay

Q러닝이 제대로 지나가지 않은 상태 s의 transition을 우선적으로 학습시키는 기법

**기준**

- 가치함수의 벨만 방정식의 절댓값 오차

$$
|[R(t+1) + \gamma * max_a[Q(s(t_1),a)]-Q(s(t),a(t))]
$$

- 절댓값 오차가 큰 transition을 우선적으로 Experience Replay에서 노출시켜서 가치함수 신경망의 출력 오차를 최적화

### Prioritized Experience Replay 구현

코드 참고

---

## 6.5 A2C 구현

### A2C

분산학습형 심층강화학습 알고리즘

- 에이전트를 여러개 사용해서 학습을 진행
- 모든 에이전트가 같은 신경망을 공유

Advantage 학습

- Q함수를 학습할 때 2단계 이상 미래의 행동가치까지 계산에 넣는 것이 핵심

$$
Q(s_t,a_t)\ ->R(t+1)+\gamma * R(t+2)+(\gamma^2)*max_a [Q(s_{t+2})]
$$

- 하지만 무조건 멀리 본다고 좋은 것이 아니기에 적절한 단계를 선택하는 것이 중요

Actor-Critic

- 정책반복, 가치반복 알고리즘의 조합
- 입력
  - 상태변수
- 출력
  - Actor
    - 행동의 가짓수 
      - 정책반복 알고리즘의 출력과 동일
        - 상태를 입력받고 각 행동이 얼마나 좋은지 출력
        - 이 출력을 소프트맥스 함수로 변환 -> 각 행동이 해당 상태에서 적절한 행동이 될 확률 (정책)이 됨
  - Critic
    - 상태가치
      - 해당 상태에서 앞으로 받을 수 있는 할인 총보상의 기댓값

### A2C 구현

코드 참고

