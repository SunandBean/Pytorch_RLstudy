# Ch3. 역진자 문제를 위한 강화학습 구현

## 3.1 로컬 PC에 강화학습 개별환경 갖추기

### 파이썬 실행 환경인 아나콘다 설치

Anaconda를 설치한 후에 Anaconda Navigator를 이용하여 가상환경 생성 및 IDE 설치

### 강화학습에 사용할 라이브러리 설치

라이브러리

pip install gym

pip install matplotlib

pip install JSAnimation

pip uninstall pyglet -y

pip install pyglet==1.2.4

conda install -c conda-forge ffmpeg



---

## 3.2 역진자 태스크 "CartPole"

### CartPole이란?

OpenAI에서 제공하는 역진자 제어 태스크!

### CartPole 구현

코드 참고

---

## 3.3 다변수, 연속값 상태를 표형식으로 나타내기

### CartPole의 상태

총 4개의 변수이고 observation 변수에 저장되어있다.

- 수레의 위치 (-2.4 ~ 2.4)
- 수레의 속도 (-Inf ~ Inf)
- 봉의 각도 (-41.8도 ~ 41.8도)
- 봉의 각속도 (-Inf ~ Inf)

위의 상태들은 모두 연속된 값이기 때문에 이를 표형식으로 나타내기 위해서는 이산값으로 변환해야 한다.

각각을 6개의 구간을 갖는 이산변수로 변환 => **6^4 = 1296개로 수레의 상태**를 나타낼 수 있음

CartPole의 행동 : 수레를 **오른쪽으로 밀기**, **왼쪽으로 밀기**

따라서 CartPole의 Q함수는 **1296행 2열**로 된 표로 나타낼 수 있다!

### 상태의 이산변수 변환 구현

코드 참고

---

## 3.4 Q러닝 구현

이 절부터 클래스를 정의해서 구현! 총 3개의 클래스를 구현할 것

- Agent
  - CartPole의 수레에 해당
  - Q함수의 수정을 맡을 메서드 - update_Q_function
  - 다음에 취할 행동을 결정하는 메서드 - get_action
- Brain
  - Agent 클래스의 두뇌 역할, Q테이블을 이용해 Q러닝 구현!
  - Agent가 관측한 상태를 이산화하는 메서드 - bins, digitize_state
  - Q테이블을 수정하는 update_Q_table
  - Q테이블을 이용해 행동을 결정하는 decide_action
- Environment
  - OpenAI Gym이 실행되는 실행 환경
  - CartPole을 실행하며, 실행을 맡을 메서드 - run

각 클래스 사이의 정보 흐름

1. 행동을 결정
   1. Agent는 현재 상태를 Brain 클래스에 전달.
   2. Brain 클래스는 전달받은 상태변수를 이산변수로 변환
   3. Q테이블을 참조해서 행동을 결정
   4. 결정한 행동을 Agent에 전달
2. 행동을 취해 환경을 한 단계 진행
   1. Agent는 Environment에 행동을 전달
   2. Environment는 행동을 받아 실행한 결과가 되는 다음 상태와, 상태가 바뀌면서 얻게 되는 즉각보상을 Agent에 반환
3. Q테이블 수정
   1. Agent는 현재 상태와 행동, 상태가 바뀌면서 얻게 된 즉각보상, 다음 상태를 Brain에 전달
   2. Brain은 Agent에서 받은 정보를 바탕으로 Q테이블 수정

- 1에서 3을 반복!



코드 참고!