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

역진자 제어 태스크

### CartPole 구현





---

## 3.3 다변수, 연속값 상태를 표형식으로 나타내기

### CartPole의 상태

총 4개의 변수이고 observation 변수에 저장되어있다.

- 수레의 위치 (-2.4 ~ 2.4)
- 수레의 속도 (-Inf ~ Inf)
- 봉의 각도 (-41.8도 ~ 41.8도)
- 봉의 각속도 (-Inf ~ Inf)

### 상태의 이산변수 변환 구현

위의 상태들은 모두 연속된 값이기 때문에 이를 표형식으로 나타내기 위해서는 이산값으로 변환해야 한다.

각각을 6개의 구간을 갖는 이산변수로 변환 => 6^4 = 1296개로 수레의 상태를 나타낼 수 있음

---

## 3.4 Q러닝 구현

