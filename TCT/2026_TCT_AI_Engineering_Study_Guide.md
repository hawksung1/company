# 2026 TCT AI Engineering 핵심 요약 가이드

> [!TIP]
> 본 가이드는 **2026년 TCT AI Engineering 시험(공통 60% + 선택 40%)**의 출제 범위와 **2023년 기출문제**를 바탕으로 작성되었습니다.
> 내일 시험은 **인터넷 검색 및 참고자료 지참이 가능**하므로, 본 문서를 Ctrl+F 키워드 검색용으로 적극 활용하시기 바랍니다.

---

##  목차
1. [시험 개요 및 합격 전략](#1-시험-개요-및-합격-전략)
2. [[공통] 데이터 준비 및 품질관리 (5점)](#2-공통-데이터-준비-및-품질관리-5점)
3. [[공통] 모델학습 및 일반화 기초 (20점)](#3-공통-모델학습-및-일반화-기초-20점)
4. [[공통] 성능평가 및 검증 (15점)](#4-공통-성능평가-및-검증-15점)
5. [[공통] Agent 기초 (20점)](#5-공통-agent-기초-20점)
6. [[선택] AI Engineering (40점)](#6-선택-ai-engineering-40점)
7. [[기출 연계] MLOps, LLMOps 및 시스템 아키텍처](#7-기출-연계-mlops-llmops-및-시스템-아키텍처)
8. [[기출 연계] LLM 디코딩 파라미터 및 RLHF](#8-기출-연계-llm-디코딩-파라미터-및-rlhf)

---

## 1. 시험 개요 및 합격 전략
* **시험 방식**: 원격 응시, 선다형 20문항 (100점 만점), 120분
* **전략**: 
  1. 인터넷 검색이 가능하므로 복잡한 수식을 외우기보다 **각 개념의 정의, 장단점, 적용 시나리오**를 파악하는 것이 중요합니다.
  2. 기출 문제를 보면 상황 제시형(예: "철수가 상담 데이터 요약 모델을 서빙하는 상황...")이 많으므로 문제의 요구조건(성능 위주인지, 자원 제약인지 등)을 잘 파악해야 합니다.
  3. 올해 시험에는 최신 트렌드인 **Agentic AI, RAG, LLM 성능 메트릭(TTFT, TPOT)**이 대폭 반영되므로 해당 파트를 집중적으로 리마인드하십시오.

---

## 2. [공통] 데이터 준비 및 품질관리 (5점)
### 데이터 불균형 (Class Imbalance) 해결 기법
소수 클래스의 데이터가 너무 적을 때 모델 성능을 보존하기 위해 적용하는 기법입니다.

* **오버샘플링 (Over-sampling)**:
  * **Random Over-sampling**: 소수 클래스 데이터를 단순 복제하여 증식. 오버피팅(Overfitting) 위험이 있고 다양성을 보장하지 못함.
  * **SMOTE (Synthetic Minority Over-sampling Technique)**: 소수 클래스 데이터 중 인접한 이웃(KNN) 사이에 가상의 데이터를 생성하는 기법. 정보 보존 및 일반화에 우수함.
  * **ADASYN (Adaptive Synthetic Sampling)**: 소수 클래스 데이터 중 **다수 클래스 데이터의 밀도가 높은(분류하기 힘든) 영역**에 더 많은 합성 데이터를 생성하는 SMOTE 개선 기법.
* **언더샘플링 (Under-sampling)**:
  * **Random Under-sampling**: 다수 클래스 데이터를 무작위로 제거. 유용한 정보 손실 위험이 큼.
  * **Tomek link**: 서로 다른 클래스에 속하면서 거리가 가장 가까운 쌍(Tomek link)을 찾아, 이 중 다수 클래스 데이터를 제거하여 경계선을 명확히 하는 기법. 데이터를 새로 생성하지 않음.
* **비용 민감 학습 (Cost-Sensitive Learning)**:
  * 소수 클래스 데이터에 더 큰 가중치(Class Weight) 또는 더 큰 Loss 값을 부여하여 불균형을 극복하는 기법.

---

## 3. [공통] 모델학습 및 일반화 기초 (20점)
### 가중치 규제 (Weight Regularization)
모델의 과적합(Overfitting)을 방지하기 위해 손실함수에 가중치 크기에 비례하는 패널티를 더하는 기법입니다.

* **L1 규제 (Lasso)**:
  * 패널티 항: 가중치 절대값의 합 ($\lambda \sum |w|$)
  * 특징: 불필요한 가중치를 **정확히 0**으로 만들어 **희소 모델(Sparse Model)**을 유도하고 특성 선택(Feature Selection) 효과를 냄.
* **L2 규제 (Ridge, Weight Decay)**:
  * 패널티 항: 가중치 제곱의 합 ($\lambda \sum w^2$)
  * 특징: 가중치의 크기가 클수록 더 강한 패널티를 주어 가중치 값을 전반적으로 작고 고르게 만듦. 가중치를 완전히 0으로 만들지는 않지만, **이상치(Outlier)에 더 민감하게 반응**함.

### 최적화 알고리즘 (Optimizers)
* **SGD**: 미니배치 데이터에 대해 단순 경사하강법 적용. 방향성 없이 진동이 심함.
* **Momentum**: 과거의 기울기 변화량(관성)을 반영하여 진동을 줄이고 빠르게 이동.
* **AdaGrad**: 자주 업데이트된 매개변수의 학습률은 낮추고, 드물게 업데이트된 매개변수의 학습률은 높임. 학습이 길어지면 학습률이 0에 수렴해버리는 문제 발생.
* **RMSprop**: AdaGrad의 단점을 해결하기 위해 지수 이동 평균을 사용하여 최근 기울기 위주로 학습률을 조절.
* **Adam (Adaptive Moment Estimation)**: Momentum과 RMSprop의 장점을 결합하여 가장 널리 쓰임.

### Transformer 아키텍처 주요 구성요소
* **Positional Encoding**: Transformer는 입력 데이터를 순차적이 아닌 **병렬로 한 번에 입력**받기 때문에 단어의 순서(위치) 정보가 유실됨. 이를 보정하기 위해 입력 임베딩에 위치 정보를 추가함.
* **Self-Attention**: 입력 문장 내의 모든 단어 간 관계성을 파악하여 가중치(Attention Value)를 계산함.
* **Masked Multi-head Attention**: 디코더(Decoder)에서 사용됨. 미래 시점의 토큰을 보지 못하도록 **Masking(마스킹)** 처리하여 타겟 단어 이전의 단어들만 참조하게 만듦.
* **Multi-head Attention**: Attention 연산을 여러 개의 head로 쪼개어 병렬로 처리함으로써, 모델이 시퀀스의 서로 다른 다양한 측면과 관계성을 포착할 수 있게 함.

---

## 4. [공통] 성능평가 및 검증 (15점)
### 분류(Classification) 평가지표
이진 분류 예측 결과를 기반으로 오차행렬(Confusion Matrix)을 작성하여 아래 지표들을 산출합니다.

| | 실제 Positive | 실제 Negative |
|---|---|---|
| **예측 Positive** | **TP** (True Positive) | **FP** (False Positive) |
| **예측 Negative** | **FN** (False Negative) | **TN** (True Negative) |

* **Accuracy (정확도)**: $\frac{TP+TN}{TP+FN+FP+TN}$
* **Precision (정밀도)**: $\frac{TP}{TP+FP}$
  * *비즈니스 예시*: **정상 거래를 사기(Positive)로 잘못 탐지하여 현업에 방해를 주면 안 되는 사기 탐지 시스템**의 경우 FP를 최소화해야 하므로 **Precision**이 적합함.
* **Recall (재현율, 민감도)**: $\frac{TP}{TP+FN}$
  * *비즈니스 예시*: **실제 폐암 환자(Positive)를 정상으로 잘못 분류(FN)하여 방치하는 의료 진단 시스템**의 경우 FN을 최소화해야 하므로 **Recall**이 적합함.
* **F1-Score**: Precision과 Recall의 조화평균. 데이터 클래스 불균형이 심할 때 정밀도와 재현율의 균형을 평가하기에 적절함.

### 회귀(Regression) 평가지표
* **MSE (Mean Squared Error)**: 오차 제곱의 평균. 오차가 큰 이상치에 민감함.
* **RMSE (Root MSE)**: MSE에 루트를 씌워 실제 타겟 값과 단위를 맞춤.
* **MAE (Mean Absolute Error)**: 오차 절대값의 평균. 이상치에 덜 민감하고 직관적임.
* **MAPE (Mean Absolute Percentage Error)**: 오차를 백분율 비율로 계산하여 상대적 오차 크기를 가늠하기에 적합함.
* **R-squared ($R^2$, 결정계수)**: 모델의 설명력을 나타내며 1에 가까울수록 성능이 우수함.

---

## 5. [공통] Agent 기초 (20점)
### Agent 기본 구조
1. **Orchestrator (Planning)**: 유저 입력을 이해하고, 해야 할 행동의 단계(Task Breakdown)를 설계 및 통제하는 중앙 제어 장치.
2. **Memory**: 단기 기억(Context Window 내 이전 대화 내역) 및 장기 기억(Vector DB 등 외부 저장소를 이용한 정보 복원).
3. **Tools (Action)**: 모델이 스스로 해결할 수 없는 일(계산기, DB 조회, 웹 서칭 등)을 외부 API 호출을 통해 해결할 수 있도록 연결된 도구 목록.

### Context Engineering & Prompt Engineering
* **Prompt 기반 RAG**: 유저 질문과 유사도가 높은 문서를 Vector DB에서 탐색하여 프롬프트의 컨텍스트로 주입해 답변을 생성함으로써 **환각(Hallucination)**을 방지하는 기술.
* **Multi-Model RAG**: 텍스트 정보뿐 아니라 이미지(VLM 활용), 테이블 등의 다양한 모달리티를 벡터화하여 함께 검색 및 제공하는 고도화된 RAG.
* **DSPy (Declarative Self-improving Python)**: 
  * 수작업으로 프롬프트를 작성하는 대신, **Signatures**와 **Modules**를 정의하여 파이프라인의 구조를 세팅함.
  * **Optimizers (예: MIPROv2, BootstrapFewShot)**를 통해 설정한 validation metric(예: 코드 작동 여부, SQL 실행 성공 여부)을 극대화하는 프롬프트 문구와 Few-shot 예시를 알고리즘 기반으로 자동 튜닝함.
* **NL2SQL**: 자연어를 SQL 쿼리로 변환하는 프롬프트 및 아키텍처. 스키마 정보와 테이블 간 관계(Join Key 등)를 컨텍스트로 제공하고, 실행 오류 발생 시 오류 메시지를 다시 모델의 입력 컨텍스트로 전달해 피드백을 주는 **Error Refinement Loop**를 자주 결합함.

---

## 6. [선택] AI Engineering (40점)
### AI Agent 아키텍처링 및 설계
* **Single-Agent vs Multi-Agent**:
  * 복잡한 문제를 해결하기 위해 다양한 역할을 가진 Agent(예: 기획자, 개발자, 검증자)를 정의하고, 각 Agent 간 협업 구조(Collaboration Flow)와 상태 관리(State Management)를 체계화하여 아키텍처링함.
* **Routing**: 사용자의 의도(Intent)에 따라 특정 도구(Tool)나 적합한 Agent 그룹으로 질의를 동적 분배하는 라우터 구조.
* **Agentic AI 최적화**: 
  * **정확도/일관성 (품질 최적화)**: 미세조정(Fine-Tuning), 프롬프트 옵티마이저 적용, 풍부한 메타데이터가 적용된 Context Engineering.
  * **속도/처리량 (성능 최적화)**: 학습/추론 엔진 최적화(vLLM 사용, 양자화), 적절한 모델 Routing.

### LLM 성능 최적화 평가지표
생성형 AI 모델 추론 및 서빙의 성능을 최적화하기 위한 핵심 지표입니다.

```
       [Request Sent] 
             │
             ▼ (Prefill Phase - processing input prompt)
       [First Token Outputted]   ◄─── TTFT (Time To First Token)
             │
             ▼ (Decoding Phase - token-by-token generation)
       [Second Token]
             │                   ◄─── TPOT (Time Per Output Token)
             ▼
       [Final Token / Completion] ◄─── E2E Latency (Total Time)
```

1. **TTFT (Time To First Token)**:
   * **정의**: 요청 전송 후 첫 번째 토큰이 출력될 때까지 걸린 시간.
   * **중요성**: 사용자 경험(UX) 측면에서 시스템 반응이 빠르다고 느끼게 만드는 핵심 지표 (챗봇 응답 즉시성).
   * **요인**: 입력 프롬프트 길이(Prefill 처리 시간), 서버 대기 큐 지연.
2. **TPOT (Time Per Output Token)**:
   * **정의**: 첫 번째 토큰 생성 이후 각각의 후속 토큰이 생성되는 평균 시간.
   * **중요성**: 텍스트가 부드럽고 끊김 없이 화면에 출력되는 속도를 결정.
   * **요인**: 모델 아키텍처 크기, GPU 메모리 대역폭.
3. **E2E Latency**:
   * **정의**: 전체 요청에서 응답 완료까지 소요된 총 시간.
   * **공식**: $E2E\ Latency \approx TTFT + (TPOT \times 생성\ 토큰\ 수)$
4. **Throughput (TPS, Tokens Per Second)**:
   * **정의**: 시스템이 단위 시간(1초)당 처리하고 생성해낸 총 토큰 수.
   * **Throughput 극대화 방법**: **배칭(Batching)**을 적용하여 한 번에 여러 개의 유저 요청을 처리. 하지만 이 경우 개별 요청의 **TTFT는 지연**될 수 있는 Trade-off가 있음.

---

## 7. [기출 연계] MLOps, LLMOps 및 시스템 아키텍처
### 데이터 드리프트 vs 컨셉 드리프트
운영 중인 모델 성능 저하의 대표적 요인입니다.

* **데이터 드리프트 (Data Drift)**:
  * **정의**: 모델 **입력값의 분포(Distribution)**가 시간에 따라 변하는 현상.
  * **감지 방법**: 라벨링 데이터가 없어도 수집된 입력 데이터 자체만으로 감지 가능. 이전 분포와 현재 분포를 **KL-Divergence, PSI(Population Stability Index)** 등의 유사도 메트릭으로 비교해 확인.
* **컨셉 드리프트 (Concept Drift)**:
  * **정의**: 입력 데이터 분포와 관계없이, **입력값과 타겟 라벨 간의 관계(의미/비즈니스 룰)** 자체가 변하는 현상. (예: 코로나 이전과 이후의 소비 행동 패턴 변화).
  * **감지 방법**: 예측 결과와 실제 정답 간의 성능 저하를 추적하며, 반드시 새로운 데이터에 대한 **라벨링 작업이 수반되어야 명확히 감지**할 수 있음.

### MLOps 핵심 구성요소
* **Data Processing**: 학습/추론 단계에서 다량의 데이터를 수집/연계/가공하는 파이프라인.
* **Model Training**: 분산 학습 및 가속화된 알고리즘 엔진 작동 영역.
* **Model Registry (저장소)**: 버전 관리, 메타데이터 보관.
* **Model Serving / Deployment**:
  * **Champion/Challenger Test (Shadow Deployment)**:
    * 현재 활성화되어 서비스 중인 모델(**Champion**)과 검증하려는 신규 후보 모델(**Challenger**)을 모두 서빙 환경에 배치함.
    * 동일한 유저 요청을 수신하지만, 예측값은 오직 **Champion 모델만 반환**하고 Challenger 모델의 예측값은 모니터링 및 정확도 비교/부하 테스트를 위해 백그라운드로 로깅만 수행하는 방식.
    * 운영 부하 성능 검증 및 안전한 정확도 비교에 우수함.
  * **A/B Test**: 실제 사용자 트래픽의 일부 비율을 신규 모델로 분배하여 직접 반응을 비교하는 방식.
* **Model Monitoring**: 드리프트 감지 및 성능 저하 모니터링.

---

## 8. [기출 연계] LLM 디코딩 파라미터 및 RLHF
### LLM 추론(Decoding) 파라미터
* **Temperature**:
  * 다음 토큰 확률 분포의 평탄도를 조절함.
  * 값을 **낮추면(예: 0.1)** 확률이 높은 안정적이고 일관된 어휘만 선택(결정론적), 값을 **높이면(예: 0.9)** 더욱 창의적이고 다양한 어휘가 도출됨.
* **Top-p (Nucleus Sampling)**:
  * 누적 확률 분포가 $p$ 이하가 되는 상위 토큰의 후보군 내에서만 샘플링함. (예: $p=0.9$면 누적 확률 90% 이내의 후보 토큰들만 대상).
* **Top-k**:
  * 단순 개수 기준 상위 $k$개의 토큰 중에서만 샘플링함.
* **do_sample**:
  * `False`로 설정 시 샘플링을 수행하지 않고 항상 확률이 가장 높은 단어만 선택하는 **Greedy Search**로 동작하여 언제나 동일한 결과를 도출함.
* **max_length**:
  * 입력(Prompt) 토큰 수와 출력(Response) 토큰 수의 **합산 최댓값**임. 만약 입력값의 길이가 길다면 이 값이 너무 작을 때 응답이 중간에 잘려버릴 수 있음.

### RLHF (Reinforcement Learning from Human Feedback)
* **개념**: 인간이 생성된 두 개의 응답 중 더 좋은 응답을 선호도로 선택한 데이터를 학습하여 **Reward Model (보상 모델)**을 구축하고, 이 보상 모델의 점수를 극대화하도록 생성 모델을 **강화학습(PPO 등)** 기법으로 최적화함.
* **장점**: 구체적으로 정량적 정의를 내리기 어려운 주관적인 평가 기준(예: "더 친절하게 대답하기", "도움이 되는 정보성 답변")을 사람이 직접 피드백을 줌으로써 모델 정렬(Alignment)을 시킬 수 있음.
