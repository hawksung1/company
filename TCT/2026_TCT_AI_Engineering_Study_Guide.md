# 2026 TCT AI Engineering 핵심 요약 가이드 (시험 대비 최종본)

> [!IMPORTANT]
> **본 문서는 인터넷 검색은 가능하나 사내망(Wire, Singlex 등) 접속이 통제되는 시험 환경을 고려하여 작성되었습니다.**
> 사내 교육과정(딥러닝 실무, 언어지능 실무, 시각지능 실무, 효율 향상 기법 등)의 모든 핵심 개념과 수식, 아키텍처 설명을 본 문서에 단권화하여 **Ctrl + F 키워드 검색만으로 즉시 확인 가능**하도록 구성하였습니다.

---

## 1. 딥러닝 기초 및 학습 테크닉 (Deep Learning Basic & Advanced)

### 활성화 함수 (Activation Functions)
입력과 가중치의 합을 출력값으로 변환하여 모델에 **비선형성(Non-linearity)**을 부여하는 함수입니다.

* **Sigmoid (시그모이드)**: $f(x) = \frac{1}{1 + e^{-x}}$
  * **특징**: 출력 범위 $(0, 1)$. 확률 값 매핑에 유용.
  * **단점**: **Gradient Vanishing (기울기 소멸)** 문제 발생. $x$가 크거나 작으면 기울기가 0에 수렴. 출력의 평균이 0이 아님(Non-zero centered).
* **Tanh (쌍곡탄젠트)**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
  * **특징**: 출력 범위 $(-1, 1)$. 원점 기준 대칭(Zero-centered)이라 시그모이드보다 학습 수렴 속도가 빠름.
  * **단점**: 여전히 양극단에서 **Gradient Vanishing** 문제가 존재함.
* **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$
  * **특징**: $x > 0$ 영역에서 기울기가 항상 1이므로 **Gradient Vanishing을 크게 완화**함. 연산 속도가 매우 빠름.
  * **단점**: **Dying ReLU** 현상 발생. $x \le 0$일 때 기울기가 0이 되어, 한 번 음수 값을 출력한 뉴런이 더 이상 업데이트되지 않고 영원히 죽을 수 있음.
* **Leaky ReLU**: $f(x) = \max(\alpha x, x)$ (보통 $\alpha = 0.01$)
  * **특징**: $x \le 0$인 영역에서도 아주 작은 기울기 $\alpha$를 흘려보내 **Dying ReLU 현상을 해결**.
* **ELU (Exponential Linear Unit)**: $f(x) = x \text{ (if } x > 0 \text{)}, \alpha(e^x - 1) \text{ (if } x \le 0 \text{)}$
  * **특징**: $x \le 0$인 구간에서 부드러운 곡선 형태를 취해 기울기가 항상 연속적임. 노이즈에 강하며 zero-centered에 가깝게 동작함.

---

### 가중치 초기화 (Weight Initialization)
초기 가중치 설정은 기울기 소멸/폭주를 방지하는 첫 걸음입니다. 모두 0으로 초기화할 경우 모든 뉴런이 동일한 연산을 수행(대칭성 문제)하여 다층 구조의 이점이 사라집니다.

* **Xavier (Glorot) 초기화**:
  * **공식**: $W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$ 또는 $U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$
  * **적용**: **Sigmoid, Tanh** 활성화 함수를 사용하는 레이어에 최적.
* **He (Kaiming) 초기화**:
  * **공식**: $W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$ 또는 $U\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$
  * **적용**: **ReLU, Leaky ReLU** 계열의 활성화 함수를 사용하는 레이어에 최적.

---

### 정규화 레이어 (Normalization Layers)
피처의 스케일을 조정하여 안정적인 학습을 돕고 내부 공변량 변화(Internal Covariate Shift)를 완화합니다.

```
Batch Norm:     [Batch, Height, Width] 차원에서 평균/분산 계산 (채널별 독립)
Layer Norm:     [Channel, Height, Width] 차원에서 평균/분산 계산 (샘플별 독립)
Instance Norm:  [Height, Width] 차원에서 평균/분산 계산 (샘플 & 채널별 독립)
Group Norm:     [Group, Height, Width] 차원에서 평균/분산 계산 (채널을 G개 그룹으로 묶음)
```

1. **Batch Normalization (배치 정규화 - BN)**:
   * **수식**: $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, $y = \gamma \hat{x} + \beta$
   * **특징**: 배치 차원($N$)을 기준으로 평균과 분산을 구함. 
   * **단점**: 배치 크기에 의존적임. 미니배치 크기가 작으면(예: GPU 메모리 한계로 2 또는 4인 경우) 통계값 변동이 심해져 성능이 급격히 저하됨. RNN/Transformer에는 적합하지 않음.
2. **Layer Normalization (레이어 정규화 - LN)**:
   * **특징**: 배치와 상관없이 **단일 샘플 내의 피처 차원(채널/히든 스테이트)**을 기준으로 정규화.
   * **적용**: NLP, 시퀀스 모델, **Transformer(Self-Attention)** 계열에 표준적으로 사용됨. 미니배치 크기로부터 독립적임.
3. **Instance Normalization (인스턴스 정규화 - IN)**:
   * **특징**: 각 샘플의 채널별로 이미지 가로/세로 방향으로만 정규화 진행. 주로 스타일 트랜스퍼(Style Transfer), 이미지 생성 분야에서 화풍이나 스타일 편차를 지우는 용도로 활용.
4. **Group Normalization (그룹 정규화 - GN)**:
   * **특징**: 채널 차원을 $G$개의 그룹으로 나누어 그룹 단위로 평균/분산을 구함. 배치 크기가 극도로 작을 때 BN의 대안으로 활용하기 좋음.

---

### 오버피팅 완화 (Regularization) & 편향-분산 트레이드오프
* **L1 규제 (Lasso)**: 손실함수에 가중치 절대값의 합($\lambda \sum |w|$)을 추가. 가중치를 **완전히 0**으로 만들어 특성 선택(Feature Selection)을 유도함.
* **L2 규제 (Ridge / Weight Decay)**: 손실함수에 가중치 제곱의 합($\lambda \sum w^2$)을 추가. 큰 가중치를 억제하여 모델을 매끄럽게 만듦. 이상치에 민감함.
* **Dropout (드롭아웃)**: 학습 시 에포크마다 설정 비율만큼 뉴런을 무작위로 비활성화하여 특정 뉴런에 대한 의존도를 분산시킴.
* **Early Stopping (조기 종료)**: 검증 손실(Validation Loss)이 일정 에포크 이상 감소하지 않고 증가 추세로 돌아설 때 오버피팅이 시작된 것으로 보고 학습을 조기 중단함.
* **Ensemble (앙상블)**:
  * **Bagging (배깅)**: 중복을 허용한 복원 추출(Bootstrap)을 통해 여러 학습 데이터를 만들고, 각각 독립적인 모델을 병렬 학습시킨 뒤 투표/평균으로 합산 (예: Random Forest). **모델의 분산(Variance)을 줄임**.
  * **Boosting (부스팅)**: 이전 모델이 틀린 오차를 보완하도록 순차적으로 약한 모델을 결합 (예: XGBoost, LightGBM). **모델의 편향(Bias)을 줄임**.
* **Bias-Variance Trade-off**:
  * **Bias(편향)**: 모델 예측값과 실제 정답의 차이. 높으면 과소적합(Underfitting).
  * **Variance(분산)**: 다른 데이터셋을 썼을 때 예측값이 흔들리는 정도. 높으면 과적합(Overfitting).
  * 수식: $\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$

---

### 모델 캘리브레이션 (Calibration)
모델의 최종 출력 확률(Confidence)이 실제 클래스 분포와 부합하도록 조정하는 기법입니다.

* **ECE (Expected Calibration Error)**: 모델이 예측한 신뢰도(Confidence)와 실제 달성한 정확도(Accuracy)의 차이를 가중 평균하여 구한 오차 메트릭.
* **Temperature Scaling (온도 스케일링)**:
  * **기법**: 모델 학습이 완료된 후, 출력 로짓(Logit) 벡터 $z$를 특정한 스칼라 상수 $T$ (Temperature)로 나누어 Softmax 연산에 입력: $\hat{p} = \text{Softmax}(z / T)$
  * **역할**: $T > 1$이면 확률 분포가 부드러워져(Overconfidence 완화) 예측 신뢰도와 정확도가 잘 매칭되도록 캘리브레이션을 돕고, $T < 1$이면 선명해짐. 분류 경계선이나 파라미터 값은 전혀 변하지 않음.
* **Label Smoothing (라벨 스무딩)**:
  * **기법**: 원-핫 인코딩된 타겟 라벨 $[1, 0, 0]$을 $[1-\alpha, \frac{\alpha}{K-1}, \frac{\alpha}{K-1}]$ 형태로 변환하여 모델이 극단적인 100% 확신을 갖지 않도록 유도. 과적합 방지 및 일반화 성능 확보에 우수.

---

## 2. 시각 지능 실무 (Computer Vision)

### 컴퓨터 비전의 도전 과제 (Challenges) & 특징
* **도전 과제**: 시점 변화(Viewpoint variation), 조명(Illumination), 스케일(Scale), 기형(Deformation), 겹침(Occlusion), 잡동사니(Clutter), Motion(움직임), 동 분류 객체 내 변화(Intra-class variation), 지역적 애매함(Local Ambiguity).
* **Gestalt 이론**: 인지 구조가 개별 픽셀이 아닌 전체적인 형태(Similarity, Proximity, Continuity 등)를 먼저 그룹화하여 받아들인다는 시각 인지 이론.

---

### 대표 CNN 아키텍처 및 핵심 연산
1. **AlexNet**: 5개 합성곱 레이어 + 3개 전결합 레이어로 구성. 최초로 ReLU 적용, 과적합 방지를 위해 Dropout과 데이터 증강을 대대적으로 도입. LRN(Local Response Normalization) 사용.
2. **VGGNet (3x3 필터의 이점)**:
   * **원리**: 5x5 합성곱 1개 대신 **3x3 합성곱 2개**를 쌓고, 7x7 합성곱 1개 대신 **3x3 합성곱 3개**를 쌓아 구성.
   * **이유**:
     1. **파라미터 수 감소**: 7x7 합성곱의 파라미터는 $7^2 C^2 = 49 C^2$인 반면, 3x3 합성곱 3개는 $3 \times (3^2 C^2) = 27 C^2$으로 대폭 절감됨.
     2. **비선형성 증가**: 활성화 함수를 더 자주 거쳐 표현력이 풍부해짐.
     3. **Receptive Field(수용 영역)**는 동일하게 유지됨.
3. **InceptionNet (GoogLeNet)**:
   * **구조**: 한 레이어에서 다양한 크기($1\times1$, $3\times3$, $5\times5$)의 필터와 Max Pooling을 병렬 수행하는 Inception 모듈 채택.
   * **1x1 Convolution의 핵심 역할**: 합성곱 연산 전에 채널 수를 줄이는 차원 축소(Dimensionality Reduction)를 수행하여 **연산 파라미터 수를 획기적으로 압축**함. 비선형성 추가 효과도 있음.
4. **ResNet (Residual Learning)**:
   * **해결 문제**: 망이 깊어질 때 정확도가 정체되거나 저하되는 Degradation(퇴화) 문제 해결.
   * **개념**: 지름길 경로(Skip Connection)를 추가하여 입력 $x$를 그대로 전달하고, 레이어는 잔차 $F(x) = H(x) - x$만 학습하도록 만듦. 미분 시 그래디언트가 $F'(x) + 1$이 되므로 1이라는 보존값 덕분에 아무리 깊어도 기울기가 사라지지 않고 전파됨.
5. **DenseNet**: 
   * **개념**: 레이어의 입력을 이후 모든 레이어의 입력과 연결(Concatenation)하는 조밀한 연결 방식. 피처 맵 재사용성을 극대화하여 매개변수 효율성이 훌륭함.

---

### 경량화 CNN (Lightweight CNN)
모바일, 임베디드 기기 등 자원 제약 환경을 타겟팅한 구조입니다.

* **SqueezeNet**: **Fire Module** 구조 채택. 1x1 conv로 채널을 압축(Squeeze)한 후, 1x1 및 3x3 conv로 채널을 복원(Expand)하는 설계를 통해 파라미터 수를 1/50 수준으로 절감.
* **MobileNet (Depthwise Separable Convolution)**:
  * **원리**: 일반적인 합성곱 연산을 두 개의 단계로 나눔.
    1. **Depthwise Convolution**: 입력 데이터의 채널별로 각각 독립적인 필터 연산을 1회 수행.
    2. **Pointwise Convolution**: 채널 연산이 끝난 맵에 $1\times1$ 합성곱을 수행하여 채널을 통합 및 조합.
  * **연산량 비교**:
    * 일반 합성곱: $D_K \times D_K \times M \times N \times D_W \times D_H$ (필터 크기 $D_K$, 입력 채널 $M$, 출력 채널 $N$, 이미지 가로세로 $D_W, D_H$)
    * Depthwise Separable: $(D_K \times D_K \times M \times D_W \times D_H) + (1 \times 1 \times M \times N \times D_W \times D_H)$
    * 절감 비율: $\frac{\text{Separable}}{\text{Standard}} \approx \frac{1}{N} + \frac{1}{D_K^2}$. 보통 3x3 필터를 사용할 경우 **연산량이 대략 1/8~1/9로 크게 감소**함.

---

### 전이 학습 (Transfer Learning) 매트릭스 전략
사전 학습된 가중치(Backbone)를 새로운 도메인 데이터셋에 적용하는 4대 시나리오별 튜닝 가이드입니다.

| 데이터셋 크기 | 사전 학습 모델과의 유사도 | 적용 전략 및 기법 |
|---|---|---|
| **Large (큼)** | **High (유사함)** | **Partial Fine-Tuning**: 상위(Out) 레이어 위주로 조금만 튜닝 진행. 데이터가 충분하므로 오버피팅 부담이 적음. |
| **Large (큼)** | **Low (상이함)** | **Full Fine-Tuning**: 전체 가중치를 새로 업데이트. 데이터량이 풍부하므로 새로운 도메인 도메인에 맞춰 완전히 미세조정해도 안정적임. |
| **Small (작음)** | **High (유사함)** | **Frozen Feature Extractor**: 백본 가중치는 완전히 고정(Freeze)하고 맨 위 분류기(Linear Classifier) 레이어만 새로 학습시킴. 오버피팅 방지에 효과적. |
| **Small (작음)** | **Low (상이함)** | **가장 어려운 상황**: 데이터가 적은데 도메인도 다름. 백본의 초기/중간 단 레이어만 고정하고 이후 레이어는 규제를 강하게 넣어 파인튜닝하거나, 증강을 적극 활용함. |

---

## 3. 자연어 처리 실무 (Natural Language Processing)

### 자연어 전처리 및 분석 단계
1. **형태소 분석 (Morphological Analysis)**: 어절을 의미를 가지는 최소 단위인 형태소로 분류.
2. **구문 분석 (Syntactic Analysis)**:
   * **구구조 분석 (Phrase Structure Parsing)**: 문장을 구(Noun Phrase, Verb Phrase 등) 단위의 계층 트리 구조로 분석.
   * **의존 구문 분석 (Dependency Parsing)**: 단어들 간의 지배-의존 관계를 분석하여 관계 화살표로 표현.
3. **의미 분석 (Semantic Analysis)**: 단어나 문장의 고유한 의미 및 역할을 확정.
   * **의미역 결정 (SRL, Semantic Role Labeling)**: 문장의 서술어를 기준으로 주변 논항들의 구체적 의미 관계(누가, 언제, 무엇을 등 - Agent, Patient)를 규명하는 기술.
4. **화용 분석 (Pragmatic Analysis)**: 주변 상황, 대화 문맥을 고려하여 실제 말하는 의도와 어조를 규명.
* **핵심 태스크**:
  * **상호참조해결 (Coreference Resolution)**: "민수는 의사다. 그는 친절하다"에서 대명사 "그"가 "민수"를 지시함을 매핑하는 태스크.
  * **개체명 인식 (NER)**: 텍스트에서 인물, 기관, 시간, 장소 등의 고유 명사를 분류 및 추출.

---

### 순환 신경망 (RNN, LSTM, GRU)
시퀀스 데이터 처리를 위한 루프 구조의 신경망입니다.

* **Vanilla RNN**: 이전 히든 스테이트 $h_{t-1}$와 현재 입력 $x_t$를 받아 $h_t = \tanh(W h_{t-1} + U x_t + b)$를 연산. 문장이 길어질수록 멀리 있는 과거 정보가 유실되는 **Long-Term Dependency(장기 의존성)** 문제 및 역전파 시 기울기가 사라지는 **Gradient Vanishing** 문제가 심각함.
* **LSTM (Long Short-Term Memory)**:
  * **원리**: 별도의 메모리 통로인 **Cell State ($C_t$)**를 두고 가감 연산으로만 정보를 보존함으로써 기울기 유실을 방지. 3개의 게이트로 제어.
    1. **Forget Gate ($f_t$)**: 이전 메모리를 얼마나 잊을지 결정. $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$
    2. **Input Gate ($i_t, \tilde{C}_t$)**: 새 정보를 얼마나 반영할지 결정. $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$, $\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$
    3. **Cell State 업데이트**: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
    4. **Output Gate ($o_t$)**: 어떤 값을 다음 은닉 스테이트로 보낼지 결정. $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$, $h_t = o_t * \tanh(C_t)$
* **GRU (Gated Recurrent Unit)**:
  * **특징**: LSTM의 복잡한 구조를 단축. Cell State와 Hidden State를 하나로 병합.
  * **구성**: **Reset Gate**와 **Update Gate** 2개만 사용하여 파라미터 수를 줄이고 연산 효율성을 높임.

---

### BERT (Bidirectional Encoder Representations from Transformers)
Transformer의 Encoder 레이어를 깊게 쌓은 양방향 사전 학습 언어 모델입니다.

* **사전 학습 태스크 (Pre-training Tasks)**:
  1. **MLM (Masked Language Model)**: 입력 문장 중 15% 토큰을 무작위로 타겟팅하여, 그중 80%는 `[MASK]` 토큰으로 치환, 10%는 엉뚱한 임의 단어, 10%는 원본 그대로 유지한 후 해당 위치의 원본 단어를 예측하도록 학습. 양방향 컨텍스트 학습 유도.
  2. **NSP (Next Sentence Prediction)**: 두 문장 A와 B가 주어졌을 때, B가 A의 실제 뒤이어 나오는 문장인지 여부(IsNext / NotNext)를 이진 분류로 학습.
* **토크나이징 기법**:
  * **BPE (Byte Pair Encoding)**: 가장 자주 등장하는 문자 바이트 쌍을 반복 결합하여 서브워드(Subword) 사전을 구성하는 빈도 기반 알고리즘. Out-Of-Vocabulary(OOV) 문제를 줄여줌.
  * **WordPiece**: BPE와 달리 병합 시 단순히 빈도가 높은 것 대신, **우도(Likelihood)가 가장 크게 증가하는 단어 쌍**을 병합하여 사전을 채우는 기법 (Google BERT 기본 토크나이저).
* **BERT의 주요 변형 모델 (Variants)**:
  * **ALBERT (A Lite BERT)**: BERT의 파라미터 감축 기법 적용.
    1. *Factorized Embedding Parameterization*: 어휘 임베딩의 차원 $E$와 은닉 상태 차원 $H$를 분리하여 파라미터 수를 줄임 ($V \times H \rightarrow V \times E + E \times H$).
    2. *Cross-Layer Parameter Sharing*: 모든 인코더 레이어간 파라미터(Self-Attention, FFN)를 공유해 크기를 축소.
  * **DistilBERT**: Knowledge Distillation(지식 증류)을 적용하여 원본 BERT의 출력 분포를 모사하도록 학생 모델을 학습시킴. 크기는 40% 작지만 97%의 이해력 유지.
  * **BART**: Encoder-Decoder 구조를 취해 입력의 노이즈를 제어하고 인코딩한 후 다시 오토리그레시브하게 복원하는 구조. 요약 및 번역, 생성형 태스크에 최적화.

---

## 4. 딥러닝 및 서빙 효율화 기법 (AI Efficiency & Optimization)

### 자가 지도 학습 (Self-Supervised Learning)
레이블이 없는 데이터로부터 데이터 자체의 구조나 성질을 통해 레이블을 자동으로 유도하여 표현(Representation)을 학습하는 기법입니다. (예: BERT의 MLM)

* **SimCLR**: 시각 지능 분야의 대조 학습(Contrastive Learning) 대표 프레임워크.
  * **메커니즘**: 1장의 원본 이미지에 서로 다른 두 가지 데이터 증강(Data Augmentation - 회전, 자르기 등)을 가하여 긍정 쌍(Positive pair)을 만들고, 다른 이미지와는 부정 쌍(Negative pair)을 만듦. 
  * **목표**: 합성곱 신경망을 통과한 두 긍정 쌍 임베딩 간의 유사도(Cos Similarity)는 극대화하고, 부정 쌍들과의 유사도는 최소화하는 방향으로 손실 함수(NT-Xent Loss)를 계산해 표상 학습 수행.

---

### 모델 압축 기술 (Model Compression)
학습된 모델의 크기를 경량화하여 추론 메모리를 아끼고 속도를 향상시킵니다.

1. **가지치기 (Pruning)**:
   * **Unstructured Pruning (비구조적 가지치기)**: 중요도(예: 가중치의 절대값 크기)가 낮은 개별 가중치 파라미터 값을 단순 0으로 변환. 0으로 변한 가중치가 산발적으로 흩어지므로 희소 행렬(Sparse Matrix) 연산 가속기가 없는 범용 하드웨어에서는 실제 추론 속도 개선율이 미미함.
   * **Structured Pruning (구조적 가지치기)**: 아예 불필요한 뉴런, 채널, 혹은 특정 레이어 블록 전체를 칼로 도려내듯 통째로 삭제. 가중치 행렬 크기 자체가 줄어들어 일반 GPU에서도 즉각적인 실질 추론 가속화 체감이 가능함.
2. **지식 증류 (Knowledge Distillation - KD)**:
   * **원리**: 크고 정교한 교사 모델(Teacher)의 "Dark Knowledge(어둠의 지식)"를 작고 빠른 학생 모델(Student)로 전수.
   * **수식**: $\mathcal{L} = (1 - \alpha) \mathcal{L}_{CE}(y, \sigma(z_s)) + \alpha T^2 \mathcal{L}_{KL}(\sigma(z_t / T), \sigma(z_s / T))$
   * **설명**: 학생 모델은 실제 정답 라벨(Hard Target)뿐만 아니라, 교사 모델이 온도 파라미터 $T$를 통해 완화해 뽑아낸 로짓 확률 분포(Soft Target)를 함께 모사하여 학습 속도와 성능을 대폭 끌어올림.
3. **양자화 (Quantization)**:
   * 32비트 실수형(FP32) 파라미터를 16비트(FP16), 8비트(INT8), 혹은 4비트(INT4) 정수형 등으로 매핑하여 비트 수 자체를 단축하는 기법. 대역폭 병목 해소 및 VRAM 요구량 축소에 절대적 기여.

---

### 분산 학습 및 다중 GPU 병렬화 (Distributed Training)
단일 GPU 메모리에 다 올릴 수 없는 대규모 데이터셋 또는 모델 크기(LLM 등)를 극복하기 위한 다중 가속기 활용 기법입니다.

* **데이터 병렬화 (Data Parallelism)**:
  * **개념**: 동일한 모델 가중치를 모든 GPU에 복제해 올린 후, 학습 데이터 미니배치만 서로 다르게 쪼개어 연산.
  * **Parameter Server**: 중앙 서버(Parameter Server)가 가중치를 유지관리하며, 연산을 수행하는 Worker 가속기들로부터 그래디언트를 받아 전역 가중치를 갱신하여 다시 Worker들에게 전송하는 허브 앤 스포크 방식. 병목 가능성 있음.
  * **Ring-AllReduce (Decentralized)**: 중앙 집중 서버 없이 GPU끼리 고리(Ring) 형태로 서로 이웃한 GPU와 그래디언트 조각을 주고받으며 누적 연산하는 분산 합산 방식. 네트워크 효율을 극대화하여 멀티 GPU 인프라의 표준 분산 엔진으로 쓰임.
* **모델 병렬화 (Model Parallelism)**:
  * **파이프라인 병렬화 (Pipeline Parallelism - Inter-layer)**: 레이어 단위로 잘라서 서로 다른 GPU에 배치 (예: Layer 1~10은 GPU 0, 11~20은 GPU 1). 선행 GPU의 연산이 끝날 때까지 후속 GPU가 대기해야 하는 **Bubble Overhead**가 한계이며, 이를 해결하기 위해 미니배치를 마이크로배치(Micro-batch)로 쪼개어 공급하는 스케줄링(예: GPipe)을 사용함.
  * **텐서 병렬화 (Tensor Parallelism - Intra-layer)**: 단일 레이어 내 가중치 행렬 자체를 쪼개어 여러 GPU에서 동시 행렬 연산 처리 (예: Megatron-LM). 멀티 헤드 어텐션의 어텐션 투영 행렬을 세로/가로로 쪼개어 분산 연산한 뒤 All-Reduce로 결과를 결합. 매우 빠른 초고속 상호 연동 대역폭(NVLink 등) 인프라가 필수적임.

---

## 5. 생성형 AI 및 에이전트 공학 (LLM & Agentic AI)

### LLM 추론 디코딩 파라미터
* **Temperature (온도)**:
  * 로짓을 스케일링하여 확률 분포의 샤프니스를 조절.
  * $T \rightarrow 0$: 가장 확률이 높은 토큰만 결정론적으로 출력 (Greedy Search와 동일). 코딩, 수학, 번역 등 정확하고 명확한 대답이 필요한 작업에 권장.
  * $T \rightarrow 1$: 본래 학습된 토큰 확률을 그대로 사용. 소설 창작, 브레인스토밍 등 다양하고 창의적인 어휘 도출에 권장.
* **Top-p (Nucleus Sampling)**:
  * 누적 확률값이 설정한 임계치 $p$ 이내가 되는 상위 집합 토큰 중에서만 샘플링. (예: $p=0.9$이면 누적 90% 비중을 갖는 핵심 어휘들 사이에서만 선택하므로 이상한 단어를 배제하는 안정성 확보 가능).
* **Top-k**:
  * 확률이 가장 높은 상위 $k$개의 토큰 후보만 남기고 나머지는 무시한 채 샘플링 진행.
* **do_sample**:
  * `False`로 지정하면 무작위 샘플링을 비활성화하고 항상 최고 확률 단어만 취하는 Greedy Search로 고정됨. 일관된 응답 출력을 위해 사용.

---

### RAG (Retrieval-Augmented Generation, 검색 증강 생성)
LLM의 자체 지식 한계 및 환각(Hallucination)을 막기 위해 외부 Vector DB에서 관련 지식을 임베딩 검색하여 컨텍스트로 전달해 답변하는 구조입니다.

```
[Query] ──► [Embedding Model] ──► [Vector Index Search (HNSW/IVF)]
                                          │
                                          ▼  (Candidate chunks)
                                    [Re-Ranker] (Cross-Encoder)
                                          │
                                          ▼  (Top-K chunks)
[LLM Prompt] ◄── [Context Insertion] ◄────┘
```

* **청킹 전략 (Chunking Strategies)**:
  * **Fixed-size**: 고정된 문자 수 또는 토큰 단위로 자르기. 경계선 파괴 위험.
  * **Recursive Character**: 줄바꿈(`\n`), 마침표(`.`), 공백순으로 계층적으로 쪼개 문맥을 살림.
  * **Semantic Chunking**: 임베딩 간 코사인 유사도를 활용하여 의미의 변화(Semantic Drift)가 발생하는 경계점을 찾아 동적으로 자름.
* **Vector Indexing**:
  * **HNSW (Hierarchical Navigable Small World)**: 다층 그래프 구조로 데이터 노드를 연결하여 근사 최근접 이웃(ANN)을 매우 빠른 속도로 정확히 탐색함. 메모리 소비량이 크지만 실시간 검색 속도가 우수함.
  * **IVF (Inverted File Index)**: 클러스터링을 통해 탐색 범위를 축소하여 인덱스 용량을 대폭 아끼지만, 정확도가 소폭 떨어질 수 있음.
* **검색 평가 지표**:
  * **MRR (Mean Reciprocal Rank)**: 정답 문서가 검색 목록 중 몇 번째 순위에 위치하는지 역수를 구함: $\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{rank_i}$
  * **MAP (Mean Average Precision)**: 관련성 높은 문서들의 배치 순서와 전체적인 정확성을 종합 평가.
* **고급 RAG 컴포넌트**:
  * **Re-Ranking (리랭킹)**: Bi-Encoder 기반의 벡터 유사도로 빠르게 Top-50 후보를 선정한 다음, 연산량은 높지만 정확한 **Cross-Encoder 기반 Re-Ranker** 모델로 유저 질문과 후보 청크 간의 다이렉트 매칭 스코어를 계산해 상위 3~5개만 최종 선별함.
  * **Query Rewriting (질의 재작성)**: 유저의 불명확한 다단문 질문을 LLM을 사용하여 여러 개의 명확한 검색용 검색어로 변환해 분산 검색하는 기법.

---

### Agentic AI 설계 및 DSPy 최적화
중앙 LLM의 판단력을 바탕으로 스스로 외부 도구(Tools)를 호출하고 행동 계획(Planning)을 수립하며 장단기 기억(Memory)을 활용하는 인공지능 에이전트 구조입니다.

* **ReAct (Reason + Action) 프레임워크**:
  * **개념**: 에이전트가 "Thought (사고) -> Action (행동/도구 호출) -> Observation (결과 관측)" 루프를 순환 수행하며 의사결정 경로를 명시적으로 풀어가는 기법.
* **Multi-Agent 시스템**:
  * 하나의 에이전트가 모든 일을 처리하지 않고, 특정 전문성(예: Code Writer, Code QA-Verifier)을 주입받은 개별 에이전트들이 통신(State Sharing)을 거쳐 최종 아웃풋을 빌딩하는 협력 모델.
* **DSPy (Declarative Self-improving Python)**:
  * **핵심 기능**: 프롬프트를 텍스트로 열심히 작성해 삽질하는 개발 패러다임을 **프로그래밍 코드와 알고리즘식 자동 옵티마이저** 방식으로 전환.
  * **구성 요소**:
    * **Signature**: 에이전트의 입력과 출력을 선언적으로 정의 (예: `question -> answer`, `context, question -> sql`).
    * **Module**: CoT(ChainOfThought), Predict, Program 등의 로직을 모듈 형태로 감쌈.
    * **Optimizer (BootstrapFewShot, MIPROv2)**: 데이터셋과 평가지표(Metric)를 기반으로, 어떤 프롬프트 문구와 Few-shot 예시들이 결합할 때 Metric이 극대화되는지를 최적화 루프를 돌며 가중치/프롬프트 튜닝하듯 자동으로 학습 및 확정함.
* **NL2SQL 아키텍처**:
  * 자연어를 받아 해당 관계형 DB 스키마(CREATE TABLE DDL 및 코멘트)를 컨텍스트로 참조하여 SQL을 생성.
  * **Error Refinement Loop**: 생성된 SQL을 테스트 DB 환경에서 직접 가짜 실행(Dry Run)해본 후, 구문 오류나 실행 오류 결과 메시지를 다시 모델 프롬프트의 오류 컨텍스트로 피드백 전달하여 자가 수정하게 만드는 루프를 필수 탑재.

---

## 6. MLOps, LLMOps 및 성능 최적화 평가지표

### 드리프트 감지 및 모니터링 (Drift Detection)
운영 중인 머신러닝 시스템의 성능을 유지하기 위해 반드시 모니터링해야 하는 요소입니다.

* **데이터 드리프트 (Data Drift)**:
  * **정의**: 모델 **입력값의 분포(Distribution)**가 시간에 따라 변하는 현상. ($P(X_{old}) \neq P(X_{new})$)
  * **핵심**: 정답 라벨($Y$)이 없어도 수집된 입력 피처 데이터만으로 상시 감지 가능.
  * **감지 방법**: 예전 피처 분포와 현재 유저가 넣는 피처 분포를 통계적 거리 함수(**KL-Divergence, PSI(Population Stability Index), KS-test**)로 계산해 임계값을 이탈할 시 알람 발생.
* **컨셉 드리프트 (Concept Drift)**:
  * **정의**: 입력값 자체의 분포 변화와 무관하게, **입력값과 정답 라벨 사이의 실제 룰이나 비즈니스 정답 관계(Concept)**가 영구적으로 달라지는 현상. ($P(Y_{old}|X) \neq P(Y_{new}|X)$)
  * **핵심**: 실제 유저 구매 데이터나 사후 확정된 라벨링 결과를 결합하여 계산해야 하므로 **라벨 수집이 필수적이며 감지에 지연이 발생**함.
  * **감지 방법**: 일정 윈도우 기간 내 모니터링되는 실시간 예측 Accuracy/F1-score의 하락 추세 감지.

---

### MLOps 성숙도 3단계 (Level 0, 1, 2)
구글 클라우드에서 정의한 MLOps 성숙도 레벨입니다.

1. **Level 0 (Manual Process)**:
   * 모델 학습, 배포, 검증까지 모든 단계가 수동으로 진행됨. 데이터 과학자가 오프라인으로 뽑은 데이터를 학습시킨 후 모델 아티팩트만 엔지니어에게 전달해 서빙. 재학습이 드물거나 수동으로만 수행됨.
2. **Level 1 (ML Pipeline Automation)**:
   * 데이터 검증, 전처리, 모델 학습, 모델 검증 과정이 자동으로 연결된 **파이프라인 트리거 형태**로 구축됨. 운영 중에 들어오는 새로운 데이터 유입이나 드리프트 감지 알람에 따라 파이프라인이 자동 구동되어 새로운 버전의 모델 아티팩트를 자동 생산함.
3. **Level 2 (CI/CD Pipeline Automation)**:
   * 모델 학습 파이프라인 자체의 코드 변경(새로운 피처 추가, 알고리즘 변경 등)이 빌드, 테스트되고 운영 환경에 무중단 지속 통합/배포(CI/CD)되는 완성도 높은 소프트웨어 엔지니어링 수준.

---

### 서빙 및 부하 검증 기법 (Shadow Deployment)
배포 시 정합성과 속도를 운영 위험 없이 검증하는 방안입니다.

* **Shadow Deployment (Champion/Challenger)**:
  * **방식**: 기존의 검증 완료된 모델(**Champion**)과 신규 개발된 대안 모델(**Challenger**)을 나란히 서빙 환경에 올린 다음, 라이브 트래픽 유저 요청을 복제(Mirroring)하여 두 모델 모두에게 동시에 던집니다.
  * **출력**: 실제 외부 유저에게는 오직 **Champion 모델의 출력값만 반환**하고, Challenger 모델의 연산값과 소요 속도는 내부 데이터베이스에 로깅하여 사후 성능(정밀도, Latency, 서버 부하 상태)을 완벽히 모니터링 및 성능 비교합니다.
  * **장점**: 유저에게 잘못된 비정상 결과를 노출할 위험이 0%이며 안전하게 실트래픽 부하 테스트가 가능함.

---

### ML System 테스트 4단계
ML 기반 소프트웨어 릴리즈 전 검증해야 할 4가지 품질 차원입니다.

1. **Infrastructure Test**: 시스템 구동 환경(GPU 가속 드라이버, CUDA 호환성, 필요한 패키지 라이브러리 및 하드웨어 가용 스토리지)이 안정적인지 체크.
2. **Functionality Test**: 입력 데이터가 API 명세에 따라 들어올 때, 모델이 예외 상황(Null값, 자료형 불일치 등)을 터뜨리지 않고 규격화된 예측 데이터 출력을 내어주는지 여부 확인.
3. **Training Test**: 모델 재학습 시 loss가 이론대로 감소하고 수렴하는지, 기울기 소멸/폭주나 NaN(Not a Number) 값이 나타나지 않는지 파이프라인 학습 진행 품질을 테스트.
4. **Evaluation Test**: 실제 검증 데이터셋에 대해 우리가 목표한 최소 성능 기준(예: Accuracy > 0.85, F1-Score > 0.8)을 충족하는지 테스트.
