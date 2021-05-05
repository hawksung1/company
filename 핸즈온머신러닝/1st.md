### 목표
- ch5
- ch8
- ch9

## Ch. 5 SVM

### 선형 분류 SVM
- 데이터를 폭이 있는 선(도로)으로 분류
- 도로의 폭이 넒어지면 유연한 모델이지만 마진오류가 커짐

- 장점
- 단점
    - 스케일에 민감

- SVC 함수 종류
    - LinerSVC
        - 빠른 수렴
    - SVC(kernel="linear", C=1)
        - predict_proba 제공
    - SGDClassifier(loss="hinge", alpha=1/(m*C))
        - 큰 데이터셋에 적절.
- 함수 적용 순서
    1. 스케일러(Standard)
    2. Linear SVC
### 비선형 분류 SVM
- 스케일링을 적용하여 선형으로 분류
    - x = x^2 등
- 함수 적용 순서
    1. 데이터 변형(PolynomialFeatures)
    2. 스케일러
    3. Linear SVC
- 다항식 커널
    1. 스케일러
    2. poly SVC
- 유사도 함수 적용
    - RBF kernel
        - 훈련세트가 크면 X
>> 결국 하이퍼파라미터는(커널 종류 포함) 그리드로 돌려봐야 함.
### 회귀 SVM
- 도로 안에 많은 샘플이 들어가도록 학습
>> https://github.com/rickiepark/handson-ml2/blob/master/05_support_vector_machines.ipynb

## Ch. 8 차원 축소
- 차원의 저주
    - 차원이 클수록 과대적합의 위험성
- 매니폴드 
    - 2D 평면처럼 보이지만 3D로 말려있음
- 주의점
    - 분산보존
- PCA
    - 분산이 최대인 축(PC) - SVD를 이용해 찾음.
    - 계속 앞의 모두와 수직인 축 찾음
    - 적절한 차원수 선택
        - 시각화 2-3
        - elbow 함수 이용, 분산 0.95

    - IncrementalPCA
        - 훈련 데이터셋이 RAM 보다 클 때
    - KernelPCA
        - 그리드 써치 사용
    - 
- LLE(지역 선형 임베딩)
    - 잡음이 많지 않은 꼬인 매니폴드 데이터에 용이
- random_projection
- multidimensional scaling
- lsomap
- t-SNE
- LDA

## Ch. 9 비지도 학습
- k means
    - 장점
        - 빠르다
        - 확장에 용이
    - 단점
        - 클러스터의 크기가 많이 다르면 X
        - 원형인 경우 X
        - 최적의 솔루션을 찾기 힘들다.(k 개수)
- k means ++
    - smart init
    - k means 보다 빠름
- MiniBarchKMeans
    - 대량 데이터에 유리
- 최적의 클러스터 선택
    - 이너셔 그래프(엘보)
    - 실루엣 점수
- 이미지 분할
    - 시맨틱 분할
        - 동일한 물체의 픽셀 = 같은 그룹
- 군집을 이용한 전처리
    - 
- 군집 준지도 학습
    1. 샘플에 레이블
    2. 레이블 전파 - 같은 군집 같은 레이블
- DBSCAN
    - 동작
        1. 각 샘플에서 입실론 내에 샘플이 몇개 놓였는지 확인
        2. e 이웃 수 >= min_samples --> 핵심샘플
        3. 핵심샘플 이웃 = 동일 클러스터
    - 단점
        - 새로운 샘플에 적용 X
        - 클러스터간 밀집도 치아가 크면 X
    - 장점
        - 클러스터 모양, 개수에 상관 X
        - 이상치에 안정적
        - 적은 수의 하이퍼파라미터
- 병합군집
- BITCH
- 평균-이동
- 유사도 전파
- 스펙트럼 군집
- 가우시안 혼합
    - ????????????
    - 생성모델
    - 이상치 탐지
        - 밀도가 낮은지역에 있는 모든 샘플
    - 클러스터 개수 선택
        - AIC
        - BIC
- 베이즈 가우시안 혼합 모델
    - 자동으로 불필요한 클러스터 삭제
    - 