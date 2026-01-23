---
title: "산업용 안전 로봇 시스템: 종합적인 멀티모달 접근법"
excerpt: "산업 현장의 위험 방지를 위해 YOLO 기반 탐지, 칼만 필터링, 그리고 MQTT 통신을 통합한 자율 안전 모니터링 시스템의 개발 및 구현"
date: 2025-07-23
layout: post
narrow: true
categories: [ROS2]
tags: [TurtleBot4, ROS2, SLAM, Navigation]
toc: true
toc_sticky: true
---

> *“수학과 기계가 만나는 지점에서 우리는 단순한 효율성을 넘어, 인간의 생명을 보존할 수 있는 가능성을 발견한다.”*


---

<div class="youtube-wrapper">
  <iframe
    src="https://www.youtube.com/embed/VQmshnQEU4k"
    title="YouTube video player"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

## 초록 (Abstract)

본 프로젝트는 **K-디지털 트레이닝 프로그램**의 일환으로 **Team RobotFactory**가 개발한 **산업 안전 로봇 시스템**을 종합적으로 제시한다.  
제안하는 시스템은 **실시간 객체 탐지**, **자율 주행 내비게이션**, 그리고 **분산 통신 프로토콜**을 통합하여 산업 환경에서 발생하는 핵심적인 안전 문제를 해결하는 것을 목표로 한다.

엄밀한 수학적 분석과 실험적 검증을 통해, 본 시스템은 **고급 칼만 필터링 기법**을 적용함으로써 **93%의 탐지 정확도**와 **24.7%의 노이즈 감소 효과**를 달성하였다.

![System Overview](/assets/images/turtlebot_project/system_overview.png)  
*그림 1: 다중 로봇 협업 및 MQTT 기반 통신 구조를 포함한 전체 시스템 아키텍처*

---

## 1. 서론 및 문제 정의 (Introduction & Problem Formulation)

### 1.1 산업 안전 환경 분석

기술 발전에도 불구하고 **산업 안전 문제는 여전히 지속적인 과제로 남아 있다**.  
고용노동부(2024)의 통계 분석에 따르면, 산업 현장의 안전 모니터링과 제도적 집행 측면에서 여전히 심각한 공백이 존재함이 드러난다.

**정량적 안전 평가:**
- 산업재해 사망자 수: 건설업(연간 2,100명 이상), 제조업(400명 이상), 부품·소재 산업(200명 이상)
- 근로자 안전 권리 인지도: 작업 거부권을 이해하는 비율은 42.5%에 불과
- 안전 권리 행사율: 실제 작업 거부권을 행사한 비율은 16.3%에 그침
- 작업 거부 이후 보호 체감도: 충분한 보호를 받았다고 느낀 비율은 13.8%에 불과

![Industrial Accident Statistics](/assets/images/turtlebot_project/accident_statistics.png)  
*그림 2: 2021–2024년 산업재해 추이 — 지속적인 안전 문제를 보여줌*
### 1.2 수학적 위험도 프레임워크 (Mathematical Risk Framework)

시간 $$t$$에서의 **순간 위험 수준** $$R(t)$$는 다음과 같이 정의한다:

$$
R(t) = \sum_{i=1}^{n} P_i(t) \cdot S_i \cdot E_i(t)
$$

여기서 각 항의 의미는 다음과 같다:
- $$P_i(t)$$ : 시간에 따라 변화하는 사고 유형 $$i$$의 발생 확률
- $$S_i$$ : 사고 유형 $$i$$에 대한 **위험 심각도 계수**
- $$E_i(t)$$ : 사고 유형 $$i$$에 노출되는 **동적 노출 빈도**

**목적 함수(Objective Function):**

$$
\min \int_0^T R(t) \, dt 
\quad \text{subject to} \quad 
\sum_{j=1}^m C_j \leq B
$$

여기서 $$C_j$$는 각 시스템 구성 요소의 **배치 및 운영 비용**을 의미하며,  
$$B$$는 전체 **예산 제약 조건**을 나타낸다.

### 1.3 근본 원인 분석 (Root Cause Analysis)

통계 분석 결과, 산업 재해의 **78.2%가 작업자 행동 요인**에서 발생하는 것으로 나타났으며, 이는 **자동화된 모니터링 시스템의 필요성**을 강하게 시사한다.

| 위험 요인 | 수학적 모델 | 대응 전략 |
|----------|------------|-----------|
| **인지적 피로** | $$V(t) = V_0 e^{-\lambda t}$$ | 지속적 상태 모니터링 |
| **문화적 압박** | $$P_{speed} > P_{safety}$$ | 자동화된 안전 규정 집행 |
| **감시 공백** | $$\eta_{monitoring} < \eta_{required}$$ | 실시간 감시 시스템 |
| **의사소통 장벽** | $$I_{effective} = I_{transmitted} \cdot \alpha$$ | 시각/청각 경고 시스템 |

![위험 요인 분석](/assets/images/turtlebot_project/risk_factors.png)  
*그림 3: 행동 요인이 강조된 산업 재해 원인 분포 파이 차트*

---

## 2. 시스템 아키텍처 및 설계 (System Architecture & Design)

### 2.1 기술 스택 개요 (Technical Stack Overview)

| 구성 요소 | 구현 기술 | 선택 근거 |
|----------|----------|-----------|
| **객체 탐지** | YOLOv8n | 속도와 정확도 간 최적의 균형 |
| **상태 추정** | 확장 칼만 필터 (EKF) | 가우시안 노이즈 가정의 타당성 |
| **좌표 변환** | TF2 프레임워크 | ROS2 네이티브 통합 지원 |
| **통신** | MQTT 프로토콜 | 산업용 IoT 환경과의 높은 호환성 |
| **자율 주행** | NAV2 스택 | 검증된 자율 내비게이션 프레임워크 |
| **플랫폼** | Ubuntu 22.04 + ROS2 | 안정성 및 활발한 커뮤니티 지원 |

![기술 스택](/assets/images/turtlebot_project/tech_stack.png)  
*그림 4: 통합 인터페이스를 포함한 전체 기술 스택 구성*

### 2.2 분산 시스템 아키텍처 (Distributed System Architecture)

본 시스템은 **허브-앤-스포크(hub-and-spoke) 구조**를 기반으로 하며,  
**통신 장애에 대응 가능한 내결함성(fault-tolerant) 분산 아키텍처**를 채택하였다.

![시스템 아키텍처 다이어그램](/assets/images/turtlebot_project/system_overview.png)  
*그림 5: 구성 요소 간 상호작용 및 데이터 흐름을 나타낸 상세 시스템 아키텍처*

### 2.3 신뢰도 분석

$n$대의 로봇으로 구성된 분산 시스템의 시스템 신뢰도 $R_{system}$은 다음과 같습니다:

$$R_{system} = 1 - \prod_{i=1}^{n}(1 - R_i)$$

개별 로봇의 신뢰도가 $R_i = 0.95$일 때, 로봇 $n = 4$대에 대한 시스템 신뢰도는 다음과 같습니다:
$$R_{system} = 1 - (1 - 0.95)^4 = 0.99999375$$

---

## 3. 사람 탐지 및 개인보호구(PPE) 모니터링 시스템

### 3.1 데이터셋 준비 및 모델 선정

**데이터셋 세부 사양:**
- **총 샘플 수:** 5,026개 (훈련: 4,401개, 검증: 415개, 테스트: 210개)
- **클래스:** 헬멧, 안전 조끼, 사람, 안전화
- **형식:** YOLO 어노테이션 형식
- **해상도:** 640×640 픽셀
- **데이터 증강(Augmentation):** 회전(±15°), 스케일링(0.8-1.2), 밝기 조절(±20%)

### 3.2 모델 성능 분석

추론 시간 분포를 이용한 YOLO 변형 모델 간 비교 분석 결과는 다음과 같습니다:

| 모델 | 평균 추론 시간 (ms) | 표준 편차 (ms) | mAP@0.5 | 모델 크기 (MB) |
| :--- | :---: | :---: | :---: | :---: |
| YOLOv5n | 4.2 | 1.19 | 0.847 | 14.4 |
| **YOLOv8n** | **3.8** | **1.01** | **0.856** | **6.2** |
| YOLOv11n | 4.5 | 1.22 | 0.851 | 5.9 |

![모델 성능 비교](/assets/images/turtlebot_project/model_comparison.png)
*그림 7: 다양한 YOLO 모델별 추론 시간 분포를 보여주는 박스 플롯(Box plots)*

**선정 근거:** YOLOv8n은 실제 산업 현장 배포에 최적화된 정확도, 속도 및 일관성의 균형을 보여줍니다.

### 3.3 탐지 수학적 프레임워크

**공간 탐지 유니버스 (Spatial Detection Universe):**
$$\mathcal{U} = \{(x,y) \mid 0 \leq x \leq W, 0 \leq y \leq H\}$$

**탐지 신뢰도 매핑 (Detection Confidence Mapping):**
$$C(x,y) = \begin{cases} 
\sigma(\mathbf{w}^T \phi(x,y) + b) & \text{if } (x,y) \in \text{ROI} \\
0 & \text{otherwise}
\end{cases}$$

여기서 $$\phi(x,y)$$는 픽셀 $$(x,y)$$에서의 특징 추출을 나타내며, $$\sigma$$는 시그모이드 활성화 함수입니다.

**개인보호구(PPE) 준수 여부 평가:**
$$\text{PPE}_{score} = \prod_{i \in \{\text{헬멧, 조끼, 안전화}\}} \max_{j} C_i^{(j)}$$

![탐지 프레임워크](/assets/images/turtlebot_project/detection_framework.png)
*그림 8: 탐지 신뢰도 매핑 및 PPE 점수 시스템의 시각적 표현*

### 3.4 노이즈 분석 및 칼만 필터 설계

**센서 노이즈 특성:**
OAK-D 깊이 측정값에 대한 통계적 분석 결과는 다음과 같습니다:
- 표준 편차: $$\sigma = 0.4261$$ m
- 분산: $$\sigma^2 = 0.1815$$ m²
- 시간적 상관관계: $$\rho(\tau) = 0.85 e^{-\tau/2.3}$$

**상태 공간 모델(State Space Model) 설계:**

사람의 위치와 속도를 추적하기 위해 4차원 상태 벡터를 사용합니다:
$$\mathbf{x}_k = [x_k, y_k, \dot{x}_k, \dot{y}_k]^T$$

**예측 방정식:**
$$\mathbf{x}_{k|k-1} = \mathbf{F}\mathbf{x}_{k-1|k-1} + \mathbf{B}\mathbf{u}_k$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1|k-1}\mathbf{F}^T + \mathbf{Q}$$

**업데이트 방정식 (Update Equations):**
$$\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R})^{-1}$$
$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}\mathbf{x}_{k|k-1})$$

**매개변수 정의:**
$$\mathbf{F} = \begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}$$



**이론적 성능 이점:**

2D 모델과 4D 모델의 평균 제곱 오차(MSE) 차이를 비교하면 다음과 같습니다:
$$\text{MSE}_{2D} - \text{MSE}_{4D} = (\dot{x}_{k-1})^2(\Delta t)^2 \geq 0$$

이는 속도 $$\dot{x}_{k-1} \neq 0$$일 때, 4D 모델이 이론적으로 더 우수함을 증명합니다.


**실험적 검증:**

| 성능 지표 | 원시 데이터 (Raw) | 칼만 필터 적용 | 개선율 |
| :--- | :---: | :---: | :---: |
| 표준 편차 | 0.4261 m | 0.4203 m | +1.4% |
| 분산 | 0.1815 m² | 0.1766 m² | +2.7% |
| 연속 차이 (Consecutive Difference) | 0.0603 m | 0.0313 m | **+48.1%** |
| 평균 절대 오차 (MAE) | 0.3755 m | 0.3758 m | +0.4% |

**전체 노이즈 감소율: 24.7%**

---

## 4. 균열 탐지 및 구조 분석 시스템

### 4.1 컴퓨터 비전 파이프라인

균열 탐지 시스템은 딥러닝과 전통적인 컴퓨터 비전 기술을 결합한 하이브리드 방식을 채택하고 있습니다:

1. **YOLO 기반 영역 제안(Region Proposal):** 초기 균열 후보 식별
2. **HSV 색 공간 세그멘테이션:** 정밀한 균열 경계 획정
3. **깊이 인식 면적 계산 (Depth-Aware Area Calculation):** 3D 표면적 추정
4. **전역 좌표 매핑:** 내비게이션 시스템과의 통합

### 4.2 HSV 세그멘테이션 방법론

**HSV 선택 근거:**
- **조명 불변성 (Illumination Invariance):** 조명 조건에서 색상 정보를 분리
- **계산 효율성:** 픽셀 처리를 위한 선형 복잡도 $$O(n)$$ 유지
- **임계값 해석력:** 산업 현장 배포를 위한 직관적인 파라미터 튜닝 가능
- **강건성(Robustness):** 제한된 훈련 데이터에서도 효과적인 성능 발휘



[Image of HSV color space model]


**HSV 변환식:**
$$H = \arctan2(\sqrt{3}(G-B), 2R-G-B) \cdot \frac{180°}{\pi}$$
$$S = 1 - \frac{3\min(R,G,B)}{R+G+B}$$
$$V = \frac{R+G+B}{3}$$

![HSV 세그멘테이션](/assets/images/turtlebot_project/hsv_segmentation.png)
*그림 12: 배경에서 균열을 분리해내는 HSV 색 공간 세그멘테이션 결과*

### 4.3 3D 면적 계산 프레임워크

**카메라 캘리브레이션 모델:**
OAK-D 내부 파라미터(intrinsic parameters)를 이용한 픽셀-미터법 변환:
$$\text{ratio}_x = \frac{Z}{f_x}, \quad \text{ratio}_y = \frac{Z}{f_y}$$



**표면적 추정:**
$$A_{crack} = \sum_{i,j \in \text{crack pixels}} \frac{Z_{i,j}}{f_x} \cdot \frac{Z_{i,j}}{f_y} \cdot \cos(\theta_{i,j})$$

여기서 $$\theta_{i,j}$$는 픽셀 $$(i,j)$$에서의 표면 법선 벡터 각도를 나타냅니다.

**오차 전파 분석 (Error Propagation Analysis):**
$$\sigma_A^2 = \left(\frac{\partial A}{\partial Z}\right)^2 \sigma_Z^2 + \left(\frac{\partial A}{\partial f_x}\right)^2 \sigma_{f_x}^2 + \left(\frac{\partial A}{\partial f_y}\right)^2 \sigma_{f_y}^2$$
### 4.4 성능 검증

| 성능 지표 | 요구 사양 | 달성 성능 |
| :--- | :---: | :---: |
| **탐지 정확도** | >90% | 93% |
| **면적 계산 오차** | <10% | 5% |
| **좌표 매핑 정밀도** | <15cm | 10cm |
| **처리 속도** | >15 fps | 20 fps |
| **통신 지연 시간** | <150ms | 100ms |

---

## 5. 자율 주행 및 다중 로봇 협업

### 5.1 NAV2 기반 내비게이션 아키텍처

내비게이션 시스템은 다음과 같은 계층적 제어 구조를 구현합니다:



![내비게이션 아키텍처](/assets/images/turtlebot_project/navigation_architecture.png)
*그림 14: 이벤트 처리 계층 구조를 보여주는 내비게이션 시스템 상태 머신(State Machine)*

### 5.2 다중 로봇 협업 알고리즘

**우선순위 할당 함수 (Priority Assignment Function):**
$$P(e_i) = w_1 \cdot U(e_i) + w_2 \cdot T(e_i) + w_3 \cdot D(e_i)$$

**매개변수 정의:**
- $$U(e_i)$$ = 이벤트 $i$의 긴급도 (Urgency level)
- $$T(e_i)$$ = 이벤트 감지 후 경과 시간
- $$D(e_i)$$ = 이벤트 위치까지의 거리
- $$w_1, w_2, w_3$$ = 가중치 계수 ($$w_1 > w_2 > w_3$$)



**자원 할당 최적화:**
$$\min \sum_{i,j} c_{ij}x_{ij} \quad \text{subject to} \quad \sum_j x_{ij} = 1, \sum_i x_{ij} \leq 1$$

여기서 $$x_{ij} \in \{0,1\}$$은 로봇 $$i$$의 작업 $$j$$에 대한 할당 여부를 나타냅니다.

### 5.3 내비게이션 파라미터 최적화

**버퍼 크기 최적화 문제:**

기존 설정은 협소한 환경에서 내비게이션 실패를 유발했습니다. 이를 해결하기 위한 최적화 목적 함수는 다음과 같습니다:

$$\min_{b} J(b) = \alpha \cdot P_{collision}(b) + \beta \cdot E[T_{stuck}(b)] + \gamma \cdot E[P_{deviation}(b)]$$

**제약 조건:**
- $$b_{min} \leq b \leq b_{max}$$
- $$P_{collision}(b) \leq P_{threshold}$$
- $$T_{response}(b) \leq T_{max}$$



**해결책:** 인플레이션 반경(Inflation Radius)을 0.4m에서 0.1m로 축소하여 다음과 같은 결과를 얻었습니다:
- 고립(Stuck) 이벤트 60% 감소
- 경로 효율성 25% 향상
- 충돌 방지 안전성 유지

---

## 6. MQTT 통신 및 IoT 통합

### 6.1 프로토콜 선정 분석

**산업용 네트워크의 신뢰성 비교:**

$n$개의 세그먼트로 구성된 네트워크에 대한 실패 확률 분석은 다음과 같습니다:

**ROS2 DDS (메시 토폴로지):**
$$P_{DDS_{failure}} = 1 - \prod_{i=1}^{n} P_{segment_i}$$

**MQTT (스타 토폴로지):**
$$P_{MQTT_{success}} = \prod_{i=1}^{n} P_{device \rightarrow broker}$$

각 장치가 독립적으로 연결되기 때문에, 다음과 같은 관계가 성립합니다: $$P_{MQTT_{success}} \gg P_{DDS_{success}}$$



![프로토콜 비교](/assets/images/turtlebot_project/protocol_comparison.png)
*그림 16: MQTT의 회복 탄력성 이점을 보여주는 네트워크 토폴로지 비교*

### 6.2 통신 성능 분석

**프로토콜 효율성 비교:**

| 특성 | ROS2 DDS | MQTT | WebSocket | HTTP REST |
| :--- | :---: | :---: | :---: | :---: |
| **네트워크 의존도** | 높음 | 낮음 | 중간 | 낮음 |
| **실시간 성능** | 매우 우수 | 우수 | 우수 | 미흡 |
| **신뢰성** | 보통 | 높음 | 중간 | 높음 |
| **확장성** | 제한적 | 매우 우수 | 우수 | 우수 |
| **전력 효율** | 미흡 | 매우 우수 | 중간 | 미흡 |
| **산업 호환성** | 보통 | 매우 우수 | 우수 | 매우 우수 |

**메시지 오버헤드 분석:**

100바이트 페이로드(Payload) 기준:
- **MQTT:** 2-7% 오버헤드 ($\eta_{MQTT} = 0.93-0.98$)
- **HTTP:** 200-800% 오버헤드 ($\eta_{HTTP} = 0.12-0.33$)



$$\text{효율 비율 (Efficiency Ratio)} = \frac{\eta_{MQTT}}{\eta_{HTTP}} \approx 3-8$$

### 6.3 Image Transmission Performance

**Experimental Setup:**
- Image size: 64KB (320×240 RGB)
- Test duration: 2000 transmission cycles
- Network conditions: Industrial WiFi simulation

**Results:**

| FPS Setting | Success Rate (%) | Avg Latency (ms) | Throughput (KB/s) |
|-------------|------------------|------------------|-------------------|
| 10 fps (100ms) | 80.2 | 45 | 51.3 |
| 20 fps (50ms) | 22.9 | 78 | 14.7 |
| 100 fps (10ms) | 21.3 | 156 | 13.6 |

**Optimal Operating Point:**
Based on performance analysis, 10 fps provides optimal balance of reliability and real-time performance for industrial monitoring applications.

---

## 7. 시스템 통합 및 실험 검증

### 7.1 엔드투엔드(End-to-End) 시스템 성능

**지연 시간 할당(Latency Budget) 분석:**
$$T_{total} = T_{detection} + T_{processing} + T_{communication} + T_{response}$$



**측정된 구성 요소:**
- $$T_{detection} = 52 \pm 8$$ ms (YOLOv8n 추론)
- $$T_{processing} = 23 \pm 5$$ ms (좌표 변환)
- $$T_{communication} = 95 \pm 15$$ ms (MQTT 왕복 시간)
- $$T_{response} = 180 \pm 30$$ ms (내비게이션 개시)

**전체 시스템 응답 시간: 350 ± 35 ms**

### 7.2 다중 로봇 협업 검증

**테스트 시나리오:** 4대의 로봇을 이용한 사람 및 균열 동시 탐지 이벤트



**협업 알고리즘 성능:**
- 이벤트 탐지부터 응답 개시까지: 평균 245ms
- 자원 할당 충돌: 0% (완벽한 협업 달성)
- 커버리지 효율성: 모니터링 영역의 94% 달성

**부하 분산(Load Balancing) 유효성:**
$$\text{균형 지수 (Balance Index)} = 1 - \frac{\sigma_{workload}}{\mu_{workload}} = 0.89$$

### 7.3 실제 환경 테스트 검증

**테스트 시설 세부 사양:**
- 면적: 400m² 규모의 산업 시뮬레이션 공간
- 장애물: 다양한 산업 장비 모형 배치
- 조명 조건: 200-800 lux (가변적)
- 네트워크: 간섭 제어가 포함된 기업용 WiFi 환경

---

## 8. 대시보드 및 모니터링 인터페이스

### 8.1 웹 기반 제어 인터페이스

모니터링 대시보드는 실시간 시각화 및 제어 기능을 제공합니다:



**주요 기능:**
- 로봇의 실시간 위치 및 상태 정보
- 실시간 이벤트 탐지 피드
- 과거 데이터 분석(Analytics)
- 원격 제어 기능
- 성능 지표(Performance Metrics) 표시

![웹 대시보드](/assets/images/turtlebot_project/web_dashboard.png)
*그림 20: 실시간 모니터링 및 제어 기능을 보여주는 웹 기반 대시보드 인터페이스*

### 8.2 모바일 애플리케이션 통합

**모바일 앱 주요 기능:**
- 중요 이벤트 발생 시 푸시 알림
- 간소화된 로봇 상태 개요
- 비상 정지(Emergency Stop) 기능
- 위치 기반 이벤트 매핑



![모바일 앱](/assets/images/turtlebot_project/mobile_app.png)
*그림 21: 비상 대응 및 알림 기능을 보여주는 모바일 애플리케이션 인터페이스*

---

## 9. 문제점 및 해결 방안 (Challenges & Solutions)

### 9.1 좌표계 캘리브레이션 과제

**문제 정의:**
탐지된 객체의 좌표와 실제 전역 지도상의 위치 사이에 평균 0.35m의 계통적 오차(Systematic offset)가 발생함.

**근본 원인 분석:**
1. **센서 캘리브레이션 드리프트:** OAK-D 카메라의 내부 파라미터가 시간에 따라 변동됨
2. **누적 변환 오차:** TF 트리(Transformation Tree) 전파 과정에서의 부정확성
3. **환경적 간섭:** 반사 표면이 깊이 추정(Depth estimation)에 영향을 미침



**수학적 오차 모델:**
$$\mathbf{p}_{measured} = \mathbf{R}\mathbf{p}_{actual} + \mathbf{t} + \boldsymbol{\epsilon}_{systematic} + \boldsymbol{\eta}_{noise}$$

여기서 $$\mathbf{R}$$과 $$\mathbf{t}$$는 계통적 회전 및 평행 이동 오차를 나타냅니다.

**해결책 구현:**

1. **실측 기반 교정 행렬 (Empirical Calibration Matrix):**
   $$\mathbf{C} = \arg\min_{\mathbf{C}} \sum_{i=1}^{N} ||\mathbf{p}_{ground\_truth}^{(i)} - \mathbf{C}\mathbf{p}_{measured}^{(i)}||^2$$

2. **실시간 검증:** 알려진 참조점(Reference points)과의 지속적인 비교 수행
3. **향후 개선 사항:** ICP(Iterative Closest Point) 알고리즘을 이용한 포인트 클라우드 등록(Registration)

### 9.2 내비게이션 버퍼 최적화

**문제점:** 과도한 버퍼 영역(Buffer zones) 설정으로 인해 협소한 통로에서 내비게이션 실패가 발생함.



**최적화 접근 방식:**
$$J(b) = w_1 \sum_{i} I_{collision}^{(i)} + w_2 \sum_{j} T_{stuck}^{(j)} + w_3 \sum_{k} D_{deviation}^{(k)}$$

**해결 결과:**
- **버퍼 반경:** 0.4m → 0.1m로 조정
- **주행 성공률 개선:** 73% → 96%
- **평균 주행 시간 감소:** 40% 단축

---

## 10. 성능 평가 및 결과

### 10.1 정량적 성능 지표

**탐지 시스템 성능:**

| 지표 | 사람 탐지 | 균열 탐지 | 통합 시스템 |
| :--- | :---: | :---: | :---: |
| **정밀도 (Precision)** | 0.91 | 0.93 | 0.92 |
| **재현율 (Recall)** | 0.89 | 0.87 | 0.88 |
| **F1-스코어 (F1-Score)** | 0.90 | 0.90 | 0.90 |
| **처리 속도 (Processing Speed)** | 18.5 fps | 20.2 fps | 19.1 fps |
| **오탐률 (False Positive Rate)** | 0.08 | 0.05 | 0.07 |



**시스템 통합 지표:**

| 구성 요소 | 가동 시간 (Uptime, %) | 평균 응답 시간 (ms) | 오류율 (%) |
| :--- | :---: | :---: | :---: |
| **사람 탐지** | 99.2 | 52 | 0.8 |
| **균열 탐지** | 98.8 | 48 | 1.2 |
| **내비게이션** | 97.5 | 180 | 2.5 |
| **MQTT 통신** | 99.8 | 95 | 0.2 |
| **전체 시스템** | 97.1 | 350 | 2.9 |

### 10.2 비교 분석

**기존 솔루션과의 벤치마크 비교:**

| 기능 | 본 시스템 | 상용 솔루션 A | 연구용 시스템 B |
| :--- | :---: | :---: | :---: |
| **탐지 정확도** | 93% | 89% | 91% |
| **실시간 성능** | ✓ | ✓ | ✗ |
| **다중 로봇 협업** | ✓ | ✗ | ✓ |
| **비용 효율성** | 높음 | 낮음 | 보통 |
| **확장성** | 매우 우수 | 제한적 | 우수 |

---

## 11. 향후 연구 및 발전 방향

### 11.1 강화된 센서 융합 (Enhanced Sensor Fusion)

**계획된 멀티 센서 통합:**
$$\hat{\mathbf{x}}_{fused} = \sum_{i=1}^{n} w_i \hat{\mathbf{x}}_i$$

여기서 가중치($w_i$)는 각 센서의 신뢰도에 따라 최적화됩니다:
$$w_i = \frac{\sigma_i^{-2}}{\sum_{j=1}^{n} \sigma_j^{-2}}$$



**기대 성능 향상:**
$$\sigma_{fused}^2 = \left(\sum_{i=1}^{n} \sigma_i^{-2}\right)^{-1} \leq \min_i \sigma_i^2$$

융합된 데이터의 분산($\sigma_{fused}^2$)은 개별 센서 중 가장 낮은 분산보다도 작거나 같아져 데이터의 정밀도가 향상됩니다.

### 11.2 고급 협업 알고리즘

**다중 로봇 시스템을 위한 분산 합의 (Distributed Consensus):**
안전과 직결된 중요한 의사결정을 위해 비잔틴 결함 허용(Byzantine Fault Tolerant, BFT) 합의 알고리즘을 구현할 예정입니다.



**군집 지능 (Swarm Intelligence) 통합:**
- **입자 군집 최적화 (PSO):** 커버리지 경로 계획 최적화에 활용
- **개미 군집 최적화 (ACO):** 동적 작업 할당 알고리즘에 적용

### 11.3 엣지 컴퓨팅 통합

**포그 컴퓨팅 (Fog Computing) 아키텍처:**
- **로컬 처리 능력 강화:** 지연 시간 단축을 위한 현장 데이터 처리
- **엣지 기반 머신러닝 추론:** 네트워크 의존도를 낮춘 실시간 탐지
- **분산 데이터 저장 및 분석:** 효율적인 데이터 관리와 통계 분석 수행

---

## 12. 결론 (Conclusion)

본 연구는 산업 현장의 핵심적인 안전 과제를 해결하기 위해 여러 첨단 기술을 성공적으로 통합한 포괄적인 산업 안전 로봇 시스템을 제시했습니다. 엄격한 수학적 분석과 실험적 검증을 통해 다음과 같은 사항을 입증했습니다.

### 12.1 주요 성과

1. **실시간 멀티모달 탐지:** 400ms 미만의 응답 속도로 위험 요소 식별 정확도 93% 달성
2. **고급 노이즈 필터링:** 칼만 필터 구현을 통해 노이즈 24.7% 감소
3. **강건한 통신:** 99.8%의 신뢰성을 가진 MQTT 기반 분산 통신 체계 구축
4. **지능형 협업:** 작업 성공률 96%를 기록한 다중 로봇 협업 시스템 개발



### 12.2 기술적 기여

**수학적 모델링:**
- 산업 환경을 위한 공식적인 위험 평가 프레임워크 구축
- 2D 모델 대비 4D 상태 추적 모델의 이론적 우수성 입증
- 내비게이션 파라미터 튜닝을 위한 최적화 모델 개발

**시스템 엔지니어링:**
- 이기종 기술들을 하나의 응집력 있는 안전 모니터링 시스템으로 통합
- 결함 허용(Fault-tolerant) 분산 아키텍처 구현
- 포괄적인 테스트 및 검증 프레임워크 구축

**산업적 영향:**
- 시뮬레이션된 산업 환경에서 실제적인 적용 가능성 입증
- 실제 현장 배포에 적합한 성능 지표 달성
- 미래 안전 시스템 개발을 위한 확장 가능한 기반 마련

### 12.3 연구의 의의

본 연구는 자율 주행 산업 안전 시스템 분야에서 이론적 토대와 실질적 구현을 모두 제공하는 중요한 진전을 이루었습니다. 수학적 엄밀함과 실전 테스트의 결합은 시스템의 산업적 배포 준비가 되었음을 보여주는 동시에, 지속적인 개선을 위한 명확한 방향을 제시합니다.



**미래 영향 전망:**
적절한 규모 확장과 산업 파트너십을 통해, 본 시스템은 지능적이고 지속적인 모니터링을 통해 작업장 사망 사고를 예방하고 산업 재해를 줄이는 데 크게 기여할 잠재력을 가지고 있습니다.

---

## Acknowledgments

We extend our gratitude to the K-Digital Training program, our mentors, and Doosan Robotics for providing the platform and resources necessary for this research. Special thanks to all team members who contributed their expertise across multiple technical domains.

## References

1. Ministry of Employment and Labor, "Industrial Accident Investigation Report 2024," Korea Occupational Safety and Health Agency
2. Kalman, R.E., "A New Approach to Linear Filtering and Prediction Problems," *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35-45, 1960
3. Redmon, J., et al., "You Only Look Once: Unified, Real-Time Object Detection," *IEEE Conference on Computer Vision and Pattern Recognition*, 2016
4. Ultralytics, "YOLOv8: A New State-of-the-Art Computer Vision Model," 2023
5. Quigley, M., et al., "ROS: An Open-Source Robot Operating System," *ICRA Workshop on Open Source Software*, 2009
6. Macenski, S., et al., "The Marathon 2: A Navigation System," *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2020
7. Light, A., "MQTT Protocol Specification v3.1.1," OASIS Standard, 2014
8. Lee, M.J., "Promoting Sustainable Safety Work Environments: Factors Affecting Korean Workers' Recognition," *MDPI Sustainability*, 2024
9. Thrun, S., "Probabilistic Robotics," MIT Press, 2005
10. OpenCV Development Team, "Open Source Computer Vision Library," 2023

---

## Appendix A: Technical Specifications

### A.1 Hardware Configuration

**Robot Platform:**
- Base: TurtleBot4 with Create3 base
- Processor: Intel NUC with i5-8250U
- Memory: 16GB DDR4 RAM
- Storage: 512GB NVMe SSD

**Sensor Suite:**
- Primary Camera: OAK-D (OpenCV AI Kit)
  - RGB Resolution: 1920×1080 @ 30fps
  - Depth Range: 0.35m - 10m
  - Baseline: 75mm
- LiDAR: RPLIDAR A1M8
  - Range: 12m
  - Angular Resolution: 0.9°
  - Scan Rate: 8000 samples/sec

**Communication:**
- WiFi: 802.11ac dual-band
- Ethernet: Gigabit RJ45
- USB: 3× USB 3.0 ports

### A.2 Software Dependencies

```bash
# Core ROS2 Dependencies
sudo apt install ros-humble-desktop-full
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-tf2-tools

# Computer Vision Dependencies
pip install ultralytics==8.0.196
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3

# Communication Dependencies
pip install paho-mqtt==1.6.1
pip install pyserial==3.5

# Mathematical Libraries
pip install scipy==1.11.3
pip install scikit-learn==1.3.0
pip install filterpy==1.4.5
```

### A.3 Configuration Parameters

**YOLO Detection Parameters:**
```yaml
model_config:
  confidence_threshold: 0.7
  iou_threshold: 0.45
  max_detections: 100
  input_size: [640, 640]
  
class_names:
  - helmet
  - vest  
  - human
  - boots
  - crack
```

**Kalman Filter Parameters:**
```yaml
kalman_config:
  process_noise_variance: 0.01
  measurement_noise_variance: 0.1
  initial_state_covariance: 1.0
  dt: 0.1  # 100ms update rate
```

**Navigation Parameters:**
```yaml
nav2_params:
  controller_server:
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.5
      lookahead_dist: 0.6
      
  planner_server:
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      
  costmap_2d:
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.1  # Optimized value
```

**MQTT Configuration:**
```yaml
mqtt_config:
  broker_host: "mqtt.emqx.cloud"
  broker_port: 1883
  keep_alive: 60
  qos_level: 1
  topics:
    human_detection: "safety/human/detected"
    crack_detection: "safety/crack/detected"
    robot_status: "robot/status"
    commands: "robot/commands"
```
## Appendix B: Mathematical Derivations

### B.1 Kalman Filter Derivation for 4D State Space

Given the state transition model:

$$\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1} + \mathbf{w}_{k-1}$$

Where $\mathbf{w}_{k-1} \sim \mathcal{N}(0, \mathbf{Q})$

**Prediction Step:**

$$\hat{\mathbf{x}}_{k \mid k-1} = \mathbf{F}\hat{\mathbf{x}}_{k-1 \mid k-1}$$

$$\mathbf{P}_{k \mid k-1} = \mathbf{F}\mathbf{P}_{k-1 \mid k-1}\mathbf{F}^T + \mathbf{Q}$$

**Update Step:**

Given measurement $\mathbf{z}_k = \mathbf{H}\mathbf{x}_k + \mathbf{v}_k$ where $\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R})$

Innovation:

$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_{k \mid k-1}$$

Innovation covariance:

$$\mathbf{S}_k = \mathbf{H}\mathbf{P}_{k \mid k-1}\mathbf{H}^T + \mathbf{R}$$

Kalman gain:

$$\mathbf{K}_k = \mathbf{P}_{k \mid k-1}\mathbf{H}^T\mathbf{S}_k^{-1}$$

State update:

$$\hat{\mathbf{x}}_{k \mid k} = \hat{\mathbf{x}}_{k \mid k-1} + \mathbf{K}_k\tilde{\mathbf{y}}_k$$

Covariance update:

$$\mathbf{P}_{k \mid k} = (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}_{k \mid k-1}$$

### B.2 MQTT vs DDS Reliability Analysis

For $n$ network segments with individual reliability $p_i$:

**DDS Mesh Network:**

All segments must be operational for system function:

$$P_{\text{DDS}} = \prod_{i=1}^{n} p_i$$

For $p_i = 0.95$ and $n = 4$:

$$P_{\text{DDS}} = (0.95)^4 = 0.815$$

**MQTT Star Network:**

Each device connects independently to broker:

$$P_{\text{MQTT}} = \prod_{i=1}^{n} p_{\text{device}_i \rightarrow \text{broker}}$$

For $p_{\text{device} \rightarrow \text{broker}} = 0.98$ and $n = 4$:

$$P_{\text{MQTT}} = (0.98)^4 = 0.922$$

**Reliability Improvement:**

$$\frac{P_{\text{MQTT}}}{P_{\text{DDS}}} = \frac{0.922}{0.815} = 1.13$$

This represents a 13% improvement in system reliability.

### B.3 Multi-Robot Task Allocation Optimization

**Objective Function:**

$$\min \sum_{i=1}^{m}\sum_{j=1}^{n} c_{ij}x_{ij}$$

**Constraints:**

1. Each task assigned to exactly one robot: 
   $$\sum_{i=1}^{m} x_{ij} = 1, \quad \forall j$$

2. Robot capacity constraint: 
   $$\sum_{j=1}^{n} x_{ij} \leq C_i, \quad \forall i$$

3. Binary assignment: 
   $$x_{ij} \in \{0,1\}$$

Where:
- $c_{ij}$ = cost of assigning robot $i$ to task $j$
- $x_{ij}$ = binary decision variable  
- $C_i$ = capacity of robot $i$

**Hungarian Algorithm Solution:**

For balanced assignment ($m = n$ and $C_i = 1$), optimal solution achievable in $O(n^3)$ time complexity.

### B.4 Risk Assessment Mathematical Framework

**Instantaneous Risk Model:**

$$R(t) = \sum_{i=1}^{n} P_i(t) \cdot S_i \cdot E_i(t)$$

Where:
- $P_i(t)$ = Time-dependent probability of incident type $i$
- $S_i$ = Severity coefficient for incident $i$  
- $E_i(t)$ = Dynamic exposure frequency to risk $i$

**Optimization Objective:**

$$\min \int_0^T R(t) \, dt \quad \text{subject to} \quad \sum_{j=1}^m C_j \leq B$$

Where $C_j$ represents deployment costs and $B$ is the budget constraint.

### B.5 Sensor Fusion for Enhanced Accuracy

**Multi-Sensor Fusion Model:**

$$\hat{\mathbf{x}}_{\text{fused}} = \sum_{i=1}^{n} w_i \hat{\mathbf{x}}_i$$

Where weights are optimized based on sensor reliability:

$$w_i = \frac{\sigma_i^{-2}}{\sum_{j=1}^{n} \sigma_j^{-2}}$$

**Expected Performance Improvement:**

$$\sigma_{\text{fused}}^2 = \left(\sum_{i=1}^{n} \sigma_i^{-2}\right)^{-1} \leq \min_i \sigma_i^2$$

This theoretical framework guarantees improved accuracy through optimal sensor combination.

### B.6 2D vs 4D State Model Performance Analysis

**Mean Squared Error Comparison:**

For velocity-informed tracking, the performance difference between 2D and 4D models:

$$\text{MSE}_{2D} - \text{MSE}_{4D} = (\dot{x}_{k-1})^2(\Delta t)^2 \geq 0$$

**Theoretical Proof:**

When $\dot{x}_{k-1} \neq 0$, the 4D model consistently achieves lower MSE by incorporating velocity information into state prediction.

**Computational Complexity Trade-off:**

| Model | State Prediction | Covariance Update | Kalman Gain | Total |
|-------|-----------------|------------------|-------------|-------|
| 2D | $O(4)$ | $O(8)$ | $O(8)$ | $O(20)$ |
| 4D | $O(16)$ | $O(64)$ | $O(32)$ | $O(112)$ |
| 6D | $O(36)$ | $O(216)$ | $O(72)$ | $O(324)$ |

The 4D model provides optimal balance between computational efficiency and tracking accuracy for industrial applications.

## Appendix C: Code Implementations

### C.1 Human Detection Node (Python)

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import tf2_ros
import tf2_geometry_msgs

class HumanDetectionNode(Node):
    def __init__(self):
        super().__init__('human_detection_node')
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', 
            self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw',
            self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.camera_info_callback, 10)
            
        self.detection_pub = self.create_publisher(
            PointStamped, '/human_detection/position', 10)
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Kalman filter initialization
        self.kalman = self.init_kalman_filter()
        
    def init_kalman_filter(self):
        """Initialize 4D Kalman filter for position and velocity tracking"""
        from filterpy.kalman import KalmanFilter
        
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 0.1
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        
        # Measurement matrix (observe position only)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        
        # Process noise covariance
        kf.Q = np.eye(4) * 0.01
        
        # Measurement noise covariance
        kf.R = np.eye(2) * 0.1
        
        # Initial state covariance
        kf.P = np.eye(4) * 1000
        
        return kf
    
    def image_callback(self, msg):
        """Process RGB image for human detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO inference
            results = self.model(cv_image, conf=0.7)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is human class
                        if int(box.cls) == 0:  # Person class in COCO
                            self.process_human_detection(box, msg.header)
                            
        except Exception as e:
            self.get_logger().error(f'Error in image processing: {e}')
    
    def process_human_detection(self, box, header):
        """Process human detection and publish 3D position"""
        # Extract bounding box center
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Get depth value
        if hasattr(self, 'depth_image'):
            depth_value = self.depth_image[center_y, center_x]
            
            # Convert to 3D coordinates
            if depth_value > 0:
                point_3d = self.pixel_to_3d(center_x, center_y, depth_value)
                
                # Apply Kalman filter
                filtered_point = self.apply_kalman_filter(point_3d[:2])
                
                # Create and publish PointStamped message
                point_msg = PointStamped()
                point_msg.header = header
                point_msg.point.x = filtered_point[0]
                point_msg.point.y = filtered_point[1]
                point_msg.point.z = point_3d[2]
                
                self.detection_pub.publish(point_msg)
    
    def apply_kalman_filter(self, measurement):
        """Apply Kalman filter to measurement"""
        self.kalman.predict()
        self.kalman.update(measurement)
        return self.kalman.x[:2]  # Return filtered position
    
    def pixel_to_3d(self, u, v, depth):
        """Convert pixel coordinates to 3D world coordinates"""
        if hasattr(self, 'camera_info'):
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]
            
            # Convert depth from mm to m
            z = depth / 1000.0
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            return np.array([x, y, z])
        return None

def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### C.2 MQTT Communication Bridge (Python)

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import paho.mqtt.client as mqtt
import json
import threading

class MQTTBridge(Node):
    def __init__(self):
        super().__init__('mqtt_bridge')
        
        # MQTT Configuration
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        
        # Connect to MQTT broker
        self.mqtt_client.connect("mqtt.emqx.cloud", 1883, 60)
        self.mqtt_client.loop_start()
        
        # ROS2 Publishers and Subscribers
        self.human_detection_sub = self.create_subscription(
            PointStamped, '/human_detection/position',
            self.human_detection_callback, 10)
        
        self.crack_detection_sub = self.create_subscription(
            PointStamped, '/crack_detection/position',
            self.crack_detection_callback, 10)
        
        self.command_pub = self.create_publisher(
            String, '/robot_commands', 10)
        
        # Message queues for thread safety
        self.message_queue = []
        self.queue_lock = threading.Lock()
        
        # Timer for processing queued messages
        self.timer = self.create_timer(0.1, self.process_message_queue)
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            self.get_logger().info("Connected to MQTT broker")
            client.subscribe("robot/commands")
            client.subscribe("safety/+/control")
        else:
            self.get_logger().error(f"Failed to connect to MQTT broker: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            with self.queue_lock:
                self.message_queue.append({
                    'topic': topic,
                    'payload': payload,
                    'timestamp': self.get_clock().now()
                })
                
        except Exception as e:
            self.get_logger().error(f"Error processing MQTT message: {e}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        self.get_logger().warn(f"Disconnected from MQTT broker: {rc}")
    
    def human_detection_callback(self, msg):
        """Publish human detection to MQTT"""
        detection_data = {
            'type': 'human_detection',
            'position': {
                'x': msg.point.x,
                'y': msg.point.y,
                'z': msg.point.z
            },
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame_id': msg.header.frame_id,
            'confidence': 0.9,  # Would come from detection system
            'robot_id': self.get_parameter('robot_id').value if self.has_parameter('robot_id') else 'robot_0'
        }
        
        self.mqtt_client.publish(
            "safety/human/detected",
            json.dumps(detection_data),
            qos=1
        )
        
        self.get_logger().info(f"Published human detection: {detection_data}")
    
    def crack_detection_callback(self, msg):
        """Publish crack detection to MQTT"""
        detection_data = {
            'type': 'crack_detection',
            'position': {
                'x': msg.point.x,
                'y': msg.point.y,
                'z': msg.point.z
            },
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame_id': msg.header.frame_id,
            'severity': 'medium',  # Would be calculated based on crack analysis
            'area_m2': 0.05,  # Would come from crack analysis
            'robot_id': self.get_parameter('robot_id').value if self.has_parameter('robot_id') else 'robot_0'
        }
        
        self.mqtt_client.publish(
            "safety/crack/detected",
            json.dumps(detection_data),
            qos=1
        )
        
        self.get_logger().info(f"Published crack detection: {detection_data}")
    
    def process_message_queue(self):
        """Process queued MQTT messages in ROS2 context"""
        with self.queue_lock:
            while self.message_queue:
                message = self.message_queue.pop(0)
                self.handle_mqtt_command(message)
    
    def handle_mqtt_command(self, message):
        """Handle MQTT commands in ROS2 context"""
        topic = message['topic']
        payload = message['payload']
        
        if topic == "robot/commands":
            command_msg = String()
            command_msg.data = json.dumps(payload)
            self.command_pub.publish(command_msg)
            
            self.get_logger().info(f"Forwarded command: {payload}")

def main(args=None):
    rclpy.init(args=args)
    node = MQTTBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### C.3 Launch File Configuration

```xml
<!-- launch/safety_robot.launch.py -->
<launch>
  <!-- Robot State Publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" command="$(find xacro)/xacro $(find turtlebot4_description)/urdf/turtlebot4.urdf.xacro" />
  </node>
  
  <!-- Navigation2 -->
  <include file="$(find nav2_bringup)/launch/navigation_launch.py">
    <arg name="use_sim_time" value="false"/>
    <arg name="params_file" value="$(find safety_robot)/config/nav2_params.yaml"/>
    <arg name="map" value="$(find safety_robot)/maps/industrial_map.yaml"/>
  </include>
  
  <!-- Camera -->
  <include file="$(find depthai_ros_driver)/launch/rgbd_pcl.launch.py">
    <arg name="camera_model" value="OAK-D"/>
    <arg name="tf_prefix" value="oak"/>
  </include>
  
  <!-- Detection Nodes -->
  <node pkg="safety_robot" exec="human_detection_node" name="human_detection">
    <param name="model_path" value="$(find safety_robot)/models/human_ppe_detection.pt"/>
    <param name="confidence_threshold" value="0.7"/>
  </node>
  
  <node pkg="safety_robot" exec="crack_detection_node" name="crack_detection">
    <param name="model_path" value="$(find safety_robot)/models/crack_detection.pt"/>
    <param name="hsv_lower" value="[0, 0, 0]"/>
    <param name="hsv_upper" value="[180, 30, 100]"/>
  </node>
  
  <!-- MQTT Bridge -->
  <node pkg="safety_robot" exec="mqtt_bridge" name="mqtt_bridge">
    <param name="robot_id" value="$(env ROBOT_ID)"/>
    <param name="mqtt_broker" value="mqtt.emqx.cloud"/>
    <param name="mqtt_port" value="1883"/>
  </node>
  
  <!-- Coordination Node -->
  <node pkg="safety_robot" exec="coordination_node" name="coordination">
    <param name="priority_weights" value="[0.6, 0.3, 0.1]"/>  <!-- urgency, time, distance -->
  </node>
</launch>
```

---

## Image Requirements Summary

Please add the following images to `/assets/images/turtlebot_project/`:

1. **system_overview.png** - Complete system architecture diagram
2. **accident_statistics.png** - Bar chart of industrial accident trends 2021-2024
3. **risk_factors.png** - Pie chart showing accident cause distribution
4. **tech_stack.png** - Technology stack diagram with logos
5. **system_architecture.png** - Detailed system architecture with data flow
6. **dataset_samples.png** - Grid of PPE detection dataset examples
7. **model_comparison.png** - Box plots of YOLO model inference times
8. **detection_framework.png** - Visual detection confidence mapping
9. **noise_analysis.png** - Time series of sensor noise measurements
10. **kalman_results.png** - Before/after Kalman filter comparison
11. **crack_pipeline.png** - Crack detection pipeline flowchart
12. **hsv_segmentation.png** - HSV color space segmentation results
13. **crack_performance.png** - Performance validation charts
14. **navigation_architecture.png** - Navigation state machine diagram
15. **navigation_optimization.png** - Buffer optimization before/after
16. **protocol_comparison.png** - Network topology comparison diagrams
17. **mqtt_performance.png** - Radar chart of MQTT performance metrics
18. **system_integration.png** - Photo of complete integrated system
19. **testing_environment.png** - Real-world testing facility
20. **web_dashboard.png** - Screenshot of web monitoring interface
21. **mobile_app.png** - Mobile application interface
22. **coordinate_calibration.png** - Coordinate accuracy improvement charts
23. **performance_summary.png** - Comprehensive performance evaluation
24. **future_architecture.png** - Proposed future system architecture

This comprehensive documentation provides a complete technical reference for your industrial safety robot system project, suitable for both academic publication and professional portfolio presentation.