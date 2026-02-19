---
title: "디지털 트윈 기반 서비스 로봇 시스템: TurtleBot3를 이용한 가상 시뮬레이션과 현실 작동의 동기화"
excerpt: "가상(RViz) 및 실제 환경에서의 터틀봇3 통합 운용 구현"
date: 2025-09-04
layout: post
categories: [ROS2]
tags: [Turtlebot3, Digital Twin, ROS2, OpenCV]
toc: true
toc_sticky: true
---

> *"디지털과 물리적 세계가 하나로 움직일 때, 지능은 비로소 현실이 된다. "*

---
<div class="youtube-wrapper">
  <iframe
    src="https://www.youtube.com/embed/57pRaic92Kg"
    title="Service Robot Operation Demo"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

---

## Abstract

본 프로젝트는 **Turtlebot3**와 **ROS2**를 활용하여 자율주행 서비스 로봇의 **Digital Twin** 환경을 구축하여 가상 환경 내 자율주행과 실제 환경에서의 자율주행 서비스를 진행한 프로젝트이다. 가상 환경(Gazebo)에서 검증된 알고리즘을 실제 환경에 적용하며 발생하는 Reality Gap을 **HSB 색공간 최적화, 이미지 전처리(CLAHE, 감마 보정), 디바운싱 알고리즘** 등을 통해 해결하였습니다. 이를 통해 차단기 인식, 신호등 준수, ArUco 마커 기반 Pick-and-Place 임무를 완수하는 통합 시스템을 구현했습니다.

---

## 1. 서론

![Digital Twin](/assets/images/Digital_Twin/Digital_Twin.png)  
*그림 1. Digital Twin*

최근 모빌리티 산업의 핵심으로 부상한 자율주행 기술은 카메라, 라이다(LiDAR), 레이더(RADAR) 등 다양한 센서 데이터를 융합하여 주변 환경을 인식하고 스스로 판단하여 주행하는 복합적인 기술의 집약체이다. 테슬라의 비전 중심 방식이나 구글 웨이모의 라이다 중심 방식처럼 기업마다 기술적 접근법은 다르지만, 공통적인 과제는 **'실행의 안전성'**과 **'환경 적응력'**을 확보하는 것이다.

이러한 맥락에서 디지털 트윈(Digital Twin) 기술은 자율주행 연구의 효율성을 극대화하는 핵심 솔루션으로 주목받고 있다. 디지털 트윈은 가상 공간에 현실과 동일한 객체와 환경을 구현하고 실시간 데이터를 동기화함으로써, 실제 주행 시 발생할 수 있는 위험 요소와 시행착오를 가상 환경에서 미리 시뮬레이션할 수 있게 한다. 이는 물리적 충돌 위험 없이 다양한 엣지 케이스(Edge Case)를 학습할 수 있다는 점에서 자율주행의 완성도를 높이는 필수적인 단계이다.

본 프로젝트에서는 이러한 디지털 트윈 메커니즘을 기반으로, 오픈소스 로봇 플랫폼인 Turtlebot3와 고성능 객체 인식 알고리즘인 YOLO, 그리고 Intel RealSense Depth Camera를 결합한 자율주행 시스템을 구축한다.

먼저 고도화된 가상 환경인 Gazebo 시뮬레이터 내에서 로봇의 기동성과 객체 인식 성능을 사전 검증하고 최적화한다. 이후 가상 세계에서 도출된 주행 알고리즘과 파라미터를 실제 환경에 적용함으로써, 가상과 현실 사이의 괴리(Reality Gap)를 최소화하고 보다 정교하고 안정적인 자율주행 기술 구현을 목표로 한다.

---

## 2. 시스템 아키텍처 및 알고리즘 분석

### 2.1. 차단기 인식 (Detect Level Crossing)

![Level_Crossing](/assets/images/Digital_Twin/Level_Crossing.png)  
*그림 2. Level_Crossing*
![HSB](/assets/images/Digital_Twin/HSB.png)  
*그림 3. HSB*

| :--- | :--- | :--- |
| **red1_hue_low** | 0 | 0 |
| **red1_hue_high** | 22 | 10 |
| **red2_hue_low** | None | 170 |
| **red2_hue_high** | None | 179 |
| **saturation_low** | 173 | 60 |
| **saturation_high** | 255 | 170 |
| **lightness_low** | 106 | 120 |
| **lightness_high** | 255 | 255 |

* Detect Level Crossing 파라미터 값 수정

기존 코드의 빛 반사로 인한 오탐지 문제를 해결하기 위해 **HSB(Hue, Saturation, Brightness)** 모델을 적용했습니다. 
* **HSB 최적화**: OpenCV의 Hue 범위(0~179)를 고려하여 빨간색 영역을 0~10 및 170~179 두 구간으로 나누어 정확도를 높였습니다.
* **기하학적 판단**: `minAreaRect` 함수를 사용하여 LED 점들을 하나의 막대 객체로 인식하고, 회전 각도(0~10도 이상)를 기준으로 'Stop/Go'를 판정합니다.

### 2.2. 신호등 인식 (Detect Traffic Light)
![Traffic Light](/assets/images/Digital_Twin/Traffic_light.png)  
*그림 4. Traffic Light*

![Debouncing](/assets/images/Digital_Twin/Debouncing.png)  
*그림 5. Debouncing*

반응성과 안정성을 동시에 확보하기 위해 다음과 같은 기법을 적용했습니다.
* **디바운싱(Debouncing)**: 특정 신호가 연속된 프레임(5회 이상)에서 검출될 때만 상태를 확정하여 깜빡임과 노이즈를 방지했습니다.
* **관심 영역(ROI) 최적화**: 우측 및 하단 중심의 ROI 설정을 통해 원거리와 근거리 신호를 모두 수용할 수 있도록 개선했습니다.
* **반응성 개선**: 타이머 주기를 10Hz에서 20Hz로 상향하고 프레임 처리 로직을 효율화했습니다.

### 2.3. 차선 인식 및 전처리 (Detect Lane)
실제 환경의 조명 변화에 대응하기 위해 고도화된 전처리 파이프라인을 구축했습니다.
* **화이트 밸런스 & 감마 보정**: 이미지의 색감과 밝기를 균일하게 유지합니다.

![Clane](/assets/images/Digital_Twin/Clane.png)  
*그림 6. Clane*

* **CLAHE & 난반사 억제**: 지역 대비를 강화하고, 채도가 낮고 밝기가 높은 영역($S \le 60, V \ge 200$)의 밝기를 강제로 줄여 난반사를 차단했습니다.

---

## 3. Pick-and-Place 시퀀스

ArUco 마커 감지 시 로봇은 차선 주행을 멈추고 정밀 접근을 시작합니다.
1. **거리 기반 접근**: Z축 거리값에 따라 속도를 단계적으로 감속(0.10 → 0.05 → 0.02)하여 오차를 최소화합니다.
2. **매니퓰레이션 시퀀스**: 'Home → Box Front → Move to Box → Close → Conveyor Up/Down → Open' 순의 리팩토링된 시퀀스를 실행합니다.
3. **복귀**: 임무 완료 후 로봇팔을 초기 주행 포즈로 복귀시키고 다시 차선 주행 메세지를 퍼블리싱합니다.

---

## 4. 문제점 및 해결 방안 (Challenges & Solutions)

| 문제점 | 해결 방안 |
| :--- | :--- |
| **차단기 오탐지** | `minAreaRect`를 활용한 막대 형태 인식 및 HSB 파라미터 튜닝  |
| **신호등 미탐지** | ROI 영역 확대 및 `SimpleBlobDetector` 파라미터 완화  |
| **조명 노이즈** | 감마 보정, CLAHE, 난반사 억제 전처리 로직 추가  |
| **주행 불안정** | 디바운싱 기법 적용을 통한 상태값 안정화  |

---

## 5. 성능 평가 및 결과

본 프로젝트는 다음과 같은 성과를 거두었습니다.
* **태스크 완수**: 차단기, 신호등, Pick-and-Place 등 모든 시나리오 임무 완료
* **주행 안정성**: 고도화된 전처리를 통해 코너링 및 난반사 구간 문제 완벽 해결
* **효율성**: 별도의 추가 센서 없이 단일 카메라만으로 모든 시각 지능 구현
* **자체 평가 점수**: **8점 / 10점** 

---

## 6. 향후 연구 및 발전 방향

현재 시스템의 한계를 극복하기 위해 다음과 같은 개선을 계획하고 있습니다.
* **시스템 통합**: 여러 노드를 하나의 Launch 파일로 관리하여 운용 편의성 증대
* **제어 알고리즘 고도화**: 단순 시퀀스 제어에서 벗어나 Inverse Kinematics 및 PD 제어 도입
* **딥러닝 융합**: 단순 색상 기반 인식을 넘어 YOLO 등 객체 탐지 모델의 완전한 통합

---

## 7. 결론

디지털 트윈 기반의 시뮬레이션은 개발 주기를 단축시켰지만, 실제 환경에서의 물리적 변수는 여전히 큰 도전 과제였습니다. 하지만 HSB 공간에서의 파라미터 최적화와 정교한 전처리 과정을 통해 가상과 현실의 괴리를 성공적으로 메울 수 있었습니다. 이번 프로젝트는 ROS2 환경에서 서비스 로봇의 핵심 기능을 통합하는 귀중한 경험이 되었습니다.

---

### Acknowledgments
본 프로젝트를 지도해주신 김루진 멘토님과 두산로보틱스 K-Digital Training 관계자 분들께 감사드립니다.


---

## References

1. [Digital Twin Image](https://unity.com/kr/topics/digital-twin-definition)
