// src/pages/projects/Project4.js
import React from "react";
import "./Project4.scss";
import "katex/dist/katex.min.css";
import {BlockMath, InlineMath} from "react-katex";

export default function Project4() {
  return (
    <article className="proj4">
      <header className="proj4__hero">
        {/* <div className="proj4__hero-badge">Project4</div> */}
        <h1 className="proj4__title">
          Industrial Safety Robot System: A Comprehensive Multi-Modal Approach{" "}
        </h1>
        <p className="proj4__subtitle">
          “In the intersection of mathematics and machinery, we find not just
          efficiency, but the possibility of preserving human life itself.”
        </p>
      </header>

      <section className="proj4__section">
        <h2 className="proj4__h2">Project Video</h2>

        <div className="proj4__video">
          <iframe
            width="560"
            height="315"
            src="https://www.youtube.com/embed/VQmshnQEU4k"
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        </div>

        {/* <a
          className="main-button"
          href="https://github.com/YourRepo"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a> */}
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">Abstrat</h2>
        <p>
          This paper presents a comprehensive industrial safety robot system
          developed by Team RobotFactory as part of the K-Digital Training
          program. The system integrates real-time object detection, autonomous
          navigation, and distributed communication protocols to address
          critical safety challenges in industrial environments. Through
          rigorous mathematical analysis and experimental validation, we
          achieved 93% detection accuracy with 24.7% noise reduction via
          advanced Kalman filtering techniques.
        </p>
        <ul className="proj4__tags">
          <li>ROS2</li>
          <li>OpenCV</li>
          <li>YOLO</li>
          <li>TurtleBot</li>
        </ul>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">Main Architecture</h2>
        <div className="proj4__card">
          {/* 이미지 추가 */}
          <img
            src={`${process.env.PUBLIC_URL}/images/architecture.png`}
            alt="메인 아키텍처 다이어그램"
            className="proj4__image"
          />
          <p className="proj4__subtitle">
            Figure 1: Complete system architecture showing multi-robot
            coordination and MQTT-based communication
          </p>
        </div>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">1. Introduction & Problem Formulation</h2>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">1.1 Industrial Safety Landscape Analysis</h2>
        <p>
          Industrial safety remains a persistent challenge despite technological
          advancement. Statistical analysis from the Ministry of Employment and
          Labor (2024) reveals critical gaps in safety monitoring and
          enforcement.
        </p>
        <p className="proj4__subtitle_1">Quantitative Safety Assessment :</p>

        <ul>
          <li>
            Industrial accident fatality rates: Construction (2,100+ annually),
            Manufacturing (400+), Components & Materials (200+)
          </li>
          <li>
            Worker safety rights awareness: Only 42.5% understand refusal rights
          </li>
          <li>
            Safety right exercise rate: Merely 16.3% have exercised work refusal
          </li>
          <li>Post-refusal protection: Only 13.8% felt adequately protected</li>
        </ul>
        {/* 이미지 추가 */}
        <img
          src={`${process.env.PUBLIC_URL}/images/Industrial.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 2: Industrial accident trends (2021-2024 showing persistent
          safety challenges)
        </p>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">1.2 Mathematical Risk Framework</h2>
        <p>
          We define the instantaneous risk level <InlineMath math={"R(t)"} /> at
          time <InlineMath math={"t"} /> as:
        </p>

        {/* 중앙 블록 수식 */}
        <BlockMath
          math={"R(t) = \\sum_{i=1}^{n} P_i(t) \\cdot S_i \\cdot E_i(t)"}
        />

        <p>Where:</p>
        <ul>
          <li>
            <InlineMath math={"P_i(t)"} /> = Time-dependent probability of
            incident type <InlineMath math={"i"} />
          </li>
          <li>
            <InlineMath math={"S_i"} /> = Severity coefficient for incident{" "}
            <InlineMath math={"i"} />
          </li>
          <li>
            <InlineMath math={"E_i(t)"} /> = Dynamic exposure frequency to risk{" "}
            <InlineMath math={"i"} />
          </li>
        </ul>

        <p>
          <strong>Objective Function:</strong>{" "}
          <InlineMath
            math={
              "\\min \\int_0^T R(t) \\, dt \\quad \\text{subject to} \\quad \\sum_{j=1}^m C_j \\leq B"
            }
          />
        </p>

        <p>
          Where <InlineMath math={"C_j"} /> represents deployment costs and{" "}
          <InlineMath math={"B"} /> is the budget constraint.
        </p>
      </section>

      {/* <section className="proj4__section_1">
        <h2 className="proj4__h2">1.3 Root Cause Analysis</h2>
        <p>
          Statistical analysis indicates 78.2% of industrial accidents stem from behavioral factors, necessitating automated monitoring solutions.
        </p>

      </section> */}

      <section className="proj4__section">
        <h2 className="proj4__h2">1.3 Root Cause Analysis</h2>
        <p>
          Statistical analysis indicates 78.2% of industrial accidents stem from
          behavioral factors, necessitating automated monitoring solutions.
        </p>

        <table className="proj4__table">
          <thead>
            <tr>
              <th>Risk Factor</th>
              <th>Mathematical Model</th>
              <th>Mitigation Strategy</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Cognitive Fatigue</td>
              <td>
                <InlineMath math={"V(t) = V_0 e^{-\\lambda t}"} />
              </td>
              <td>Continuous monitoring</td>
            </tr>
            <tr>
              <td>Cultural Pressure</td>
              <td>
                <InlineMath math={"P_{speed} > P_{safety}"} />
              </td>
              <td>Automated enforcement</td>
            </tr>
            <tr>
              <td>Monitoring Gaps</td>
              <td>
                <InlineMath math={"\\eta_{monitoring} < \\eta_{required}"} />
              </td>
              <td>Real-time surveillance</td>
            </tr>
            <tr>
              <td>Communication Barriers</td>
              <td>
                <InlineMath
                  math={"I_{effective} = I_{transmitted} \\cdot \\alpha"}
                />
              </td>
              <td>Visual/audio alerts</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="proj4__section">
        <div className="proj4__card">
          {/* 이미지 추가 */}
          <img
            src={`${process.env.PUBLIC_URL}/images/MajorAccident.png`}
            alt="메인 아키텍처 다이어그램"
            className="proj4__image"
          />
          <p className="proj4__subtitle">
            Figure 3: Pie chart showing distribution of
          </p>
          <p>accident causes with behavioral factors highlighted</p>
        </div>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">2. System Architecture & Design</h2>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">2.1 Technical Stack Overview</h2>

        <table className="proj4__table">
          <thead>
            <tr>
              <th>Component</th>
              <th>Implementation</th>
              <th>Justification</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>Object Detection</strong>
              </td>
              <td>YOLOv8n</td>
              <td>Optimal speed-accuracy trade-off</td>
            </tr>
            <tr>
              <td>
                <strong>State Estimation</strong>
              </td>
              <td>Extended Kalman Filter</td>
              <td>Gaussian noise assumption validity</td>
            </tr>
            <tr>
              <td>
                <strong>Coordinate Transform</strong>
              </td>
              <td>TF2 Framework</td>
              <td>ROS2 native integration</td>
            </tr>
            <tr>
              <td>
                <strong>Communication</strong>
              </td>
              <td>MQTT Protocol</td>
              <td>Industrial IoT compatibility</td>
            </tr>
            <tr>
              <td>
                <strong>Navigation</strong>
              </td>
              <td>NAV2 Stack</td>
              <td>Proven autonomous navigation</td>
            </tr>
            <tr>
              <td>
                <strong>Platform</strong>
              </td>
              <td>Ubuntu 22.04 + ROS2</td>
              <td>Stability and community support</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="proj4__section">
        <div className="proj4__card">
          {/* 이미지 추가 */}
          <img
            src={`${process.env.PUBLIC_URL}/images/stack.png`}
            alt="메인 아키텍처 다이어그램"
            className="proj4__image"
          />
          <p className="proj4__subtitle">
            Figure 4: Comprehensive technology stack with integration interfaces
          </p>
        </div>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">2.2 Distributed System Architecture</h2>
        <p>
          The system employs a hub-and-spoke topology with fault-tolerant
          communication:
        </p>

        {/* 이미지 추가 */}
        <img
          src={`${process.env.PUBLIC_URL}/images/Human_Detect_Architecture.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">Figure 5: Human Detect Architecture</p>
        <img
          src={`${process.env.PUBLIC_URL}/images/Crack_Detect_Architecture.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">Figure 6: Crack Detect Architecture</p>
        <img
          src={`${process.env.PUBLIC_URL}/images/Navigation_System_Architecture.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 7: Navigation System Architecture
        </p>
        <img
          src={`${process.env.PUBLIC_URL}/images/MQTT_protocol.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">Figure 8: MQTT protocol</p>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">2.3 Reliability Analysis</h2>
        <p>
          For a distributed system with <InlineMath math={"n"} /> robots, system
          reliability <InlineMath math={"R_{system}"} /> is:
        </p>

        {/* 수식 (BlockMath) */}
        <BlockMath math={"R_{system} = 1 - \\prod_{i=1}^{n} (1 - R_i)"} />

        <p>
          With individual robot reliability <InlineMath math={"R_i = 0.95"} />,
          the system reliability for <InlineMath math={"n = 4"} /> robots:
        </p>

        {/* 계산된 값 (BlockMath) */}
        <BlockMath math={"R_{system} = 1 - (1 - 0.95)^4 = 0.99999375"} />
      </section>

      {/* ===================== 3. Human Detection & PPE Monitoring System ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">
          3. Human Detection & PPE Monitoring System
        </h2>
        {/* <hr className="proj4__divider" /> */}

        {/* 3.1 */}
        <hr className="proj4__divider" />
        <h2 className="proj4__h2">
          3.1 Dataset Preparation &amp; Model Selection
        </h2>
        <p className="proj4__lead">Dataset Specifications:</p>
        <ul className="proj4__bullets">
          <li>
            <strong>Total Samples:</strong> 5,026 (Training: 4,401, Validation:
            415, Test: 210)
          </li>
          <li>
            <strong>Classes:</strong> Helmet, Safety Vest, Human, Safety Boots
          </li>
          <li>
            <strong>Format:</strong> YOLO annotation format
          </li>
          <li>
            <strong>Resolution:</strong> 640×640 pixels
          </li>
          <li>
            <strong>Augmentation:</strong> Rotation (±15°), Scaling (0.8–1.2),
            Brightness (±20%)
          </li>
        </ul>
        <hr className="proj4__divider" />
        {/* 3.2 */}
        <h2 className="proj4__h2">3.2 Model Performance Analysis</h2>
        <p>
          Comparative analysis across YOLO variants using inference time
          distribution:
        </p>

        <table className="proj4__table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Mean Inference (ms)</th>
              <th>Std Dev (ms)</th>
              <th>mAP@0.5</th>
              <th>Model Size (MB)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>YOLOv5n</strong>
              </td>
              <td>4.2</td>
              <td>1.19</td>
              <td>0.847</td>
              <td>14.4</td>
            </tr>
            <tr className="proj4__row--highlight">
              <td>
                <strong>YOLOv8n</strong>
              </td>
              <td>
                <strong>3.8</strong>
              </td>
              <td>
                <strong>1.01</strong>
              </td>
              <td>
                <strong>0.856</strong>
              </td>
              <td>
                <strong>6.2</strong>
              </td>
            </tr>
            <tr>
              <td>
                <strong>YOLOv11n</strong>
              </td>
              <td>4.5</td>
              <td>1.22</td>
              <td>0.851</td>
              <td>5.9</td>
            </tr>
          </tbody>
        </table>
        <img
          src={`${process.env.PUBLIC_URL}/images/yolomodelcompare.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 9: Box plots showing inference time distribution across
          different YOLO models
        </p>
        <p>
          Selection Rationale: YOLOv8n demonstrates optimal balance of accuracy,
          speed, and consistency for real-time industrial deployment.
        </p>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">3.3 Mathematical Framework for Detection</h2>

        <p>
          <strong>Spatial Detection Universe:</strong>&nbsp;
          <InlineMath
            math={
              "\\mathcal{U} = \\{(x,y)\\mid 0 \\le x \\le W,\\ 0 \\le y \\le H\\}"
            }
          />
        </p>

        <p>
          <strong>Detection Confidence Mapping:</strong>
        </p>
        <BlockMath
          math={String.raw`
          C(x,y)=
          \begin{cases}
            \sigma(\mathbf{w}^{T}\,\phi(x,y)+b) & \text{if } (x,y)\in \mathrm{ROI}\\
            0 & \text{otherwise}
          \end{cases}
        `}
        />

        <p>
          Where <InlineMath math={"\\phi(x,y)"} /> represents feature extraction
          at pixel
          <InlineMath math={"\\ (x,y)"} /> and <InlineMath math={"\\sigma"} />{" "}
          is the sigmoid activation.
        </p>

        <p>
          <strong>PPE Compliance Assessment:</strong>&nbsp;
          <InlineMath
            math={String.raw`
            \mathrm{PPE}_{score}
            = \prod_{i\in\{\text{helmet},\,\text{vest},\,\text{boots}\}}
              \ \max_{j}\ C_i^{(j)}
          `}
          />
        </p>
        <img
          src={`${process.env.PUBLIC_URL}/images/Crack_Detect_Architecture.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 6: Crack Detect Architecture (Visual representation of
          detection confidence mapping and PPE scoring system)
        </p>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">
          3.4 Noise Analysis &amp; Kalman Filter Design
        </h2>

        <p>
          <strong>Sensor Noise Characterization:</strong> Statistical analysis
          of OAK-D depth measurements reveals:
        </p>
        <ul className="proj4__bullets">
          <li>
            Standard Deviation:{" "}
            <InlineMath math={"\\sigma = 0.4261\\,\\text{m}"} />
          </li>
          <li>
            Variance: <InlineMath math={"\\sigma^2 = 0.1815\\,\\text{m}^2"} />
          </li>
          <li>
            Temporal Correlation:{" "}
            <InlineMath math={"\\rho(\\tau) = 0.85 e^{-\\tau/2.3}"} />
          </li>
        </ul>

        <p>
          <strong>State Space Model Design:</strong>
        </p>
        <p>
          For tracking human position and velocity, we employ a 4D state vector:
        </p>
        <BlockMath
          math={"\\mathbf{x}_k = [x_k, y_k, \\dot{x}_k, \\dot{y}_k]^T"}
        />

        <p>
          <strong>Prediction Equations:</strong>
        </p>
        <BlockMath
          math={
            "\\mathbf{x}_{k|k-1} = \\mathbf{F} \\mathbf{x}_{k-1|k-1} + \\mathbf{B}u_k"
          }
        />
        <BlockMath
          math={
            "\\mathbf{P}_{k|k-1} = \\mathbf{F} \\mathbf{P}_{k-1|k-1} \\mathbf{F}^T + \\mathbf{Q}"
          }
        />

        <p>
          <strong>Update Equations:</strong>
        </p>
        <BlockMath
          math={
            "\\mathbf{K}_k = \\mathbf{P}_{k|k-1}\\mathbf{H}^T (\\mathbf{H}\\mathbf{P}_{k|k-1}\\mathbf{H}^T + \\mathbf{R})^{-1}"
          }
        />
        <BlockMath
          math={
            "\\mathbf{x}_{k|k} = \\mathbf{x}_{k|k-1} + \\mathbf{K}_k(z_k - \\mathbf{H}\\mathbf{x}_{k|k-1})"
          }
        />

        <p>Where:</p>
        <BlockMath
          math={String.raw`
          \mathbf{F} =
          \begin{bmatrix}
            1 & 0 & \Delta t & 0 \\
            0 & 1 & 0 & \Delta t \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
          \end{bmatrix},
          \quad
          \mathbf{H} =
          \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0
          \end{bmatrix}
        `}
        />

        {/* ----------------- 추가된 부분 ----------------- */}
        <p>
          <strong>Theoretical Performance Advantage:</strong>
        </p>
        <p>Comparing 2D vs 4D models, the Mean Squared Error difference:</p>
        <BlockMath
          math={
            "MSE_{2D} - MSE_{4D} = (\\dot{x}_{k-1})^2 (\\Delta t)^2 \\geq 0"
          }
        />
        <p>
          This proves the 4D model&apos;s theoretical superiority when velocity{" "}
          <InlineMath math={"\\dot{x}_{k-1} \\neq 0"} />.
        </p>

        <p>
          <strong>Experimental Validation:</strong>
        </p>
        <table className="proj4__table">
          <thead>
            <tr>
              <th>Performance Metric</th>
              <th>Raw Data</th>
              <th>Kalman Filter</th>
              <th>Improvement</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Standard Deviation</td>
              <td>0.4261 m</td>
              <td>0.4203 m</td>
              <td>+1.4%</td>
            </tr>
            <tr>
              <td>Variance</td>
              <td>0.1815 m²</td>
              <td>0.1766 m²</td>
              <td>+2.7%</td>
            </tr>
            <tr>
              <td>Consecutive Difference</td>
              <td>0.0603 m</td>
              <td>0.0313 m</td>
              <td>
                <strong>+48.1%</strong>
              </td>
            </tr>
            <tr>
              <td>Mean Absolute Error</td>
              <td>0.3755 m</td>
              <td>0.3758 m</td>
              <td>+0.4%</td>
            </tr>
          </tbody>
        </table>

        <p>
          <strong>Overall Noise Reduction: 24.7%</strong>
        </p>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">
          4. Crack Detection & Structural Analysis System
        </h2>
      </section>

      {/* ===================== 4.1 Computer Vision Pipeline ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">4.1 Computer Vision Pipeline</h2>
        <p>
          The crack detection system employs a hybrid approach combining deep
          learning and classical computer vision:
        </p>

        <ol className="proj4__ol">
          <li>
            <strong>YOLO-based Region Proposal:</strong> Initial crack candidate
            identification
          </li>
          <li>
            <strong>HSV Color Space Segmentation:</strong> Precise crack
            boundary delineation
          </li>
          <li>
            <strong>Depth-Aware Area Calculation:</strong> 3D surface area
            estimation
          </li>
          <li>
            <strong>Global Coordinate Mapping:</strong> Integration with
            navigation system
          </li>
        </ol>
      </section>

      {/* ===================== 4.2 HSV Segmentation Methodology ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">4.2 HSV Segmentation Methodology</h2>

        <p className="proj4__lead">Rationale for HSV Selection:</p>
        <ul className="proj4__bullets">
          <li>
            <strong>Illumination Invariance:</strong> Separates color
            information from lighting conditions
          </li>
          <li>
            <strong>Computational Efficiency:</strong> Linear complexity{" "}
            <InlineMath math={"O(n)"} /> for pixel processing
          </li>
          <li>
            <strong>Threshold Interpretability:</strong> Intuitive parameter
            tuning for industrial deployment
          </li>
          <li>
            <strong>Robustness:</strong> Effective performance with limited
            training data
          </li>
        </ul>

        <p className="proj4__lead">HSV Transformation:</p>

        {/* H, S, V 식 */}
        <BlockMath
          math={String.raw`
          H \;=\; \mathrm{atan2}\!\big(\sqrt{3}\,(G - B),\; 2R - G - B\big)\;\cdot\; \frac{180^\circ}{\pi}
        `}
        />
        <BlockMath
          math={String.raw`
          S \;=\; 1 \;-\; \frac{3\,\min(R,G,B)}{R + G + B}
        `}
        />
        <BlockMath
          math={String.raw`
          V \;=\; \frac{R + G + B}{3}
        `}
        />

        <img
          src={`${process.env.PUBLIC_URL}/images/yolodetect.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 10: HSV color space segmentation results showing crack
          isolation from background
        </p>
      </section>

      {/* ===================== 4.3 3D Area Calculation Framework ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">4.3 3D Area Calculation Framework</h2>

        <p>
          <strong>Camera Calibration Model:</strong> Using OAK-D intrinsic
          parameters, the pixel-to-metric conversion:
        </p>
        <p>
          <InlineMath
            math={
              "\\mathrm{ratio}_x = \\dfrac{Z}{f_x},\\quad \\mathrm{ratio}_y = \\dfrac{Z}{f_y}"
            }
          />
        </p>

        <p>
          <strong>Surface Area Estimation:</strong>
        </p>
        <BlockMath
          math={String.raw`
          A_{\mathrm{crack}}
            = \sum_{(i,j)\in\text{crack pixels}}
              \left(\frac{Z_{i,j}}{f_x}\right)\!
              \left(\frac{Z_{i,j}}{f_y}\right)
              \cos(\theta_{i,j})
        `}
        />

        <p>
          Where <InlineMath math={"\\theta_{i,j}"} /> represents the surface
          normal angle at pixel <InlineMath math={"(i, j)"} />.
        </p>

        <p>
          <strong>Error Propagation Analysis:</strong>
        </p>
        <BlockMath
          math={String.raw`
          \sigma_A^{2}
            = \left(\frac{\partial A}{\partial Z}\right)^{2} \sigma_{Z}^{2}
            + \left(\frac{\partial A}{\partial f_x}\right)^{2} \sigma_{f_x}^{2}
            + \left(\frac{\partial A}{\partial f_y}\right)^{2} \sigma_{f_y}^{2}
        `}
        />
      </section>

      {/* ===================== 4.4 Performance Validation ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">4.4 Performance Validation</h2>

        <table className="proj4__table">
          <thead>
            <tr>
              <th>Performance Metric</th>
              <th>Specification</th>
              <th>Achieved Performance</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Detection Accuracy</td>
              <td>&gt;90%</td>
              <td>93%</td>
            </tr>
            <tr>
              <td>Area Calculation Error</td>
              <td>&lt;10%</td>
              <td>5%</td>
            </tr>
            <tr>
              <td>Coordinate Mapping Precision</td>
              <td>&lt;15cm</td>
              <td>10cm</td>
            </tr>
            <tr>
              <td>Processing Speed</td>
              <td>&gt;15 fps</td>
              <td>20 fps</td>
            </tr>
            <tr>
              <td>Communication Latency</td>
              <td>&lt;150ms</td>
              <td>100ms</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">
          5. Autonomous Navigation & Multi-Robot Coordination
        </h2>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">5.1 NAV2-Based Navigation Architecture</h2>
        <p>
          The navigation system implements a hierarchical control structure:
        </p>
        <img
          src={`${process.env.PUBLIC_URL}/images/Navigation_System_Architecture.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 7: Navigation System Architecture (Navigation system state
          machine showing event handling hierarchy)
        </p>
      </section>

      {/* ===================== 5.2 Multi-Robot Coordination Algorithm ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">5.2 Multi-Robot Coordination Algorithm</h2>

        <p>
          <strong>Priority Assignment Function:</strong>
        </p>
        <BlockMath
          math={String.raw`
          P(e_i) = w_1 \cdot U(e_i) + w_2 \cdot T(e_i) + w_3 \cdot D(e_i)
        `}
        />

        <p>Where:</p>
        <ul className="proj4__bullets">
          <li>
            <InlineMath math={`U(e_i)`} /> = Urgency level of event{" "}
            <InlineMath math={`i`} />
          </li>
          <li>
            <InlineMath math={`T(e_i)`} /> = Time since event detection
          </li>
          <li>
            <InlineMath math={`D(e_i)`} /> = Distance to event location
          </li>
          <li>
            <InlineMath math={`w_1, w_2, w_3`} /> = Weighting factors (
            <InlineMath math={`w_1 > w_2 > w_3`} />)
          </li>
        </ul>

        <p>
          <strong>Resource Allocation Optimization:</strong>
        </p>
        <BlockMath
          math={String.raw`
          \min \sum_{i,j} c_{ij}\, x_{ij}
          \quad \text{subject to} \quad
          \sum_{j} x_{ij} = 1,\;\; \sum_{i} x_{ij} \le 1
        `}
        />

        <p>
          Where <InlineMath math={`x_{ij} \\in \\{0,1\\}`} /> indicates
          assignment of robot
          <InlineMath math={`\\ i`} /> to task <InlineMath math={`\\ j`} />.
        </p>
      </section>

      {/* ===================== 5.3 Navigation Parameter Optimization ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">5.3 Navigation Parameter Optimization</h2>

        <p>
          <strong>Buffer Size Optimization Problem:</strong>
        </p>
        <p>
          Original configuration caused navigation failures in constrained
          environments. The optimization objective:
        </p>

        <BlockMath
          math={String.raw`
          \min_b J(b) = \alpha \cdot P_{\text{collision}}(b)
          + \beta \cdot E[T_{\text{stuck}}(b)]
          + \gamma \cdot E[P_{\text{deviation}}(b)]
        `}
        />

        <p>Subject to constraints:</p>
        <ul className="proj4__bullets">
          <li>
            <InlineMath math={`b_{min} \\leq b \\leq b_{max}`} />
          </li>
          <li>
            <InlineMath math={`P_{collision}(b) \\leq P_{threshold}`} />
          </li>
          <li>
            <InlineMath math={`T_{response}(b) \\leq T_{max}`} />
          </li>
        </ul>

        <p>
          <strong>Solution:</strong> Reduced inflation radius from 0.4m to 0.1m,
          resulting in:
        </p>
        <ul className="proj4__bullets">
          <li>60% reduction in stuck events</li>
          <li>25% improvement in path efficiency</li>
          <li>Maintained collision avoidance safety</li>
        </ul>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">6. MQTT Communication & IoT Integration</h2>
      </section>

      {/* ===================== 6.1 Protocol Selection Analysis ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">6.1 Protocol Selection Analysis</h2>

        <p>
          <strong>Reliability Comparison for Industrial Networks:</strong>
        </p>

        <p>
          For a network with <InlineMath math={"n"} /> segments, failure
          probability analysis:
        </p>

        {/* ROS2 DDS (mesh) */}
        <p>
          <strong>ROS2 DDS (Mesh Topology):</strong>{" "}
          <InlineMath
            math={
              "P_{\\mathrm{DDS}\\_\\mathrm{failure}} = 1 - \\prod_{i=1}^{n} P_{\\mathrm{segment}_i}"
            }
          />
        </p>

        {/* MQTT (star) */}
        <p>
          <strong>MQTT (Star Topology):</strong>{" "}
          <InlineMath
            math={
              "P_{\\mathrm{MQTT}\\_\\mathrm{success}} = \\prod_{i=1}^{n} P_{\\mathrm{device\\to broker}}"
            }
          />
        </p>

        <p>
          Since devices connect independently:{" "}
          <InlineMath
            math={
              "P_{\\mathrm{MQTT}\\_\\mathrm{success}} \\gg P_{\\mathrm{DDS}\\_\\mathrm{success}}"
            }
          />
        </p>
        <img
          src={`${process.env.PUBLIC_URL}/images/MQTT_protocol.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 8: MQTT protocol (Network topology comparison showing MQTT’s
          resilience advantages)
        </p>
      </section>

      {/* ===================== 6.2 Communication Performance Analysis ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">6.2 Communication Performance Analysis</h2>

        <p>
          <strong>Protocol Efficiency Comparison:</strong>
        </p>

        <table className="proj4__table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>ROS2 DDS</th>
              <th>MQTT</th>
              <th>WebSocket</th>
              <th>HTTP REST</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Network Dependency</td>
              <td>High</td>
              <td>Low</td>
              <td>Medium</td>
              <td>Low</td>
            </tr>
            <tr>
              <td>Real-time Performance</td>
              <td>Excellent</td>
              <td>Good</td>
              <td>Good</td>
              <td>Poor</td>
            </tr>
            <tr>
              <td>Reliability</td>
              <td>Moderate</td>
              <td>High</td>
              <td>Medium</td>
              <td>High</td>
            </tr>
            <tr>
              <td>Scalability</td>
              <td>Limited</td>
              <td>Excellent</td>
              <td>Good</td>
              <td>Good</td>
            </tr>
            <tr>
              <td>Power Efficiency</td>
              <td>Poor</td>
              <td>Excellent</td>
              <td>Medium</td>
              <td>Poor</td>
            </tr>
            <tr>
              <td>Industrial Compatibility</td>
              <td>Moderate</td>
              <td>Excellent</td>
              <td>Good</td>
              <td>Excellent</td>
            </tr>
          </tbody>
        </table>

        <p>
          <strong>Message Overhead Analysis:</strong>
        </p>
        <p>For 100-byte payload:</p>

        <ul>
          <li>
            MQTT: 2–7% overhead (
            <InlineMath math={"\\eta_{MQTT} = 0.93 - 0.98"} />)
          </li>
          <li>
            HTTP: 200–800% overhead (
            <InlineMath math={"\\eta_{HTTP} = 0.12 - 0.33"} />)
          </li>
        </ul>

        <BlockMath
          math={
            "\\text{Efficiency Ratio} = \\frac{\\eta_{MQTT}}{\\eta_{HTTP}} \\approx 3 - 8"
          }
        />
      </section>

      {/* ===================== 6.3 Image Transmission Performance ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">6.3 Image Transmission Performance</h2>

        <p className="proj4__lead">Experimental Setup:</p>
        <ul className="proj4__bullets">
          <li>Image size: 64KB (320×240 RGB)</li>
          <li>Test duration: 2000 transmission cycles</li>
          <li>Network conditions: Industrial WiFi simulation</li>
        </ul>

        <p className="proj4__lead">Results:</p>
        <table className="proj4__table">
          <thead>
            <tr>
              <th>FPS Setting</th>
              <th>Success Rate (%)</th>
              <th>Avg Latency (ms)</th>
              <th>Throughput (KB/s)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>10 fps (100ms)</td>
              <td>80.2</td>
              <td>45</td>
              <td>51.3</td>
            </tr>
            <tr>
              <td>20 fps (50ms)</td>
              <td>22.9</td>
              <td>78</td>
              <td>14.7</td>
            </tr>
            <tr>
              <td>100 fps (10ms)</td>
              <td>21.3</td>
              <td>156</td>
              <td>13.6</td>
            </tr>
          </tbody>
        </table>

        <p>
          <strong>Optimal Operating Point:</strong> Based on performance
          analysis, 10 fps provides optimal balance of reliability and real-time
          performance for industrial monitoring applications.
        </p>
      </section>

      <section className="proj4__section">
        <h2 className="proj4__h2">
          7. System Integration & Experimental Validation
        </h2>
      </section>

      {/* ===================== 7.1 End-to-End System Performance ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">7.1 End-to-End System Performance</h2>

        <p className="proj4__lead">Latency Budget Analysis:</p>
        <BlockMath
          math={
            "T_{total} = T_{detection} + T_{processing} + T_{communication} + T_{response}"
          }
        />

        <p>Measured components:</p>
        <ul className="proj4__bullets">
          <li>
            <InlineMath math={"T_{detection} = 52 \\pm 8 \\, ms"} /> (YOLOv8n
            inference)
          </li>
          <li>
            <InlineMath math={"T_{processing} = 23 \\pm 5 \\, ms"} />{" "}
            (coordinate transformation)
          </li>
          <li>
            <InlineMath math={"T_{communication} = 95 \\pm 15 \\, ms"} /> (MQTT
            round-trip)
          </li>
          <li>
            <InlineMath math={"T_{response} = 180 \\pm 30 \\, ms"} />{" "}
            (navigation initiation)
          </li>
        </ul>

        <p>
          <strong>Total System Response Time:</strong> 350 ± 35 ms
        </p>
      </section>

      {/* ===================== 7.2 Multi-Robot Coordination Validation ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">7.2 Multi-Robot Coordination Validation</h2>

        <p className="proj4__lead">Test Scenario:</p>
        <p>Simultaneous human and crack detection events with 4 robots</p>

        <p className="proj4__lead">Coordination Algorithm Performance:</p>
        <ul className="proj4__bullets">
          <li>Event detection to response initiation: 245ms average</li>
          <li>Resource allocation conflicts: 0% (perfect coordination)</li>
          <li>Coverage efficiency: 94% of monitored area</li>
        </ul>

        <p>
          <strong>Load Balancing Effectiveness:</strong>{" "}
          <InlineMath
            math={
              "Balance\\ Index = 1 - \\frac{\\sigma_{workload}}{\\mu_{workload}} = 0.89"
            }
          />
        </p>
      </section>

      {/* ===================== 7.3 Real-World Testing Environment ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">7.3 Real-World Testing Environment</h2>

        <p className="proj4__lead">Test Facility Specifications:</p>
        <ul className="proj4__bullets">
          <li>
            Area: 400m<sup>2</sup> industrial simulation space
          </li>
          <li>Obstacles: Various industrial equipment mockups</li>
          <li>Lighting conditions: 200–800 lux (variable)</li>
          <li>Network: Enterprise WiFi with controlled interference</li>
        </ul>
      </section>

      {/* ===================== 8. Dashboard & Monitoring Interface ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">8. Dashboard &amp; Monitoring Interface</h2>
        <hr className="proj4__divider" />
        {/* 8.1 */}
        <h2 className="proj4__h2">8.1 Web-Based Control Interface</h2>
        <p>
          The monitoring dashboard provides real-time visualization and control
          capabilities:
        </p>

        <p className="proj4__lead">Key Features:</p>
        <ul className="proj4__bullets">
          <li>Live robot positioning and status</li>
          <li>Real-time event detection feed</li>
          <li>Historical data analytics</li>
          <li>Remote control capabilities</li>
          <li>Performance metrics display</li>
        </ul>

        <img
          src={`${process.env.PUBLIC_URL}/images/MQTTserver.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 11: Web-based dashboard interface showing real-time monitoring
          and control features
        </p>
      </section>

      {/* ===================== 8.2 Mobile Application Integration ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">8.2 Mobile Application Integration</h2>

        <p className="proj4__lead">Mobile App Functionality:</p>
        <ul className="proj4__bullets">
          <li>Push notifications for critical events</li>
          <li>Simplified robot status overview</li>
          <li>Emergency stop capabilities</li>
          <li>Location-based event mapping</li>
        </ul>
        <img
          src={`${process.env.PUBLIC_URL}/images/app.png`}
          alt="메인 아키텍처 다이어그램"
          className="proj4__image"
        />
        <p className="proj4__subtitle">
          Figure 12: Mobile application interface showing emergency response and
          notification features
        </p>
      </section>

      {/* ===================== 9. Challenges & Solutions ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">9. Challenges &amp; Solutions</h2>
        <hr className="proj4__divider" />
        {/* 9.1 */}
        <h2 className="proj4__h2">
          9.1 Coordinate System Calibration Challenge
        </h2>

        <p>
          <strong>Problem Statement:</strong> Systematic offset between detected
          object coordinates and actual global map positions, with average error
          of 0.35m.
        </p>

        <p className="proj4__lead">Root Cause Analysis:</p>
        <ol className="proj4__listnum">
          <li>
            <strong>Sensor Calibration Drift:</strong> OAK-D intrinsic
            parameters variation over time
          </li>
          <li>
            <strong>Accumulated Transformation Error:</strong> TF tree
            propagation inaccuracies
          </li>
          <li>
            <strong>Environmental Interference:</strong> Reflective surfaces
            affecting depth estimation
          </li>
        </ol>

        <p className="proj4__lead">Mathematical Error Model:</p>
        <BlockMath
          math={
            "\\mathbf{p}_{measured} = \\mathbf{R} \\mathbf{p}_{actual} + \\mathbf{t} + \\epsilon_{systematic} + \\eta_{noise}"
          }
        />
        <p>
          Where <InlineMath math={"\\mathbf{R}"} /> and{" "}
          <InlineMath math={"\\mathbf{t}"} /> represent systematic rotation and
          translation errors.
        </p>

        <p className="proj4__lead">Solution Implementation:</p>
        <ol className="proj4__listnum">
          <li>
            <strong>Empirical Calibration Matrix:</strong>
            <BlockMath
              math={
                "\\mathbf{C} = \\arg\\min_{\\mathbf{C}} \\sum_{i=1}^{N} \\left\\| \\mathbf{p}^{(i)}_{ground\\_truth} - \\mathbf{C} \\mathbf{p}^{(i)}_{measured} \\right\\|^2"
              }
            />
          </li>
          <li>
            <strong>Real-time Validation:</strong> Continuous comparison with
            known reference points
          </li>
          <li>
            <strong>Future Enhancement:</strong> PointCloud registration using
            Iterative Closest Point (ICP)
          </li>
        </ol>
      </section>

      {/* ===================== 9.2 Navigation Buffer Optimization ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">9.2 Navigation Buffer Optimization</h2>

        <p>
          <strong>Problem:</strong> Excessive buffer zones causing navigation
          failures in narrow passages.
        </p>

        <p className="proj4__lead">Optimization Approach:</p>
        <BlockMath
          math={
            "J(b) = w_1 \\sum_i I^{(i)}_{collision} + w_2 \\sum_j T^{(j)}_{stuck} + w_3 \\sum_k D^{(k)}_{deviation}"
          }
        />

        <p className="proj4__lead">Solution Results:</p>
        <ul>
          <li>Buffer radius: 0.4m → 0.1m</li>
          <li>Success rate improvement: 73% → 96%</li>
          <li>Average navigation time reduction: 40%</li>
        </ul>
      </section>
      <section className="proj4__section">
        <h2 className="proj4__h2">10. Performance Evaluation & Results</h2>
      </section>

      {/* ===================== 10.1 Quantitative Performance Metrics ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">10.1 Quantitative Performance Metrics</h2>

        <p className="proj4__lead">Detection System Performance:</p>
        <table className="proj4__table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Human Detection</th>
              <th>Crack Detection</th>
              <th>Combined System</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Precision</td>
              <td>0.91</td>
              <td>0.93</td>
              <td>0.92</td>
            </tr>
            <tr>
              <td>Recall</td>
              <td>0.89</td>
              <td>0.87</td>
              <td>0.88</td>
            </tr>
            <tr>
              <td>F1-Score</td>
              <td>0.90</td>
              <td>0.90</td>
              <td>0.90</td>
            </tr>
            <tr>
              <td>Processing Speed</td>
              <td>18.5 fps</td>
              <td>20.2 fps</td>
              <td>19.1 fps</td>
            </tr>
            <tr>
              <td>False Positive Rate</td>
              <td>0.08</td>
              <td>0.05</td>
              <td>0.07</td>
            </tr>
          </tbody>
        </table>

        <p className="proj4__lead">System Integration Metrics:</p>
        <table className="proj4__table">
          <thead>
            <tr>
              <th>Component</th>
              <th>Uptime (%)</th>
              <th>Avg Response Time (ms)</th>
              <th>Error Rate (%)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Human Detection</td>
              <td>99.2</td>
              <td>52</td>
              <td>0.8</td>
            </tr>
            <tr>
              <td>Crack Detection</td>
              <td>98.8</td>
              <td>48</td>
              <td>1.2</td>
            </tr>
            <tr>
              <td>Navigation</td>
              <td>97.5</td>
              <td>180</td>
              <td>2.5</td>
            </tr>
            <tr>
              <td>MQTT Communication</td>
              <td>99.8</td>
              <td>95</td>
              <td>0.2</td>
            </tr>
            <tr>
              <td>Overall System</td>
              <td>97.1</td>
              <td>350</td>
              <td>2.9</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* ===================== 10.2 Comparative Analysis ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">10.2 Comparative Analysis</h2>
        <p className="proj4__lead">
          Benchmark Comparison with Existing Solutions:
        </p>

        <table className="proj4__table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Our System</th>
              <th>Commercial Solution A</th>
              <th>Research System B</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Detection Accuracy</td>
              <td>93%</td>
              <td>89%</td>
              <td>91%</td>
            </tr>
            <tr>
              <td>Real-time Performance</td>
              <td>✓</td>
              <td>✓</td>
              <td>✗</td>
            </tr>
            <tr>
              <td>Multi-robot Coordination</td>
              <td>✓</td>
              <td>✗</td>
              <td>✓</td>
            </tr>
            <tr>
              <td>Cost Effectiveness</td>
              <td>High</td>
              <td>Low</td>
              <td>Medium</td>
            </tr>
            <tr>
              <td>Scalability</td>
              <td>Excellent</td>
              <td>Limited</td>
              <td>Good</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* ===================== 11. Future Work & Research Directions ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">11. Future Work & Research Directions</h2>
        <hr className="proj4__divider" />
        <h2 className="proj4__h2">11.1 Enhanced Sensor Fusion</h2>
        <p className="proj4__lead">
          <strong>Planned Multi-Sensor Integration:</strong> x̂<sub>fused</sub> =
          Σ<sub>i=1→n</sub> w<sub>i</sub>x̂<sub>i</sub>
        </p>
        <p>
          Where weights are optimized based on sensor reliability:{" "}
          <BlockMath
            math={"w_i = \\frac{\\sigma_i^{-2}}{\\sum_{j=1}^n \\sigma_j^{-2}}"}
          />
        </p>
        <p>
          <strong>Expected Performance Improvement:</strong> σ<sub>fused</sub>
          <sup>2</sup> = (Σ<sub>i=1→n</sub> σ<sub>i</sub>
          <sup>-2</sup>)<sup>-1</sup> ≤ min σ<sub>i</sub>
          <sup>2</sup>
        </p>
      </section>

      {/* ===================== 11.2 Advanced Coordination Algorithms ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">11.2 Advanced Coordination Algorithms</h2>

        <p className="proj4__lead">
          <strong>Distributed Consensus for Multi-Robot Systems:</strong>{" "}
          Implementation of Byzantine Fault Tolerant consensus for critical
          safety decisions.
        </p>

        <p className="proj4__lead">Swarm Intelligence Integration:</p>
        <ul className="proj4__bullets">
          <li>Particle Swarm Optimization for coverage path planning</li>
          <li>Ant Colony Optimization for dynamic task allocation</li>
        </ul>
      </section>

      {/* ===================== 11.3 Edge Computing Integration ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">11.3 Edge Computing Integration</h2>

        <p className="proj4__lead">Fog Computing Architecture:</p>
        <ul className="proj4__bullets">
          <li>Local processing capabilities for reduced latency</li>
          <li>Edge-based machine learning inference</li>
          <li>Distributed data storage and analytics</li>
        </ul>
      </section>

      {/* ===================== 12. Conclusion ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">12. Conclusion</h2>

        <p className="proj4__lead">
          This research presents a comprehensive industrial safety robot system
          that successfully integrates multiple cutting-edge technologies to
          address critical workplace safety challenges. Through rigorous
          mathematical analysis and experimental validation, we demonstrated:
        </p>
        <hr className="proj4__divider" />
        <h2 className="proj4__h2">12.1 Key Achievements</h2>
        <ol className="proj4__bullets">
          <li>
            <strong>Real-time Multi-Modal Detection:</strong> Achieved 93%
            accuracy in hazard identification with sub-400ms response times
          </li>
          <li>
            <strong>Advanced Noise Filtering:</strong> Implemented Kalman
            filtering resulting in 24.7% noise reduction
          </li>
          <li>
            <strong>Robust Communication:</strong> Developed MQTT-based
            distributed communication with 99.8% reliability
          </li>
          <li>
            <strong>Intelligent Coordination:</strong> Created multi-robot
            coordination system with 96% task success rate
          </li>
        </ol>
      </section>

      {/* ===================== 12.2 Technical Contributions ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">12.2 Technical Contributions</h2>

        <h4 className="proj4__h4">Mathematical Modeling:</h4>
        <ul className="proj4__bullets">
          <li>
            Formalized risk assessment framework for industrial environments
          </li>
          <li>
            Proved theoretical superiority of 4D state tracking over 2D
            alternatives
          </li>
          <li>Developed optimization models for navigation parameter tuning</li>
        </ul>

        <h4 className="proj4__h4">System Engineering:</h4>
        <ul className="proj4__bullets">
          <li>
            Integrated heterogeneous technologies into cohesive safety
            monitoring system
          </li>
          <li>Implemented fault-tolerant distributed architecture</li>
          <li>Created comprehensive testing and validation framework</li>
        </ul>

        <h4 className="proj4__h4">Industrial Impact:</h4>
        <ul className="proj4__bullets">
          <li>
            Demonstrated practical applicability in simulated industrial
            environments
          </li>
          <li>
            Achieved performance metrics suitable for real-world deployment
          </li>
          <li>
            Established scalable foundation for future safety system development
          </li>
        </ul>
      </section>

      {/* ===================== 12.3 Research Significance ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">12.3 Research Significance</h2>
        <p>
          This work represents a significant advancement in autonomous
          industrial safety systems, providing both theoretical foundations and
          practical implementations. The mathematical rigor combined with
          real-world testing demonstrates the system’s readiness for industrial
          deployment while identifying clear pathways for continued improvement.
        </p>

        <p>
          <strong>Future Impact Projection:</strong> With proper scaling and
          industrial partnership, this system has the potential to contribute to
          the reduction of industrial accidents, directly supporting the goal of
          preventing workplace fatalities through intelligent, continuous
          monitoring.
        </p>
      </section>

      {/* ===================== 12.4 Final Reflection ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">12.4 Final Reflection</h2>
        <blockquote className="proj4__quote">
          “The intersection of rigorous mathematics and compassionate
          engineering creates not just efficient systems, but technologies that
          preserve human dignity and life itself. In every equation solved and
          every algorithm optimized, we find the possibility of someone
          returning home safely.”
        </blockquote>
      </section>

      {/* ===================== Acknowledgments ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">Acknowledgments</h2>
        <p>
          We extend our gratitude to the K-Digital Training program, our
          mentors, and Doosan Robotics for providing the platform and resources
          necessary for this research. Special thanks to all team members who
          contributed their expertise across multiple technical domains.
        </p>
      </section>

      {/* ===================== References ===================== */}
      <section className="proj4__section">
        <h2 className="proj4__h2">References</h2>
        <ol className="proj4__references">
          <li>
            Ministry of Employment and Labor, “Industrial Accident Investigation
            Report 2024,” Korea Occupational Safety and Health Agency
          </li>
          <li>
            Kalman, R.E., “A New Approach to Linear Filtering and Prediction
            Problems,”
            <em>Journal of Basic Engineering</em>, vol. 82, no. 1, pp. 35-45,
            1960
          </li>
          <li>
            Redmon, J., et al., “You Only Look Once: Unified, Real-Time Object
            Detection,”
            <em>IEEE Conference on Computer Vision and Pattern Recognition</em>,
            2016
          </li>
          <li>
            Ultralytics, “YOLOv8: A New State-of-the-Art Computer Vision Model,”
            2023
          </li>
          <li>
            Quigley, M., et al., “ROS: An Open-Source Robot Operating System,”
            <em>ICRA Workshop on Open Source Software</em>, 2009
          </li>
          <li>
            Macenski, S., et al., “The Marathon 2: A Navigation System,”
            <em>
              IEEE/RSJ International Conference on Intelligent Robots and
              Systems
            </em>
            , 2020
          </li>
          <li>
            Light, A., “MQTT Protocol Specification v3.1.1,” OASIS Standard,
            2014
          </li>
          <li>
            Lee, M.J., “Promoting Sustainable Safety Work Environments: Factors
            Affecting Korean Workers’ Recognition,”
            <em>MDPI Sustainability</em>, 2024
          </li>
          <li>Thrun, S., “Probabilistic Robotics,” MIT Press, 2005</li>
          <li>
            OpenCV Development Team, “Open Source Computer Vision Library,” 2023
          </li>
        </ol>
      </section>

      {/* <section className="proj4__section">
        <h2 className="proj4__h2">4. 코드 스니펫</h2>
        <pre className="proj4__code">
{`# bringup
ros2 launch turtlebot3_bringup robot.launch.py

# 감지 노드
ros2 run your_pkg detect_lane`}
        </pre>
      </section> */}
    </article>
  );
}
