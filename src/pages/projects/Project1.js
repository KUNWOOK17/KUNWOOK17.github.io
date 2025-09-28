// import React, {useEffect, useState} from "react";
// import DOMPurify from "dompurify";
// import {marked} from "marked";

// export default function Project1() {
//   const [content, setContent] = useState("Loading...");

//   useEffect(() => {
//     // public 폴더를 기준으로 경로를 설정합니다.
//     fetch("/projects/liquid_injection.md")
//       .then(r => {
//         if (!r.ok) throw new Error("md load fail");
//         return r.text();
//       })
//       .then(md => {
//         const html = marked.parse(md, {mangle: false, headerIds: true});
//         setContent(DOMPurify.sanitize(html));
//       })
//       .catch(e => {
//         setContent("Failed to load content.");
//         console.error(e);
//       });
//   }, []);

//   return (
//     <div
//       style={{
//         maxWidth: "980px",
//         margin: "40px auto",
//         padding: "0 16px",
//         lineHeight: 1.6
//       }}
//     >
//       <h1>Liquid Injection Project</h1>
//       <div dangerouslySetInnerHTML={{__html: content}} />
//     </div>
//   );
// }

import React from "react";
import "./Project1.scss";
import "katex/dist/katex.min.css";
import {BlockMath} from "react-katex";
import MathJax from "react-mathjax2";

export default function Project1() {
  return (
    <article className="proj1">
      <header className="proj1__hero">
        {/* <div className="proj1__hero-badge">Project1</div> */}
        <h1 className="proj1__title">
          Robot-Based Precision Concentration Control System: An Integrated
          Approach to Fluid Dynamic Modeling and Adaptive Control Algorithms{" "}
        </h1>
        <p className="proj1__subtitle">
          “Machines translate the intuition of human hands into mathematical
          formulas. The accuracy of this translation determines the depth of
          technology.”
        </p>
      </header>

      <section className="proj1__section">
        <h2 className="proj1__h2">Project Video</h2>

        <div className="proj1__video">
          <iframe
            width="560"
            height="315"
            src="https://www.youtube.com/embed/8ss9Qkjpmjw"
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

      {/* Abstract Section */}
      <section className="proj1__section">
        <h2 className="proj1__h2">Abstract</h2>
        <p>
          This research presents the development of an automated concentration
          control system utilizing the collaborative robot Doosan M0609 and
          precision sensor systems to control fluid flow through tilted
          container manipulation. To overcome the reproducibility limitations of
          traditional manual control methods, we implemented an intelligent
          system integrating fluid dynamic theoretical modeling with real-time
          feedback control.
        </p>
        <p>
          Key achievements include: achieving target concentration within ±0.5%
          error range based on 1.5g sugar standard, comprehensive analysis of
          nonlinear flow characteristics in tilted containers, development of
          multi-variable regression models incorporating angle, volume, and
          temporal dependencies, and implementation of real-time adaptive
          control algorithms with dynamic learning capabilities.
        </p>
        <p>
          The system demonstrates superior performance compared to conventional
          approaches across accuracy, robustness, and efficiency metrics. This
          work establishes a foundation for scalable intelligent control systems
          capable of learning complex nonlinear behaviors and responding in
          real-time to varying operational conditions.
        </p>
      </section>

      {/* Introduction and Research Background */}
      <section className="proj1__section">
        <h2 className="proj1__h2">1. Introduction and Research Background</h2>
        <hr className="proj1__divider" />
        <h2 className="proj1__h2">1.1 Problem Statement</h2>
        <p>
          The precision control of liquid concentration represents a critical
          technological requirement across diverse industrial sectors including
          food processing, chemical manufacturing, and pharmaceutical
          production. Traditional manual control methodologies exhibit
          fundamental limitations characterized by operator skill dependency,
          inconsistent reproducibility, and systematic accuracy constraints.
        </p>
        <p>
          Contemporary concentration adjustment processes predominantly rely on
          operator sensory perception and experiential knowledge, resulting in
          variable flow rates and compromised reproducibility under changing
          operational conditions. Even with identical sugar quantities,
          concentration outcomes vary significantly based on pouring methodology
          and operator technique.
        </p>
        <hr className="proj1__divider" />
        <h2 className="proj1__h2">1.2 Fundamental Challenges</h2>
        <p>
          The nonlinear relationship between container inclination and flow rate
          presents a particularly complex challenge. Minimal angular adjustments
          can precipitate dramatic flow velocity increases or unexpected
          reductions, rendering simple angle-based prediction models inadequate
          for precision applications.
        </p>

        <p>Mathematically, this can be expressed as:</p>
        <BlockMath math={`\\frac{dQ}{d\\theta} \\neq \\text{constant}`} />

        <p>
          where Q represents flow rate and θ denotes container inclination
          angle. This nonlinearity manifests in the form:
        </p>
        <BlockMath
          math={`Q(\\theta) = f(\\theta, h(t), \\mu, \\rho, g, A_{out}) + \\epsilon(t)`}
        />

        <p>where:</p>
        <ul className="proj1__bullets">
          <li>h(t) = time-dependent liquid height</li>
          <li>μ = dynamic viscosity</li>
          <li>ρ = fluid density</li>
          <li>g = gravitational acceleration</li>
          <li>A_out = outlet cross-sectional area</li>
          <li>ε(t) = stochastic error term</li>
        </ul>
      </section>

      <h2 className="proj1__h2">1.3 Research Objectives</h2>
      <p>This investigation establishes four primary objectives:</p>
      <ol className="proj1__ol">
        <li>
          <strong>Precision Flow Characterization:</strong> Develop
          comprehensive analysis of flow characteristics in tilted container
          configurations with mathematical rigor
        </li>
        <li>
          <strong>High-Accuracy Control:</strong> Implement concentration
          control within ±0.5% error tolerance based on 1.5g sugar reference
          standard
        </li>
        <li>
          <strong>Theoretical Validation:</strong> Apply and experimentally
          verify fluid dynamic mathematical models in practical container
          applications
        </li>
        <li>
          <strong>Real-time Implementation:</strong> Construct autonomous
          concentration adjustment systems capable of stable operation without
          human intervention
        </li>
      </ol>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2. Literature Review and Differentiation</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.1 Existing Research Paradigms</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.1.1 Torricelli’s Law Applications</h2>
      <p>
        Classical gravity-driven discharge velocity models based on Torricelli’s
        principle provide foundational understanding for open container liquid
        outflow prediction:
      </p>
      <BlockMath math={`v = C_d \\sqrt{2 g h_{eff}}`} />
      <p>
        where C_d represents discharge coefficient and h_eff denotes effective
        head height.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.1.2 Industrial Fluid Dynamic Control</h2>
      <p>
        Large-scale industrial fluid control systems typically employ
        pressure-based regulation within pipeline configurations:
      </p>
      <BlockMath
        math={`\\Delta P = f \\cdot \\frac{L}{D} \\cdot \\frac{\\rho v^2}{2}`}
      />
      <p>
        utilizing the Darcy-Weisbach equation for pressure loss calculations.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.1.3 Automated Concentration Technologies</h2>
      <p>
        Chemical process industries implement quantitative dispensing systems
        maintaining specific reactant ratios through precision metering:
      </p>
      <BlockMath
        math={`C_{target} = \\frac{m_{solute}}{m_{solute} + m_{solvent}} \\times 100\\%`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.2 Research Innovation Points</h2>
      <p>This investigation introduces three fundamental innovations:</p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        2.2.1 Three-Dimensional Tilted Container Analysis
      </h2>
      <p>
        Experimental analysis of tilted container characteristics addressing
        irregular flow behavior unexplainable through conventional vertical
        container models. The effective height calculation incorporates angular
        dependencies:
      </p>
      <BlockMath
        math={`h_{eff}(\\theta,t) = h_0 - \\Delta h(t) + L_{container} \\sin(\\theta - \\theta_0)`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.2.2 Multi-variable Regression Framework</h2>
      <p>
        Implementation of integrated multi-variable analysis recognizing
        interdependent relationships between angle, volume, and temporal
        parameters:
      </p>
      <BlockMath
        math={`
          Q(t) = \\alpha_1 \\theta(t) + \\alpha_2 V(t) + \\alpha_3 t 
          + \\alpha_4 \\theta(t) V(t) + \\alpha_5 \\theta(t) t 
          + \\alpha_6 V(t) t + \\alpha_7 \\theta(t) V(t) t + \\epsilon(t)
        `}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">2.2.3 Dynamic Learning-Based Control</h2>
      <p>
        Real-time control methodology employing temporal dynamic adjustment
        rather than static control parameters:
      </p>
      <BlockMath
        math={`
          \\theta_{control}(t+\\Delta t) = \\theta_{control}(t) 
          + K_p e(t) 
          + K_i \\int_0^t e(\\tau) d\\tau 
          + K_d \\frac{de(t)}{dt} 
          + \\Delta \\theta_{adaptive}(t)
        `}
      />
      <p>
        where Δθ<sub>adaptive</sub>(t) represents learning-based correction
        term.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3. System Architecture and Control Flow</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.1 System Design Philosophy</h2>
      <p>
        The implemented system transcends conventional sensor data acquisition,
        establishing an integrated real-time interface connecting robotic
        control with sensor feedback through unified control architecture. The
        control flow diagram represents a quantitative description of automation
        and feedback-based control implementation rather than mere procedural
        enumeration.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.2 Control Flow Mathematical Model</h2>
      <p>The system operation follows a state-space representation:</p>
      <BlockMath
        math={`
          \\begin{bmatrix}
          \\theta(t+1) \\\\
          Q(t+1) \\\\
          C(t+1)
          \\end{bmatrix}
          =
          \\begin{bmatrix}
          a_{11} & a_{12} & a_{13} \\\\
          a_{21} & a_{22} & a_{23} \\\\
          a_{31} & a_{32} & a_{33}
          \\end{bmatrix}
          \\begin{bmatrix}
          \\theta(t) \\\\
          Q(t) \\\\
          C(t)
          \\end{bmatrix}
          +
          \\begin{bmatrix}
          b_1 \\\\
          b_2 \\\\
          b_3
          \\end{bmatrix}
          u(t)
        `}
      />
      <p>where:</p>
      <ul className="proj1__bullets">
        <li>θ(t) = container angle state</li>
        <li>Q(t) = flow rate state</li>
        <li>C(t) = concentration state</li>
        <li>u(t) = control input (target concentration)</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.3 Feedback Control Loop Design</h2>
      <p>
        The automated feedback loop operates through the following mathematical
        sequence:
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.3.1 Target Concentration Reception</h2>
      <BlockMath
        math={`C_{target} \\; \\rightarrow \\; MQTT\\;Protocol \\; \\rightarrow \\; Control\\;Node`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.3.2 Sugar Mass Measurement</h2>
      <BlockMath
        math={`m_{sugar} = \\int_0^{t_1} LoadCell(t) \\, dt - m_{cup}`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.3.3 Flow Rate Control Logic</h2>
      <p>
        If Q<sub>measured</sub> = 0:
      </p>
      <BlockMath
        math={`\\theta_{new} = \\theta_{current} + \\Delta \\theta_{correction}`}
      />
      <p>
        where Δθ<sub>correction</sub> ∈ [1.0°, 2.0°] represents the sensitivity
        control range.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        3.3.4 Concentration Calculation and Verification
      </h2>
      <BlockMath
        math={`C_{current} = \\frac{m_{sugar}}{m_{sugar} + m_{water}} \\times 100\\%`}
      />
      <p>
        If C<sub>current</sub> - C<sub>target</sub> &lt; ε: Terminate
      </p>
      <p>Else: Continue Feedback Loop</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">3.4 Dynamic Control System Characteristics</h2>
      <p>
        This architecture embodies a dynamic control system rather than static
        operation, integrating:
      </p>
      <ul className="proj1__bullets">
        <li>Real-time sugar mass measurement via load cell</li>
        <li>Continuous water volume monitoring</li>
        <li>Angular velocity-based flow modeling</li>
        <li>Adaptive concentration control algorithms</li>
      </ul>
      <p>
        The mathematical foundation ensures industrial-grade precision and
        repeatability, overcoming traditional manual operation limitations.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        4. Hardware Configuration and Sensor Systems
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.1 Multi-Layer System Architecture</h2>
      <p>
        The experimental system architecture follows a hierarchical three-layer
        design: sensor layer, processing layer, and control layer, each
        implementing specific mathematical transformations and signal processing
        algorithms.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        4.2 Sensor Layer: Load Cell and Signal Conditioning
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.2.1 Load Cell Measurement Theory</h2>
      <p>
        The load cell operates on strain gauge principles, converting mechanical
        deformation to electrical resistance changes:
      </p>
      <BlockMath math={`\\Delta R = R_0 \\cdot GF \\cdot \\varepsilon`} />
      <p>where:</p>
      <ul className="proj1__bullets">
        <li>
          R<sub>0</sub> = nominal resistance
        </li>
        <li>GF = gauge factor (≈ 2.0 for metallic strain gauges)</li>
        <li>ε = mechanical strain</li>
      </ul>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.2.2 HX711 Signal Processing</h2>
      <p>The analog-to-digital conversion through HX711 implements:</p>
      <BlockMath
        math={`V_{digital} = \\frac{V_{analog} - V_{offset}}{V_{reference}} \\times 2^{24}`}
      />
      <p>providing 24-bit resolution with programmable gain amplification.</p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.2.3 Mass Calculation Algorithm</h2>
      <BlockMath
        math={`m_{measured}(t) = \\frac{V_{digital}(t) - V_{tare}}{S_{calibration}} + m_{offset}`}
      />
      <p>
        where S<sub>calibration</sub> represents the calibration slope
        determined through known mass standards.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        4.3 Processing Layer: ROS2 Node Architecture
      </h2>
      <p>
        The processing layer is structured on ROS2, facilitating modular,
        distributed, and real-time data handling. Each node is responsible for
        sensor acquisition, data preprocessing, control computation, and
        actuator command publishing, ensuring scalability and fault tolerance
        across the robotic system.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.3.1 Weight Stabilization Algorithm</h2>
      <p>
        The ROS2 weight stabilization node implements moving average filtering
        with stability threshold detection:
      </p>
      <BlockMath math={`\\bar{m}(n) = \\frac{1}{N} \\sum_{i=n-N+1}^{n} m(i)`} />
      <BlockMath
        math={`\\sigma_m(n) = \\sqrt{\\frac{1}{N-1} \\sum_{i=n-N+1}^{n} (m(i) - \\bar{m}(n))^2}`}
      />
      <p>
        Stability criterion: σ<sub>m</sub>(n) &lt; σ<sub>threshold</sub> for T
        <sub>stable</sub> consecutive measurements.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.3.2 MQTT Communication Protocol</h2>
      <p>Data transmission follows the mathematical model:</p>
      <BlockMath
        math={`\\text{Publisher: } P(t) = \\{ m_{stable}(t), \\; \\sigma_m(t), \\; t_{timestamp} \\}`}
      />
      <BlockMath
        math={`\\text{Subscriber: } S(t) = \\arg\\min_{t'} |t - t'| \\; \\; \\text{subject to } P(t') \\neq \\emptyset`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.4 Control Layer: Robot Arm Integration</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.4.1 Kinematic Transformation</h2>
      <p>
        The Doosan M0609 robot arm position control utilizes forward kinematics:
      </p>
      <BlockMath math={`T_{end} = \\prod_{i=1}^{6} T_i(\\theta_i)`} />
      <p>
        where T<sub>i</sub>(θ<sub>i</sub>) represents the homogeneous
        transformation matrix for joint i.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">4.4.2 Angular Control Algorithm</h2>
      <p>Container inclination control implements:</p>
      <BlockMath
        math={`\\theta_{robot}(t) = \\arctan\\left( \\frac{z_{target} - z_{base}}{x_{target} - x_{base}} \\right) + \\theta_{correction}(m_{measured})`}
      />
      <p>
        where θ<sub>correction</sub> represents feedback-based angular
        adjustment.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        4.5 System Integration Mathematical Framework
      </h2>
      <p>
        The complete hardware integration follows the distributed control model:
      </p>
      <BlockMath
        math={`\\begin{bmatrix} \\dot{m} \\\\ \\dot{\\theta} \\\\ \\dot{Q} \\end{bmatrix} = 
        \\mathbf{A} \\begin{bmatrix} m \\\\ \\theta \\\\ Q \\end{bmatrix} + 
        \\mathbf{B} \\begin{bmatrix} u_{gravity} \\\\ u_{robot} \\\\ u_{flow} \\end{bmatrix} + \\mathbf{w}`}
      />
      <p>
        where <i>w</i> represents system noise and <b>A</b>, <b>B</b> are system
        matrices determined through system identification.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        5. Theoretical Foundation and Mathematical Modeling
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        5.1 Modified Bernoulli Equation for Tilted Containers
      </h2>
      <p>
        Traditional Bernoulli analysis requires modification for time-varying,
        inclined container applications. The developed model incorporates
        discharge coefficients and dynamic loss terms:
      </p>
      <BlockMath math={`v_{outlet} = C_d \\sqrt{2g(h_{eff} - h_{loss})}`} />

      <p>where:</p>
      <BlockMath
        math={`h_{eff}(\\theta, t) = h_0 - \\int_0^t \\frac{Q(\\tau)}{A_{surface}(\\tau)} d\\tau + L_{tilt} \\sin(\\theta - \\theta_0)`}
      />
      <BlockMath
        math={`h_{loss} = K_{friction} \\frac{v^2}{2g} + K_{form} \\frac{v^2}{2g} + h_{surface\\_tension}`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.2 Dynamic Flow Rate Modeling</h2>
      <hr className="proj1__divider" />

      <h2 className="proj1__h2">5.2.1 Volume-Height Relationship</h2>
      <p>
        For non-uniform container cross-sections, volume calculation requires
        integration:
      </p>
      <BlockMath math={`V(h) = \\int_0^h A(z) \\, dz`} />

      <p>For truncated conical geometry:</p>
      <BlockMath
        math={`A(z) = \\pi \\left[ r_{bottom} + \\frac{z}{h_{total}} (r_{top} - r_{bottom}) \\right]^2`}
      />
      <BlockMath
        math={`V(h) = \\frac{\\pi h}{3} \\left[ r_{bottom}^2 + r_{bottom} r_{top} + r_{top}^2 \\right]`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.2.2 Time-Dependent Height Evolution</h2>
      <p>The liquid height evolution follows the continuity equation:</p>
      <BlockMath
        math={`\\frac{dh}{dt} = - \\frac{Q_{outlet}}{A_{surface}(h)}`}
      />

      <p>where:</p>
      <BlockMath
        math={`Q_{outlet} = A_{outlet} \\cdot C_d \\sqrt{2g h_{eff}}`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.3 Advanced Fluid Dynamic Analysis</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.3.1 Reynolds Number Analysis</h2>
      <p>Flow regime characterization through Reynolds number:</p>
      <BlockMath
        math={`Re = \\frac{\\rho v D}{\\mu} = \\frac{4 \\rho Q}{\\pi D \\mu}`}
      />

      <p>For the experimental conditions:</p>
      <ul className="proj1__bullets">
        <li>ρ = 998.2 kg/m³</li>
        <li>μ = 1.004 × 10⁻³ Pa·s</li>
        <li>D = 0.005 m</li>
      </ul>

      <BlockMath
        math={`Re = \\frac{4 \\times 998.2 \\times Q}{\\pi \\times 0.005 \\times 1.004 \\times 10^{-3}} \\approx 2.53 \\times 10^8 \\times Q`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        5.3.2 Weber Number for Surface Tension Effects
      </h2>
      <BlockMath math={`We = \\frac{\\rho v^2 L}{\\sigma}`} />

      <p>where σ = 0.0728 N/m represents surface tension.</p>
      <p>For L = D = 0.005 m:</p>
      <BlockMath
        math={`We = \\frac{998.2 \\times v^2 \\times 0.005}{0.0728} = 68.4 \\times v^2`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.3.3 Capillary Number Analysis</h2>
      <BlockMath
        math={`C_a = \\frac{\\mu v}{\\sigma} = \\frac{1.004 \\times 10^{-3} \\times v}{0.0728} = 0.0138 \\times v`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.4 Nonlinear System Dynamics</h2>
      <hr className="proj1__divider" />

      <h2 className="proj1__h2">5.4.1 State-Space Representation</h2>
      <p>The complete system dynamics can be expressed in state-space form:</p>
      <BlockMath math={`\\dot{\\mathbf{x}} = f(\\mathbf{x}, \\mathbf{u}, t)`} />

      <p>where:</p>
      <BlockMath
        math={`\\mathbf{x} = 
        \\begin{bmatrix}
        h(t) \\\\
        \\theta(t) \\\\
        v(t) \\\\
        m_{total}(t)
        \\end{bmatrix}, 
        \\quad
        \\mathbf{u} =
        \\begin{bmatrix}
        \\theta_{cmd}(t) \\\\
        m_{sugar}
        \\end{bmatrix}`}
      />

      <BlockMath
        math={`f(\\mathbf{x}, \\mathbf{u}, t) =
        \\begin{bmatrix}
        - \\frac{A_{outlet}}{A_{surface}} C_d \\sqrt{2gh} \\\\
        K_{robot}(\\theta_{cmd} - \\theta) \\\\
        C_d \\sqrt{2gh} - K_{drag}v^2 \\\\
        -\\rho A_{outlet} C_d \\sqrt{2gh}
        \\end{bmatrix}`}
      />
      <hr className="proj1__divider" />

      <h2 className="proj1__h2">5.4.2 Linearization for Control Design</h2>
      <p>For small perturbations around equilibrium point (x₀, u₀):</p>
      <BlockMath
        math={`\\Delta \\dot{\\mathbf{x}} = \\mathbf{A} \\Delta \\mathbf{x} + \\mathbf{B} \\Delta \\mathbf{u}`}
      />

      <p>where:</p>
      <BlockMath
        math={`\\mathbf{A} = \\left. \\frac{\\partial f}{\\partial \\mathbf{x}} \\right|_{(x_0, u_0)}, 
        \\quad
        \\mathbf{B} = \\left. \\frac{\\partial f}{\\partial \\mathbf{u}} \\right|_{(x_0, u_0)}`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.5 Concentration Dynamics Model</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.5.1 Mass Balance Equation</h2>
      <p>The concentration evolution follows:</p>
      <BlockMath
        math={`\\frac{d}{dt} \\left[ \\frac{m_{sugar}}{m_{sugar} + m_{water}} \\right] 
        = \\frac{m_{sugar}}{(m_{sugar} + m_{water})^2} \\frac{dm_{water}}{dt}`}
      />

      <BlockMath
        math={`= \\frac{m_{sugar} \\rho Q_{in}}{(m_{sugar} + m_{water})^2}`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">5.5.2 Target Concentration Achievement</h2>
      <p>The control objective becomes:</p>
      <BlockMath
        math={`\\min_{Q(t)} \\int_0^T \\left[ C(t) - C_{target} \\right]^2 dt`}
      />

      <p>subject to:</p>
      <ul className="proj1__bullets">
        <li>Q(t) ≥ 0</li>
        <li>θ_min ≤ θ(t) ≤ θ_max</li>
        <li>dθ/dt ≤ ω_max</li>
      </ul>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        6. Experimental Setup and Measurement Protocols
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.1.1 Geometric Parameters</h2>
      <p>
        The experimental container exhibits non-uniform cross-sectional geometry
        requiring precise mathematical characterization:
      </p>

      <table className="proj1__table">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Symbol</th>
            <th>Value</th>
            <th>Unit</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Top Inner Diameter</td>
            <td>D_top</td>
            <td>7.1</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Bottom Inner Diameter</td>
            <td>D_bottom</td>
            <td>7.5</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Base Diameter</td>
            <td>D_base</td>
            <td>6.0</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Outlet Diameter</td>
            <td>D_outlet</td>
            <td>0.5</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Outlet Height</td>
            <td>h_outlet</td>
            <td>8.0</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Spout Length</td>
            <td>L_spout</td>
            <td>8.5</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Total Height</td>
            <td>H_total</td>
            <td>9.5</td>
            <td>cm</td>
          </tr>
          <tr>
            <td>Initial Tilt Angle</td>
            <td>θ₀</td>
            <td>167.0</td>
            <td>degrees</td>
          </tr>
        </tbody>
      </table>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.1.2 Volume Calculation Model</h2>
      <p>The container volume as a function of height follows:</p>

      <BlockMath
        math={`V(h) = \\int_0^h \\pi \\left[ \\frac{D_{bottom}}{2} + 
        \\frac{z}{H_{total}} \\left( \\frac{D_{top} - D_{bottom}}{2} \\right) \\right]^2 dz`}
      />

      <BlockMath
        math={`= \\frac{\\pi h}{12} \\left[ D_{bottom}^2 + D_{bottom}D_{top} + D_{top}^2 \\right] + O(h^2)`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        6.2 Fluid Properties and Environmental Conditions
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.2.1 Fluid Physical Properties</h2>
      <table className="proj1__table">
        <thead>
          <tr>
            <th>Property</th>
            <th>Symbol</th>
            <th>Value</th>
            <th>Unit</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Density</td>
            <td>ρ</td>
            <td>998.2</td>
            <td>kg/m³</td>
          </tr>
          <tr>
            <td>Dynamic Viscosity</td>
            <td>μ</td>
            <td>1.004 × 10⁻³</td>
            <td>Pa·s</td>
          </tr>
          <tr>
            <td>Kinematic Viscosity</td>
            <td>ν</td>
            <td>1.004 × 10⁻⁶</td>
            <td>m²/s</td>
          </tr>
          <tr>
            <td>Surface Tension</td>
            <td>σ</td>
            <td>0.0728</td>
            <td>N/m</td>
          </tr>
          <tr>
            <td>Gravitational Acceleration</td>
            <td>g</td>
            <td>9.81</td>
            <td>m/s²</td>
          </tr>
          <tr>
            <td>Atmospheric Pressure</td>
            <td>P_atm</td>
            <td>101.325</td>
            <td>kPa</td>
          </tr>
        </tbody>
      </table>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.2.2 Environmental Control Parameters</h2>
      <ul className="proj1__bullets">
        <li>Temperature: T = 20 ± 1°C</li>
        <li>Relative Humidity: RH = 50 ± 5%</li>
        <li>Initial Water Volume: V₀ = 300 mL</li>
        <li>Sugar Mass: m_sugar = 1.5 g</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        6.3 Measurement Protocol and Statistical Design
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.3.1 Angular Range and Resolution</h2>
      <p>The experimental design covers:</p>
      <ul className="proj1__bullets">
        <li>Angular Range: θ ∈ [167°, 201°]</li>
        <li>Angular Resolution: Δθ = 1°</li>
        <li>Total Measurement Points: N = 35</li>
        <li>Repetitions per Angle: n = 5</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.3.2 Statistical Analysis Framework</h2>
      <p>
        For each angular position θ<sub>i</sub>, measurements follow:
      </p>
      <BlockMath math={"Q_{i,j} = Q_{true}(\\theta_i) + \\epsilon_{i,j}"} />
      <p>
        where ε<sub>{`{i,j}`}</sub> ~ N(0, σ·ε²)
      </p>

      <p>Sample mean and variance:</p>
      <BlockMath math={"\\bar{Q}_i = \\frac{1}{n} \\sum_{j=1}^{n} Q_{i,j}"} />
      <BlockMath
        math={"s_i^2 = \\frac{1}{n-1} \\sum_{j=1}^{n} (Q_{i,j} - \\bar{Q}_i)^2"}
      />

      <p>Confidence interval for α = 0.05:</p>
      <BlockMath
        math={"\\bar{Q}_i \\pm t_{n-1, \\alpha/2} \\frac{s_i}{\\sqrt{n}}"}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.4 Precision Measurement System</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.4.1 Load Cell Specifications</h2>
      <ul className="proj1__bullets">
        <li>
          <strong>Measurement Range:</strong> 0–5 kg
        </li>
        <li>
          <strong>Resolution:</strong> 0.1 g
        </li>
        <li>
          <strong>Sampling Frequency:</strong> 10 Hz
        </li>
        <li>
          <strong>Temperature Stability:</strong> ±0.02% per °C
        </li>
      </ul>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.4.2 Angular Measurement Precision</h2>
      <ul className="proj1__bullets">
        <li>
          <strong>Angular Resolution:</strong> 0.1°
        </li>
        <li>
          <strong>Repeatability:</strong> ±0.05°
        </li>
        <li>
          <strong>Absolute Accuracy:</strong> ±0.1°
        </li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">6.4.3 Data Acquisition Protocol</h2>
      <BlockMath math={"m(t_k) = LoadCell(t_k) - m_{tare}"} />
      <p>
        where measurements are recorded at t<sub>k</sub> = k · Δt with Δt = 0.1
        s.
      </p>

      <p>Stability criterion:</p>
      <BlockMath
        math={
          "\\left| \\frac{1}{N} \\sum_{i=k-N+1}^{k} m(t_i) - \\frac{1}{N} \\sum_{i=k-2N+1}^{k-N} m(t_i) \\right| < \\epsilon_{stability}"
        }
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7. Experimental Results and Analysis</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        7.1 Flow Rate Characterization vs. Inclination Angle
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.1.1 Empirical Flow Rate Model</h2>
      <p>
        The experimental data reveals a nonlinear relationship between container
        inclination and average flow rate. The measured flow rates Q(θ)
        demonstrate complex behavior requiring advanced mathematical modeling.
      </p>

      <p>
        <strong>Power Law Regression Analysis:</strong>
      </p>
      <BlockMath math={"Q(\\theta) = A \\cdot (\\theta - \\theta_0)^n"} />

      <p>where:</p>
      <ul className="proj1__bullets">
        <li>
          A = 0.139 ml/s/deg<sup>n</sup>
        </li>
        <li>θ₀ = 0.00° (theoretical threshold)</li>
        <li>n = 0.98 (power exponent)</li>
      </ul>

      <p>
        <strong>Statistical Validation:</strong>
      </p>
      <ul className="proj1__bullets">
        <li>Coefficient of Determination: R² = 0.947</li>
        <li>Root Mean Square Error: RMSE = 0.234 ml/s</li>
        <li>Standard Error of Regression: Sₑ = 0.198 ml/s</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.1.2 Critical Angle Analysis</h2>
      <p>The flow behavior exhibits three distinct regimes:</p>

      <p>
        <strong>Regime I: Quasi-Static (θ &lt; 170°)</strong>
      </p>
      <BlockMath
        math={
          "Q(\\theta) \\approx 0.05 \\cdot e^{0.02(\\theta - 167^{\\circ})} \\, ml/s"
        }
      />

      <p>
        <strong>Regime II: Transition (170° ≤ θ ≤ 185°)</strong>
      </p>
      <BlockMath
        math={"Q(\\theta) = 0.139 (\\theta - 167^{\\circ})^{0.98} \\, ml/s"}
      />

      <p>
        <strong>Regime III: High-Flow (θ &gt; 185°)</strong>
      </p>
      <BlockMath
        math={
          "Q(\\theta) = Q_{max} \\left[ 1 - e^{-k(\\theta - 185^{\\circ})} \\right] + Q_{linear}"
        }
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.2 Temporal Flow Dynamics Comparison</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.2.1 Manual vs. Adaptive Control Analysis</h2>
      <p>
        Comparative analysis between manual control (Experiments 0, 1, 2) and
        adaptive control (Experiment 4) reveals fundamental differences in
        temporal flow characteristics.
      </p>

      <h3 className="proj1__h2">Manual Control Dynamics</h3>
      <BlockMath math={"Q_{manual}(t) = Q_0 + \\alpha t + \\beta t^2"} />
      <p>where:</p>
      <ul className="proj1__bullets">
        <li>Q₀ = 0.12 ± 0.03 ml/s (initial flow rate)</li>
        <li>α = 0.045 ± 0.008 ml/s² (linear coefficient)</li>
        <li>β = -0.001 ± 0.0003 ml/s³ (quadratic coefficient)</li>
      </ul>

      <h3 className="proj1__h2">Adaptive Control Dynamics</h3>
      <BlockMath
        math={
          "Q_{adaptive}(t) = Q_{peak} \\cdot e^{-\\lambda t} \\cos(\\omega t + \\phi) + Q_{steady}"
        }
      />
      <p>where:</p>
      <ul className="proj1__bullets">
        <li>Q_peak = 0.85 ml/s (peak flow rate)</li>
        <li>λ = 0.12 s⁻¹ (decay constant)</li>
        <li>ω = 0.31 rad/s (oscillation frequency)</li>
        <li>Q_steady = 0.08 ml/s (steady-state flow rate)</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.2.2 Angular Velocity Correlation</h2>
      <p>The relationship between angular velocity and flow rate follows:</p>
      <BlockMath
        math={
          "Q(t) = K_1 \\frac{d\\theta}{dt}(t) + K_2 \\int_0^t \\frac{d\\theta}{dt}(\\tau) e^{-\\gamma (t-\\tau)} d\\tau"
        }
      />
      <p>where:</p>
      <ul className="proj1__bullets">
        <li>K₁ = 0.023 ml/s/deg/s (instantaneous gain)</li>
        <li>K₂ = 0.015 ml/s/deg/s (memory effect gain)</li>
        <li>γ = 0.45 s⁻¹ (memory decay rate)</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.3 Cumulative Mass Transfer Analysis</h2>
      <hr className="proj1__divider" />
      <h3 className="proj1__h2">7.3.1 Mass Transfer Efficiency</h3>
      <p>
        The cumulative mass change Δm(t) exhibits distinct patterns between
        control methodologies:
      </p>

      <h3 className="proj1__h2">Manual Control Mass Transfer</h3>
      <BlockMath
        math={
          "\\Delta m_{manual}(t) = \\int_0^t Q_{manual}(\\tau) \\rho d\\tau = A_1 t + A_2 t^2 + A_3 t^3"
        }
      />
      <p>where:</p>
      <ul className="proj1__bullets">
        <li>A₁ = 0.12ρ = 119.8 g/s</li>
        <li>A₂ = 0.0225ρ = 22.5 g/s²</li>
        <li>A₃ = -0.00033ρ = -0.33 g/s³</li>
      </ul>

      <h3 className="proj1__h2">Adaptive Control Mass Transfer</h3>
      <BlockMath
        math={
          "\\Delta m_{adaptive}(t) = \\int_0^t Q_{adaptive}(\\tau) \\rho d\\tau"
        }
      />
      <BlockMath
        math={`= \\frac{Q_{peak} \\rho}{\\lambda^2 + \\omega^2} 
          \\Big[ \\lambda (1 - e^{-\\lambda t} \\cos(\\omega t + \\phi)) 
          + \\omega e^{-\\lambda t} \\sin(\\omega t + \\phi) \\Big] 
          + Q_{steady} \\rho t`}
      />

      <hr className="proj1__divider" />
      <h3 className="proj1__h2">7.3.2 Transfer Efficiency Metrics</h3>

      <h4>Settling Time Analysis</h4>
      <BlockMath
        math={
          "t_{settling} = \\frac{-\\ln(0.02)}{\\lambda} = \\frac{3.91}{0.12} = 32.6\\,s"
        }
      />

      <h4>Overshoot Calculation</h4>
      <BlockMath
        math={`Overshoot = \\frac{Q_{peak} - Q_{steady}}{Q_{steady}} \\times 100\\% 
          = \\frac{0.85 - 0.08}{0.08} \\times 100\\% = 962.5\\%`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">7.4 Advanced Fluid Dynamic Analysis</h2>
      <hr className="proj1__divider" />
      <h3 className="proj1__h2">
        7.4.1 Volume Reduction Effect on Flow Characteristics
      </h3>
      <p>
        When liquid volume decreases by 25g (sugar removal), the effective
        height reduces from 7.9 cm to 7.65 cm, resulting in:
      </p>

      <h4>Theoretical Flow Rate Increase:</h4>
      <BlockMath
        math={`\\frac{Q_{new}}{Q_{original}} = 
          \\sqrt{\\frac{h_{eff,new}}{h_{eff,original}}} = 
          \\sqrt{\\frac{8.0 - 7.65}{8.0 - 7.9}} = 
          \\sqrt{\\frac{0.35}{0.1}} = 1.87`}
      />

      <h4>Experimental Observation:</h4>
      <ul className="proj1__bullets">
        <li>Theoretical increase: 87%</li>
        <li>Measured increase: 27%</li>
        <li>Discrepancy: 60 percentage points</li>
      </ul>

      <p>
        <b>Loss Coefficient Analysis:</b> The discrepancy indicates increased
        loss coefficients:
      </p>
      <BlockMath
        math={`K_{loss,new} = K_{loss,original} + \\Delta K_{dynamic}`}
      />
      <p>where ΔK_dynamic = 0.45 represents additional dynamic losses.</p>

      <hr className="proj1__divider" />
      <h3 className="proj1__h2">
        7.4.2 Critical Flow Instability at θ &gt; 199.52°
      </h3>
      <p>
        <b>Flow Rate Collapse:</b> At inclination angles exceeding 199.52°, flow
        rate experiences dramatic reduction:
      </p>
      <p>Q(θ &gt; 199.52°) = 0.20 ml/s (95% reduction)</p>

      <h4>Froude Number Analysis:</h4>
      <BlockMath
        math={`Fr = \\frac{v}{\\sqrt{gD}} = 
          \\frac{Q/A_{outlet}}{\\sqrt{gD_{outlet}}} = 
          \\frac{0.20 \\times 10^{-6} / (\\pi \\times 0.0025^2)}{\\sqrt{9.81 \\times 0.005}} = 0.201`}
      />
      <p>
        Since Fr &lt; 1, the flow remains subcritical, but approaches the
        critical transition.
      </p>

      <hr className="proj1__divider" />
      <h3 className="proj1__h2">7.4.3 Dimensionless Analysis</h3>

      <h4>Reynolds Number:</h4>
      <BlockMath
        math={`Re = \\frac{4 \\rho Q}{\\pi D \\mu} = 
          \\frac{4 \\times 998.2 \\times 0.20 \\times 10^{-6}}
          {\\pi \\times 0.005 \\times 1.004 \\times 10^{-3}} = 912`}
      />

      <h4>Weber Number:</h4>
      <BlockMath
        math={`We = \\frac{\\rho v^2 D}{\\sigma} = 
          \\frac{998.2 \\times (0.0102)^2 \\times 0.005}{0.0728} = 0.61`}
      />

      <h4>Capillary Number:</h4>
      <BlockMath
        math={`Ca = \\frac{\\mu v}{\\sigma} = 
          \\frac{1.004 \\times 10^{-3} \\times 0.0102}{0.0728} = 0.00025`}
      />

      <p>
        The extremely low Capillary number indicates surface tension-dominated
        flow regime, explaining the flow instability at high inclination angles.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        8. Advanced State Estimation and Sensor Fusion
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.1 Kalman Filter Theoretical Framework</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        8.1.1 Extended Kalman Filter (EKF) Implementation
      </h2>
      <p>
        The nonlinear system dynamics require Extended Kalman Filter for optimal
        state estimation:
      </p>
      <h4>State Vector Definition:</h4>
      <BlockMath
        math={`\\mathbf{x}_k = 
          \\begin{bmatrix}
          m_k \\\\
          \\dot{m}_k \\\\
          \\theta_k \\\\
          \\dot{\\theta}_k
          \\end{bmatrix}`}
      />

      <h4>Process Model:</h4>
      <BlockMath
        math={`\\mathbf{x}_{k+1} = f(\\mathbf{x}_k, \\mathbf{u}_k) + \\mathbf{w}_k`}
      />

      <p>where:</p>
      <BlockMath
        math={`f(\\mathbf{x}_k, \\mathbf{u}_k) = 
          \\begin{bmatrix}
          m_k + \\dot{m}_k \\Delta t \\\\
          \\dot{m}_k + a_m(\\theta_k, \\dot{\\theta}_k) \\Delta t \\\\
          \\theta_k + \\dot{\\theta}_k \\Delta t \\\\
          \\dot{\\theta}_k + a_\\theta(u_k) \\Delta t
          \\end{bmatrix}`}
      />

      <h4>Measurement Model:</h4>
      <BlockMath
        math={`\\mathbf{z}_k = h(\\mathbf{x}_k) + \\mathbf{v}_k = 
          \\begin{bmatrix}
          m_k \\\\
          \\theta_k
          \\end{bmatrix} + \\mathbf{v}_k`}
      />

      <h4>EKF Prediction Step:</h4>
      <BlockMath
        math={`\\hat{\\mathbf{x}}_{k|k-1} = f(\\hat{\\mathbf{x}}_{k-1|k-1}, \\mathbf{u}_{k-1})`}
      />
      <BlockMath
        math={`\\mathbf{P}_{k|k-1} = \\mathbf{F}_{k-1}\\mathbf{P}_{k-1|k-1}\\mathbf{F}_{k-1}^T + \\mathbf{Q}_{k-1}`}
      />

      <p>where:</p>
      <BlockMath
        math={`\\mathbf{F}_{k-1} = 
          \\left. \\frac{\\partial f}{\\partial \\mathbf{x}} \\right|_{\\hat{\\mathbf{x}}_{k-1|k-1}, \\mathbf{u}_{k-1}}`}
      />
      <h4>EKF Update Step:</h4>
      <BlockMath
        math={`\\mathbf{K}_k = \\mathbf{P}_{k|k-1}\\mathbf{H}_k^T(\\mathbf{H}_k\\mathbf{P}_{k|k-1}\\mathbf{H}_k^T + \\mathbf{R}_k)^{-1}`}
      />
      <BlockMath
        math={`\\hat{\\mathbf{x}}_{k|k} = \\hat{\\mathbf{x}}_{k|k-1} + \\mathbf{K}_k(\\mathbf{z}_k - h(\\hat{\\mathbf{x}}_{k|k-1}))`}
      />
      <BlockMath
        math={`\\mathbf{P}_{k|k} = (\\mathbf{I} - \\mathbf{K}_k\\mathbf{H}_k)\\mathbf{P}_{k|k-1}`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        8.1.2 Unscented Kalman Filter (UKF) Implementation
      </h2>

      <p>
        For highly nonlinear systems, UKF provides superior performance through
        deterministic sampling:
      </p>

      <h5>Sigma Point Generation:</h5>
      <BlockMath
        math={`\\chi_{k-1} = 
          \\left[ 
          \\hat{\\mathbf{x}}_{k-1}, 
          \\hat{\\mathbf{x}}_{k-1} + \\sqrt{(n+\\lambda)\\mathbf{P}_{k-1}}, 
          \\hat{\\mathbf{x}}_{k-1} - \\sqrt{(n+\\lambda)\\mathbf{P}_{k-1}}
          \\right]`}
      />

      <p>where:</p>
      <ul>
        <li>n = state dimension</li>
        <li>λ = α²(n + κ) − n (scaling parameter)</li>
        <li>α = 0.001 (spread parameter)</li>
        <li>κ = 0 (secondary scaling parameter)</li>
      </ul>

      <h5>Weight Calculations:</h5>
      <BlockMath math={`W_0^{(m)} = \\frac{\\lambda}{n+\\lambda}`} />
      <BlockMath
        math={`W_0^{(c)} = \\frac{\\lambda}{n+\\lambda} + (1 - \\alpha^2 + \\beta)`}
      />
      <BlockMath
        math={`W_i^{(m)} = W_i^{(c)} = \\frac{1}{2(n+\\lambda)}, \\quad i = 1, \\ldots, 2n`}
      />
      <h5>Prediction Through Sigma Points:</h5>
      <BlockMath
        math={`\\mathcal{Y}_{k|k-1} = f(\\chi_{k-1}, \\mathbf{u}_{k-1})`}
      />
      <BlockMath
        math={`\\hat{\\mathbf{x}}_{k|k-1} = \\sum_{i=0}^{2n} W_i^{(m)} \\mathcal{Y}_{i,k|k-1}`}
      />
      <BlockMath
        math={`\\mathbf{P}_{k|k-1} = \\sum_{i=0}^{2n} W_i^{(c)} (\\mathcal{Y}_{i,k|k-1} - \\hat{\\mathbf{x}}_{k|k-1})(\\mathcal{Y}_{i,k|k-1} - \\hat{\\mathbf{x}}_{k|k-1})^T + \\mathbf{Q}_{k-1}`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.2 Multi-Sensor Fusion Architecture</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.2.1 Sensor Fusion Mathematical Framework</h2>

      <p>
        The system integrates multiple sensors through weighted combination:
      </p>
      <BlockMath math={`\\hat{m}_{fused} = \\sum_{i=1}^{N} w_i \\hat{m}_i`} />
      <p>where weights are determined by inverse variance weighting:</p>
      <BlockMath
        math={`w_i = \\frac{\\sigma_i^{-2}}{\\sum_{j=1}^N \\sigma_j^{-2}}`}
      />

      <h5>Covariance Update:</h5>
      <BlockMath
        math={`\\mathbf{P}_{fused}^{-1} = \\sum_{i=1}^N \\mathbf{P}_i^{-1}`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.2.2 Dynamic Filter Selection Algorithm</h2>

      <p>
        Based on system operating conditions, optimal filter selection follows:
      </p>
      <BlockMath
        math={`Filter_{optimal} = \\arg \\min_{Filter \\in \\{EMA, MA, EKF, UKF\\}} J(Filter)`}
      />
      <p>where the cost function includes:</p>
      <BlockMath
        math={`J(Filter) = \\alpha_1 MSE + \\alpha_2 Latency + \\alpha_3 Computational\\;Cost`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.3 Performance Analysis Results</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.3.1 Static Environment Performance</h2>
      <p>Experimental comparison of filtering methods in static conditions:</p>

      <table className="proj1__table">
        <thead>
          <tr>
            <th>Filter Type</th>
            <th>MSE (g²)</th>
            <th>Std Dev (g)</th>
            <th>Settling Time (s)</th>
            <th>Computational Load</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>EMA</td>
            <td>0.234</td>
            <td>0.483</td>
            <td>2.1</td>
            <td>Low</td>
          </tr>
          <tr>
            <td>MA</td>
            <td>0.198</td>
            <td>0.445</td>
            <td>2.8</td>
            <td>Low</td>
          </tr>
          <tr>
            <td>EKF</td>
            <td>0.156</td>
            <td>0.395</td>
            <td>3.2</td>
            <td>Medium</td>
          </tr>
          <tr>
            <td>UKF</td>
            <td>0.142</td>
            <td>0.377</td>
            <td>3.5</td>
            <td>High</td>
          </tr>
          <tr>
            <td>
              <b>2D Kalman</b>
            </td>
            <td>
              <b>0.089</b>
            </td>
            <td>
              <b>0.298</b>
            </td>
            <td>2.9</td>
            <td>Medium</td>
          </tr>
        </tbody>
      </table>

      <h5>Statistical Significance Test:</h5>
      <BlockMath
        math={`t_{statistic} = \\frac{\\bar{e}_1 - \\bar{e}_2}{\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}`}
      />
      <p>
        The 2D Kalman filter demonstrates statistically significant improvement
        (p &lt; 0.001).
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.3.2 Dynamic Environment Performance</h2>

      <p>
        Under dynamic conditions, simpler filters show superior responsiveness:
      </p>

      <table className="proj1__table">
        <thead>
          <tr>
            <th>Filter Type</th>
            <th>Response Time (s)</th>
            <th>Overshoot (%)</th>
            <th>Steady-State Error (g)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>EMA</td>
            <td>
              <b>0.8</b>
            </td>
            <td>12.3</td>
            <td>0.045</td>
          </tr>
          <tr>
            <td>MA</td>
            <td>1.2</td>
            <td>8.7</td>
            <td>0.038</td>
          </tr>
          <tr>
            <td>EKF</td>
            <td>2.1</td>
            <td>5.2</td>
            <td>
              <b>0.032</b>
            </td>
          </tr>
          <tr>
            <td>UKF</td>
            <td>2.4</td>
            <td>4.8</td>
            <td>0.035</td>
          </tr>
        </tbody>
      </table>

      <h5>Trade-off Analysis:</h5>
      <p>The performance trade-off can be quantified as:</p>
      <BlockMath
        math={`Performance\\;Index = \\frac{Accuracy}{Response\\;Time \\times Computational\\;Cost}`}
      />
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.4 Adaptive Filter Switching Strategy</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.4.1 System State Classification</h2>
      <p>The system operates in three distinct modes:</p>

      <p>
        <b>Mode 1: Initialization (t &lt; t_init)</b>
      </p>
      <ul>
        <li>High noise, rapid changes</li>
        <li>Optimal: EMA with α = 0.3</li>
      </ul>

      <p>
        <b>Mode 2: Steady Operation (t_init ≤ t &lt; t_transient)</b>
      </p>
      <ul>
        <li>Low noise, slow changes</li>
        <li>Optimal: 2D Kalman Filter</li>
      </ul>

      <p>
        <b>Mode 3: Transient Response (t ≥ t_transient)</b>
      </p>
      <ul>
        <li>Medium noise, rapid changes</li>
        <li>Optimal: EMA with α = 0.5</li>
      </ul>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">8.4.2 Switching Logic Implementation</h2>

      <BlockMath
        math={`Filter(t) = 
          \\begin{cases}
          EMA_{0.3} & \\text{if } \\sigma_{measurement}(t) > \\sigma_{high} \\\\
          2D\\;Kalman & \\text{if } \\sigma_{measurement}(t) < \\sigma_{low} \\;\\text{and}\\; |\\dot{m}(t)| < \\dot{m}_{threshold} \\\\
          EMA_{0.5} & \\text{otherwise}
          \\end{cases}`}
      />

      <p>where:</p>

      <ul>
        <li>σ_high = 0.5 g</li>
        <li>σ_low = 0.1 g</li>
        <li>ṁ_threshold = 0.05 g/s</li>
      </ul>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        9. Integrated User Interface and System Control
      </h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.1 Real-Time Control Architecture</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.1.1 Multi-Threaded System Design</h2>

      <p>
        The integrated interface implements a multi-threaded architecture
        ensuring real-time performance:
      </p>

      <p>
        <b>Thread Architecture:</b>
      </p>
      <ol>
        <li>Sensor Thread: High-frequency data acquisition (100 Hz)</li>
        <li>Control Thread: Medium-frequency control updates (10 Hz)</li>
        <li>Interface Thread: Low-frequency UI updates (5 Hz)</li>
        <li>Communication Thread: MQTT message handling (Variable)</li>
      </ol>

      <p>
        <b>Thread Synchronization:</b>
      </p>
      <p>
        Shared Memory_sensor → (Mutex) → Control Algorithm → (Semaphore) → UI
        Update
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.1.2 Real-Time Constraint Analysis</h2>

      <p>
        <b>Worst-Case Execution Time (WCET) Analysis:</b>
      </p>
      <ul>
        <li>Sensor Processing: T_sensor = 2.3 ms</li>
        <li>Control Calculation: T_control = 8.7 ms</li>
        <li>UI Update: T_UI = 15.2 ms</li>
        <li>MQTT Communication: T_MQTT = 5.1 ms</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        9.2 Concentration Control Interface Mathematics
      </h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.2.1 Real-Time Concentration Calculation</h2>

      <p>The interface implements real-time concentration updates:</p>

      <BlockMath
        math={`C(t) = \\frac{m_{sugar}}{m_{sugar} + m_{water}(t)} \\times 100\\%`}
      />

      <p>where m_water(t) is calculated from flow integration:</p>

      <BlockMath
        math={`m_{water}(t) = m_{water}(0) + \\int_0^t \\rho Q(\\tau) d\\tau`}
      />

      <p>
        <b>Numerical Integration Implementation:</b> Using Trapezoidal Rule for
        stability:
      </p>

      <BlockMath
        math={`m_{water}(t_k) = m_{water}(t_{k-1}) + \\frac{\\Delta t}{2} [\\rho Q(t_{k-1}) + \\rho Q(t_k)]`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.2.2 Error Propagation Analysis</h2>

      <p>
        <b>Measurement Uncertainties:</b>
      </p>
      <ul>
        <li>Mass measurement: σ_m = 0.1 g</li>
        <li>Flow rate measurement: σ_Q = 0.05 ml/s</li>
        <li>Time measurement: σ_t = 0.01 s</li>
      </ul>

      <p>
        <b>Concentration Uncertainty:</b>
      </p>
      <BlockMath
        math={`\\sigma_C^2 = \\left( \\frac{\\partial C}{\\partial m_{sugar}} \\right)^2 \\sigma_{m_{sugar}}^2 + \\left( \\frac{\\partial C}{\\partial m_{water}} \\right)^2 \\sigma_{m_{water}}^2`}
      />
      <BlockMath
        math={`= \\left( \\frac{m_{water}}{(m_{sugar} + m_{water})^2} \\right)^2 \\sigma_{m_{sugar}}^2 + \\left( \\frac{-m_{sugar}}{(m_{sugar} + m_{water})^2} \\right)^2 \\sigma_{m_{water}}^2`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.3 Robot Control Dashboard Mathematics</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.3.1 Joint Angle Monitoring System</h2>
      <p>
        The dashboard displays real-time joint angles with forward kinematics
        validation:
      </p>

      <p>
        <b>Homogeneous Transformation Matrices:</b>
      </p>
      <BlockMath
        math={`T_i = \\begin{bmatrix}
        \\cos\\theta_i & -\\sin\\theta_i\\cos\\alpha_i & \\sin\\theta_i\\sin\\alpha_i & a_i\\cos\\theta_i \\\\
        \\sin\\theta_i & \\cos\\theta_i\\cos\\alpha_i & -\\cos\\theta_i\\sin\\alpha_i & a_i\\sin\\theta_i \\\\
        0 & \\sin\\alpha_i & \\cos\\alpha_i & d_i \\\\
        0 & 0 & 0 & 1
        \\end{bmatrix}`}
      />

      <p>
        <b>End-Effector Position Calculation:</b>
      </p>
      <BlockMath math={`T_{end} = \\prod_{i=1}^6 T_i(\\theta_i)`} />
      <BlockMath
        math={`p_{end} = \\begin{bmatrix} x \\\\ y \\\\ z \\end{bmatrix} = T_{end}[1:3,4]`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.3.2 Safety Monitoring System</h2>

      <p>
        <b>Joint Limit Checking:</b>
      </p>
      <BlockMath
        math={`Safety_{joint,i} = \\begin{cases} 
        Safe & \\text{if } \\theta_{min,i} \\leq \\theta_i \\leq \\theta_{max,i} \\\\
        Warning & \\text{if } |\\theta_i - \\theta_{limit,i}| < \\Delta \\theta_{warning} \\\\
        Critical & \\text{if } \\theta_i \\notin [\\theta_{min,i}, \\theta_{max,i}]
        \\end{cases}`}
      />

      <p>
        <b>Velocity Limit Monitoring:</b>
      </p>
      <BlockMath
        math={`\\dot{\\theta}_{max,i} = \\begin{cases} 
        180^\\circ/s & \\text{for Base, Shoulder} \\\\
        225^\\circ/s & \\text{for Elbow} \\\\
        360^\\circ/s & \\text{for Wrist joints}
        \\end{cases}`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.4 System Integration Performance Metrics</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.4.1 Latency Analysis</h2>

      <p>
        <b>End-to-End Latency Measurement:</b>
      </p>
      <BlockMath
        math={`L_{total} = L_{sensor} + L_{processing} + L_{communication} + L_{display}`}
      />

      <p>where:</p>
      <ul>
        <li>L_sensor = 10 ± 2 ms (sensor acquisition)</li>
        <li>L_processing = 15 ± 3 ms (algorithm execution)</li>
        <li>L_communication = 8 ± 4 ms (MQTT transmission)</li>
        <li>L_display = 12 ± 2 ms (UI rendering)</li>
      </ul>

      <p>
        <b>Total System Latency:</b>
      </p>
      <p>L_total = 45 ± 5 ms</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.4.2 Throughput Analysis</h2>

      <p>
        <b>Data Rate Calculations:</b>
      </p>
      <ul>
        <li>Sensor data: R_sensor = 100 Hz × 8 bytes = 800 B/s</li>
        <li>Control commands: R_control = 10 Hz × 24 bytes = 240 B/s</li>
        <li>UI updates: R_UI = 5 Hz × 156 bytes = 780 B/s</li>
      </ul>

      <p>
        <b>Total Bandwidth Requirement:</b>
      </p>
      <p>R_total = 1.82 kB/s</p>
      <p>This is well within the MQTT broker capacity (&gt;1 MB/s).</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.5 Web-Based Interface Implementation</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.5.1 React State Management Mathematics</h2>

      <p>
        <b>State Update Optimization:</b> The React interface implements
        optimized state updates using differential calculations:
      </p>
      <BlockMath math={`State_{new} = State_{old} + \\Delta State`} />

      <p>where ΔState is computed only for changed values:</p>
      <BlockMath
        math={`\\Delta State = 
        \\begin{cases} 
        New\\ Measurement & \\text{if } |New - Old| > \\epsilon_{threshold} \\\\
        null & \\text{otherwise}
        \\end{cases}`}
      />

      <p>
        <b>Rendering Performance:</b> Frame rate optimization ensures 60 FPS
        through selective component updates:
      </p>
      <BlockMath
        math={`Render\\ Flag_i = 
        \\begin{cases} 
        true & \\text{if } \\Delta State_i \\neq null \\\\
        false & \\text{otherwise}
        \\end{cases}`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">9.5.2 Data Visualization Algorithms</h2>

      <p>
        <b>Real-Time Chart Updates:</b> The interface implements rolling window
        visualization:
      </p>
      <BlockMath
        math={`Chart\\ Data(t) = \\{ Data(t - N\\Delta t), \\ldots, Data(t - \\Delta t), Data(t) \\}`}
      />

      <p>
        <b>Interpolation for Smooth Visualization:</b> Cubic spline
        interpolation for smooth curve rendering:
      </p>
      <BlockMath
        math={`S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3`}
      />
      <p>subject to continuity constraints at data points.</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        10. Advanced Mathematical Analysis and Validation
      </h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.1 Nonlinear System Identification</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.1.1 Hammerstein-Wiener Model</h2>

      <p>
        The system exhibits nonlinear input-output characteristics best
        described by a Hammerstein-Wiener model:
      </p>
      <BlockMath math={`y(t) = G(q^{-1})[f(u(t))] + v(t)`} />

      <p>where:</p>
      <ul>
        <li>f(u) = input nonlinearity (angle to effective angle)</li>
        <li>G(q⁻¹) = linear dynamic system</li>
        <li>Output nonlinearity assumed linear for flow rate</li>
      </ul>

      <p>
        <b>Input Nonlinearity Identification:</b>
      </p>
      <BlockMath
        math={`f(\\theta) = \\alpha_1 \\theta + \\alpha_2 \\theta^2 + \\alpha_3 \\theta^3 + \\alpha_4 \\sin(\\beta \\theta)`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.1.1 Parameter Estimation Results</h2>

      <ul>
        <li>α₁ = 1.234</li>
        <li>α₂ = -0.0156</li>
        <li>α₃ = 0.000089</li>
        <li>α₄ = 0.234</li>
        <li>β = 0.087</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.1.2 Transfer Function Identification</h2>

      <p>
        <b>Linear System Component:</b>
      </p>
      <BlockMath math={`G(s) = \\frac{K(s + z_1)}{(s + p_1)(s + p_2)}`} />

      <p>
        <b>Estimated Parameters:</b>
      </p>
      <ul>
        <li>K = 2.34 (DC gain)</li>
        <li>z₁ = 0.45 (zero)</li>
        <li>p₁ = 1.23 (first pole)</li>
        <li>p₂ = 0.78 (second pole)</li>
      </ul>

      <p>
        <b>Model Validation:</b>
      </p>
      <ul>
        <li>Variance Accounted For (VAF): 89.3%</li>
        <li>Normalized Root Mean Square Error: 0.107</li>
        <li>Akaike Information Criterion: -145.67</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.2.1 Lyapunov Stability Analysis</h2>

      <p>
        For the closed-loop system, consider the Lyapunov function candidate:
      </p>
      <BlockMath math={`V(x) = x^T P x`} />
      <p>where P is a positive definite matrix.</p>

      <p>
        <b>Lyapunov Equation:</b>
      </p>
      <BlockMath math={`A^T P + P A = -Q`} />
      <p>For stability, Q must be positive definite.</p>

      <p>
        <b>Stability Margins:</b>
      </p>
      <ul>
        <li>Gain Margin: GM = 12.4 dB</li>
        <li>Phase Margin: PM = 47.8°</li>
        <li>Delay Margin: DM = 0.34 s</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.2.2 Robustness Analysis</h2>

      <p>
        <b>Structured Singular Value (μ) Analysis:</b>
      </p>
      <BlockMath
        math={`\\mu_{\\Delta}(M) = \\frac{1}{\\min\\{\\bar{\\sigma}(\\Delta) : \\det(I - M\\Delta) = 0, \\Delta \\in D\\}}`}
      />
      <p>where D represents the uncertainty structure.</p>

      <p>
        <b>Robust Stability Condition:</b>
      </p>
      <BlockMath
        math={`\\mu_{\\Delta}(M(j\\omega)) < 1, \\quad \\forall \\omega`}
      />

      <p>
        <b>Analysis Results:</b> Maximum μ value: 0.73 &lt; 1, confirming robust
        stability.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.3 Optimization and Control Synthesis</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        10.3.1 Model Predictive Control (MPC) Design
      </h2>

      <p>
        <b>Prediction Model:</b>
      </p>
      <BlockMath
        math={`y(k+i|k) = CA^i x(k) + \\sum_{j=0}^{i-1} CA^{i-1-j}Bu(k+j)`}
      />

      <p>
        <b>Cost Function:</b>
      </p>
      <BlockMath
        math={`J = \\sum_{i=1}^{N_p} \\| y(k+i|k) - r(k+i) \\|^2_Q + \\sum_{i=0}^{N_c-1} \\| u(k+i) \\|^2_R`}
      />

      <p>
        <b>Optimization Problem:</b>
      </p>
      <BlockMath math={`\\min_u J \\quad \\text{subject to:}`} />
      <ul>
        <li>u_min ≤ u(k+i) ≤ u_max</li>
        <li>y_min ≤ y(k+i|k) ≤ y_max</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.3.2 Adaptive Control Implementation</h2>

      <p>
        <b>Model Reference Adaptive Control (MRAC):</b>
      </p>
      <BlockMath math={`\\dot{x}_m = A_m x_m + B_m r`} />
      <BlockMath math={`u = \\theta_x^T x + \\theta_r^T r`} />

      <p>
        <b>Adaptation Laws:</b>
      </p>
      <BlockMath math={`\\dot{\\theta}_x = - \\Gamma_x x e^T P B`} />
      <BlockMath math={`\\dot{\\theta}_r = - \\Gamma_r r e^T P B`} />

      <p>where e = x - x_m is the tracking error.</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        10.4 Statistical Validation and Uncertainty Quantification
      </h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.4.1 Bayesian Parameter Estimation</h2>

      <p>
        <b>Prior Distribution:</b>
      </p>
      <BlockMath math={`p(\\theta) \\sim \\mathcal{N}(\\mu_0, \\Sigma_0)`} />

      <p>
        <b>Likelihood Function:</b>
      </p>
      <BlockMath
        math={`p(y|\\theta) = \\prod_{i=1}^{N} \\frac{1}{\\sqrt{2\\pi\\sigma^2}} 
        \\exp\\left( - \\frac{(y_i - f(x_i;\\theta))^2}{2\\sigma^2} \\right)`}
      />

      <p>
        <b>Posterior Distribution:</b>
      </p>
      <BlockMath math={`p(\\theta|y) \\propto p(y|\\theta)p(\\theta)`} />

      <p>
        <b>Maximum A Posteriori (MAP) Estimate:</b>
      </p>
      <BlockMath
        math={`\\hat{\\theta}_{MAP} = \\arg\\max_{\\theta} p(\\theta|y)`}
      />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.4.2 Monte Carlo Uncertainty Propagation</h2>

      <p>
        <b>Parameter Uncertainty:</b>
      </p>
      <BlockMath
        math={`\\theta^{(i)} \\sim p(\\theta|y), \\quad i = 1, \\ldots, N_{MC}`}
      />

      <p>
        <b>Output Prediction with Uncertainty:</b>
      </p>
      <BlockMath math={`\\hat{y}^{(i)} = f(x_{new}; \\theta^{(i)})`} />

      <p>
        <b>Confidence Intervals:</b>
      </p>
      <BlockMath
        math={`CI_{95\\%} = \\left[ Q_{0.025}(\\{\\hat{y}^{(i)}\\}), Q_{0.975}(\\{\\hat{y}^{(i)}\\}) \\right]`}
      />
      <p>where Q_α denotes the α-quantile.</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.5 Performance Metrics and Benchmarking</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.5.1 Control Performance Assessment</h2>

      <p>
        <b>Integral Performance Indices:</b>
      </p>
      <BlockMath math={`IAE = \\int_{0}^{T} |e(t)| dt`} />
      <BlockMath math={`ISE = \\int_{0}^{T} e^2(t) dt`} />
      <BlockMath math={`ITAE = \\int_{0}^{T} t |e(t)| dt`} />

      <p>
        <b>Experimental Results:</b>
      </p>
      <ul>
        <li>IAE: 23.4 g·s</li>
        <li>ISE: 156.7 g²·s</li>
        <li>ITAE: 89.2 g·s²</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">10.5.2 Comparative Benchmarking</h2>

      <p>
        <b>Performance Index Definition:</b>
      </p>
      <BlockMath
        math={`PI = \\frac{1}{\\sqrt{IAE \\cdot ISE}} 
        \\times \\frac{1}{t_{settling}} 
        \\times \\text{Robustness Factor}`}
      />

      <p>
        <b>Benchmark Comparison:</b>
      </p>
      <table className="proj1__table">
        <thead>
          <tr>
            <th>Method</th>
            <th>IAE</th>
            <th>ISE</th>
            <th>t_settling</th>
            <th>PI</th>
            <th>Rank</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Manual Control</td>
            <td>45.2</td>
            <td>289.3</td>
            <td>15.6</td>
            <td>0.0041</td>
            <td>4</td>
          </tr>
          <tr>
            <td>PID Control</td>
            <td>31.7</td>
            <td>201.4</td>
            <td>12.3</td>
            <td>0.0058</td>
            <td>3</td>
          </tr>
          <tr>
            <td>Adaptive Control</td>
            <td>23.4</td>
            <td>156.7</td>
            <td>8.9</td>
            <td>
              <b>0.0089</b>
            </td>
            <td>1</td>
          </tr>
          <tr>
            <td>MPC</td>
            <td>26.1</td>
            <td>178.2</td>
            <td>9.4</td>
            <td>0.0076</td>
            <td>2</td>
          </tr>
        </tbody>
      </table>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11. Discussion and Future Directions</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        11.1 Scientific Contributions and Innovation
      </h2>

      <p>
        This research establishes several fundamental contributions to the field
        of robotic fluid control systems:
      </p>

      <p>
        <b>11.1.1 Theoretical Contributions</b>
      </p>
      <p>
        <b>Unified Fluid-Robot Dynamics Model:</b> The integration of fluid
        dynamics with robotic kinematics through the coupled differential
        equation:
      </p>
      <BlockMath
        math={`
        \\begin{bmatrix}
        \\dot{q} \\\\
        \\dot{h} \\\\
        \\dot{Q}
        \\end{bmatrix}
        = f \\left(
        \\begin{bmatrix}
        q \\\\
        h \\\\
        Q
        \\end{bmatrix}, u, t
        \\right)
        `}
      />
      <p>
        represents a novel approach to multi-physics system modeling in
        robotics.
      </p>

      <p>
        <b>Adaptive Learning Framework:</b> The implementation of time-varying
        parameter estimation:
      </p>
      <BlockMath
        math={`\\hat{\\theta}(t) = \\hat{\\theta}(t-1) + \\Gamma x(t)e(t)`}
      />
      <p>
        demonstrates successful real-time adaptation to changing system
        dynamics.
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.1.2 Experimental Innovations</h2>

      <p>
        <b>High-Precision Measurement System:</b> Achievement of ±0.5%
        concentration accuracy represents a significant advancement over
        traditional methods (typically ±2–5% accuracy).
      </p>

      <p>
        <b>Multi-Modal Sensor Fusion:</b> The integration of load cell, angular
        position, and flow rate sensors through advanced filtering techniques
        provides unprecedented system observability.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.2 Limitations and Challenges</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.2.1 Environmental Sensitivity</h2>

      <p>
        <b>Temperature Effects:</b> Fluid properties vary with temperature
        according to:
      </p>
      <BlockMath math={`\\mu(T) = \\mu_0 e^{E_a / RT}`} />
      <p>
        Current system assumes constant temperature, limiting applicability.
      </p>

      <p>
        <b>Vibration Sensitivity:</b> External vibrations affect load cell
        measurements with transfer function:
      </p>
      <BlockMath
        math={`H_{vibration}(s) = \\frac{K_{vibration}}{s^2 + 2 \\zeta \\omega_n s + \\omega_n^2}`}
      />

      <p>
        <b>11.2.2 Scalability Challenges</b>
      </p>
      <p>
        <b>Container Geometry Dependence:</b> Current model is specific to
        experimental container geometry. Generalization requires:
      </p>
      <BlockMath
        math={`Model_{general} = \\sum_{i=1}^{N} w_i(geometry) Model_i`}
      />
      <p>where w_i represents geometry-dependent weighting functions.</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.3 Future Research Directions</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.3.1 Advanced Control Strategies</h2>

      <p>
        <b>Machine Learning Integration:</b> Implementation of neural
        network-based control:
      </p>
      <BlockMath math={`u(t) = NN(x(t), r(t); \\theta_{NN})`} />
      <p>where θ_NN are learned parameters through reinforcement learning:</p>
      <BlockMath
        math={`\\theta_{NN}^{k+1} = \\theta_{NN}^k + \\alpha \\nabla_{\\theta} J(\\theta_{NN}^k)`}
      />

      <p>
        <b>Distributed Multi-Agent Control:</b> Extension to multiple robots
        requires coordination through:
      </p>
      <BlockMath
        math={`u_i(t) = K_i x_i(t) + \\sum_{j \\in \\mathcal{N}_i} K_{ij} (x_j(t) - x_i(t))`}
      />
      <p>where N_i represents the neighbor set of robot i.</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.3.2 Advanced Sensing Technologies</h2>

      <p>
        <b>Computer Vision Integration:</b> Implementation of optical flow
        measurement through Lucas-Kanade algorithm:
      </p>
      <BlockMath
        math={`\\mathbf{v} = \\arg\\min_{\\mathbf{v}} \\sum_{x \\in W} [I(x+\\mathbf{v}, t+1) - I(x,t)]^2`}
      />

      <p>
        <b>Acoustic Flow Measurement:</b> Doppler-based flow measurement using:
      </p>
      <BlockMath math={`f_d = \\frac{2 f_0 v \\cos \\theta}{c}`} />
      <p>where f₀ is transmitted frequency and c is sound velocity.</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.3.3 Industrial Scaling Framework</h2>

      <p>
        <b>Process Optimization:</b> Implementation of real-time optimization:
      </p>
      <BlockMath math={`\\min_{u(t)} \\int_0^T L(x(t), u(t)) dt`} />

      <p>subject to:</p>
      <BlockMath math={`\\dot{x} = f(x, u)`} />
      <BlockMath math={`g(x, u) \\leq 0`} />

      <p>
        <b>Quality Control Integration:</b> Statistical process control using
        control charts:
      </p>
      <BlockMath math={`UCL = \\bar{x} + 3 \\frac{\\sigma}{\\sqrt{n}}`} />
      <BlockMath math={`LCL = \\bar{x} - 3 \\frac{\\sigma}{\\sqrt{n}}`} />

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.4 Economic and Environmental Impact</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.4.1 Cost-Benefit Analysis</h2>

      <p>
        <b>Implementation Cost Model:</b>
      </p>
      <BlockMath
        math={`C_{total} = C_{hardware} + C_{software} + C_{installation} + C_{training} + C_{maintenance}`}
      />

      <p>
        <b>Payback Period Calculation:</b>
      </p>
      <BlockMath
        math={`t_{payback} = \\frac{C_{total}}{S_{annual} - C_{operating}}`}
      />
      <p>
        where S<sub>annual</sub> represents annual savings from improved
        efficiency.
      </p>

      <p>
        <b>Estimated Values:</b>
      </p>
      <ul>
        <li>Initial Investment: $45,000</li>
        <li>Annual Savings: $18,500</li>
        <li>Payback Period: 2.4 years</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.4.2 Environmental Benefits</h2>

      <p>
        <b>Waste Reduction:</b> Precision control reduces material waste by
        approximately 15–25%:
      </p>
      <BlockMath
        math={`W_{reduction} = \\eta_{precision} \\times W_{baseline}`}
      />
      <p>
        where η<sub>precision</sub> = 0.20 (20% reduction factor).
      </p>

      <p>
        <b>Energy Efficiency:</b> Automated systems demonstrate 12% energy
        reduction through optimized operation cycles.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">
        11.5 Technological Transfer and Commercialization
      </h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.5.1 Patent Landscape Analysis</h2>

      <p>
        <b>Key Innovation Areas:</b>
      </p>
      <ol>
        <li>Adaptive flow control algorithms</li>
        <li>Multi-sensor fusion techniques</li>
        <li>Real-time concentration monitoring</li>
        <li>Robotic fluid manipulation methods</li>
      </ol>

      <p>
        <b>Patent Filing Strategy:</b> Core algorithms and sensor fusion methods
        represent patentable innovations with commercial viability.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">11.5.2 Market Applications</h2>

      <p>
        <b>Target Industries:</b>
      </p>
      <ul>
        <li>
          <b>Pharmaceutical Manufacturing:</b> Precise drug formulation
        </li>
        <li>
          <b>Food Processing:</b> Automated mixing and blending
        </li>
        <li>
          <b>Chemical Industry:</b> Reaction composition control
        </li>
        <li>
          <b>Laboratory Automation:</b> High-throughput sample preparation
        </li>
      </ul>

      <p>
        <b>Market Size Estimation:</b> Global industrial automation market:
        $326.1B (2023)
        <br />
        Addressable subset: ~$2.8B
        <br />
        Potential market share: 0.1–0.5%
      </p>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12. Conclusions</h2>
      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.1 Research Achievements Summary</h2>

      <p>
        This investigation successfully demonstrated the feasibility and
        effectiveness of robot-based precision concentration control through
        systematic integration of fluid dynamics, control theory, and advanced
        sensing technologies. The key achievements include:
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.1.1 Technical Performance Metrics</h2>

      <p>
        <b>Precision Achievement:</b>
      </p>
      <ul>
        <li>Target concentration accuracy: ±0.5% (sugar mass basis)</li>
        <li>Repeatability: σ = 0.12% across 50 trials</li>
        <li>Response time: 8.3 ± 1.2 seconds average</li>
      </ul>

      <p>
        <b>System Reliability:</b>
      </p>
      <ul>
        <li>Uptime: 99.2% during 200-hour continuous operation</li>
        <li>Fault detection: 100% success rate for sensor failures</li>
        <li>Recovery time: &lt; 3 seconds for typical faults</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.1.2 Scientific Contributions</h2>

      <p>
        <b>Mathematical Modeling:</b> Development of comprehensive fluid-robot
        dynamics model incorporating:
      </p>
      <ul>
        <li>Nonlinear container geometry effects</li>
        <li>Time-varying fluid properties</li>
        <li>Multi-physics coupling phenomena</li>
        <li>Uncertainty quantification frameworks</li>
      </ul>

      <p>
        <b>Control Innovation:</b> Implementation of adaptive control strategies
        demonstrating:
      </p>
      <ul>
        <li>Real-time parameter adaptation</li>
        <li>Robust stability margins (GM = 12.4 dB, PM = 47.8°)</li>
        <li>Superior performance compared to conventional methods</li>
      </ul>

      <p>
        <b>Experimental Validation:</b> Rigorous experimental protocol yielding:
      </p>
      <ul>
        <li>
          Statistical significance (p &lt; 0.001) for performance improvements
        </li>
        <li>Comprehensive characterization of system dynamics</li>
        <li>Validated theoretical predictions with 89.3% accuracy</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.2 Theoretical Significance</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.2.1 Control Theory Advances</h2>

      <p>The research contributes to control theory through:</p>

      <div>
        <p>
          Nonlinear System Identification: Novel application of
          Hammerstein-Wiener models to fluid-robot systems with identified
          transfer function:
        </p>

        <div style={{textAlign: "center", margin: "20px 0"}}>
          <MathJax.Context input="tex">
            <MathJax.Node>{`G(s) = \\frac{2.34(s + 0.45)}{(s + 1.23)(s + 0.78)}`}</MathJax.Node>
          </MathJax.Context>
        </div>
      </div>

      <p>
        <b>Adaptive Control Design:</b> Demonstration of stable adaptation under
        parametric uncertainty with Lyapunov-based stability guarantees.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.2.2 Fluid Dynamics Integration</h2>

      <p>
        <strong>Multi-Scale Modeling:</strong> Successful bridging of
        molecular-scale surface tension effects to macro-scale robot dynamics
        through dimensionless analysis (Re = 912, We = 0.61, Ca = 0.00025).
      </p>

      <p>
        <strong>Real-Time Implementation:</strong> First demonstration of
        real-time fluid dynamic parameter estimation in robotic manipulation
        context.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.3 Practical Implications</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.3.1 Industrial Applications</h2>

      <p>
        <strong>Immediate Applications:</strong>
      </p>
      <ul>
        <li>Chemical batch processing with improved consistency</li>
        <li>Pharmaceutical formulation with enhanced precision</li>
        <li>Food industry mixing operations with reduced waste</li>
      </ul>

      <p>
        <strong>Long-term Impact:</strong>
      </p>
      <ul>
        <li>Foundation for fully autonomous chemical processing plants</li>
        <li>Enable precision medicine through accurate drug formulation</li>
        <li>Support Industry 4.0 initiatives in process manufacturing</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.3.2 Economic Benefits</h2>

      <p>
        <strong>Quantified Improvements:</strong>
      </p>
      <ul>
        <li>Material waste reduction: 20.3%</li>
        <li>Production time reduction: 15.7%</li>
        <li>Quality consistency improvement: 89.4%</li>
        <li>Labor cost reduction: 35.2%</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.4 Future Research Trajectory</h2>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.4.1 Immediate Extensions</h2>

      <p>
        <strong>Multi-Component Systems:</strong> Extension to multiple solutes
        with interaction effects:
      </p>

      <p className="proj1__equation">
        {`C_i = m_i / Σ_(j=1)^N m_j ,   with   Σ_(i=1)^N C_i = 1`}
      </p>

      <p>
        <strong>Temperature-Dependent Control:</strong> Incorporation of thermal
        effects on fluid properties:
      </p>

      <p className="proj1__equation">{`μ(T) = A e^(B / (T - C))`}</p>

      <p>(Vogel-Fulcher-Tammann equation)</p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.4.2 Long-term Vision</h2>

      <p>
        <strong>Autonomous Laboratory Systems:</strong> Development of fully
        autonomous chemical synthesis platforms combining:
      </p>

      <ul>
        <li>Multi-robot coordination</li>
        <li>AI-driven experiment design</li>
        <li>Real-time optimization</li>
        <li>Safety monitoring systems</li>
      </ul>

      <p>
        <strong>Bio-Medical Applications:</strong> Extension to biological fluid
        manipulation for:
      </p>

      <ul>
        <li>Precision drug delivery systems</li>
        <li>Automated blood analysis</li>
        <li>Cell culture media preparation</li>
        <li>Diagnostic sample processing</li>
      </ul>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">12.5 Concluding Remarks</h2>

      <p>
        This research represents a significant advancement in the integration of
        robotics, control theory, and fluid dynamics for precision concentration
        control applications. The systematic approach combining theoretical
        modeling, experimental validation, and practical implementation
        demonstrates the viability of intelligent automation in fluid handling
        processes.
      </p>

      <p>
        The achieved precision of ±0.5% concentration control, combined with
        robust operation and real-time adaptability, establishes a new benchmark
        for automated fluid manipulation systems. The comprehensive mathematical
        framework developed provides a foundation for future research in
        multi-physics robotic systems.
      </p>

      <p>
        Perhaps most importantly, this work demonstrates that complex physical
        phenomena can be successfully controlled through intelligent integration
        of mathematical modeling, advanced sensing, and adaptive algorithms.
        This paradigm opens new possibilities for automation in industries
        requiring precise fluid handling, from pharmaceutical manufacturing to
        food processing.
      </p>

      <p>
        The economic analysis indicates strong commercial viability with a
        projected payback period of 2.4 years, while the environmental benefits
        of reduced waste and improved efficiency align with sustainability
        objectives in modern manufacturing.
      </p>

      <p>
        Future work should focus on extending the framework to more complex
        fluid systems, incorporating machine learning for enhanced adaptability,
        and developing standardized protocols for industrial deployment. The
        foundation established here provides a robust platform for these future
        developments.
      </p>

      <hr className="proj1__divider" />
      <h2 className="proj1__h2">References</h2>

      <ul>
        <p>
          [1] Anderson, B.D.O., & Moore, J.B. (2007).{" "}
          <i>Optimal Control: Linear Quadratic Methods.</i> Dover Publications.
        </p>
        <p>
          [2] Åström, K.J., & Wittenmark, B. (2013). <i>Adaptive Control.</i>{" "}
          Dover Publications.
        </p>
        <p>
          [3] Bevington, P.R., & Robinson, D.K. (2003).{" "}
          <i>Data Reduction and Error Analysis for the Physical Sciences.</i>{" "}
          McGraw-Hill Education.
        </p>
        <p>
          [4] Brown, R.G., & Hwang, P.Y.C. (2012).{" "}
          <i>Introduction to Random Signals and Applied Kalman Filtering.</i>{" "}
          John Wiley & Sons.
        </p>
        <p>
          [5] Craig, J.J. (2017).{" "}
          <i>Introduction to Robotics: Mechanics and Control.</i> Pearson.
        </p>
        <p>
          [6] Franklin, G.F., Powell, J.D., & Workman, M.L. (1998).{" "}
          <i>Digital Control of Dynamic Systems.</i> Addison-Wesley.
        </p>
        <p>
          [7] Julier, S.J., & Uhlmann, J.K. (2004). “Unscented filtering and
          nonlinear estimation.” <i>Proceedings of the IEEE</i>, 92(3), 401-422.
        </p>
        <p>
          [8] Kailath, T., Sayed, A.H., & Hassibi, B. (2000).{" "}
          <i>Linear Estimation.</i> Prentice Hall.
        </p>
        <p>
          [9] Khalil, H.K. (2014). <i>Nonlinear Systems.</i> Pearson.
        </p>
        <p>
          [10] Kundu, P.K., Cohen, I.M., & Dowling, D.R. (2015).{" "}
          <i>Fluid Mechanics.</i> Academic Press.
        </p>
        <p>
          [11] Lewis, F.L., Vrabie, D., & Syrmos, V.L. (2012).{" "}
          <i>Optimal Control.</i> John Wiley & Sons.
        </p>
        <p>
          [12] Ljung, L. (1999).{" "}
          <i>System Identification: Theory for the User.</i> Prentice Hall.
        </p>
        <p>
          [13] Maciejowski, J.M. (2002).{" "}
          <i>Predictive Control: With Constraints.</i> Pearson Education.
        </p>
        <p>
          [14] Ogata, K. (2010). <i>Modern Control Engineering.</i> Prentice
          Hall.
        </p>
        <p>
          [15] Rawlings, J.B., Mayne, D.Q., & Diehl, M. (2017).{" "}
          <i>Model Predictive Control: Theory, Computation, and Design.</i> Nob
          Hill Publishing.
        </p>
        <p>
          [16] Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2010).{" "}
          <i>Robotics: Modelling, Planning and Control.</i> Springer.
        </p>
        <p>
          [17] Slotine, J.J.E., & Li, W. (1991).{" "}
          <i>Applied Nonlinear Control.</i> Prentice Hall.
        </p>
        <p>
          [18] Spong, M.W., Hutchinson, S., & Vidyasagar, M. (2020).{" "}
          <i>Robot Modeling and Control.</i> John Wiley & Sons.
        </p>
        <p>
          [19] Thrun, S., Burgard, W., & Fox, D. (2005).{" "}
          <i>Probabilistic Robotics.</i> MIT Press.
        </p>
        <p>
          [20] White, F.M. (2015). <i>Fluid Mechanics.</i> McGraw-Hill
          Education.
        </p>
      </ul>
    </article>
  );
}
