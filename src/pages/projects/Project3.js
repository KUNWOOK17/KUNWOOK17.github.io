import React from "react";
import "./Project3.scss";
import "katex/dist/katex.min.css";
// import {BlockMath} from "react-katex";
// import MathJax from "react-mathjax2";

export default function Project3() {
  return (
    <article className="proj1">
      <header className="proj1__hero">
        {/* <div className="proj1__hero-badge">Project1</div> */}
        <h1 className="proj1__title">
          Digital Twin–Enabled Service Robot System: Synchronizing Virtual Simulation and Real-World Operation Using TurtleBot3{" "}
        </h1>
        <p className="proj1__subtitle">
          “When the digital and the physical move as one, intelligence becomes reality.”
        </p>
      </header>

      <section className="proj1__section">
        <h2 className="proj1__h2">Project Video</h2>

        <div className="proj1__video">
          <iframe
            width="560"
            height="315"
            src="https://www.youtube.com/embed/57pRaic92Kg?si=93_UXE_XCY-zQCw1"
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        </div>
      </section>

    </article>
  );
}
