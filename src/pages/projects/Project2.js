import React from "react";
import "./Project2.scss";
import "katex/dist/katex.min.css";
// import {BlockMath} from "react-katex";
// import MathJax from "react-mathjax2";

export default function Project2() {
  return (
    <article className="proj1">
      <header className="proj1__hero">
        {/* <div className="proj1__hero-badge">Project1</div> */}
        <h1 className="proj1__title">
          AI-Driven Collaborative Robot Work Assistant: A Vision-Based Approach Using YOLO Object Detection and Human-Robot Interaction Intelligence{" "}
        </h1>
        <p className="proj1__subtitle">
          “When machines begin to see and understand, collaboration becomes intuition.”
        </p>

      </header>

      <section className="proj1__section">
        <h2 className="proj1__h2">Project Video</h2>

        <div className="proj1__video">
          <iframe
            width="560"
            height="315"
            src="https://www.youtube.com/embed/Fas-aIrPaJc?si=HwX70-jfb9fs8ik9"
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
