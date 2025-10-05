import React, {useContext} from "react";
import "./StartupProjects.scss";
import {bigProjects} from "../../portfolio";
import {Fade} from "react-reveal";
import StyleContext from "../../contexts/StyleContext";

export default function StartupProject({ changePage }) {
  const { isDark } = useContext(StyleContext);

  if (!bigProjects.display) {
    return null;
  }

  // URL → 내부 페이지 이름 매핑
  const mapUrlToPage = (url = "") => {
    if (url.includes("liquid_injection")) return "project1";
    if (url.includes("YOLO")) return "project2";
    if (url.includes("RViz")) return "project3";
    if (url.includes("turtlebot")) return "project4";
    return "main";
  };

  return (
    <Fade bottom duration={1000} distance="20px">
      <div className="main" id="projects">
        <div>
          <h1 className="skills-heading">{bigProjects.title}</h1>
          <p
            className={
              isDark
                ? "dark-mode project-subtitle"
                : "subTitle project-subtitle"
            }
          >
            {bigProjects.subtitle}
          </p>

          <div className="projects-container">
            {bigProjects.projects.map((project, i) => {
              const footer = project.footerLink?.[0]; // 첫 번째 footerLink 사용
              const targetPage = footer ? mapUrlToPage(footer.url) : "main";

              return (
                <div
                  key={i}
                  className={
                    isDark
                      ? "dark-mode project-card project-card-dark"
                      : "project-card project-card-light"
                  }
                >
                  {project.image && (
                    <div className="project-image">
                      <img
                        src={project.image}
                        alt={project.projectName}
                        className="card-image"
                      />
                    </div>
                  )}

                  <div className="project-detail">
                    <h5
                      className={isDark ? "dark-mode card-title" : "card-title"}
                    >
                      {project.projectName}
                    </h5>
                    <p
                      className={
                        isDark ? "dark-mode card-subtitle" : "card-subtitle"
                      }
                    >
                      {project.projectDesc}
                    </p>

                    {footer && (
                      <div className="project-card-footer">
                        <span
                          className={
                            isDark
                              ? "dark-mode project-tag"
                              : "project-tag"
                          }
                          onClick={() => changePage(targetPage)}
                        >
                          {footer.name || "Visit Website"}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </Fade>
  );
}