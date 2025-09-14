// // import React, {useContext} from "react";
// // import "./StartupProjects.scss";
// // import {bigProjects} from "../../portfolio";
// // import {Fade} from "react-reveal";
// // import StyleContext from "../../contexts/StyleContext";

// // export default function StartupProject() {
// //   function openUrlInNewTab(url) {
// //     if (!url) {
// //       return;
// //     }
// //     var win = window.open(url, "_self");
// //     win.focus();
// //   }

// //   const {isDark} = useContext(StyleContext);
// //   if (!bigProjects.display) {
// //     return null;
// //   }
// //   return (
// //     <Fade bottom duration={1000} distance="20px">
// //       <div className="main" id="projects">
// //         <div>
// //           <h1 className="skills-heading">{bigProjects.title}</h1>
// //           <p
// //             className={
// //               isDark
// //                 ? "dark-mode project-subtitle"
// //                 : "subTitle project-subtitle"
// //             }
// //           >
// //             {bigProjects.subtitle}
// //           </p>

// //           <div className="projects-container">
// //             {bigProjects.projects.map((project, i) => {
// //               return (
// //                 <div
// //                   key={i}
// //                   className={
// //                     isDark
// //                       ? "dark-mode project-card project-card-dark"
// //                       : "project-card project-card-light"
// //                   }
// //                 >
// //                   {project.image ? (
// //                     <div className="project-image">
// //                       <img
// //                         src={project.image}
// //                         alt={project.projectName}
// //                         className="card-image"
// //                       ></img>
// //                     </div>
// //                   ) : null}
// //                   <div className="project-detail">
// //                     <h5
// //                       className={isDark ? "dark-mode card-title" : "card-title"}
// //                     >
// //                       {project.projectName}
// //                     </h5>
// //                     <p
// //                       className={
// //                         isDark ? "dark-mode card-subtitle" : "card-subtitle"
// //                       }
// //                     >
// //                       {project.projectDesc}
// //                     </p>
// //                     {project.footerLink ? (
// //                       <div className="project-card-footer">
// //                         {project.footerLink.map((link, i) => {
// //                           return (
// //                             <span
// //                               key={i}
// //                               className={
// //                                 isDark ? "dark-mode project-tag" : "project-tag"
// //                               }
// //                               onClick={() => openUrlInNewTab(link.url)}
// //                             >
// //                               {link.name}
// //                             </span>
// //                           );
// //                         })}
// //                       </div>
// //                     ) : null}
// //                   </div>
// //                 </div>
// //               );
// //             })}
// //           </div>
// //         </div>
// //       </div>
// //     </Fade>
// //   );
// // }

// // // src/containers/StartupProjects/StartupProject.js

// // import React, {useContext} from "react";
// // import "./StartupProjects.scss";
// // import {bigProjects} from "../../portfolio";
// // import {Fade} from "react-reveal";
// // import StyleContext from "../../contexts/StyleContext";

// // export default function StartupProject({changePage}) {
// //   const {isDark} = useContext(StyleContext);
// //   if (!bigProjects.display) {
// //     return null;
// //   }
// //   return (
// //     <Fade bottom duration={1000} distance="20px">
// //       <div className="main" id="projects">
// //         <div>
// //           <h1 className="skills-heading">{bigProjects.title}</h1>
// //           <p
// //             className={
// //               isDark
// //                 ? "dark-mode project-subtitle"
// //                 : "subTitle project-subtitle"
// //             }
// //           >
// //             {bigProjects.subtitle}
// //           </p>

// //           <div className="projects-container">
// //             {bigProjects.projects.map((project, i) => {
// //               return (
// //                 <div
// //                   key={i}
// //                   className={
// //                     isDark
// //                       ? "dark-mode project-card project-card-dark"
// //                       : "project-card project-card-light"
// //                   }
// //                 >
// //                   {project.image ? (
// //                     <div className="project-image">
// //                       <img
// //                         src={project.image}
// //                         alt={project.projectName}
// //                         className="card-image"
// //                       ></img>
// //                     </div>
// //                   ) : null}
// //                   <div className="project-detail">
// //                     <h5
// //                       className={isDark ? "dark-mode card-title" : "card-title"}
// //                     >
// //                       {project.projectName}
// //                     </h5>
// //                     <p
// //                       className={
// //                         isDark ? "dark-mode card-subtitle" : "card-subtitle"
// //                       }
// //                     >
// //                       {project.projectDesc}
// //                     </p>
// //                     {project.footerLink ? (
// //                       <div className="project-card-footer">
// //                         <span
// //                           className={isDark ? "dark-mode project-tag" : "project-tag"}
// //                           onClick={() => changePage("project1")}
// //                         >
// //                           View Details
// //                         </span>
// //                         <span
// //                           className={isDark ? "dark-mode project-tag" : "project-tag"}
// //                           onClick={() => changePage("project4")}
// //                         >
// //                           Visit Website
// //                         </span>
// //                       </div>
// //                     ) : null}
// //                   </div>
// //                 </div>
// //               );
// //             })}
// //           </div>
// //         </div>
// //       </div>
// //     </Fade>
// //   );
// // }

// // src/containers/StartupProjects/StartupProject.js

import React, {useContext} from "react";
import "./StartupProjects.scss";
import {bigProjects} from "../../portfolio";
import {Fade} from "react-reveal";
import StyleContext from "../../contexts/StyleContext";

export default function StartupProject({changePage}) {
  const {isDark} = useContext(StyleContext);
  if (!bigProjects.display) {
    return null;
  }
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
              // project.projectName 값을 사용하여 어떤 페이지로 이동할지 결정합니다.
              const pageName = project.projectName.includes("Project 1")
                ? "project1"
                : "project4";

              return (
                <div
                  key={i}
                  className={
                    isDark
                      ? "dark-mode project-card project-card-dark"
                      : "project-card project-card-light"
                  }
                >
                  {project.image ? (
                    <div className="project-image">
                      <img
                        src={project.image}
                        alt={project.projectName}
                        className="card-image"
                      ></img>
                    </div>
                  ) : null}
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
                    {project.footerLink ? (
                      <div className="project-card-footer">
                        <span
                          className={
                            isDark ? "dark-mode project-tag" : "project-tag"
                          }
                          onClick={() => changePage(pageName)} // 위에서 결정한 페이지로 이동
                        >
                          Visit Website
                        </span>
                      </div>
                    ) : null}
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


// // src/containers/StartupProjects/StartupProject.js
// import React, {useContext} from "react";
// import "./StartupProjects.scss";
// import {bigProjects} from "../../portfolio";
// import {Fade} from "react-reveal";
// import StyleContext from "../../contexts/StyleContext";

// export default function StartupProject() {
//   const {isDark} = useContext(StyleContext);
//   if (!bigProjects.display) return null;

//   return (
//     <Fade bottom duration={1000} distance="20px">
//       <div className="main" id="projects">
//         <div>
//           <h1 className="skills-heading">{bigProjects.title}</h1>
//           <p className={isDark ? "dark-mode project-subtitle" : "subTitle project-subtitle"}>
//             {bigProjects.subtitle}
//           </p>

//           <div className="projects-container">
//             {bigProjects.projects.map((project, i) => (
//               <div
//                 key={i}
//                 className={isDark ? "dark-mode project-card project-card-dark" : "project-card project-card-light"}
//               >
//                 {project.image ? (
//                   <div className="project-image">
//                     <img src={project.image} alt={project.projectName} className="card-image" />
//                   </div>
//                 ) : null}

//                 <div className="project-detail">
//                   <h5 className={isDark ? "dark-mode card-title" : "card-title"}>
//                     {project.projectName}
//                   </h5>
//                   <p className={isDark ? "dark-mode card-subtitle" : "card-subtitle"}>
//                     {project.projectDesc}
//                   </p>

//                   {project.footerLink && (
//                     <div className="project-card-footer">
//                       {project.footerLink.map((link, j) => (
//                         <a
//                           key={j}
//                           className={isDark ? "dark-mode project-tag" : "project-tag"}
//                           href={link.url}
//                           rel="noopener"
//                         >
//                           {link.name || "Visit Website"}
//                         </a>
//                       ))}
//                     </div>
//                   )}
//                 </div>
//               </div>
//             ))}
//           </div>
//         </div>
//       </div>
//     </Fade>
//   );
// }
