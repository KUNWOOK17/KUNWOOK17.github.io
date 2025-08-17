// // // src/pages/Project1.js
// // import React from "react";

// // export default function Project1() {
// //   return (
// //     <div style={{ padding: "2rem" }}>
// //       <h1>Doosan Robotics Project 1</h1>
// //       <p>Implementation of a robot automation process system using ROS2.</p>
// //       <p>(여기에 상세 설명/이미지/결과 등을 추가)</p>
// //     </div>
// //   );
// // }

// import React from "react";
// import ReactMarkdown from "react-markdown";
// import remarkGfm from "remark-gfm";
// import rehypeRaw from "rehype-raw";

// export default function Project1() {
//   const [content, setContent] = React.useState("");

//   React.useEffect(() => {
//     // public/ 아래 파일은 절대경로로 접근 가능
//     fetch("/projects/liquid_injection.md")
//       .then((res) => res.text())
//       .then((text) => setContent(text))
//       .catch((err) => console.error("Failed to load markdown:", err));
//   }, []);

//   return (
//     <main style={{maxWidth: 980, margin: "40px auto", padding: "0 16px"}}>
//       <h1 style={{marginBottom: 12}}>TurtleBot Project</h1>
//       <ReactMarkdown
//         remarkPlugins={[remarkGfm]}
//         rehypePlugins={[rehypeRaw]}
//       >
//         {content}
//       </ReactMarkdown>
//     </main>
//   );
// }

// import React, {useEffect, useState} from "react";
// import DOMPurify from "dompurify";
// import {marked} from "marked";
// import LiquidInjectionMd from "./liquid_injection.md"; // 마크다운 파일 import

// export default function Project1() {
//   const [content, setContent] = useState("Loading...");

//   useEffect(() => {
//     fetch(LiquidInjectionMd) // import한 파일을 fetch 함수의 경로로 사용
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

import React, {useEffect, useState} from "react";
import DOMPurify from "dompurify";
import {marked} from "marked";

export default function Project1() {
  const [content, setContent] = useState("Loading...");

  useEffect(() => {
    // public 폴더를 기준으로 경로를 설정합니다.
    fetch("/projects/liquid_injection.md")
      .then(r => {
        if (!r.ok) throw new Error("md load fail");
        return r.text();
      })
      .then(md => {
        const html = marked.parse(md, {mangle: false, headerIds: true});
        setContent(DOMPurify.sanitize(html));
      })
      .catch(e => {
        setContent("Failed to load content.");
        console.error(e);
      });
  }, []);

  return (
    <div
      style={{
        maxWidth: "980px",
        margin: "40px auto",
        padding: "0 16px",
        lineHeight: 1.6
      }}
    >
      <h1>Liquid Injection Project</h1>
      <div dangerouslySetInnerHTML={{__html: content}} />
    </div>
  );
}
