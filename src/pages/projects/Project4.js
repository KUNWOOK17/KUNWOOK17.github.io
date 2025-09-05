// // // src/pages/Project4.js
// // import React from "react";

// // export default function Project4() {
// //   return (
// //     <div style={{ padding: "2rem" }}>
// //       <h1>Doosan Robotics Project 4</h1>
// //       <p>Development of an autonomous navigation robot system based on SLAM.</p>
// //       <p>(여기에 상세 설명/이미지/결과 등을 추가)</p>
// //     </div>
// //   );
// // }

// // export default function Project4() {
// //   return (
// //     <div style={{ padding: "2rem" }}>
// //       <h1>Project 4</h1>
// //       <p>SLAM-based autonomous navigation detail page.</p>
// //     </div>
// //   );
// // }

// // src/pages/Project1.js
// import React from "react";
// import ReactMarkdown from "react-markdown";
// import remarkGfm from "remark-gfm";
// import rehypeRaw from "rehype-raw";

// export default function Project4() {
//   const [content, setContent] = React.useState("");

//   React.useEffect(() => {
//     // public/ 아래 파일은 절대경로로 접근 가능
//     fetch("/projects/turtlebot.md")
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
// import TurtlebotMd from "./turtlebot.md"; // turtlebot.md 파일을 import

// export default function Project4() {
//   const [content, setContent] = useState("Loading...");

//   useEffect(() => {
//     // import한 변수 TurtlebotMd를 fetch 함수의 경로로 사용
//     // fetch(TurtlebotMd)
//     fetch('/home/kunwookpark/kunwook17/src/pages/projects/turtlebot.md')
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
//       <h1>TurtleBot Project</h1>
//       <div dangerouslySetInnerHTML={{__html: content}} />
//     </div>
//   );
// }

// import React, {useEffect, useState} from "react";
// import DOMPurify from "dompurify";
// import {marked} from "marked";

// export default function Project4() {
//   const [content, setContent] = useState("Loading...");

//   useEffect(() => {
//     // public 폴더를 기준으로 경로를 설정합니다.
//     fetch("/projects/turtlebot.md")
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
//       <h1>TurtleBot Project</h1>
//       <div dangerouslySetInnerHTML={{__html: content}} />
//     </div>
//   );
// }

import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Project4
 * - public/projects/turtlebot.md 를 불러와 페이지 내에 렌더
 * - GitHub Pages 서브경로에서도 동작하도록 PUBLIC_URL 사용
 * - 링크는 현재 창에서 열리며, a11y 경고 해결(링크에 children 보장)
 * - 프론트매터(--- ... ---)가 있으면 제거
 */
export default function Project4() {
  const [md, setMd] = useState("");
  const [error, setError] = useState("");

  // md 파일 경로 (public 기준)
  const MD_PATH = `${process.env.PUBLIC_URL}/projects/turtlebot.md`;

  useEffect(() => {
    fetch(MD_PATH, { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((raw) => {
        // YAML front-matter 제거 (있을 때만)
        const cleaned = raw.replace(/^---[\s\S]*?---\s*/m, "");
        setMd(cleaned.trim());
      })
      .catch((e) => {
        console.error(e);
        setError("콘텐츠를 불러오지 못했습니다.");
      });
  }, [MD_PATH]);

  return (
    <div
      style={{
        maxWidth: 980,
        margin: "40px auto",
        padding: "0 16px",
        lineHeight: 1.7,
      }}
    >
      {/* 기존 프로젝트 소개 섹션이 있다면 여기에 배치 */}
      <h1 style={{ marginBottom: 16 }}>TurtleBot Project</h1>

      {/* 마크다운 렌더 영역 */}
      {error ? (
        <div>{error}</div>
      ) : !md ? (
        <div>불러오는 중…</div>
      ) : (
        <article className="markdown-body">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // ✅ a11y 경고 해결: children을 반드시 렌더
              a: ({ node, children, ...props }) => {
                const hasChildText =
                  Array.isArray(children) ? children.length > 0 : !!children;
                return (
                  <a {...props}>
                    {hasChildText ? children : props.href}
                  </a>
                );
              },
              img: ({ node, ...props }) => (
                // 이미지가 컨테이너 넓이를 넘지 않도록
                <img
                  {...props}
                  style={{ maxWidth: "100%", height: "auto" }}
                  loading="lazy"
                  alt={props.alt ?? ""}
                />
              ),
              table: ({ node, ...props }) => (
                <div style={{ overflowX: "auto" }}>
                  <table {...props} />
                </div>
              ),
            }}
          >
            {md}
          </ReactMarkdown>
        </article>
      )}

      {/* 간단 스타일 - 필요시 SCSS로 옮겨도 됨 */}
      <style>{`
        .markdown-body h1, .markdown-body h2, .markdown-body h3 {
          margin-top: 28px;
          margin-bottom: 12px;
        }
        .markdown-body p {
          margin: 12px 0;
        }
        .markdown-body pre, .markdown-body code {
          background: #f6f8fa;
          padding: 2px 6px;
          border-radius: 4px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        .markdown-body pre {
          padding: 12px;
          overflow: auto;
        }
        .markdown-body ul, .markdown-body ol {
          padding-left: 1.4rem;
        }
        .markdown-body blockquote {
          border-left: 4px solid #e5e7eb;
          margin: 12px 0;
          padding: 8px 12px;
          color: #4b5563;
          background: #fafafa;
        }
        .markdown-body table {
          border-collapse: collapse;
          width: 100%;
        }
        .markdown-body th, .markdown-body td {
          border: 1px solid #e5e7eb;
          padding: 6px 8px;
          text-align: left;
        }
      `}</style>
    </div>
  );
}

