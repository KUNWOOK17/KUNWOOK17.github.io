import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * MarkdownViewer
 * - public/ 경로의 .md 파일을 fetch 하여 렌더링
 * - GitHub Pages 배포 환경에서도 동작
 *
 * props:
 *   src: string  // 예) "/assets/posts/liquid-injection.md"
 *   className?: string
 *   style?: React.CSSProperties
 */
export default function MarkdownViewer({ src, className, style }) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const url =
      src.startsWith("http") || src.startsWith("/")
        ? src
        : `${process.env.PUBLIC_URL}/${src.replace(/^\//, "")}`;

    fetch(url, { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then(setText)
      .catch((e) => setError(String(e)));
  }, [src]);

  if (error) return <div style={{ padding: 24 }}>로드 에러: {error}</div>;
  if (!text) return <div style={{ padding: 24 }}>불러오는 중…</div>;

  return (
    <article
      className={className}
      style={{ padding: 24, maxWidth: 980, margin: "0 auto", ...style }}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </article>
  );
}
