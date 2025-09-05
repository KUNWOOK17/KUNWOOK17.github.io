import React from "react";
import "./App.scss";
import Main from "./containers/Main";

// 라우팅을 위해 react-router-dom 추가
import { BrowserRouter, Routes, Route } from "react-router-dom";
import MarkdownPost from "./pages/projects/MarkdownPost";

function App() {
  return (
    <BrowserRouter basename={process.env.PUBLIC_URL}>
      <Routes>
        {/* 기본 홈 페이지 */}
        <Route path="/" element={<Main />} />

        {/* 마크다운 포스트 페이지 */}
        <Route path="/post/liquid-injection" element={<MarkdownPost />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
