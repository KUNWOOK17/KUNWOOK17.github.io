import React from "react";
import "./BlogCard.scss";

// ✅ props로 changePage 함수를 받도록 수정했습니다.
export default function BlogCard({ blog, isDark, changePage }) {

  // 외부 URL을 새 탭에서 열거나, 내부 라우트로 페이지를 변경하는 함수
  function handleCardClick() {
    if (blog.route) {
      // portfolio.js에 route 속성이 있는 경우, changePage 함수를 호출
      changePage(blog.route);
    } else if (blog.url) {
      // url 속성이 있는 경우, 새 탭에서 URL을 엽니다.
      var win = window.open(blog.url, "_blank");
      win.focus();
    } else {
      console.log(`URL or route for ${blog.title} not found`);
    }
  }

  return (
    // ✅ onClick 이벤트 핸들러를 handleCardClick 함수로 변경
    <div onClick={handleCardClick}>
      <div className={isDark ? "blog-container dark-mode" : "blog-container"}>
        <div
          className={
            isDark ? "dark-mode blog-card blog-card-shadow" : "blog-card"
          }
        >
          {/* ✅ href 속성을 제거했습니다. */}
          <h3 className={isDark ? "small-dark blog-title" : "blog-title"}>
            {blog.title}
          </h3>
          <p className={isDark ? "small-dark small" : "small"}>
            {blog.description}
          </p>
          <div className="go-corner">
            <div className="go-arrow">→</div>
          </div>
        </div>
      </div>
    </div>
  );
}