// src/containers/topbutton/Top.js
import React, {useEffect} from "react";
import "./Top.scss";

export default function Top() {
  // 맨 위로
  function TopEvent() {
    window.scrollTo({top: 0, behavior: "smooth"});
  }

  // 버튼 표시/숨김
  function scrollFunction() {
    const btn = document.getElementById("topButton");
    if (!btn) return; // ✅ 버튼이 아직 없으면 그냥 종료 (스플래시/전환 중 안전)

    const scrolled =
      document.body.scrollTop > 20 || document.documentElement.scrollTop > 20;

    // 원래 visibility만 쓰던 로직에 opacity/pointerEvents를 보강해 부드럽게
    btn.style.visibility = scrolled ? "visible" : "hidden";
    btn.style.opacity = scrolled ? "1" : "0";
    btn.style.pointerEvents = scrolled ? "auto" : "none";
  }

  useEffect(() => {
    // ✅ 전역 할당(window.onscroll) 대신 안전하게 등록/해제
    window.addEventListener("scroll", scrollFunction, {passive: true});
    window.addEventListener("load", scrollFunction);

    // 첫 렌더 직후 상태 동기화
    scrollFunction();

    return () => {
      window.removeEventListener("scroll", scrollFunction);
      window.removeEventListener("load", scrollFunction);
    };
  }, []);

  return (
    <button
      onClick={TopEvent}
      id="topButton"
      title="Go to top"
      aria-label="Go to top"
    >
      <i className="fas fa-hand-point-up" aria-hidden="true"></i>
    </button>
  );
}
