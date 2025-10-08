// src/components/HeroRobotModelViewer.jsx
import "@google/model-viewer";

export default function HeroRobotModelViewer() {
  const modelSrc = process.env.PUBLIC_URL + "/models/Robot.glb"; // ✅ 배포/로컬 모두 안전

  return (
    <model-viewer
      src={modelSrc}
      alt="3D Robot"
      camera-controls
      auto-rotate
      shadow-intensity="0.7"
      bounds="tight"

      /* 조명 세팅(배포에서도 일관성) */
      environment-image="neutral"   // ✅ 기본 IBL 확실히 지정
      exposure="0.4"                // ✅ 과노출 방지 (0.8~1.0 사이로 조정)

      /* 크기/프레이밍(네가 쓰던 값 유지) */
      scale="2 2 2"
      camera-target="0m -2.6m 0m"
      camera-orbit="0deg 88deg 8.5m"
      field-of-view="60deg"

      /* 초기 값 흔들림 방지(선택) */
      min-field-of-view="60deg"
      max-field-of-view="60deg"
      disable-zoom
    />
  );
}
