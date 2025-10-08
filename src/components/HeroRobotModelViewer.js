import "@google/model-viewer";

export default function HeroRobotModelViewer() {
  return (
    <model-viewer
      src="/models/Robot.glb"
      alt="3D Robot"
      camera-controls
      auto-rotate
      shadow-intensity="0.7"
      bounds="tight"

      /* 로봇 가로/세로 크기(실제 스케일) */
      scale="2.2 2.2 2.2"

      /* 프레이밍/위치 */
      camera-target="0m -2.6m 0m"
      camera-orbit="0deg 88deg 8.5m"
      field-of-view="60deg"

      /* 초기값 흔들림 방지(선택: 줌 고정 레일) */
      min-field-of-view="60deg"
      max-field-of-view="60deg"
      disable-zoom
    />
  );
}
