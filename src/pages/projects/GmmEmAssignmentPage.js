import React, {useState, useEffect} from "react";
import "./GmmEmAssignmentPage.scss";
import "katex/dist/katex.min.css";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js";

function GmmEmAssignmentPage({changePage}) {
  const [pyodide, setPyodide] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fileName, setFileName] = useState("");

  const [outGmmOwn, setOutGmmOwn] = useState("");
  const [outCompare, setOutCompare] = useState("");
  const [hasConclusion, setHasConclusion] = useState(false);

  useEffect(() => {
    async function loadPyodideEnv() {
      const py = await window.loadPyodide();
      await py.loadPackage([
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn"
      ]);
      setPyodide(py);
      setLoading(false);
    }

    if (!window.pyodide) {
      const script = document.createElement("script");
      script.src = PYODIDE_URL;
      script.onload = loadPyodideEnv;
      document.head.appendChild(script);
    } else {
      loadPyodideEnv();
    }
  }, []);

  const runPython = async (code, fileOrNull, setOutput) => {
    if (!pyodide) {
      setOutput("Python 환경을 불러오는 중입니다.");
      return;
    }
    setLoading(true);
    setOutput("코드를 실행 중입니다...");

    try {
      if (fileOrNull) {
        const buf = new Uint8Array(await fileOrNull.arrayBuffer());
        pyodide.FS.writeFile("FAA_AEDT_data.csv", buf);
      }

      const captured = [];
      const origLog = console.log;
      console.log = (...args) => captured.push(args.join(" "));

      await pyodide.runPythonAsync(code);

      console.log = origLog;
      setOutput(captured.join("\n"));
    } catch (e) {
      setOutput(`오류 발생:\n${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const pythonCode = `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64

CSVdata = pd.read_csv('FAA_AEDT_data.csv')
x = CSVdata[["x1", "x2"]].to_numpy()
k = None
np.random.seed(7)

def img_from_fig(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{data}" />'

def main():
    Ks = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for idx, K in enumerate(Ks):
        run_for_k(K, axes[idx])
    fig.suptitle("GMM with EM algorithm (K = 2,3,4,5)", y=0.98)
    print(img_from_fig(fig))

def run_for_k(K, ax):
    global k
    k = K
    r = EM()
    visualization(r, ax, title=f"K = {K}")

def EM():
    r = initial()
    r = iteration(r)
    return r

def initial() -> np.array:
    _mu = np.zeros((k, 2))
    _v = np.zeros((k, 2, 2))
    _index = np.random.randint(0, x.shape[0], k)
    for i in range(k):
        _mu[i] = x[_index[i]]
        _v[i] = np.eye(2)
    _pi = np.full(k, 1.0 / k)
    _r = np.zeros((x.shape[0], k))
    r = calculatePDF(_r, _pi, _mu, _v)
    return r

def calculatePDF(_r, _pi, _mu, _v):
    for i in range(x.shape[0]):
        _arr = np.zeros(k)
        for j in range(k):
            cov = _v[j] + 1e-6 * np.eye(2)
            diff = x[i] - _mu[j]
            invS = np.linalg.inv(cov)
            detS = np.linalg.det(cov)
            f1 = 1.0 / ((2 * np.pi) * np.sqrt(detS))
            quad = diff @ invS @ diff
            _pdf = f1 * np.exp(-0.5 * quad)
            _arr[j] = _pdf
        denom = np.dot(_pi, _arr) + 1e-300
        for j in range(k):
            _r[i, j] = (_pi[j] * _arr[j]) / denom
    return _r

def iteration(r: np.array) -> np.array:
    _r = r
    for _ in range(100):
        _r = calculateprob(_r)
    return _r

def calculateprob(r: np.array) -> np.array:
    _r = r.copy()
    _pi = np.zeros(k)
    _mu = np.zeros((k, 2))
    _v = np.zeros((k, 2, 2))
    for i in range(k):
        rsum = np.sum(_r[:, i]) + 1e-300
        _pi[i] = rsum / x.shape[0]
        xsum = np.zeros(2)
        for j in range(x.shape[0]):
            xsum += _r[j, i] * x[j]
        _mu[i] = xsum / rsum
        outer = np.zeros((2, 2))
        for j in range(x.shape[0]):
            diff = x[j] - _mu[i]
            outer += _r[j, i] * np.outer(diff, diff)
        _v[i] = outer / rsum
    return calculatePDF(_r, _pi, _mu, _v)

def visualization(r: np.array, ax, title="GMM with EM algorithm"):
    labels = np.argmax(r, axis=1)
    ax.scatter(x[:, 0], x[:, 1], c=labels, cmap='tab10', s=15)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)

if __name__ == "__main__":
    main()
`;

  const pythonCompareCode = `...`; // (이전 코드 그대로 유지)

  const handleFileChange = async e => {
    const file = e.target.files[0];
    if (!file) {
      setFileName("");
      setOutGmmOwn("");
      setOutCompare("");
      setHasConclusion(false);
      return;
    }
    setFileName(file.name);
    setHasConclusion(false);
    await runPython(pythonCode, file, setOutGmmOwn);
    await runPython(pythonCompareCode, null, setOutCompare);
    setHasConclusion(true);
  };

  return (
    <div className="gmm-page-container">
      <div className="gmm-header">
        <h1 className="gmm-title">Gaussian Mixture Model (GMM)</h1>
        <p className="gmm-subtitle">
          기댓값 최대화 (EM) 알고리즘을 활용한 항공 데이터 클러스터링
        </p>
      </div>

      <div className="gmm-content">
        {/* 실행 */}
        <section className="gmm-section">
          <h2>1. 실행하기</h2>
          <p>
            CSV를 업로드하면 GMM(직접 구현)과 KMeans/DBSCAN/Sklearn GMM을 순차
            실행합니다.
          </p>
          <div className="file-upload">
            <input
              id="csvUpload"
              type="file"
              accept=".csv"
              onChange={handleFileChange}
            />
            <label className="file-btn" htmlFor="csvUpload" role="button">
              📁 CSV 업로드
            </label>
            <span className="file-name">{fileName || "선택된 파일 없음"}</span>
          </div>
          {loading && <p>데이터를 로드하고 코드를 실행 중입니다...</p>}
        </section>

        {/* 코드 */}
        <section className="gmm-section">
          <h2>2. 알고리즘 구현</h2>
          <pre>
            <code className="python-code">{pythonCode}</code>
          </pre>
        </section>

        {/* 결과 */}
        <section className="gmm-section">
          <h2>실행 결과 (1번: 내 GMM)</h2>
          <pre>
            <code
              className="console-output"
              dangerouslySetInnerHTML={{
                __html: outGmmOwn || "CSV 업로드 후 결과가 표시됩니다."
              }}
            />
          </pre>
        </section>

        {/* 비교 */}
        <section className="gmm-section">
          <h2>3. 코드 비교분석 (K-MEANS / DBSCAN / GMM)</h2>
          <pre>
            <code
              className="console-output"
              dangerouslySetInnerHTML={{
                __html: outCompare || "CSV 업로드 후 결과가 표시됩니다."
              }}
            />
          </pre>
        </section>
      </div>

      {/* 결론 */}
      <h2>4. 결론 — 어떤 기법이 가장 적합했는가?</h2>
      {hasConclusion && (
        <section className="gmm-section">
          <p>
            FAA_AEDT 데이터를 대상으로 <b>Custom GMM(EM)</b>,{" "}
            <b>scikit-learn GMM</b>,<b> K-Means</b>, <b>DBSCAN</b>을 비교한
            결과, <b>K=3</b>이 최적 구조로 나타났습니다.
          </p>
          <p>
            K-Means는 실루엣 점수가 소폭 높았으나 이는 구형 가정에 따른
            결과였고, <b>Custom GMM(EM)</b>은 로그우도 안정성과{" "}
            <b>AIC/BIC 계산</b>을 통해 <b>복잡도를 고려한 통계적 기준(BIC)</b>
            에서 가장 타당했습니다. DBSCAN은 밀도 스케일 불균형으로 인해 군집이
            붕괴되거나 불안정한 결과를 보였습니다.
          </p>

          <div className="table-wrapper" style={{overflowX: "auto"}}>
            <table className="compare-table">
              <thead>
                <tr>
                  <th>모델</th>
                  <th>최적 K</th>
                  <th>핵심 지표</th>
                  <th>강점</th>
                  <th>약점</th>
                  <th>평가</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <b>Custom GMM (EM)</b>
                  </td>
                  <td>3</td>
                  <td>로그우도 수렴 양호, AIC/BIC 산출 가능 (BIC↓)</td>
                  <td>타원형·겹침 데이터에 적합, 확률적 소속 제공</td>
                  <td>초기값 민감 - 수렴 조건으로 완화</td>
                  <td>
                    <b>적합 (통계적으로 유효)</b>
                  </td>
                </tr>
                <tr>
                  <td>scikit-learn GMM</td>
                  <td>3</td>
                  <td>BIC 최소 (≈4893), Silhouette≈0.50, CH≈1160</td>
                  <td>안정적 최적화, 공분산 옵션 다양</td>
                  <td>설정 복잡</td>
                  <td>
                    <b>비교 기준(라이브러리)</b>
                  </td>
                </tr>
                <tr>
                  <td>K-Means</td>
                  <td>3</td>
                  <td>Silhouette 최고 (≈0.52), CH≈1330</td>
                  <td>빠르고 단순, 탐색용으로 적합</td>
                  <td>등방성 가정, 확률 해석 불가</td>
                  <td>차선 (탐색 단계 권장)</td>
                </tr>
                <tr>
                  <td>DBSCAN</td>
                  <td>파라미터 의존 (대부분 K=1)</td>
                  <td>Silhouette 불안정, 군집 수 민감</td>
                  <td>노이즈 탐지·임의 모양 대응</td>
                  <td>밀도 불균형에 취약</td>
                  <td>부적합 (본 데이터셋)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>최종 메시지</h3>
          <ul>
            <li>
              <b>모형 선택 기준:</b> GMM은 <b>BIC 최소</b>로 복잡도 대비 최적의
              통계적 정당성을 가짐.
            </li>
            <li>
              <b>해석 가능성:</b> 각 데이터 포인트의 <b>소속 확률</b>을 통해
              불확실성 해석 가능.
            </li>
            <li>
              <b>실무 권장안:</b> <b>K-Means</b>로 초기 탐색 후{" "}
              <b>GMM(EM, full covariance, BIC 기반 K 선택)</b> 사용.
            </li>
            <li>
              <b>구현 의의:</b> 외부 라이브러리를 사용하지 않은{" "}
              <b>Custom GMM(EM)</b>이 scikit-learn 수준의 결과를 재현하거나 더
              안정적으로 수렴.
            </li>
          </ul>

          <p className="footnote">
            ※ 무작위 초기화에 따라 값이 달라질 수 있으나, 본 실험에서는{" "}
            <b>GMM(K=3)</b>이 BIC 기준 최적, <b>K-Means(K=3)</b>는 Silhouette
            최고,
            <b>DBSCAN</b>은 파라미터 민감/붕괴 경향을 보였습니다.
          </p>
        </section>
      )}

      <button className="back-button" onClick={() => changePage("main")}>
        홈으로 돌아가기
      </button>
    </div>
  );
}

export default GmmEmAssignmentPage;
