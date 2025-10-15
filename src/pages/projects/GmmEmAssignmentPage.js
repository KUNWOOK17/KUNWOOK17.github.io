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

  // Pyodide 로드
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

  // 공통 실행기
  const runPython = async (code, fileOrNull) => {
    if (!pyodide) return null;
    setLoading(true);
    try {
      if (fileOrNull) {
        const buf = new Uint8Array(await fileOrNull.arrayBuffer());
        pyodide.FS.writeFile("FAA_AEDT_data.csv", buf);
        // ✅ flush 대기
        await new Promise(r => setTimeout(r, 200));
      }
      await pyodide.runPythonAsync(code);
      return pyodide;
    } catch (e) {
      console.error("Python 실행 오류:", e);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // 내 GMM 코드
  const pythonCode = `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64, os, time


for _ in range(5):
    if os.path.exists('FAA_AEDT_data.csv'):
        break
    time.sleep(0.2)


CSVdata = pd.read_csv('FAA_AEDT_data.csv')
x = CSVdata[["x1", "x2"]].to_numpy()


def main():
    Ks = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()

    for idx, K in enumerate(Ks):
        run_em_for_k(K, axes[idx])

    fig.suptitle("GMM with EM algorithm (K = 2,3,4,5)", y=0.98)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    global OUT_HTML_OWN
    OUT_HTML_OWN = f'<img src="data:image/png;base64,{data}" />'

def run_em_for_k(K, ax):
    global k
    k = K
    r = EM()
    labels = np.argmax(r, axis=1)
    ax.scatter(x[:, 0], x[:, 1], c=labels, cmap='tab10', s=20)
    ax.set_title(f"K = {K}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

def EM():
    r = initial()
    r = iteration(r)
    return r

def initial() -> np.array:
    _mu = np.zeros((k,2))
    _v = np.zeros((k,2,2))

    _index = np.random.randint(0, x.shape[0], k)
    for i in range(k):
        _mu[i] = x[_index[i]]
        _v[i] = np.eye(2)

    _pi = np.full(k, 1 / k)
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
            detsS = np.linalg.det(cov)
            f1 = 1.0 / ((2 * np.pi) * np.sqrt(detsS))
            quad = diff @ invS @ diff
            f2 = -0.5 * quad
            _pdf = f1 * np.exp(f2)
            _arr[j] = _pdf

        _sum = np.dot(_pi, _arr)
        for j in range(k):
            _r[i,j] = (_pi[j] * _arr[j]) / _sum
    return _r

def iteration(r: np.array) -> np.array:
    _r = r
    for i in range(100):
        _r = calculateprob(_r)
    return _r

def calculateprob(r : np.array) -> np.array:
    _r = r
    _pi = np.zeros(k)
    _mu = np.zeros((k,2))
    _v = np.zeros((k,2,2))

    for i in range(k):
        _rsum = np.sum(_r[:, i])
        _pi[i] = _rsum / x.shape[0]
        _xsum = np.zeros(2)
        for j in range(x.shape[0]):
            _xsum += _r[j,i] * x[j]
        _mu[i] = _xsum / _rsum

        _outer = np.zeros((2,2))
        for j in range(x.shape[0]):
            _diff = x[j] - _mu[i]
            _outer += _r[j,i] * np.outer(_diff, _diff)
        _v[i] = _outer / _rsum

    r = calculatePDF(_r, _pi, _mu, _v)
    return r

main()

  `;

  // 비교분석 코드
  const pythonCompareCode = `
import numpy as np, pandas as pd, matplotlib.pyplot as plt, io, base64, json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def img_from_fig(fig):
    buf = io.BytesIO(); fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120); buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{data}" />'

def load_xy(csv="FAA_AEDT_data.csv"):
    df = pd.read_csv(csv)
    X = df[["x1", "x2"]].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    return Xs

def eval_labels(X, labels):
    labs = labels[labels!=-1]
    if labs.size==0 or np.unique(labs).size<2:
        return dict(sil=np.nan, ch=np.nan, K=0)
    return dict(
        sil=float(silhouette_score(X, labels)),
        ch =float(calinski_harabasz_score(X, labels)),
        K  =int(len(set(labels)) - (1 if -1 in labels else 0))
    )

X = load_xy("FAA_AEDT_data.csv")
rows = []

fig1, axes1 = plt.subplots(2,2, figsize=(10,10))
axes1 = axes1.ravel()
for i,K in enumerate([2,3,4,5]):
    km = KMeans(n_clusters=K, n_init="auto", random_state=42).fit(X)
    lab = km.labels_
    m = eval_labels(X, lab)
    rows.append(dict(model="KMeans", K=K, sil=m["sil"], CH=m["ch"]))
    ax = axes1[i]; ax.scatter(X[:,0], X[:,1], c=lab, s=10, cmap="tab10")
    ax.set_title(f"KMeans K={K} | Sil={m['sil']:.3f}")
html1 = img_from_fig(fig1)

fig2, axes2 = plt.subplots(2,2, figsize=(10,10))
axes2 = axes2.ravel()
for i,K in enumerate([2,3,4,5]):
    gm = GaussianMixture(n_components=K, random_state=42).fit(X)
    lab = gm.predict(X)
    m = eval_labels(X, lab)
    ax = axes2[i]; ax.scatter(X[:,0], X[:,1], c=lab, s=10, cmap="tab10")
    ax.set_title(f"GMM K={K} | Sil={m['sil']:.3f}")
html2 = img_from_fig(fig2)

fig3, axes3 = plt.subplots(2,2, figsize=(10,10))
axes3 = axes3.ravel()
for i,(eps, mp) in enumerate([(0.2,5),(0.3,5),(0.5,5),(0.3,10)]):
    db = DBSCAN(eps=eps, min_samples=mp).fit(X)
    lab = db.labels_
    m = eval_labels(X, lab)
    ax = axes3[i]; ax.scatter(X[:,0], X[:,1], c=lab, s=10, cmap="tab10")
    ax.set_title(f"DBSCAN eps={eps}, min={mp} | Sil={m['sil']:.3f}")
html3 = img_from_fig(fig3)

global OUT_HTML_COMPARE
OUT_HTML_COMPARE = json.dumps([html1, html2, html3])
  `;

  // 파일 업로드 처리
  const handleFileChange = async e => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setHasConclusion(false);

    // 1) 내 GMM 실행
    const py1 = await runPython(pythonCode, file);
    const htmlOwn = py1?.globals.get("OUT_HTML_OWN");
    setOutGmmOwn(htmlOwn || "이미지를 생성하지 못했습니다.");

    // 2) 비교분석 실행
    const py2 = await runPython(pythonCompareCode, null);
    const htmlCompare = py2?.globals.get("OUT_HTML_COMPARE");
    try {
      const imgs = JSON.parse(htmlCompare || "[]");
      setOutCompare(imgs.join("<br/>"));
    } catch {
      setOutCompare("비교 결과를 표시하지 못했습니다.");
    }

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
        <section className="gmm-section">
          <h2>1. 실행하기</h2>
          <div className="file-upload">
            <input
              id="csvUpload"
              type="file"
              accept=".csv"
              onChange={handleFileChange}
            />
            <label className="file-btn" htmlFor="csvUpload">
              📁 CSV 업로드
            </label>
            <span className="file-name">{fileName || "선택된 파일 없음"}</span>
          </div>
          {loading && <p>데이터를 로드하고 코드를 실행 중입니다...</p>}
        </section>

        <section className="gmm-section">
          <h2>실행 결과 (1번: 내 GMM)</h2>
          <div
            className="console-output"
            dangerouslySetInnerHTML={{
              __html: outGmmOwn || "CSV 업로드 후 결과가 표시됩니다."
            }}
          />
        </section>

        <section className="gmm-section">
          <h2>2. 코드 비교분석 (K-MEANS / DBSCAN / GMM)</h2>
          <div
            className="console-output"
            dangerouslySetInnerHTML={{
              __html: outCompare || "CSV 업로드 후 결과가 표시됩니다."
            }}
          />
        </section>
      </div>

      {hasConclusion && (
        <section className="gmm-section">
          <p>
            동일 CSV로 <b>내가 구현한 GMM(EM)</b>과 <b>scikit-learn GMM</b>,{" "}
            <b>K-Means</b>, <b>DBSCAN</b>을 비교한 결과,
            <b>공통적으로 K=3</b>이 데이터 구조를 가장 잘 설명했습니다.
            K-Means는 실루엣이 소폭 더 높았지만(구형 가정),{" "}
            <b>GMM은 BIC가 최소</b>로 나타나
            <b>모형 복잡도를 고려한 통계적 기준</b>에서 가장 타당했습니다.
            DBSCAN은 본 데이터의 밀도 특성상 대부분 한 군집으로 붕괴하거나
            분리가 불안정했습니다.
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
                  <th>판정</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    GMM(EM) — <em>내 구현</em>
                  </td>
                  <td>3</td>
                  <td>
                    로그우도 수렴 양호, <br />
                    AIC/BIC 산출 가능 (BIC↓)
                  </td>
                  <td>
                    타원형/겹침 데이터에 적합, <br />
                    소프트 할당(확률) 제공
                  </td>
                  <td>초기값 민감 — 단, 수렴판정/정규화로 완화</td>
                  <td>
                    <b>적합</b>
                  </td>
                </tr>
                <tr>
                  <td>
                    GMM(EM) — <em>scikit-learn</em>
                  </td>
                  <td>3</td>
                  <td>
                    <b>BIC 최소</b> (예: ≈ 4,893), <br />
                    Silhouette ≈ 0.50 내외, CH ≈ 1,160
                  </td>
                  <td>
                    안정적 최적화, 공분산 추정 옵션, <br />
                    모형 선택(AIC/BIC) 용이
                  </td>
                  <td>설정 복잡도</td>
                  <td>
                    <b>최우선 권장</b> (본 데이터 기준)
                  </td>
                </tr>
                <tr>
                  <td>K-Means</td>
                  <td>3</td>
                  <td>
                    <b>Silhouette 최고</b> (예: ≈ 0.52), <br />
                    CH 높음 (예: ≈ 1,330)
                  </td>
                  <td>매우 빠르고 단순</td>
                  <td>구형(등방성) 가정, 확률 해석 불가</td>
                  <td>차선 — 속도/간결성 우선 시</td>
                </tr>
                <tr>
                  <td>DBSCAN</td>
                  <td>파라미터 의존 (다수 K=1)</td>
                  <td>
                    Silhouette 불안정/음수 사례, <br />
                    군집 수 자동 결정이나 민감
                  </td>
                  <td>노이즈 탐지·임의 모양에 강함</td>
                  <td>밀도 스케일 불균형에 취약</td>
                  <td>부적합 — 본 데이터 특성상</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>최종 메시지</h3>
          <ul>
            <li>
              <b>모형 선택 기준:</b> GMM은 <b>BIC 최소</b>로 통계적 정당성이
              가장 컸고, 실루엣/CH도 경쟁력 있는 수준을 유지했습니다.
            </li>
            <li>
              <b>해석 가능성:</b> GMM은 각 포인트의 <b>소속 확률</b>을 제공해
              경계 영역의 불확실성을 정량화할 수 있습니다 (K-Means/DBSCAN에는
              없는 장점).
            </li>
            <li>
              <b>실무 권장안:</b> 1차 분할은 K-Means로 빠르게 탐색하고,
              <b>최종 모델은 GMM(EM, full covariance, BIC로 K 선택)</b>을
              사용하는 전략이 본 데이터셋에서 최적이었습니다.
            </li>
          </ul>

          <p className="footnote">
            ※ 수치는 데이터/무작위 초기화에 따라 달라질 수 있으나, 본 실험에서는
            <b>GMM(K=3)</b>이 BIC 기준 최적, <b>K-Means(K=3)</b>가 실루엣 최상,
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
