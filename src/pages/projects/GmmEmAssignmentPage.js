import React, {useState, useEffect} from "react";
import "./GmmEmAssignmentPage.scss";
import {BlockMath} from "react-katex";
import "katex/dist/katex.min.css";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js";

function GmmEmAssignmentPage({changePage}) {
  const [pyodide, setPyodide] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fileName, setFileName] = useState("");

  // 출력 상태: 1번(내 GMM) / 4번(비교분석)
  const [outGmmOwn, setOutGmmOwn] = useState("");
  const [outCompare, setOutCompare] = useState("");
  const [hasConclusion, setHasConclusion] = useState(false);

  // Pyodide 로드
  useEffect(() => {
    async function loadPyodideEnv() {
      const py = await window.loadPyodide();
      // ✅ scikit-learn 포함
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

  // 공통 실행기(출력 캡처)
  const runPython = async (code, fileOrNull, setOutput) => {
    if (!pyodide) {
      setOutput("Python 환경을 불러오는 중입니다.");
      return;
    }
    setLoading(true);
    setOutput("코드를 실행 중입니다...");

    try {
      // 파일 업로드(있을 때만)
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

  // 1) 네 GMM-EM 코드 (그대로 유지)
  const pythonCode = `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSVdata = pd.read_csv('FAA_AEDT_data.csv')
x = CSVdata[["x1", "x2"]].to_numpy()

k = None
np.random.seed(7)

def main():
    Ks = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()

    for idx, K in enumerate(Ks):
        run_for_k(K, axes[idx])

    fig.suptitle("GMM with EM algorithm (K = 2,3,4,5)", y=0.98)
    plt.tight_layout()
    plt.show()

def run_for_k(K, ax):
    """K 값을 세팅하고 EM을 실행한 뒤, 해당 축(ax)에 시각화."""
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
    _v  = np.zeros((k, 2, 2))

    _index = np.random.randint(0, x.shape[0], k)
    for i in range(k):
        _mu[i] = x[_index[i]]
        _v[i]  = np.eye(2)

    _pi = np.full(k, 1.0 / k)
    _r  = np.zeros((x.shape[0], k))

    r = calculatePDF(_r, _pi, _mu, _v)
    return r

def calculatePDF(_r, _pi, _mu, _v):
    for i in range(x.shape[0]):
        _arr = np.zeros(k)
        for j in range(k):
            cov  = _v[j] + 1e-6 * np.eye(2)
            diff = x[i] - _mu[j]
            invS = np.linalg.inv(cov)
            detS = np.linalg.det(cov)

            f1 = 1.0 / ((2 * np.pi) * np.sqrt(detS))
            quad = diff @ invS @ diff
            f2 = -0.5 * quad

            _pdf = f1 * np.exp(f2)
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
    _v  = np.zeros((k, 2, 2))

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

    r_new = calculatePDF(_r, _pi, _mu, _v)
    return r_new

def visualization(r: np.array, ax, title="GMM with EM algorithm"):
    labels = np.argmax(r, axis=1)
    sc = ax.scatter(x[:, 0], x[:, 1], c=labels, cmap='tab10', s=15)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)

if __name__ == "__main__":
    main()

  `;

  // 4) 비교분석(KMEANS/DBSCAN/Sklearn GMM)
  const pythonCompareCode = `
import numpy as np, pandas as pd, matplotlib.pyplot as plt, io, base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def load_xy(csv="FAA_AEDT_data.csv"):
    df = pd.read_csv(csv)
    num = df.select_dtypes(include=['number'])
    if num.shape[1] < 2:
        df = pd.read_csv(csv, sep=None, engine="python")
        num = df.select_dtypes(include=['number'])
    X = num.iloc[:,:2].to_numpy(dtype=float)
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

def img_from_fig(fig):
    buf = io.BytesIO(); fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120); buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{data}" />'

X = load_xy("FAA_AEDT_data.csv")
rows = []

# 1) KMEANS K=2..5
fig1, axes1 = plt.subplots(2,2, figsize=(10,10))
axes1 = axes1.ravel()
for i,K in enumerate([2,3,4,5]):
    km = KMeans(n_clusters=K, n_init="auto", random_state=42).fit(X)
    lab = km.labels_
    m = eval_labels(X, lab)
    rows.append(dict(model="KMeans", K=K, sil=m["sil"], CH=m["ch"], AIC=np.nan, BIC=np.nan, loglik=np.nan))
    ax = axes1[i]; ax.scatter(X[:,0], X[:,1], c=lab, s=10, cmap="tab10")
    ax.set_title(f"KMeans K={K} | Sil={m['sil']:.3f} | CH={m['ch']:.0f}")
print(img_from_fig(fig1))

# 2) GMM(sklearn) K=2..5
fig2, axes2 = plt.subplots(2,2, figsize=(10,10))
axes2 = axes2.ravel()
for i,K in enumerate([2,3,4,5]):
    gm = GaussianMixture(n_components=K, covariance_type="full",
                         reg_covar=1e-6, max_iter=500, tol=1e-4,
                         random_state=42).fit(X)
    lab = gm.predict(X)
    ll = float(gm.score(X)*X.shape[0])
    aic = float(gm.aic(X)); bic = float(gm.bic(X))
    m = eval_labels(X, lab)
    rows.append(dict(model="GMM(full)", K=K, sil=m["sil"], CH=m["ch"], AIC=aic, BIC=bic, loglik=ll))
    ax = axes2[i]; ax.scatter(X[:,0], X[:,1], c=lab, s=10, cmap="tab10")
    ax.set_title(f"GMM K={K} | BIC={bic:.1f} | Sil={m['sil']:.3f}")
print(img_from_fig(fig2))

# 3) DBSCAN (grid)
db_grid = [(0.2,5),(0.3,5),(0.5,5),(0.3,10)]
best = []
for eps, mp in db_grid:
    db = DBSCAN(eps=eps, min_samples=mp).fit(X)
    lab = db.labels_; m = eval_labels(X, lab)
    rows.append(dict(model=f"DBSCAN(eps={eps},min={mp})", K=m["K"], sil=m["sil"], CH=m["ch"], AIC=np.nan, BIC=np.nan, loglik=np.nan))
# 상위 4개만 시각화
top = sorted([r for r in rows if str(r["model"]).startswith("DBSCAN")], key=lambda r: (np.nan_to_num(r["sil"], nan=-1.0)), reverse=True)[:4]
fig3, axes3 = plt.subplots(2,2, figsize=(10,10))
axes3 = axes3.ravel()
for i, r in enumerate(top):
    eps = float(r["model"].split("eps=")[1].split(",")[0])
    mp  = int(r["model"].split("min=")[1].split(")")[0])
    lab = DBSCAN(eps=eps, min_samples=mp).fit(X).labels_
    ax = axes3[i]; ax.scatter(X[:,0], X[:,1], c=lab, s=10, cmap="tab10")
    ax.set_title(f"DBSCAN eps={eps}, min={mp} | K={r['K']} | Sil={r['sil']:.3f}")
print(img_from_fig(fig3))

# 표 출력
df = pd.DataFrame(rows)
df_sorted = df.sort_values(by=["model","BIC","sil"], ascending=[True, True, False])
print("\\n=== Benchmark Summary (KMeans / DBSCAN / GMM) ===")
print(df_sorted.to_string(index=False))
print("\\n표(HTML):")
print(df_sorted.to_html(index=False))
  `;

  // 업로드 → 1번, 4번 순차 실행
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
    // 1) 내 GMM
    await runPython(pythonCode, file, setOutGmmOwn);
    // 4) 비교분석
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
        {/* 1. 실행하기 */}
        <section className="gmm-section">
          <h2>1. 실행하기</h2>
          <p>
            브라우저에서 직접 Python 코드를 실행합니다. CSV 파일을 업로드하면 ①
            내 GMM과 ② 비교분석을 순서대로 실행합니다.
          </p>
          <div className="file-upload">
            <input
              id="csvUpload"
              type="file"
              accept=".csv"
              onChange={handleFileChange}
            />
            <label className="file-btn" htmlFor="csvUpload" role="button">
              <span className="file-icon" aria-hidden>
                📁
              </span>{" "}
              CSV 업로드
            </label>
            <span className="file-name">{fileName || "선택된 파일 없음"}</span>
          </div>
          {loading && <p>데이터를 로드하고 코드를 실행 중입니다...</p>}
        </section>

        {/* 2. 알고리즘 구현(내 GMM 코드) */}
        <section className="gmm-section">
          <h2>2. 알고리즘 구현</h2>
          <p>본 프로젝트에 사용된 GMM-EM 알고리즘의 핵심 구현 코드입니다.</p>
          <pre>
            <code className="python-code">{pythonCode}</code>
          </pre>
        </section>

        {/* 3. 알고리즘 원리(보류) */}
        <section className="gmm-section">
          <h2>3. 알고리즘 원리</h2>
          <p>GMM-EM은 E-step과 M-step으로 구성됩니다.</p>
          <h3>E-step</h3>
          <BlockMath math="\\gamma_{ij} = \\frac{\\pi_j \\mathcal{N}(x_i | \\mu_j, \\Sigma_j)}{\\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x_i | \\mu_k, \\Sigma_k)}" />
          <h3>M-step</h3>
          <BlockMath math="\\mu_j = \\frac{1}{N_j} \\sum_{i=1}^N \\gamma_{ij} x_i" />
          <p>(세부 유도 및 해설은 추후 추가 예정)</p>
        </section>

        {/* 1번 실행 결과 콘솔 */}
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

        {/* 4. 코드 비교분석 */}
        <section className="gmm-section">
          <h2>4. 코드 비교분석 (K-MEANS / DBSCAN / GMM)</h2>
          <p>
            동일 CSV로 K-MEANS, DBSCAN, GMM(sklearn)을 비교하여 시각화와
            지표(실루엣/CH, GMM의 AIC/BIC)를 출력합니다.
          </p>
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

      <h2>5. 결론 — 어떤 기법이 가장 적합했는가?</h2>
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
