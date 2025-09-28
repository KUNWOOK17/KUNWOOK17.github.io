// // import React from 'react';
// // import './GmmEmAssignmentPage.scss';

// // function GmmEmAssignmentPage({ changePage }) {
// //   return (
// //     <div className="gmm-page-container">
// //       <div className="gmm-header">
// //         <h1 className="gmm-title">Gaussian Mixture Model (GMM) 과제</h1>
// //         <p className="gmm-subtitle">
// //           기댓값 최대화 (EM) 알고리즘을 활용한 항공 데이터 클러스터링
// //         </p>
// //       </div>

// //       <div className="gmm-content">
// //         <section className="gmm-section">
// //           <h2>1. 서론: 프로젝트 개요</h2>
// //           <p>
// //             이 프로젝트는 GMM-EM 알고리즘을 직접 구현하여 항공 데이터를 클러스터링하는 것을 목표로 합니다.
// //             제공된 FAA A-EDT 데이터셋은 ...
// //           </p>
// //         </section>

// //         <section className="gmm-section">
// //           <h2>2. 데이터 분석</h2>
// //           <p>
// //             데이터는 1000개의 데이터 포인트와 2개의 차원으로 구성되어 있으며, 초기 산점도는 두 개의 뚜렷한 군집을 보여줍니다.
// //           </p>
// //           {/* 그래프 이미지 추가 */}
// //         </section>

// //         <section className="gmm-section">
// //           <h2>3. 알고리즘 구현</h2>
// //           <p>
// //             GMM-EM 알고리즘은 크게 기댓값(E) 단계와 최대화(M) 단계로 나뉩니다. 각 단계의 원리는 ...
// //           </p>
// //           <h3>E-step (기댓값 단계)</h3>
// //           <p>
// //             각 데이터 포인트가 특정 클러스터에 속할 확률을 계산합니다.
// //             수식: gamma_ij = (pi_j * N(x_i | mu_j, Sigma_j)) / sum_k(pi_k * N(x_i | mu_k, Sigma_k))
// //           </p>
// //           <h3>M-step (최대화 단계)</h3>
// //           <p>
// //             계산된 확률을 이용해 클러스터의 매개변수를 업데이트합니다.
// //             수식: mu_j = (1/N_j) * sum_i(gamma_ij * x_i)
// //           </p>
// //         </section>
        
// //         <section className="gmm-section">
// //           <h2>4. 클러스터링 결과 시각화 (K=2, 3, 4, 5)</h2>
// //           <p>
// //             다양한 K 값에 대한 클러스터링 결과를 시각화하여 알고리즘의 성능을 확인했습니다.
// //           </p>
// //         </section>
        
// //         <section className="gmm-section">
// //           <h2>5. 최적의 K 값 선정</h2>
// //           <p>
// //             AIC와 BIC 지표를 사용하여 K 값에 대한 모델 적합성을 평가했습니다. 그 결과 ...
// //           </p>
// //         </section>

// //       </div>

// //       <button className="back-button" onClick={() => changePage("main")}>
// //         홈으로 돌아가기
// //       </button>
// //     </div>
// //   );
// // }

// // export default GmmEmAssignmentPage;
// import React, { useState, useEffect } from 'react';
// import './GmmEmAssignmentPage.scss';
// import { BlockMath } from 'react-katex';
// import 'katex/dist/katex.min.css';

// // Pyodide를 사용하기 위해 CDN에서 스크립트를 로드합니다.
// const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js';

// function GmmEmAssignmentPage({ changePage }) {
//   const [pyodide, setPyodide] = useState(null);
//   const [output, setOutput] = useState('');
//   const [loading, setLoading] = useState(true);

//   // Pyodide 로드
//   useEffect(() => {
//     async function loadPyodide() {
//       const pyodide = await window.loadPyodide();
//       await pyodide.loadPackage(['numpy', 'pandas', 'scipy', 'matplotlib']);
//       setPyodide(pyodide);
//       setLoading(false);
//     }
//     if (!window.pyodide) {
//       const script = document.createElement('script');
//       script.src = PYODIDE_URL;
//       script.onload = loadPyodide;
//       document.head.appendChild(script);
//     } else {
//       loadPyodide();
//     }
//   }, []);

//   const runCode = async (code, file) => {
//     if (!pyodide) {
//       setOutput('Python 환경을 불러오는 중입니다. 잠시만 기다려주세요.');
//       return;
//     }
//     setLoading(true);
//     setOutput('코드를 실행 중입니다...');

//     try {
//       // 파일 처리: Pyodide 파일 시스템에 업로드
//       if (file) {
//         const fileContent = await file.arrayBuffer();
//         pyodide.FS.writeFile('FAA_AEDT_data.csv', new Uint8Array(fileContent));
//       }

//       // 표준 출력(console.log)을 캡처
//       const capturedOutput = [];
//       const originalLog = console.log;
//       console.log = (...args) => capturedOutput.push(args.join(' '));

//       // Python 코드 실행
//       await pyodide.runPythonAsync(code);

//       console.log = originalLog; // 원본으로 복원
//       setOutput(capturedOutput.join('\n'));
//     } catch (error) {
//       setOutput(`오류 발생:\n${error.message}`);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const pythonCode = `
// import numpy as np
// import pandas as pd
// import matplotlib.pyplot as plt
// import io
// import base64

// def load_csv_2d(path):
//     df = pd.read_csv(path)
//     num_df = df.select_dtypes(include=['number'])
//     if num_df.shape[1] < 2:
//         df = pd.read_csv(path, sep=None, engine="python")
//         num_df = df.select_dtypes(include=['number'])
//     if num_df.shape[1] < 2:
//         raise ValueError("CSV에 숫자형 컬럼이 2개 이상 필요합니다.")
//     X = num_df.iloc[:, :2].to_numpy(dtype=float)
//     return X

// def gaussian_pdf(X, mean, cov):
//     D = X.shape[1]
//     L = np.linalg.cholesky(cov)
//     solve = np.linalg.solve(L, (X - mean).T)
//     quad = np.sum(solve**2, axis=0)
//     log_det = 2.0 * np.sum(np.log(np.diag(L)))
//     log_norm = -0.5 * (D * np.log(2 * np.pi) + log_det + quad)
//     return np.exp(log_norm)

// def log_likelihood(X, pis, mus, covs):
//     N, K = X.shape[0], mus.shape[0]
//     comp = np.zeros((N, K))
//     for k in range(K):
//         comp[:, k] = pis[k] * gaussian_pdf(X, mus[k], covs[k])
//     s = np.sum(comp, axis=1) + 1e-300
//     return np.sum(np.log(s))

// def e_step(X, pis, mus, covs):
//     N, K = X.shape[0], mus.shape[0]
//     resp = np.zeros((N, K))
//     for k in range(K):
//         resp[:, k] = pis[k] * gaussian_pdf(X, mus[k], covs[k])
//     resp_sum = resp.sum(axis=1, keepdims=True) + 1e-300
//     return resp / resp_sum

// def m_step(X, gamma, reg_covar=1e-6):
//     N, D = X.shape
//     K = gamma.shape[1]
//     Nk = gamma.sum(axis=0) + 1e-300
//     pis = Nk / N
//     mus = (gamma.T @ X) / Nk[:, None]
//     covs = np.zeros((K, D, D))
//     for k in range(K):
//         Xc = X - mus[k]
//         covs[k] = (Xc.T * gamma[:, k]) @ Xc / Nk[k]
//         covs[k].flat[:: D + 1] += reg_covar
//     return pis, mus, covs

// def gmm_em(X, K, max_iter=300, tol=1e-6, reg_covar=1e-6, seed=7, verbose=False):
//     N, D = X.shape
//     rng = np.random.default_rng(seed)
//     mus = X[rng.choice(N, size=K, replace=False)]
//     covs = np.array([np.cov(X.T) + reg_covar*np.eye(D) for _ in range(K)])
//     pis = np.ones(K) / K
//     ll_hist = []
//     for it in range(max_iter):
//         gamma = e_step(X, pis, mus, covs)
//         pis, mus, covs = m_step(X, gamma, reg_covar=reg_covar)
//         ll = log_likelihood(X, pis, mus, covs)
//         ll_hist.append(ll)
//         if verbose:
//             print(f"[K={K}] iter {it+1} ll={ll:.4f}")
//         if it > 0 and abs(ll_hist[-1]-ll_hist[-2]) < tol:
//             break
//     labels = np.argmax(gamma, axis=1)
//     return {"pis": pis, "mus": mus, "covs": covs, "labels": labels,
//             "ll_hist": ll_hist, "n_iter": len(ll_hist)}

// def aic_bic(X, model):
//     N, D = X.shape
//     K = model["mus"].shape[0]
//     n_params = K*D + K*(D*(D+1)//2) + (K-1)
//     ll = model["ll_hist"][-1]
//     AIC = 2*n_params - 2*ll
//     BIC = n_params*np.log(N) - 2*ll
//     return AIC, BIC

// # --- 그래프를 HTML에 표시하기 위한 함수 ---
// def plot_clusters(X, labels, mus, title):
//     fig, ax = plt.subplots(figsize=(5,4.3), dpi=120)
//     K = mus.shape[0]
//     for k in range(K):
//         m = labels==k
//         ax.scatter(X[m,0], X[m,1], s=10, alpha=0.7, label=f"cluster {k}")
//     ax.scatter(mus[:,0], mus[:,1], s=120, marker="X", edgecolor="k", linewidths=1, label="means")
//     ax.set_title(title)
//     ax.set_xlabel("X1")
//     ax.set_ylabel("X2")
//     ax.legend(fontsize=8)
//     fig.tight_layout()
//     buf = io.BytesIO()
//     fig.savefig(buf, format="png")
//     buf.seek(0)
//     img_data = base64.b64encode(buf.getvalue()).decode()
//     plt.close(fig)
//     return f'<img src="data:image/png;base64,{img_data}" />'

// def plot_ll(ll_hist, title):
//     fig, ax = plt.subplots(figsize=(4.6,3.6), dpi=120)
//     ax.plot(ll_hist, lw=2)
//     ax.set_xlabel("iteration")
//     ax.set_ylabel("log-likelihood")
//     ax.set_title(title)
//     fig.tight_layout()
//     buf = io.BytesIO()
//     fig.savefig(buf, format="png")
//     buf.seek(0)
//     img_data = base64.b64encode(buf.getvalue()).decode()
//     plt.close(fig)
//     return f'<img src="data:image/png;base64,{img_data}" />'

// # --- 메인 실행 로직 ---
// try:
//     X = load_csv_2d('FAA_AEDT_data.csv')
//     print("CSV 파일을 성공적으로 불러왔습니다.")
    
//     results = {}
//     for K in [2,3,4,5]:
//         model = gmm_em(X, K, max_iter=300, tol=1e-6, reg_covar=1e-6, seed=7)
//         AIC, BIC = aic_bic(X, model)
//         results[K] = {"model": model, "AIC": AIC, "BIC": BIC}
        
//         # 플롯을 생성하고 HTML 문자열로 변환
//         plot_html = plot_clusters(X, model["labels"], model["mus"], f"GMM-EM (K={K})")
//         log_ll_html = plot_ll(model["ll_hist"], f"Log-likelihood (K={K}, iters={model['n_iter']})")
//         print(plot_html)
//         print(log_ll_html)
//         print(f"K={K}: iters={model['n_iter']}, ll={model['ll_hist'][-1]:.3f}, AIC={AIC:.1f}, BIC={BIC:.1f}")

//     print("\\n=== Model selection (lower is better) ===")
//     for K in results:
//         print(f"K={K}: AIC={results[K]['AIC']:.1f} | BIC={results[K]['BIC']:.1f}")
//     best_aic = min(results, key=lambda k: results[k]['AIC'])
//     best_bic = min(results, key=lambda k: results[k]['BIC'])
//     print(f"\\nBest by AIC: K={best_aic} | Best by BIC: K={best_bic}")
    
// except Exception as e:
//     print(f"오류가 발생했습니다: {e}")
//   `;

//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       runCode(pythonCode, file);
//     }
//   };

//   return (
//     <div className="gmm-page-container">
//       <div className="gmm-header">
//         <h1 className="gmm-title">Gaussian Mixture Model (GMM) 과제</h1>
//         <p className="gmm-subtitle">
//           기댓값 최대화 (EM) 알고리즘을 활용한 항공 데이터 클러스터링
//         </p>
//       </div>

//       <div className="gmm-content">
//         <section className="gmm-section">
//           <h2>1. 실행하기</h2>
//           <p>
//             Pyodide를 이용하여 브라우저에서 직접 Python 코드를 실행합니다.
//             아래 버튼을 눌러 과제에 필요한 CSV 파일을 업로드해주세요.
//           </p>
//           <input type="file" onChange={handleFileChange} />
//           {loading && <p>데이터를 로드하고 코드를 실행 중입니다. 잠시만 기다려주세요...</p>}
//           <pre>
//             <code className="console-output">
//               {output || "코드를 실행하려면 CSV 파일을 업로드해주세요."}
//             </code>
//           </pre>
//         </section>

//         <section className="gmm-section">
//           <h2>2. 알고리즘 구현</h2>
//           <p>
//             본 프로젝트에 사용된 GMM-EM 알고리즘의 핵심 구현 코드는 다음과 같습니다.
//             <br />
//             (클러스터 초기화는 Pyodide 환경에 맞게 랜덤 방식으로 변경되었습니다.)
//           </p>
//           <pre>
//             <code className="python-code">
//               {pythonCode}
//             </code>
//           </pre>
//         </section>

//         <section className="gmm-section">
//           <h2>3. 알고리즘 원리</h2>
//           <p>
//             GMM-EM 알고리즘은 크게 기댓값(E) 단계와 최대화(M) 단계로 나뉩니다.
//           </p>
//           <h3>E-step (기댓값 단계)</h3>
//           <p>
//             각 데이터 포인트가 특정 클러스터에 속할 확률을 계산합니다.
//           </p>
//           <BlockMath math="\gamma_{ij} = \frac{\pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}{\sum_{k=1}^K \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}" />
//           <h3>M-step (최대화 단계)</h3>
//           <p>
//             계산된 확률을 이용해 클러스터의 매개변수를 업데이트합니다.
//           </p>
//           <BlockMath math="\mu_j = \frac{1}{N_j} \sum_{i=1}^N \gamma_{ij} x_i" />
//         </section>

//       </div>

//       <button className="back-button" onClick={() => changePage("main")}>
//         홈으로 돌아가기
//       </button>
//     </div>
//   );
// }

// export default GmmEmAssignmentPage;

import React, { useState, useEffect } from 'react';
import './GmmEmAssignmentPage.scss';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Pyodide를 사용하기 위해 CDN에서 스크립트를 로드합니다.
const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js';

function GmmEmAssignmentPage({ changePage }) {
  const [pyodide, setPyodide] = useState(null);
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(true);

  // Pyodide 로드
  useEffect(() => {
    async function loadPyodide() {
      const pyodide = await window.loadPyodide();
      await pyodide.loadPackage(['numpy', 'pandas', 'scipy', 'matplotlib']);
      setPyodide(pyodide);
      setLoading(false);
    }
    if (!window.pyodide) {
      const script = document.createElement('script');
      script.src = PYODIDE_URL;
      script.onload = loadPyodide;
      document.head.appendChild(script);
    } else {
      loadPyodide();
    }
  }, []);

  const runCode = async (code, file) => {
    if (!pyodide) {
      setOutput('Python 환경을 불러오는 중입니다. 잠시만 기다려주세요.');
      return;
    }
    setLoading(true);
    setOutput('코드를 실행 중입니다...');

    try {
      // 파일 처리: Pyodide 파일 시스템에 업로드
      if (file) {
        const fileContent = await file.arrayBuffer();
        pyodide.FS.writeFile('FAA_AEDT_data.csv', new Uint8Array(fileContent));
      }

      // 표준 출력(console.log)을 캡처
      const capturedOutput = [];
      const originalLog = console.log;
      console.log = (...args) => capturedOutput.push(args.join(' '));

      // Python 코드 실행
      await pyodide.runPythonAsync(code);

      console.log = originalLog; // 원본으로 복원
      setOutput(capturedOutput.join('\n'));
    } catch (error) {
      setOutput(`오류 발생:\n${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const pythonCode = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def load_csv_2d(path):
    df = pd.read_csv(path)
    num_df = df.select_dtypes(include=['number'])
    if num_df.shape[1] < 2:
        df = pd.read_csv(path, sep=None, engine="python")
        num_df = df.select_dtypes(include=['number'])
    if num_df.shape[1] < 2:
        raise ValueError("CSV에 숫자형 컬럼이 2개 이상 필요합니다.")
    X = num_df.iloc[:, :2].to_numpy(dtype=float)
    return X

def gaussian_pdf(X, mean, cov):
    D = X.shape[1]
    L = np.linalg.cholesky(cov)
    solve = np.linalg.solve(L, (X - mean).T)
    quad = np.sum(solve**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    log_norm = -0.5 * (D * np.log(2 * np.pi) + log_det + quad)
    return np.exp(log_norm)

def log_likelihood(X, pis, mus, covs):
    N, K = X.shape[0], mus.shape[0]
    comp = np.zeros((N, K))
    for k in range(K):
        comp[:, k] = pis[k] * gaussian_pdf(X, mus[k], covs[k])
    s = np.sum(comp, axis=1) + 1e-300
    return np.sum(np.log(s))

def e_step(X, pis, mus, covs):
    N, K = X.shape[0], mus.shape[0]
    resp = np.zeros((N, K))
    for k in range(K):
        resp[:, k] = pis[k] * gaussian_pdf(X, mus[k], covs[k])
    resp_sum = resp.sum(axis=1, keepdims=True) + 1e-300
    return resp / resp_sum

def m_step(X, gamma, reg_covar=1e-6):
    N, D = X.shape
    K = gamma.shape[1]
    Nk = gamma.sum(axis=0) + 1e-300
    pis = Nk / N
    mus = (gamma.T @ X) / Nk[:, None]
    covs = np.zeros((K, D, D))
    for k in range(K):
        Xc = X - mus[k]
        covs[k] = (Xc.T * gamma[:, k]) @ Xc / Nk[k]
        covs[k].flat[:: D + 1] += reg_covar
    return pis, mus, covs

def gmm_em(X, K, max_iter=300, tol=1e-6, reg_covar=1e-6, seed=7, verbose=False):
    N, D = X.shape
    rng = np.random.default_rng(seed)
    mus = X[rng.choice(N, size=K, replace=False)]
    covs = np.array([np.cov(X.T) + reg_covar*np.eye(D) for _ in range(K)])
    pis = np.ones(K) / K
    ll_hist = []
    for it in range(max_iter):
        gamma = e_step(X, pis, mus, covs)
        pis, mus, covs = m_step(X, gamma, reg_covar=reg_covar)
        ll = log_likelihood(X, pis, mus, covs)
        ll_hist.append(ll)
        if verbose:
            print(f"[K={K}] iter {it+1} ll={ll:.4f}")
        if it > 0 and abs(ll_hist[-1]-ll_hist[-2]) < tol:
            break
    labels = np.argmax(gamma, axis=1)
    return {"pis": pis, "mus": mus, "covs": covs, "labels": labels,
            "ll_hist": ll_hist, "n_iter": len(ll_hist)}

def aic_bic(X, model):
    N, D = X.shape
    K = model["mus"].shape[0]
    n_params = K*D + K*(D*(D+1)//2) + (K-1)
    ll = model["ll_hist"][-1]
    AIC = 2*n_params - 2*ll
    BIC = n_params*np.log(N) - 2*ll
    return AIC, BIC

# --- 그래프를 HTML에 표시하기 위한 함수 ---
def plot_clusters(X, labels, mus, title):
    fig, ax = plt.subplots(figsize=(5,4.3), dpi=120)
    K = mus.shape[0]
    for k in range(K):
        m = labels==k
        ax.scatter(X[m,0], X[m,1], s=10, alpha=0.7, label=f"cluster {k}")
    ax.scatter(mus[:,0], mus[:,1], s=120, marker="X", edgecolor="k", linewidths=1, label="means")
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_data}" />'

def plot_ll(ll_hist, title):
    fig, ax = plt.subplots(figsize=(4.6,3.6), dpi=120)
    ax.plot(ll_hist, lw=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_data}" />'

# --- 메인 실행 로직 ---
try:
    X = load_csv_2d('FAA_AEDT_data.csv')
    print("CSV 파일을 성공적으로 불러왔습니다.")
    
    results = {}
    for K in [2,3,4,5]:
        model = gmm_em(X, K, max_iter=300, tol=1e-6, reg_covar=1e-6, seed=7)
        AIC, BIC = aic_bic(X, model)
        results[K] = {"model": model, "AIC": AIC, "BIC": BIC}
        
        # 플롯을 생성하고 HTML 문자열로 변환
        plot_html = plot_clusters(X, model["labels"], model["mus"], f"GMM-EM (K={K})")
        log_ll_html = plot_ll(model["ll_hist"], f"Log-likelihood (K={K}, iters={model['n_iter']})")
        print(plot_html)
        print(log_ll_html)
        print(f"K={K}: iters={model['n_iter']}, ll={model['ll_hist'][-1]:.3f}, AIC={AIC:.1f}, BIC={BIC:.1f}")

    print("\\n=== Model selection (lower is better) ===")
    for K in results:
        print(f"K={K}: AIC={results[K]['AIC']:.1f} | BIC={results[K]['BIC']:.1f}")
    best_aic = min(results, key=lambda k: results[k]['AIC'])
    best_bic = min(results, key=lambda k: results[k]['BIC'])
    print(f"\\nBest by AIC: K={best_aic} | Best by BIC: K={best_bic}")
    
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
  `;

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      runCode(pythonCode, file);
    }
  };

  return (
    <div className="gmm-page-container">
      <div className="gmm-header">
        <h1 className="gmm-title">Gaussian Mixture Model (GMM) 과제</h1>
        <p className="gmm-subtitle">
          기댓값 최대화 (EM) 알고리즘을 활용한 항공 데이터 클러스터링
        </p>
      </div>

      <div className="gmm-content">
        <section className="gmm-section">
          <h2>1. 실행하기</h2>
          <p>
            Pyodide를 이용하여 브라우저에서 직접 Python 코드를 실행합니다.
            아래 버튼을 눌러 과제에 필요한 CSV 파일을 업로드해주세요.
          </p>
          <input type="file" onChange={handleFileChange} />
          {loading && <p>데이터를 로드하고 코드를 실행 중입니다. 잠시만 기다려주세요...</p>}
          <pre>
            <code className="console-output" dangerouslySetInnerHTML={{ __html: output || "코드를 실행하려면 CSV 파일을 업로드해주세요." }}>
            </code>
          </pre>
        </section>

        <section className="gmm-section">
          <h2>2. 알고리즘 구현</h2>
          <p>
            본 프로젝트에 사용된 GMM-EM 알고리즘의 핵심 구현 코드는 다음과 같습니다.
            <br />
            (클러스터 초기화는 Pyodide 환경에 맞게 랜덤 방식으로 변경되었습니다.)
          </p>
          <pre>
            <code className="python-code">
              {pythonCode}
            </code>
          </pre>
        </section>

        <section className="gmm-section">
          <h2>3. 알고리즘 원리</h2>
          <p>
            GMM-EM 알고리즘은 크게 기댓값(E) 단계와 최대화(M) 단계로 나뉩니다.
          </p>
          <h3>E-step (기댓값 단계)</h3>
          <p>
            각 데이터 포인트가 특정 클러스터에 속할 확률을 계산합니다.
          </p>
          <BlockMath math="\gamma_{ij} = \frac{\pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}{\sum_{k=1}^K \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}" />
          <h3>M-step (최대화 단계)</h3>
          <p>
            계산된 확률을 이용해 클러스터의 매개변수를 업데이트합니다.
          </p>
          <BlockMath math="\mu_j = \frac{1}{N_j} \sum_{i=1}^N \gamma_{ij} x_i" />
        </section>

      </div>

      <button className="back-button" onClick={() => changePage("main")}>
        홈으로 돌아가기
      </button>
    </div>
  );
}

export default GmmEmAssignmentPage;