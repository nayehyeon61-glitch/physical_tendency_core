# `G(x) + 기억`에 `∫p log p`를 입력하는 실험

2026-11-16. 이 문서는 엔트로피를 **손실에만 추가하는 실험과 구분하여**, 조건부 분포에서
계산한 엔트로피 특징을 기억·경향 네트워크에 실제로 입력하는 가설을 정의한다.

## 어떤 분포의 엔트로피인가

관측 가능한 과거를 `H_t`라 하고, 정규화된 하루 상태의 미해결 증분을 `r_j`라 한다.
첫 모델은 사영 전 조건부 Gaussian `p_j(r_j | H_t)=N(0, σ_j²)`이다.
`σ_ref,j`는 train 자료에서만 정한 같은 좌표 단위의 기준 scale이다. 그러면

\[
\int p_j\log p_j\,dr_j=-\tfrac12\log(2\pi e\sigma_j^2),\qquad
e_j=\int p_j\log p_j\,dr_j-\int p_{\mathrm{ref},j}\log p_{\mathrm{ref},j}\,dr_j
=-\log(\sigma_j/\sigma_{\mathrm{ref},j}).
\]

따라서 `e_j`는 **기준을 뺀 음의 미분 엔트로피**이며, Gaussian에서는 log-scale 특징과
정확히 같다. 이것은 `KL(p || p_ref)`가 아니다. 조건부 대각 성분이 독립일 때만 성분의
엔트로피 합이 공동 엔트로피가 된다. 유색 모델에서는 정상 주변 scale이 아니라 실제
조건부 innovation의 표준편차를 사용해야 한다. 단위·좌표가 바뀌면 원시 미분 엔트로피도
바뀌며, 여기서의 값은 대기의 열역학적 엔트로피가 아니다.

## 순환 정의를 피한 실제 입력 순서

\[
\begin{aligned}
m_t^-&=\mathcal M_\theta(m_{t-1},X_t,G(X_t)),\\
\sigma_t&=s_\theta(X_t,G(X_t),m_t^-)>0,\qquad
e_t=-\log(\sigma_t/\sigma_{\rm ref}),\\
m_t^+&=m_t^-+W_e e_t,\\
\mu_{t+1}&=\operatorname{Solver}[f_\theta(X_t,G(X_t),m_t^+)],\\
X_{t+1}&=\mathcal P(\mu_{t+1}+\sigma_t\epsilon_t),\quad\epsilon_t\sim N(0,I).
\end{aligned}
\]

한 step의 `σ_t`는 entropy feedback **전**의 기억에서 한 번 계산하고, 그 값을 실제 샘플링과
분포 점수에도 사용한다. feedback 후 다시 `σ`를 계산하여 같은 step의 `e`로 되먹이지 않는다.
Gaussian의 엔트로피는 평균에 의존하지 않으므로 feedback으로 `μ`가 바뀌어도 위 정의가
일관된다. warmup도 같은 인과 순서를 따르고 미래 정답·숨은 생성 상태를 입력하지 않는다.
`W_e`는 0으로 초기화한다. 대조군 `entropy_off`도 같은 head·projection과 초기 가중치를
가지되 입력을 0으로 하여 용량·초기 조건을 맞춘다.

이는 새로운 독립 관측량을 추가하는 것이 아니라, 모델이 추정한 불확실성을 자신의 기억에
되먹이는 **학습된 특징 경로**다. 개선되어도 엔트로피가 물리적으로 필수이거나 노이즈와의
유일한 법칙을 발견했다는 결론은 나오지 않는다. 미래 정답과의 NLL/CRPS로 분포를 학습하며,
원시 엔트로피를 무조건 최대화하지 않는다. [Gneiting & Raftery (2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
학습 후 `e=0`으로 바꾸어 성능이 떨어져도 그것은 경로 의존성을 보여줄 뿐이다. 특히 거의
상수인 `e`는 기억의 bias처럼 쓰일 수 있다. 재학습한 off 대조와 비교해야 하며, 적응적인
불확실성 정보의 효과를 분리하려면 상수 `e`로 재학습한 추가 대조가 필요하다.

## 물리적 확산과 연결되는 엄밀한 범위

상수 `D`에 대해 `dX=b(X)dt+sqrt(2D)dW`의 밀도는

\[
\partial_t p=-\nabla\cdot(bp)+D\Delta p,
\qquad D\Delta p=D\nabla\cdot(p\nabla\log p).
\]

특히 `b=-∇U`일 때 `F[p]=E_p[U]+D∫p log p`의 Wasserstein gradient flow로 해석할 수 있다.
이 관계는 밀도의 확산과 자유에너지 사이의 관계다. [Jordan, Kinderlehrer & Otto (1998)](https://epubs.siam.org/doi/10.1137/S0036141096303359).
확률 흐름 속도 `b-D∇log p`에 같은 Brownian 확산을 다시 더하면 확산을 중복 계산할 수 있다.
따라서 엔트로피 gradient와 노이즈를 계수 검토 없이 동시에 물리 경향에 추가하지 않는다.
현재 `G`에는 이류·코리올리 같은 비-gradient 힘도 있으므로 전체를 `-∇U`로 가정하지 않는다.

기억과 상관된 노이즈를 함께 쓰는 동기는 GLE/Mori–Zwanzig와 연결되지만, 기상 자료에서
평형 fluctuation–dissipation 관계가 성립한다고 강제하지 않는다. [Lei, Baker & Li (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5167214/).
물리 사영 `P` 후 분포는 일반적으로 Gaussian이 아니므로 analytic entropy/NLL은 사영 전
법칙에 한정한다. 합성 CPU 대조는 이 입력 경로의 기전 검증이며 ERA5 장기 예보 개선,
열역학적 타당성 또는 클라우드 실행 완료의 증거가 아니다.
