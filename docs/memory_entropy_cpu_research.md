# 물리 경향 + 기억 + 확률적 강제력: CPU 연구

기준: `physical_tendency_core/main`의 `22ed5fd`, 원본 작업 브랜치
`claude/project-thread-wjjayb`의 `bbd9b50`. 2026-11-16 작성.

## 먼저 바로잡을 현재 구조

현재 모델은 물리식 자체를 그대로 적분하는 `G(x)` 모델이 아니다.
물리 연산자의 경향 `G(x)`를 입력 특징으로 받는 **학습된 경향**을 Heun으로 적분한다.
`pgcfm/dyncore.py`의 `DynCore` / `OperatorFeatures`,
`pgcfm/kai_tend.py`의 `make_core` / `MemCore`가 구현이다.

- `--core-gfeat`: 현재 상태와 함께 48개의 물리 경향 특징을 입력.
- `--core-mem 8 --core-mz`: ConvGRU 기억, 과거 예보의 innovation, 과거 구간의 sequence loss.
- `--core-acc 3 5`: 고정 지수 커널로 누적한 기억을 경향의 추가 항으로 사용.
- `--core-pacc 3 5`: 물리 경향·관측 추세·innovation의 누적을 기억의 입력에 포함.
- `--core-kern 8`: 기억 상태의 시차별 학습 커널.

따라서 기억을 새로 발명하는 실험이 아니라 **기존 기억과 확률적 closure를 결합하는 실험**이다.
`docs/kai_tendency.md`에는 장기 기억의 변형들이 기존 8일 기억 기준을 넘지 못한 결과도 있다.
기억을 길게 하거나 노이즈를 추가하는 것만으로 개선을 가정하지 않는다.
별도 `claude/project-thread-f8ks9d`에는 노이즈·CRPS·제어에너지 벌점 실험이 있지만,
그 브랜치의 일부 옵션은 기억 core와 동시 사용을 막는다. 이를 main 전체에 바로 병합하지 않는다.

## 제안의 수학적 해석

현재 구조를 유지한 확장은 다음과 같다.

\[
 dX_t=f_\theta(X_t,G(X_t),m_t)dt+\eta_tdt,\qquad
 \dot m_t=\mathcal M_\theta(m_t,X_t,G(X_t)).
\]

Mori–Zwanzig/GLE 관점에서 제거된 자유도는 기억 적분과 확률적 강제력을 함께 남길 수 있다.
지수 커널의 경우 `dm/dt=(phi(X)-m)/tau`라는 보조 상태로 적분을 계산할 수 있다.
이는 정확한 MZ 분해를 식별했다는 뜻은 아니다.
[Lei, Baker & Li (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5167214/).

백색·유색 노이즈는 다른 모델이다. 첫 비교는 하루 단위의 확률적 **증분**이고,
유색 모델은 시차 1의 AR 상태를 가진다. 이를 연속 SDE의 Brownian 항과 혼동하지 않는다.
Brownian 증분은 `sqrt(dt)`, 시간 상관을 가진 힘의 적분은 별도의 `dt` 스케일을 가진다.

사용자가 제안한 항은 부호와 적분 영역을 구분해야 한다.

\[
 H[p]=-\int p(x)\log p(x)dx.
\]

`int p log p`는 음의 미분 엔트로피인 스칼라 분포 함수다. 시간 기억 적분이나 상태의 경향과
차원·단위가 달라서 경향 벡터에 그대로 더하지 않는다. `p(next state | observed history)`를
정의하고 샘플링해야 실제 앙상블 차이가 생긴다. 이 실험의 엔트로피는 정규화된 toy 변수의
분포 엔트로피이며 대기의 열역학적 엔트로피가 아니다. 좌표·단위 변환에도 값이 달라진다.

물리 potential `U`가 명확한 문제에서는 `F[p]=E_p[U]+T int p log p`라는 자유에너지가
확산과 연결된다. 그러나 날씨의 이류·코리올리 등을 포함한 `G` 전체를 `-grad U`라고 놓을 수
없다. 기존 물리 항과 potential을 중복해 넣지 않는다.
[Jordan, Kinderlehrer & Otto (1998)](https://epubs.siam.org/doi/10.1137/S0036141096303359).

## 학습 항과 진단 항

첫 학습은 Gaussian 조건부 NLL을 사용한다. 정답의 잔차와 `log sigma`를 함께 평가하므로
분산을 키우거나 0으로 만드는 것만으로 목적을 달성할 수 없다. NLL은 정답 분포에 대한
**교차 엔트로피**이며 모델 자신의 엔트로피와 같지 않다.

선택적인 추가 비교는 train 자료에서 추정한 기준 분산에 대한 **같은 평균 Gaussian KL**이다.
이는 분산의 과도한 팽창·수축을 제어하는 제한된 실험이며, 경로 공간의 KL,
Schrodinger bridge, 물리적 entropy production을 구현했다고 부르지 않는다.
원시 엔트로피의 무조건적인 최대화/최소화는 사용하지 않는다.
CRPS·coverage·spread·조건부 엔트로피·시간 상관은 각각 보고한다.
[Gneiting & Raftery (2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

## 1차 CPU 실험의 범위

`scripts/run_memory_entropy_cpu.py`는 숨은 기억과 OU 힘을 가진 작은 비선형 계에서
`G(x)`를 입력받는 신경망 closure를 비교한다. 학습 입력은 관측된 `x`와 `G(x)` 및 과거뿐이다.
생성기의 진짜 기억·노이즈는 정답 생성에만 쓰고 학습·예보에 넘기지 않는다.

| 비교 | 기억 | 확률적 증분 | 목적 |
|---|---|---|---|
| G + learned correction | 없음 | 없음 | 결정론 기준 |
| G + memory | GRU | 없음 | 기억만의 효과 |
| G + white | 없음 | 백색 | 확률성만의 효과 |
| G + memory + white | GRU | 백색 | 결합 효과 |
| G + memory + colored | GRU | AR(1) | 시간 상관의 효과 |
| 추가 KL 비교 | GRU | 유색 + 기준 분산 KL | 분포 제약의 효과 |

같은 seed의 자료와 최적화 횟수를 공유한다. 기억·확률 head가 추가되므로 매개변수 수는
완전히 같지 않으며 결과에 함께 기록한다. 단순한 용량 증가의 효과를 배제하려면 이후
매개변수 수를 맞춘 별도 대조가 필요하다.
train / validation / test는 독립적인 궤적이다. 정규화·기준 분산은 train에서만,
체크포인트 선택은 validation에서만 정한다. test는 최종 채점에만 사용한다.
자유 rollout 중에는 미래 정답이나 숨은 생성 상태를 다시 넣지 않는다.

평가: 선행별 앙상블 평균 RMSE, CRPS, 90% coverage와 spread, 분산·시간 상관,
수치적 발산률 및 조건부 엔트로피. 주변분포 CRPS만 좋아도 시간 상관이 틀릴 수 있으므로
궤적 지표를 따로 본다. 노이즈·물리 기억 커널을 각각 없앤 생성기 대조도 실행할 수 있다.
추가 노이즈가 개별 멤버의 RMSE를 낮춘다고 가정하지 않는다.

주의: 물리 기억 커널을 없애도 OU 힘이 남으면 관측된 x에는 여전히 시간 기억이 있다.
`--memory-strength 0 --white-forcing`을 함께 사용한 별도 Markov 대조에서만 두 숨은 기억을
모두 없앤다. 학습된 AR rho도 GRU가 설명한 뒤의 잔차 상관이므로 생성기의 물리적 OU 시간
상수를 복원했다고 해석하지 않는다. KL의 기준 분산 역시 train 잔차의 경험적 기준이며
열역학·fluctuation–dissipation 제약이 아니다.

이것은 **기전 검증용 합성 실험**이다. 기존 ERA5 core의 학습 결과나 실제 장기 예보의
개선 증거로 해석하지 않는다. 아래 adapter의 전체 ERA5 재학습도 별도 단계다.

## 기존 core와의 접점

`pgcfm/stochastic_memory.py`는 기존 기억 core를 감싸는 선택적 확률적 adapter와
Gaussian NLL/entropy/KL 및 앙상블 CRPS 함수를 제공한다. 기존 main의 기본 경로는 바꾸지 않는다.
자유 예보에서 memory의 innovation이 자신의 주입 노이즈를 관측 오차로 잘못 읽지 않도록
`core.fc`를 실제 반환된 멤버 상태와 일치시켜야 한다.

### 요청 보완: entropy를 실제 입력으로 연결하는 추가 비교

`EntropyConditionedMemoryCore`와 합성계의 `EntropyInputClosure`를 추가했다.
먼저 현재 X/G와 잠정 기억으로 조건부 Gaussian σ를 계산하고,
`e = ∫p log p − ∫pref log pref = −log(σ/σref)`를 계산해 기억의 학습 가능한
보정 입력으로 넣은 뒤 경향을 적분한다. σ는 같은 시각에 보정된 기억을 다시 읽지 않아 순환식이 없다.
기준 σ는 고정된 정규화 단위이며 미래 test 자료로 정하지 않는다.

`memory_entropy_off/on`은 같은 구조·매개변수·초기 가중치·배치로 학습하며 off는 e만 0이다.
`memory_entropy_mean`은 on과 같은 조건부 분포를 학습하되 자유 예보에서 샘플링만 끈 경로다.
비선형 모델의 샘플링을 끈 경로는 앙상블 평균과 다르다.
학습된 on 모델에서 가중치는 그대로 두고 entropy 입력만 끄는 개입도 기록한다.
Gaussian에서는 이 entropy가 log-scale의 다른 표현이므로 독립적인 정보나 물리 법칙의 검증은 아니다.
초기 entropy 보정과 σ head가 모두 0 변화에서 시작하므로, σ를 NLL/CRPS로 학습해야 한다.
기존 결정론 MSE 학습기에 adapter만 붙였다고 entropy 경로가 자동으로 학습되는 것은 아니다.

```bash
EXECUTION_LABEL='실제 실행 위치' PYTHON_BIN=.venv-cpu/bin/python \
  OUTDIR=runs/memory_entropy_direct_cpu bash scripts/run_memory_entropy_cloud_cpu.sh entropy
```

수분 양성·질량 사영 뒤의 분포는 Gaussian이 아니다. analytic NLL/entropy는 **사영 전 법칙**의
값이며, 사영된 최종 멤버의 분포 성능은 CRPS 등 샘플 기반 점수로 평가한다.
현재 adapter는 전구 보존형 노이즈를 구현하지 않는다. `PositiveMass(ref=None)`가 보존하는 것은
노이즈를 넣은 프레임의 수분 총량이며 결정론 예보의 총량이 아니다. ERA5 적용 시 확률적
수분원·질량 drift를 별도로 채점하고, 필요하면 보존형 공분산/사영을 따로 설계해야 한다.

## 클라우드 CPU와 다음 ERA5 단계

최소 환경과 실행 명령:

```bash
python3 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-memory-entropy-cpu.txt
PYTHON_BIN=.venv-cpu/bin/python bash scripts/run_memory_entropy_cloud_cpu.sh smoke
# 실제 클라우드 호스트에서 실행할 때에만 이 label을 사용한다.
EXECUTION_LABEL='Codex Cloud CPU' PYTHON_BIN=.venv-cpu/bin/python \
  bash scripts/run_memory_entropy_cloud_cpu.sh study
# 동일한 예산으로 생성기의 기억/노이즈를 각각 제거한 대조
EXECUTION_LABEL='Codex Cloud CPU' PYTHON_BIN=.venv-cpu/bin/python \
  bash scripts/run_memory_entropy_cloud_cpu.sh controls
```

JSON에 실제 hostname, 버전, Git revision/dirty 상태, 자료 seed, 학습 seed, 시간과
실행 label을 저장한다. 기본 3개 학습 seed는 같은 독립 train/validation/test 자료를 사용한다.
seed 간 표준편차는 신뢰구간이 아니며, 별도 자료 seed까지 바꾼 일반화 시험은 후속 단계다.
원본 JSON·CSV와 선행별 그림을 함께 읽고 유리한 선행이나 seed만 고르지 않는다.

현재 채팅의 셸은 로컬 CPU다. 저장된 클라우드 설정 `physical_guide_flowmatching`은 확인했지만
그 설정의 기본 checkout은 과거 step13이다. 클라우드 실행 시 반드시
`physical_tendency_core`의 이 연구 브랜치를 별도로 checkout하고 commit을 기록한다.
`scripts/run_memory_entropy_cloud_cpu.sh`는 같은 합성 비교를 클라우드 CPU에서 재실행하기 위한
실행 명령이며, 클라우드 자원을 생성하거나 유료 작업을 자동 예약하지 않는다.

실제 ERA5 단계는 5.625도, 짧은 iteration/적은 IC의 smoke → 3 seeds의 동일 학습 예산 →
미사용 연도 1~10일 rollout 순서다. 기존 자료의 train/validation/test 연도를 유지한다.
자료 다운로드·캐시와 수 시간의 CPU 예산이 필요하며 1.5도 전체 학습은 첫 CPU 검증 대상이 아니다.
기존 보고서에는 5.625도 core 3000 iteration이 CPU에서 약 2시간 45분 걸린 기록이 있다.
새 확률 모델의 소요 시간은 직접 측정하기 전에는 이 값과 같다고 가정하지 않는다.
