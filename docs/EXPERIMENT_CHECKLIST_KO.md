# Tendency core · entropy 입력 검증 체크 테이블

작성: 2026-11-16 (한국 시간). 최신 상태는 이 파일과 연결된 원자료로 확인한다.
**완료**는 해당 행의 완료 조건만 충족했다는 뜻이며, 실제 기상 예보 개선까지 입증했다는 뜻이 아니다.
새 기능은 `research/memory-entropy-cpu` 브랜치의 선택적 경로이며 기존 main 모델에 배포된 상태는 아니다.
현재 집중 테스트: **40 passed, 1 skipped**. skip은 전체 실제 core 연결 의존성이 없는 테스트다.

## 구조와 수학적 정의

| 체크 | 확인할 항목 | 완료 조건 | 현재 상태 / 확인 내용 | 근거 |
|---|---|---|---|---|
| ☑ | 현재 G(x) 입력 구조 | 실제 함수와 적분 경로 확인 | 완료. X·G(X)·기억으로 학습된 경향을 Heun 적분 | `pgcfm/dyncore.py`, `pgcfm/kai_tend.py` |
| ☑ | 기존 기억 구조 | GRU·sequence loss·누적 커널 확인 | 완료. 기억을 처음 추가하는 연구가 아님 | `MemCore`, `--core-acc`, `--core-pacc`, `--core-kern` |
| ☑ | p의 정의 | 어느 확률변수의 어떤 조건부분포인지 명시 | 조건부 미해결 증분의 사영 전 Gaussian 분포로 정의 | `EntropyConditionedMemoryCore` 문서 |
| ☑ | entropy 항의 부호·단위 | H=-∫p log p, 기준·정규화 단위 명시 | e=∫p log p−∫pref log pref=−log(σ/σref). 열역학적 entropy·KL과 구분 | `entropy_feature`, `centered_negative_entropy` |
| ☑ | 원래 G와 potential 중복 | 추가 potential의 물리적 정의가 없으면 중복하지 않음 | 별도 potential을 임의로 G에 더하지 않음 | [연구 설계](memory_entropy_cpu_research.md) |
| ☐ | noise–entropy 물리적 쌍대성 | 해당 계에서 성립 조건·함수미분·확산 관계까지 검증 | 미검증. Gaussian entropy 입력이 유효한지 보는 실험과 구분 | 이득이 나도 물리적 쌍대성의 증명은 아님 |

## 구현과 대조군

| 체크 | 확인할 항목 | 완료 조건 | 현재 상태 / 확인 내용 | 근거 |
|---|---|---|---|---|
| ☑ | 실제 노이즈 샘플링 | seed 재현, 멤버 차이, 강제 입력 유지 | 기존 adapter 단위 테스트 통과 | `StochasticMemoryCore`, `tests/test_stochastic_memory.py` |
| ☑ | entropy가 실제 입력에 연결되는가 | 계산한 e가 기억과 이후 경향에 영향을 줌 | 선택적 adapter·toy 구현 및 fixture 단위 검증 완료. 실제 ERA5 연결은 별도 | `EntropyConditionedMemoryCore`, `EntropyInputClosure` |
| ☑ | 같은 크기의 entropy-off 대조 | 구조·매개변수·초기 가중치·자료·학습 예산 동일, e만 0 | 1,586개 매개변수·초기 가중치·배치 동일성 테스트 완료 | `memory_entropy_off` / `memory_entropy_on` |
| ☑ | entropy와 sampling의 효과 분리 | 동일 학습 모델에서 예보 샘플링만 끈 비교 | `memory_entropy_mean` 구현·동일 학습법/샘플링 분리 테스트 완료 | mean path는 앙상블 평균과 다름 |
| ☑ | 같은 시각의 순환 의존 차단 | 기억 → σ → e → 기억 보정 → 경향 순서 | 구현·인과성 단위 테스트 완료 | 현재 σ가 같은 시각의 보정된 기억을 다시 읽지 않음 |
| ☑ | 과거 warm-up과 자유 예보 일치 | 동일 갱신 경로, 미래 정답 입력 금지 | 과거 재생·seed 재현·미래 입력 차단 단위 테스트 완료 | `reset`, `step` |
| ☑ | entropy 경로의 학습 가능성 | loss gradient가 entropy 연결·분산·기억에 유한하게 도달 | 유한 gradient 도달 단위 테스트 완료 | 초기 보정은 0, 학습 후 효과를 별도 확인 |
| ☑ | 기존 solver 기본 경로 보존 | 선택적 adapter, 비활성 경로 재현 | 기존 단위 테스트 통과, base solver 변경 없음 | `StochasticMemoryCore` deterministic mode |
| ☐ | 실제 66채널 core 전체 연결 | 실제 프로젝트 의존성으로 integration test 통과 | 미검증. 선택적 테스트는 의존성 부족으로 skip | toy·fixture 통과와 구분 |

## 실험 결과와 실행 환경

| 체크 | 확인할 항목 | 완료 조건 | 현재 상태 / 확인 내용 | 근거 |
|---|---|---|---|---|
| ☑ | 기존 기억·노이즈 비교 | 6개 모델 × 3 seeds 완료 | 로컬 CPU 본 비교 18회 완료 | [원자료](../reports/memory_entropy_cpu/main/results.json) |
| ☑ | 생성기 대조 | 물리 기억 제거·노이즈 제거·Markov 대조 | 로컬 CPU 추가 54회 완료. 합계 72회 | [기존 비교 보고서](../reports/memory_entropy_cpu/REPORT_KO.md) |
| ☑ | 기존 비교의 해석 한계 | 분포 개선과 평균 오차 개선을 구분 | 기억+백색 CRPS 약30% 개선, 평균 RMSE 개선 없음. 직접 entropy 입력의 결과가 아님 | 동일 보고서 |
| ☐ | 직접 entropy 입력의 3-seed 결과 | off/on 동일 조건 학습·최종 test·원자료 보존 | 아직 결과 없음. 실행 준비 중 | 새 비교의 결과로 기존 72회를 대신 제시하지 않음 |
| ☐ | entropy 입력의 실제 기여 | off/on 비교와 학습된 on 모델의 e=0 개입 시험 | 실행 예정 | 단순한 학습/용량 차이와 분리 |
| ☐ | 새로운 자료 seed 일반화 | 독립 자료 seed에서 재실행 | 미실행. 현재 자료 split은 고정 | 학습 seed 표준편차는 신뢰구간이 아님 |
| ☐ | 실제 ERA5 장기 예보 | 동일 연도·IC·학습 예산, 1~10일 분포·궤적 채점 | 미실행 | 합성 스칼라 실험과 구분 |
| ☐ | 보존형 노이즈와 물리 drift | 결정론 예보 대비 질량·수분 drift 확인 | 미검증. 현재 adapter는 전역 보존형 노이즈가 아님 | PositiveMass는 노이즈가 든 입력의 총량을 보존 |
| ☑ | 로컬 재현 기록 | Git commit·코드 SHA256·버전·seed·호스트 기록 | 기존 72회 clean commit `dfc14ff`로 완료 | 각 `results.json`의 `provenance` |
| ◐ | 클라우드 작업 생성 | 실제 thread/host 반환 확인 | 사용자 승인 후 생성 요청이 장기 응답 대기. 목록에서도 해당 작업을 확인하지 못함 | 로컬 결과를 클라우드 결과로 표시하지 않음 |
| ☐ | 클라우드 학습 완료 | 실제 cloud 호스트 원자료와 완료 상태 확인 | 미확인 | 생성 요청과 학습 완료는 별개 |

체크 표기: ☑ 완료 / ◐ 진행·확인 중 / ☐ 미완료·미검증.
새 결과가 나오면 해당 행을 갱신하고 원자료·실행 commit을 연결한다.
