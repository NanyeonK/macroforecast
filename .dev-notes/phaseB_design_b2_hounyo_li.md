# Phase B 설계 — B2: Hounyo & Li (2026, IJF 42, 414–433) "Forecasting economic time series in the presence of weak factors" (SsPCA)

설계자: Fable fork / 2026-07-08. 근거: 논문 전문 정독(추출텍스트 2,504줄 전량 + 표), 공식 MATLAB 재현패키지 스크립트 직접 판독(`inflation_linear_tune.m`, `SsPCA_tune.m`, `dmtest_withp.m`), 패키지 미러(post-#463 main) file:line 검증. 이 문서가 B2 레인의 binding design이다.

**중대 사전 사실**: 논문 1면 각주 — *"The numerical results presented in this manuscript were not reproduced, owing to the substantial computational cost involved"* (IJF 재현성 검사조차 재실행 못함). 따라서 B2의 성공 기준은 "저널도 못 돌린 것을 우리 패키지로 돌려 논문 표에 근접"이며, parity는 논문 수치 대비 허용오차 + 단계적 서브셋 검증으로 정의한다 (전량 1:1 실행은 계산비용상 staged).

---

## 1. 패키지 추가/수정 목록

### (a) 이미 native — 초기 타당성 보고서보다 훨씬 유리 (전부 미러에서 직접 확인)
| 논문 구성요소 | 패키지 대응 | 증거 |
|---|---|---|
| **SsPCA** (Alg. 4) | `supervised_scaled_pca` 모델 — 문서에 "Hounyo-Li supervised scaled PCA: predictive-slope scaling plus SPCA" 명기 | models/linear.py:1793, specs.py:2367 |
| **sPCA** (Alg. 2, Huang) | `scaled_pca` — "Checked against Huang's spcaest.m" (오라클 대조 이력 있음) | linear.py:1516, specs.py:2182 |
| **SPCA** (Alg. 3, Giglio) | `supervised_pca` | linear.py:2108, specs.py:2247 |
| **PLS** | `pls` | specs.py:2120 |
| **PCA 확산지수** (Alg. 1) | `far` (factor-augmented regression) | 레지스트리 |
| **PC² 비선형** (Eq. 4) | `quadratic_factors=True` 파라미터 (native!) | specs.py:2367 default_params |
| **hard/soft 사전선별** (t-stat / EN) | 모델 내장 `preselect` + `t_threshold`(1.28 기본) + `elastic_net_alpha`/`l1_ratio`(0.0002/0.5 — 논문 값과 동일 기본값!) | specs.py:2367 default_params |
| **윈도우 내 표준화** | 모델 내장 `scale=True` ("standardize predictors and target inside the model") | specs.py 파라미터 doc |
| **W = [상수, y_t]** | `control_columns` + `include_constant` | specs.py:2367 |
| 5개 서브샘플 기간 | `EvalSpec.subsamples` (start/end) — 라운드3 #457 배송분 | pipeline/spec.py |
| 예측경로 그림 (D.1–D.5) | `reporting.figures.forecast_path_plot` (#460) | reporting/figures.py |
| 롤링 240개월 + h=1,6,12,24 direct | window rolling + direct policy | window/core.py |

**→ A5(스크리닝 de-fusion)는 이 논문에는 불필요**: 사전선별이 모델 파라미터로 이미 내장. A5는 스크리닝을 *다른* 모델과 조합할 때 필요한 일반화 유닛이므로 로드맵 가치는 유지되나 B2의 전제조건이 아님. **로드맵 정정 사항.**

### (b) Phase A 유닛 중 B2가 실제로 요구하는 것
1. **A2 사용자 정의 스플리터** — 오라클에서 확정한 정확 스킴 (SsPCA_tune.m:23-28에서 직접 판독):
   - 240개월 윈도우 내 고정 fold 경계 **{train 1–80 | val 81–130}, {1–130 | 131–190}, {1–190 | 191–240}**
   - **fold 내부에서 관측치마다 확장 재추정** (`parfor o: idx_train = 1:train_end+o-1`) — 단순 1회-적합 fold가 아님
   - (K, ⌊qN⌋) 격자 **공동** 탐색, 평균 MSE 최소화, 선택 후 전체 윈도우 재적합
   - 현재 val 메서드 {last_block, poos, expanding, rolling_blocks, blocked_kfold, random_kfold}(window/core.py:1058)로는 근사만 가능(`expanding`+`n_splits=3` — 경계가 ratio 유도라 80/130/190과 불일치, fold 내 재추정 캐덴스 미보장) → A2가 **명시적 fold 경계 + fold 내 refit 정책**을 받아야 정확 재현. A2 워크플랜에 이 요구 명세를 전달할 것.
2. **A6 hac_lags + pairwise 표** — Table 3(SsPCA vs SPCA DM 행렬)는 pairwise 어댑터 직접 사용. 저자 DM은 plain DM(NW lag=h, N(0,1) 참조; `dmtest_withp.m` 판독) — 패키지 dm의 HLN 보정/t(n−1) 참조와 다름 → `test_options`로 변형 선택(무보정+정규참조) 가능해야 함. **A6에 요구 추가: hac_lags 뿐 아니라 dm의 small-sample 보정/참조분포 off 스위치.**
3. **A1 hist_mean**: 불필요 (벤치마크는 AR,BIC).

### (c) 이번 정독에서 새로 발견된 요구 (타당성 보고서에 없던 것)
1. **결측 → 0 대입 (표준화 이후)**: 저자는 EM 대신 "simply setting it to zero" (p.427 + tune 스크립트 `xt_standardized(isnan)=0`). 패키지 전처리에 `impute="zero"` 마이크로 옵션 필요 (additions_sweep Tier-3에 이미 등재 — B2가 실수요 확정). S 효과.
2. **선별 순서**: hard/soft 사전선별은 **원시(비표준화) 윈도우 X에 대한 y_{t+h} 회귀**로 수행 후 선별된 X만 표준화 (tune 스크립트 판독). 패키지 `preselect` 내부 구현이 같은 순서인지 레인에서 검증 필수 — 다르면 파라미터 동일해도 수치 이탈.
3. **AR,BIC 벤치마크**: BIC로 시차 선택(최대 12), direct projection, **타깃도 윈도우 내 표준화** 후 적합(저자 AR 스크립트 별도 존재). 패키지 `ar`의 IC 차수선택 경로 확인 필요 — 없으면 A2의 IC SearchSpec route가 선행조건, 있으면 그대로.
4. **타깃 스케일 규약**: 저자는 y를 윈도우마다 표준화해 예측·오차 계산 (Fig 노트 "transformed to stationarity and standardized"). RMSFE는 비율이라 대체로 스케일 소거되지만 윈도우별 σ가 다르면 origin 가중이 달라짐 — 재현은 **동일 구성**(모델 내 scale=True + 벤치마크도 동일 처리)으로 맞추고 문서에 명시.
5. **K, qN 격자**: K∈1..10; ⌊qN⌋ 매크로 18:6:108 (n=126일 때; SsPCA_tune.m:12 하드코딩 일치), SPC는 36:12:216; 금융 100:25:400(SPC 200:50:800). 레지스트리 기본 search_spaces(1,2,3,5,8 × 10..200)와 다름 → **arm별 search space 오버라이드**로 논문 격자 지정 (SearchSpec 사용자 격자 지원 여부 레인 검증; 미지원이면 A2에 포함).

---

## 2. 복제 범위 — exhibit-by-exhibit

**스코프 결정: 매크로 응용(§4.3)이 1급 목표, 금융 응용(§4.2)은 2급(선택 레인).** 근거: ① 패키지 정체성 = macro forecasting(오너 확정), ② 금융 데이터는 S&P 구성종목 XLS ~500파일(ZIP 내 제공되나 2011년 추출본) + 146 요인 강도 측정(Bailey 검정) 등 자산가격 전용 기계 필요, ③ custom-data 시연은 B3(crude oil)가 담당. 단 Table 3의 금융 행은 매크로 행과 같은 기계라 데이터만 준비되면 확장 가능.

### 1급 (macro) — 총 16개 exhibit
| Exhibit | 내용 | macroforecast 1:1 표현 |
|---|---|---|
| **Table 2** (p.428) | 3타깃(인플레·IP성장·Δ실업) × {PC,SPC,PC²} × 5방법 × h∈{1,6,12,24}, 전체표본 RMSFE | 5 arms(`far`,`scaled_pca`,`supervised_pca`,`supervised_scaled_pca`,`pls`) × 모델구성 3종(선형 / X²확장 / `quadratic_factors=True`), `relative_mse` vs `ar` 벤치마크; `paper_accuracy_table` |
| **Table 3** (매크로 3행, p.431) | DM: SsPCA vs SPCA | A6 `pairwise_test_table(test="dm", test_options={"hac_lags": h, "small_sample": off})` |
| **Tables D.11–D.14** | 인플레: h별 × 6표본 × {무/hard t1.28,1.65,2.58/soft Λ 3종} | 동일 파이프라인 + `preselect` 파라미터 arm 변형 + `subsamples` 5개(93:3–03:12, 03:3–13:12, 13:3–23:3, 93:3–13:12, 03:3–23:3) — **주의: 저자 서브샘플은 평가 window가 아니라 별도 재추정이므로 subsamples(평가분할)로는 근사; 정확 재현은 표본별 재실행**(§3 참조) |
| **Tables D.15–D.18** | IP growth 동일 | 〃 |
| **Tables D.19–D.22** | 실업률 동일 | 〃 |
| **Fig 4** (A/B/C) | R²_OS(%) vs 요인 수 1..10, 4 horizons | K 고정 arm 스윕(n_components=1..10, qN만 튜닝) → R²_OS = 1 − MSFE/MSFE_AR 계산 후 matplotlib(피규어 모듈 관례) |
| **Fig 5 + D.6** | SsPCA 선별 predictor 히트맵 (top-50, 361 윈도우, h=1/12) | **선별이력 필요** → 모델의 per-window 선택 컬럼 로깅. A5 로깅 유닛 or 모델 fit metadata 노출. B2에서 최소구현: custom callback/refit 루프로 `n_selected` 멤버 기록 |
| **Figures D.1–D.4** | 인플레 예측경로 (2013:9–2023:3, COVID 구간) | `forecast_path_plot` |
| **Figure D.5** | 실업 h=1 COVID 구간 예측경로(로그) | 〃 |
| **Appendix A 표** | FRED-MD 127변수 + tcode 목록 | 데이터 매니페스트 표 자동 생성 (`load_fred_md` 메타 + 저자 Macrodataset 대조) |

### 제외 (사유 명기)
- **Table 1, D.1–D.6** — Monte Carlo 시뮬레이션 (브리프 지정 제외; 복제 목표는 실증)
- **Appendix B (요인 강도, Bailey et al. 검정)** — 금융 전용 + Bailey 다중검정 통계는 패키지 범위 밖(stays-custom; additions_sweep 판정 유지)
- **금융 exhibit (Tables D.7–D.10, Figs 1–3, Table 3 S&P 행)** — 2급 선택 레인으로 분리 (동일 기계, 데이터만 교체; ZIP에 전 데이터 포함되어 착수 가능하나 매크로 완주 후)

---

## 3. 최종 pipeline 함수 설계 — `replicate_hounyo_li_2026()`

**배치**: GCLS 2021 전례 형식 — `docs/replication/hounyo_li_2026_replication.py`(실행 스크립트, 페이지 재생성) + `hounyo_li_2026_replication.md`(실행된 노트북 스타일 페이지) + `docs/replication/data/hl2026_ground_truth.csv`(논문 표 수치 기계가독본).

```python
def replicate_hounyo_li_2026(
    *,
    scope: Literal["macro", "macro+finance"] = "macro",
    samples: Sequence[str] = ("full",),          # "full","s2".."s6" — 저자 서브샘플은 개별 재실행
    horizons: Sequence[int] = (1, 6, 12, 24),
    thresholds: Sequence[str] = ("none",),        # "none","t1.28","t1.65","t2.58","en1","en2","en3"
    model_configs: Sequence[str] = ("PC",),       # "PC","SPC","PC2"
    result_store: str | Path = "runs/hl2026_store",  # 증분 실행 필수 (계산량 방어)
    stage: Literal["smoke", "table2", "full"] = "table2",
) -> HL2026Report: ...
```

핵심 구성 (검증된 실제 API):
```python
# 데이터: 저자 패널 = 1차 소스 (ZIP 추출; FRED-MD 1971:4–2023:3 사전변환본, n=126)
panel  = mf.data.load_custom_csv(hl_macrodataset_csv)       # XLS→CSV 변환 스크립트 동반
infl   = mf.data.load_custom_csv(hl_usinflation_csv)        # 타깃 3종은 저자 시리즈 그대로
# 로버스트니스 arm: 신선한 load_fred_md() 패널로 동일 스펙 재실행(부록 절)

ARMS = [
  mf.Arm("AR_BIC", model="ar", params={...bic max12...}, is_benchmark=True),
  mf.Arm("PCA",   model="far",                    params={"scale": True}),
  mf.Arm("sPCA",  model="scaled_pca",             params={"scale": True}),
  mf.Arm("SPCA",  model="supervised_pca",         params={"scale": True}),
  mf.Arm("SsPCA", model="supervised_scaled_pca",  params={"scale": True, "preselect": ..., "quadratic_factors": cfg=="PC2"}),
  mf.Arm("PLS",   model="pls"),
]
# SPC 구성: X* = [X, X²] — feature 단계 custom step(제곱 컬럼 추가) 후 동일 arms
# 튜닝: 논문 격자 K=1..10 × qN=18:6:108, A2 고정-fold 스플리터(80/130/190/240, fold내 확장재추정)
window = mf.window.spec(mode="rolling", size=240, val_method=<A2 fixed-folds>, retune_every=1)
spec = mf.pipeline_spec(data=..., targets=[infl, ip, unemp], horizons=horizons,
                        arms=ARMS, window=window, forecast_policy="direct",
                        evaluation=mf.EvalSpec(benchmark="AR_BIC", metrics=("relative_mse",),
                                               tests=("dm",), test_options={"dm": {...plain DM, lag=h...}}),
                        result_store=result_store, seed=42)
```

**parity 게이트** (스크립트가 표 생성 후 자동 대조):
- G1 (smoke): 1타깃×h=1×전체표본×무threshold, 5방법 RMSFE가 Table 2 좌열과 부호·순위 일치 + |Δ|≤0.03
- G2 (table2): Table 2 전체 36셀×5방법 — 순위 보존율 ≥90%, |Δ RMSFE|≤0.05 (저자도 미재현 논문이므로 tolerance 명시; 이탈 셀은 GCLS 페이지 방식으로 원인 분석 표)
- G3 (full): D.11–D.22 + Table 3 DM 부호/유의성 일치율 보고
- 저자 서브샘플(s2–s6)은 subsamples 평가분할이 아니라 **표본별 재실행**(저자 코드가 윈도우 자체를 다르게 슬라이스 — tune 스크립트 sample 분기 판독) — result_store가 중복 셀 재사용
- MATLAB 직접 실행 parity는 [ASSUMPTION: server1에 octave 설치 시 SsPCA_emp002.m 스팟 대조 가능] — 필수 아님

**계산량 방어** (저자 각주의 교훈): stage 파라미터로 G1→G2→G3 단계 실행; fold 내 확장재추정이 지배 비용(윈도우당 3 fold×50–60 재추정×격자 160조합×방법 5) → ResultStore 증분 + 튜닝 결과 캐시(A2 스플리터가 selector 로깅과 함께 (K,qN) 선택 이력 저장) + 서버 병렬(n_jobs, 라운드3 수리 완료).

---

## 4. 리스크 Top 5

| # | 리스크 | 완화 |
|---|---|---|
| 1 | **fold 정확성**: 고정 경계 {80,130,190,240} + fold 내 관측치별 확장 재추정 — A2가 이 semantics를 못 담으면 튜닝 선택 (K,qN)이 달라져 전 표가 흔들림 | A2 워크플랜에 본 문서 §1(b)1 요구 명세 반영; 불일치 시 근사(expanding n_splits=3) 결과를 별도 열로 공표하고 이탈 원인 표기 |
| 2 | **전처리 순서**: 선별(원시 X) → 표준화(선별된 X) → NaN=0 순서; 모델 `preselect` 내부 순서 검증 필요 | 레인 S1에 순서 검증 테스트(합성 3변수 오라클) 필수 배치; `impute="zero"` 옵션 추가 |
| 3 | **타깃 표준화 규약**: 윈도우별 y 표준화가 예측·RMSFE 산출 스케일에 개입 | 전 방법+벤치마크 동일 구성(scale=True) 강제, 문서 명시; RMSFE 비율의 스케일 근사불변성 부록 수식 1단락 |
| 4 | **DM 변형**: 저자 plain DM(N(0,1), NW lag=h) vs 패키지 HLN/t(n−1) | A6 test_options로 보정 off 스위치; 불가 시 Table 3은 양쪽 변형 병기 |
| 5 | **계산량**: 361 윈도우×fold내 재추정×격자×5방법×12표본셀 — 저널도 포기한 규모 | stage 게이트 + ResultStore + 논문 표와의 대조는 G2(전체표본)까지를 1차 완료선으로, D-표는 증분 |

**레인 규모 추정**: 코드 M (데이터 스크립트 + 스펙 + 페이지), 컴퓨트 L (G2까지 수 시간~1일, G3 수일 — 서버 백그라운드). Phase A2/A6 머지 후 착수 가능.
