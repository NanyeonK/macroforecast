# Phase B 설계 — B4: GCLS 2022 "How is Machine Learning Useful for Macroeconomic Forecasting?" (JAE 37(5), 920–964)

작성: 2026-07-08 (Fable fork, 원문 전독 기반). 근거: 원문 추출텍스트 6,158줄 전독 + TeX 소스(GCLSS_JAE_13527_2.tex, 표/그림 원본) + 공식 아카이브 MANIFEST + 패키지 미러(post round-3 main) + `docs/replication/gcls_2021_replication.md` 전례 + gcls_axis_audit 판정.

**주의 — 2021 vs 2022**: 기존 복제 완료본은 GCLS **2021**(Macroeconomic Data Transformations Matter, IJF)이며 본 설계는 **2022 JAE 4축 논문**. 이전되는 자산: 복제 문서/스크립트 레이아웃(`docs/replication/*.md` + `*.py` + `data/`), retune-vs-refit 교훈(아래 §4-R2), parity 등급제, server1의 FRED-MD 처리 경험. 이전되지 않는 것: 2021 러너/표 산출물(다른 실험).

---

## §0. 논문 해부 (원문 인용 기반)

**실험 설계** (p.923 §2.3, Eq.9–12): 46개 모델(Table 1, TeX :453)의 POOS horse race → 제곱오차/R² 패널을 4개 특성 더미에 회귀해 "ML treatment effect"를 식별. 특성: **NL**(비선형: KRR·RF), **SH**(정규화: B1/B2/B3 × ζ∈{0, ζ̂, 1}), **CV**(하이퍼파라미터 선택: AIC/BIC/POOS-CV/K-fold), **LF**(손실: quadratic vs ε̄-insensitive SVR), + **X**(data-poor Ht⁻ vs data-rich Ht⁺).

**데이터** (p.929 §4.1–4.2): FRED-MD 1960M01–2017M12, 134계열(공식 아카이브 `MainAnalysis/2018-01.csv` = 레벨 원계열, McCracken-Ng tcode 변환 지시 — readme.glss.txt). 타깃 5종: INDPRO·CPI(INF)·HOUST = I(1) log → **average growth** (Eq.4), UNRATE = I(1) no-log → **average change**, SPREAD(T10YFFM) = I(0) **level** (Eq.3).

**POOS 설계** (p.929 §4.3): 평가구간 1980M01–2017M12 = **456 origins**, h ∈ {1,3,9,12,24}, **expanding window**, 파라미터는 매월 재적합·**하이퍼파라미터는 2년마다 재최적화**, POOS-CV = 인샘플 마지막 25%, K-fold k=5 (CV 구현 상세는 논문 Appendix S5).

**평가** (p.931 §4.4): 벤치마크 상대 RMSPE(기준: 본문 서술은 ARDI,BIC 대비 DM — 부록 표 주석은 AR,BIC 상대 비율; TeX Table_*.tex 주석으로 구현 시 확정), DM 검정, MCS(Hansen), NBER 침체기 별도 표. 처리효과 회귀(Eq.11–12): R²(t,h,v,m)을 특성 더미 + ψ(t,v,h) 고정효과에 회귀, **HAC SE** (모든 그림 캡션 "SEs are HAC").

**밀도 예측** (§7, pp.940–941): QRF vs QARDI 분위수 예측 → **skewed-t 적합**(Adrian et al. 2019식, 분위수-거리 최소화) → 90%/70% 커버리지+구간길이(Table 4), log score 차이의 상수회귀 검정(Table 5, "negative favors QRF"), PIT 검정(Figure 10 = PIT-core, Rossi-Sekhposyan 계열).

---

## §1. 패키지 추가/수정 목록

### (a) Phase A 커버 확인 — 충분/델타
| 요구 | Phase A 레인 | 판정 (미러 검증) |
|---|---|---|
| AIC/BIC per-origin 선택 (AR·ARDI ,BIC/,AIC 8개 arm) | **A2** IC 경로 | 필요 그대로 — SearchSpec.method = {fixed,grid,cv_path,random,bayesian,genetic,custom}뿐 (search.py:477-487), IC 부재 확인. **델타: IC는 lag/factor 차수(p_y,p_f,n_f) 선택에 쓰임 — A2는 (i) 격자 위 IC 스코어링, (ii) retune 캐던스 준수(아래 R2)를 함께 보장해야 함** |
| POOS-CV(마지막 25%)·K-fold k=5 | native | window/core.py: "poos"(:20) + "blocked_kfold"/"random_kfold"(:28-33). **델타: GCLS의 K-fold는 표준(비블록) k-fold — random_kfold로 표현 가능하나 A2 스플리터가 "마지막 25%" POOS를 val_ratio로 정확히 고정해야** |
| 2년 재튜닝 | native | `retune_every=24` + `retune_on_retrain=False` (window/core.py:296-297) — **2021 전례의 함정 문서화됨: retrain_every=24로 잘못 쓰면 벤치마크가 456 중 19 origin만 재적합 (gcls_2021_replication.md:1055-1073)** |
| arm 축 태그 → 처리효과 회귀 | **A3** | Arm.metadata 존재(spec.py:120)·master frame 미전파 확인 — A3 설계 그대로. **델타 3건: ① Eq.11의 회귀변수는 R²(t,h,v,m)=1−e²/Σ(y−ȳ)² — axis_contribution이 e² → R² 변환 유틸 포함해야, ② 고정효과는 ψ(t,v,h) 3원 조합, ③ Table 3형 상호작용(외생변수 ξ(t−h)와 더미의 곱, Eq. p.937) 지원 — design 인자에 interaction 시리즈 허용** |
| NBER 침체/확장 분할 표 (A1–A5 우측 패널, Fig B3/B4) | **A4** | mask="nber_recession" 그대로 |
| DM/MCS/상대 RMSPE 표 | native (#457/#443) + **A6** pairwise 어댑터 | Table A1–A5 형식은 46행×(h=5)×2(전기간/침체) — A6 어댑터 확장 또는 레인 커스텀 포매터 |

### (b) 46모델 레지스트리 재구성 (Table 1 전수, TeX :453 대조)
표기: 모델명(GCLS) → macroforecast arm 구성. 전 arm 공통: 타깃변환은 TargetSpec(average growth/change/level → `direct_average`/`direct`), expanding, retune_every=24.
- **Data-poor 14**: AR,{BIC,AIC}(→`ar`+A2 IC), AR,{POOS,K-fold}(→`ar`+val_method), RRAR,{POOS,KF}(→`ridge`, features=y-lag만), RFAR×2(→`random_forest`, 동일 피처), KRRAR×2(→`kernel_ridge` RBF, specs.py:1415), SVR-AR,Lin×2·RBF×2(→`svr` kernel∈{linear,rbf}, ε tube — specs.py:1596-1640에 kernel/C/epsilon 전부 노출).
- **Data-rich 32**: ARDI,{BIC,AIC,POOS,KF}(→`far` + IC/CV로 (p_y,p_f,n_f)), RRARDI×2(→`ridge`+PCA факт피처), RFARDI×2, KRRARDI×2, **B1**(ζ̂/1/0 × POOS/KF = 6: `elastic_net`(l1_ratio CV)/`lasso`/`ridge`, features=**identity**(원계열 X+lags 전체)), **B2**(6: 동일 모델, features=**X의 PC 전부 보존**(회전만) — FeatureSpec.pca_components(specs.py:120)로 전-성분 지정; 구현 시 "전부 유지" 옵션 확인, 없으면 n=미달 시 min(N,T) 명시), **B3**(6: **H⁺(lags 포함) 전체의 PC** — 피처 파이프라인 순서 lag→PCA), SVR-ARDI Lin×2·RBF×2.
- **판정: 46개 전부 native 모델 + 피처 조합으로 구성 가능** (axis audit 재확인). custom_model 불요. 단 ARDI의 "IC로 K개 연속 인자 선택"과 B2/B3의 "전 인자 보존"의 구분(p.926 §3.2 마지막 문단)을 arm 정의에서 정확히 대비시킬 것.
- **밀도 2모델**: QRF → `quantile_regression_forest`(specs.py:3798 확인). QARDI(선형 분위수회귀+ARDI 피처) → 레지스트리 확인 후 부재 시 **quantile_regression 원자 모델 추가**(표준 추정기 — 패키지 자격) 또는 A-2차로.

### (c) 밀도 레인 경계 (1급 vs 유예)
- **1급(이번 복제)**: 포인트 예측 46모델 전체, Tables 1–3, A1–A5, Figures 1–9, Fig C3(CSSED·GR fluctuation — **#460 figures 모듈이 정확히 커버**: cumulative_loss_differential_plot + fluctuation_test_plot).
- **유예(A-2차 밀도 제네릭 이후)**: Table 4(커버리지 — skewed-t 적합 필요), Table 5(log-score 상수회귀 — log score 지표 필요), Figure 10(PIT 검정 — 조건부 PIT bootstrap 필요). 유예 근거: additions_sweep Tier-2 §11의 3개 제네릭(log-score+AG, PIT bootstrap, skew-t quantile-fit)이 선행 유닛. QRF/QARDI의 **분위수 예측 생성 자체**는 1급에서 미리 돌려 ResultStore에 적재 가능(quantile_predictions 컬럼 존재) — 평가만 유예.

### (d) 원문 전독 신규 발견 (선행 감사 미포착)
1. **X(data-rich)는 5번째 축**: 처리효과 회귀는 4특성+X 5개 더미(Fig 1 캡션 "X is making the switch...") — A3 태그 설계에 X 포함 (46 arm 중 14/32 구분).
2. **적응형(adaptive) 9종은 각주로 제외**(각주 14: "similar or deteriorated") — 46개가 최종 확정 수.
3. **CPI는 I(1) 처리**(각주 19, I(2) 아님 — Medeiros와 동일 선택) — tcode 오버라이드 필요 지점.
4. **VAR 의도적 배제**(각주 9: 반복 h-step은 direct와 비교 불가) — 우리의 #442 reroute 정직-라벨링 철학과 일치, 복제에서 반복정책 arm 금지.
5. **Fig C1**: 선형 ARDI의 선택된 회귀변수 수 시계열 — **A5 선택이력 로깅**이 정확히 생산(HP 선택 로그).
6. 본문 서술상 DM 기준 ARDI,BIC(p.931) vs 부록 표는 AR,BIC 상대 — 구현 시 TeX 표 각주로 확정(리스크 R5).

## §2. 복제 범위 — exhibit 전수
| Exhibit | 내용 | 판정 | 1:1 표현 |
|---|---|---|---|
| Table 1 | 46모델 그리드 | 재현(정의표) | arm 정의 코드에서 자동 생성 |
| Fig 1, 2 | α̇_F 분포 (h,v)/(v)/(h) | **재현** | A3 axis_contribution(R², ψ(t,v,h) FE, HAC) + 신뢰대역 플롯 |
| Fig 3, 4 | NL 모델별(KRR vs RF) v/h 분해 | 재현 | 동일, 부분집합 회귀(Eq.12) |
| Fig 5 | SH 9종 비교(기준 ARDI-CV) | 재현 | 동일 |
| Fig 6, 7 | CV 방법 비교(byX/byrec) | 재현 | 동일 + A4 침체 마스크 |
| Fig 8, 9 | LF(SVR) 효과 rec/exp | 재현 | 동일 |
| Table 2 (CV1.tex) | 특성 전체 회귀 | **재현** | axis_contribution 본체 |
| Table 3 (Inter-07/09) | NL 이질성(MacroU·ANFCI·CSUSHPI·UMCSENT 상호작용, h={9,12,24}/전체/rich/최근20년) | 재현 | axis_contribution + interaction 인자((a)델타③); 아카이브의 4개 외생 파일 사용 |
| Tables A1–A5 | 변수별 46모델 상대RMSPE+DM+MCS, 전기간/NBER | **재현 (핵심 parity 대상)** | paper_accuracy_table + mcs + A4 마스크 + A6 포매터 |
| Fig C3 | 누적/3y-rolling RMSPE + GR fluctuation | 재현 | **#460 figures 그대로** |
| Fig C1, C2 | HP 수 시계열 / NL 동인 시계열 | 재현(C1)/재현(C2, 데이터플롯) | A5 로그 / 아카이브 외생 4계열 |
| Table 4, 5, Fig 10 | 밀도(커버리지/logscore/PIT) | **유예** (§1c) | A-2차 후 후속 레인 |
| Fig B1–B7, App S1–S4 | 강건성(부분집합·실시간·rolling·절대손실·분기·캐나다) | B1/B2/B3/B4(rec/exp)=재현(동일 회귀의 부분집합), **B6 실시간=유예**(vintage 지원은 있으나[아카이브 FredMD_Vintages 완비] 계산 2배 — 2단계), B7 rolling=유예(동일), S3 분기·S4 캐나다=제외(별도 데이터셋, 프로그램 후순위), S1 절대손실=옵션(loss="absolute" 재채점만이라 저비용 — 포함 권장) |

## §3. `replicate_gcls2022()` 설계
전례 레이아웃 준수: `docs/replication/gcls_2022_replication.md` + `gcls_2022_replication.py` + `docs/replication/data/`.
```python
# gcls_2022_replication.py (골격)
PANEL = load_official_archive("MainAnalysis/2018-01.csv")      # 레벨 → McCracken-Ng tcode (CPI는 I(1) 오버라이드, 각주19)
EXOG  = load_interaction_series()                               # MacroUncertainty, NFCI/ANFCI, CSUSHPINSA, UMCSENT (아카이브 4파일)
WINDOW = mf.window.from_cutoffs(estimation_start="1960-01", test_start="1980-01",
    test_end="2017-12", mode="expanding", retrain_every=1,      # ← 매월 재적합 (2021 교훈: retune≠retrain)
    retune_every=24, retune_on_retrain=False)
ARMS = build_gcls2022_arms()   # 46개: §1(b) 그리드 — 각 Arm.tags={"X":0/1,"NL":0/1,"SH":...,"CV":...,"LF":...} (A3)
TARGETS = [TargetSpec("INDPRO", transform="average_growth", policy="direct_average"), ...5종]
spec = mf.pipeline_spec(data=PANEL, targets=TARGETS, horizons=[1,3,9,12,24], arms=ARMS,
    window=WINDOW, result_store=STORE,                          # 증분 필수 (아래 규모)
    evaluation=mf.EvalSpec(benchmark="AR,BIC", tests=("dm","mcs"),
        subsamples={"full": SubsampleWindow(), "nber_rec": SubsampleWindow(mask="nber_recession")}))
report = mf.run_pipeline(spec)
contrib = mf.analysis.axis_contribution(report, design=GCLS_DESIGN)   # Tables 2-3, Figs 1-9
emit_tables_A1_A5(report); emit_fig_C3(report); emit_parity(report)
```
**규모와 증분**: 셀 = 46 arm × 5 타깃 × 5 h = **1,150 셀 × 456 origin ≈ 52만 적합** (retune은 셀당 19회만). KRR/SVR은 T≤700에서 경량, RF/QRF·B1(N=134 원계열 EN 경로)이 지배 비용. **ResultStore 증분이 생존 조건**: arm 단위로 6~8회 분할 제출(스테이지별 스토어 재사용), server1 n_jobs 병렬.
**스테이지 게이트**: G1 smoke(2 arm×1 타깃×h1, 456 origin 완주+체크포인트) → G2 **Table A1 INDPRO 블록**(46 arm×h5, AR,BIC 상대비 parity) → G3 전 타깃 + axis_contribution(Figs 1-2·Table 2) → G4 Table 3 상호작용+강건성 부분집합. 각 게이트에 parity 등급: 결정적(AR/ARDI/ridge/lasso/KRR 폐형) **MATCH**(비율 ±0.01), 확률적(RF/QRF)·SVR(솔버 차) **CLOSE**(±0.05·부호/순위 보존), 처리효과 회귀는 계수 부호+유의성 일치 기준(구현 언어가 R→Python이므로 점추정 CLOSE).

## §4. 리스크 Top 5
- **R1 46모델 재구성 충실도** (공식 코드 부재 — MANIFEST:35-39 데이터-온리 검증): 완화 — Appendix S5/S6(CV·모델 상세; Wiley 보충자료 **입수 필요 — 현 아카이브에 미포함, [GAP]**)를 구현 전 확보; 불명 시 GCLS 2021 코드베이스(공개 R)와 저자 관례로 보간하고 편차 명시.
- **R2 retune-vs-refit 함정**: 전례 문서화 완료(gcls_2021_replication.md:1055-1073) — retrain_every=1+retune_every=24 조합을 G1 게이트에서 refit-count assert로 고정.
- **R3 계산 규모**(52만 적합, B6/B7 시 ×2~3): 완화 — ResultStore 증분+스테이지 제출+O(1,150) 셀 병렬; G2에서 벽시계 실측 후 G3 예산 재추정.
- **R4 HP 격자 재구성**(λ·σ·C·ε̄·RF mtry 등 상한/격자 미상): 완화 — S6 확보 시 그대로; 아니면 격자를 명시 기록하고 CV-선택 결과의 강건성으로 방어(처리효과는 모델 간 차이라 격자 민감도 낮음 — Table 2 부호 재현이 1차 목표).
- **R5 벤치마크 기준 불일치**(본문 ARDI,BIC vs 부록표 AR,BIC): 완화 — TeX Table_*.tex 각주 원문 확정 후 두 기준 모두 산출(비용 0, 표 두 벌).

**전례 이전 자산 요약**: 레이아웃·등급제·retune 교훈·FRED-MD tcode 경로 재사용; server1 gcls_2021 산출물은 참고만(다른 실험), 공식 2018-01.csv가 본 복제의 유일 데이터 원천(+상호작용 4파일+ALFRED vintages는 B6 유예분).
