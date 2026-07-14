# Phase B-3 복제 설계서 — Zhang, Wahab & Wang (IJF 39(2) 2023, 486–502)
"Forecasting crude oil market volatility using variable selection and common factor"
설계: Fable fork, 2026-07-08. 근거: 논문 전문 정독(추출 텍스트 2,535줄) + NAS 아카이브 코드 대조 + 패키지 미러(post-#463) file:line 검증.

## 0. 논문 해부 (전 항목 원문 인용 위치 포함)

- **타깃**: WTI **선물** 일별 수익률 기반 월별 실현분산 RV_t = Σ_j r²_{t,j} (Eq. 1, p.488). LV=ln(RV). 다기간 타깃 **LV_{t+1:t+h} = ln[(1/h)(RV_{t+1}+…+RV_{t+h})]** (Eq. 2 하단, p.488) — h-개월 **평균 RV의 로그**.
- **데이터**: EIA 일별 WTI 선물가 (p.492, fn.11); FRED-MD **2019:06 vintage** 128→127개 변수(ACOGNO 제외, p.491), McCracken-Ng tcode 적용 + **다수 변수 1개월 발표지연 lag** (p.491-492); 표본 1985:01–2018:12, 초기 학습 13년 → **OOS 1998:01–2018:12** (p.492). Spot 로버스트니스는 1986:01 시작 (Table 4, p.494).
- **벤치마크**: AR(L(h)), L(h)를 AIC로 선택 → **L=2,6,6,5 (h=1,3,6,12)** (p.488 fn.3; BIC→2,2,1,1, adj-R²→3,6,6,6, p.495 fn.15).
- **변수선택(VS) 4종** (§2.2, p.489): ①t-stat: 증강 AR(Eq. 3)에서 |t|>1.65(10%) ②ΔR²: 증강AR R² − AR R² > 1% ③lasso ④ENet(ρ=0.5, **λ=corrected AIC**(AICc, Hurvich-Tsai), p.489 fn.5-6). **최소 5개 규칙**: 컷 통과가 5개 미만이면 상위 5개 강제 선택 (p.489).
- **모델 13종** (Table 1, p.491): AR / PCA-VS×4(선택변수의 제1주성분을 AR에 추가, Eq. 9) / lasso / ENet / PCA-all(전체127의 PC1, Eq. 8) / KS-all(Eq. 4) / KS-VS×4(Eq. 10).
- **평가**: Campbell-Thompson R²_OS vs AR (Eq. 11, p.491) + **Clark-West** 단측 (p.491); 확장 윈도우 (p.490 fn.8); in-sample은 Newey-West t-stat (Table 2).
- **복제 대상 exhibits**: Table 2(IS: φ/NW-t/ΔR², 5모델×4h) · **Table 3(메인: R²_OS+CW, 12모델×4h, 선물)** · Table 4(spot 로버스트) · Table 5(대안 VS: 잔차기반 t-stat/ΔR² + time-series-validation λ) · Table 6(대안 AR lag) · Fig 1(127변수 상관 히트맵) · Fig 2(월별 선택변수 개수, NBER 음영) · Fig 3(변수×월 선택 래스터) · Fig 4(그룹별 선택빈도 히트맵, 전체+5년 서브샘플) · 본문 top-5 선택빈도(VXO 98.4%).
- **제외**: **Table 7 + 관련 유틸리티 분석(§5, Bollerslev 포트폴리오)** — 소유자 결정(finance-tool 영역). Online Appendix(QLIKE/HMSE/HMAE DM, 윈도우 크기, TVP-AR/DMA/DMS/BMA/BMS/BSS 비교, 헤징)는 2차 범위로 명시적 보류.

## 1. 패키지 추가/수정 목록

### (a) Phase A 커버 확인
| 필요 | Phase A 레인 | 비고 |
|---|---|---|
| t-stat/ΔR² 스크리닝(AR-controls, 컷+**top-5 fallback**) | **A5** (screening de-fusion) | ZWW 규칙이 A5 스펙과 정확 일치 — top-k fallback·잔차기반 변형(Table 5)까지 A5 옵션으로 |
| 선택변수 이력 (arm,origin,h) 로깅 → Figs 2-4 | **A5** (selector logging) | 사이드카에 선택 마스크 저장, Fig 2-4는 로그의 후처리 플롯 |
| NBER 음영/서브샘플 | **A4** (`load_fred_series("USREC")`) + figures.py `shade=` | Fig 2 음영 + Fig 4 서브샘플 분할(단, Fig 4 분할은 5년 균등 — 날짜창 subsamples로 표현) |
| CW·R²_OS(relative MSE) | 기존 native (evaluate.py cw; accuracy r2_oos) | Table 3-6 통계 열 |
| NW t-stat 고정 lag (Table 2) | **A6** (`hac_lags`) | in-sample 표는 아래 (c)-3 참고 |
| AR AIC lag 선택 | **A2** (IC SearchSpec 경로) | L(h) 사전고정 재현이 1차(파라미터로 명시), IC 경로는 검증용 |

### (b) 기존 백로그에서 이 논문이 승격시키는 것
1. **AICc λ 선택 (lasso/ENet)** — sweep Tier-13(S/M). ZWW의 λ 규칙 그 자체(p.489 fn.6). `lasso`/`elastic_net`은 사용자 지정 alpha만 받음(models/linear.py:571-636). **원자 유닛**: IC-기반 λ 경로를 A2의 IC 선택 유닛에 통합(모델별 λ-grid + AICc 스코어러) — 별도 래퍼 금지.
2. **`log_average_value` 타깃 변환** — 신규(S, 이 설계에서 발견). 현 `average_value`는 1기간 객체의 h-평균(feature_engineering/targets.py:74-76, shared.py `_average_future_path`)이라 **ln(mean RV)를 표현 불가**(h=1만 LV+`value`로 일치). 변환 패밀리에 `log_average_value`(원시 시리즈의 h-스텝 평균에 ln) 1개 추가 — 변동성 문헌 표준 타깃이므로 패키지 원자 유닛 자격 충분.
3. **PCA-on-selected 합성** — A5 스크리닝 스텝 뒤에 기존 PCA 피처(specs.py:326-338, `pca_components=1`)를 체이닝하면 PCA-VS 완성. **추가 개발 불요** — 단 A5가 "스크리닝 출력 → 후속 피처 스텝 입력" 체이닝을 지원하는지 A5 검수 시 확인 항목으로 (스텝 순서 조합 가능성 = 원자성의 시금석).

### (c) custom으로 남기는 것 (패키지에 넣지 않음)
1. **RV 데이터 구성** — 아래 §3 stage-1 스크립트 (bring-your-own-data 시연이 목적이므로 의도적으로 custom).
2. **Table 5의 time-series-validation λ** — Zhang, Ma et al.(2019) 알고리즘, 논문 특화 → A2의 사용자 스플리터로 주입.
3. **Table 2 in-sample 표** — 파이프라인은 OOS 중심; IS 회귀표는 statsmodels + A6 hac_lags 컨벤션으로 ~40줄 후처리 스크립트.
4. Fig 1 상관 히트맵 — pandas/matplotlib 10줄.

## 2. 복제 범위 exhibit-by-exhibit (1:1 매핑)

| Exhibit | 내용 | macroforecast 경로 | 판정 |
|---|---|---|---|
| Table 1 | 모델 정의 요약 | arms 정의 echo (문서화) | trivial |
| Table 2 | IS φ/NW-t/ΔR² ×4h | custom IS 스크립트((c)-3) + A6 lag 컨벤션 | custom-스크립트 |
| **Table 3** | **R²_OS+CW 12모델×4h 선물** | 13-arm PipelineSpec(§3) → accuracy(r2_oos)+significance(cw) → `paper_accuracy_table` | **완전 파이프라인** |
| Table 4 | spot 로버스트 | 동일 spec, 데이터만 spot RV(같은 stage-1 스크립트 `--series=spot`), 표본 1986:01 시작 | 파이프라인 재실행 |
| Table 5 | 대안 VS | A5 잔차기반 옵션 + A2 스플리터 주입 arms | 파이프라인 재실행 |
| Table 6 | 대안 AR lag | L(h)=(3,2,1,1) params 교체 | 파이프라인 재실행 |
| Fig 2 | 선택개수 시계열+NBER | A5 selector log → 플롯 스크립트(+A4 USREC 음영) | 로그 후처리 |
| Fig 3 | 선택 래스터 | A5 selector log | 로그 후처리 |
| Fig 4 | 그룹×기간 선택빈도 | A5 log + FRED-MD 8그룹 매핑(부록 그룹 번호) | 로그 후처리 |
| Table 7·§5 | 포트폴리오 유틸리티 | — | **제외 (finance-tool, 소유자 결정)** |
| Online App. | QLIKE-DM 등 | metrics.py:288 qlike 존재 → 2차 | 보류 |

핵심 정합 포인트: ①타깃 = `log_average_value` (h=1은 기존 경로와 일치 검증), ②AR lag L(h) 고정값 재현 후 A2 IC로 교차검증, ③min-5 fallback, ④발표지연 1개월 lag의 변수 목록(부록 대조), ⑤CW 단측 컨벤션.

## 3. 최종 pipeline 함수 (2-stage)

### Stage 1 — `scripts/replications/build_wti_rv_dataset.py` (custom-data 시연의 본체)
```
입력: EIA/FRED 일별 WTI (선물 RCLC1 = 1차 대상 / 스팟 DCOILWTICO = Table 4)
처리: ① 결측가 일자 제거(dropna) — **소유자 아카이브 검증 규칙** (NAS KRW_2025.ipynb cell 6:
      dropna(subset=['prc']) → log_ret → 월합; ZWW 원문 미기재이나 소유자 직접대조로 동일 처리 확인)
      ② log return (제거일 건너뛰어 체이닝) ③ r² 월합 → RV ④ LV=ln(RV)
출력: wti_rv_monthly.csv (RV, LV) + fred_md_2019_06 예측변수 127열(tcode + 1개월 lag 적용)
      + data_manifest.json: 소스 URL, 해시, **명시적 날짜-제외 규칙**(원문이 공시하지 않은 처리의 공시),
      발표지연 lag 목록, 표본 경계
parity gate: RV 요약통계·시계열을 ZWW Online Appendix 및 NAS KRW 산출물과 대조 —
      잔차 불일치 시 제외 일자 집합 차이를 의심하고 일자 diff를 리포트 (§4 리스크 1)
```

### Stage 2 — `scripts/replications/replicate_zww2023.py`
```python
def replicate_zww2023(data_dir, out_dir, horizons=(1,3,6,12), market="futures"):
    bundle = mf.data.load_custom_csv(f"{data_dir}/zww_panel_{market}.csv")   # LV + 127 predictors
    L = {1:2, 3:6, 6:6, 12:5}                                  # AIC lags (Table 6은 (3,2,1,1)로 재호출)
    arms = [
      Arm("AR",        model="ar", params={"n_lag": L}, is_benchmark=True),
      *[Arm(f"PCA-{vs}", model="ar_plus_factor",                      # = ar + 외생 F1
            features=screen(vs, min_k=5) >> pca(1))                  # A5 스크리닝 → PCA(1) 체인
        for vs in ("t_stat","delta_r2","lasso","enet")],
      Arm("Lasso", model="lasso", selection=aicc_lambda()),           # (b)-1
      Arm("ENet",  model="elastic_net", params={"l1_ratio":0.5}, selection=aicc_lambda()),
      Arm("PCA-all", model="ar_plus_factor", features=pca(1)),
      Arm("KS-all",  model="ar_plus_x", features=all_predictors()),
      *[Arm(f"KS-{vs}", model="ar_plus_x", features=screen(vs, min_k=5)) for vs in VS4],
    ]
    spec = mf.pipeline_spec(data=bundle,
        targets=[TargetSpec("RV", transform="log_average_value", policy="direct")],  # (b)-2
        horizons=horizons,
        window=expanding(start="1985-01", oos_start="1998-01", end="2018-12"),
        preprocessing=passthrough(),            # tcode는 stage 1에서 기적용 (매니페스트에 명시)
        evaluation=EvalSpec(benchmark="AR", metrics=("r2_oos",), tests=("cw",),
                            test_options={"cw": {...단측 컨벤션 확인...}}),
        result_store=out_dir/"cells")
    report = mf.run_pipeline(spec)
    write_table3(report); write_fig2_fig4(selector_logs(out_dir))     # A5 로그 후처리
    return report
```
(`ar_plus_factor`/`ar_plus_x`는 개념 표기 — 구현 시 실체는 "ar 모델 + 외생 피처" 조합이 기존 supervised 경로로 표현되는지 확인이 1차 과제; 안 되면 `custom_model` 20줄(OLS on [LV lags, F])로 주입 — 패키지 확장 불요)

산출: docs/replication/zww_2023_replication.md (GCLS 2021 문서 형식: 표별 parity 판정 + 편차 설명 + 게이트 명령)

### 부수 노트 (구현하지 않음)
같은 stage-1 파이프라인(+ NAS의 불확실성 CSV: EPU/EMV/GPR/MPU/RA/TPU/FU/RU/MU/ERU + 유가 펀더멘털)이 소유자 본인 논문 **Kang·Ryu·Webb (state-dependent oil vol)** 재현의 데이터 기반이 됨 — RF/XGB/NN/LSTM arms + A4 NBER 상태분할 + interpretation 서브시스템 시연까지 자연 확장. B3 완료 후 별도 결정.

## 4. 리스크 top 5

1. **날짜-제외 집합 불일치** — 아카이브에서 검증된 규칙은 "결측가 일자 제거"뿐; 소유자가 기억하는 특정일 제거가 이 이상(예: 0-거래량일, 이상치일)일 가능성. NAS 노트북엔 명시적 특정일 리스트 없음(전 셀 grep). **완화**: stage-1 parity gate에서 RV 요약통계/시계열을 ZWW 공시치·KRW 산출물과 대조, 불일치 시 일자 diff 자동 출력. 2020-04-20 음가격일은 ZWW 표본 밖(–2018:12)이라 본 복제엔 무영향(확장 표본 시 처리 규칙 필요).
2. **EIA 선물 시리즈/롤 컨벤션** — RCLC1(Contract 1) 가정이나 논문은 시리즈명 미기재(p.492 fn.11 EIA 웹만 인용). **완화**: 선물·스팟 둘 다 구축(스팟은 소유자 아카이브와 직접 대조 가능), Table 3(선물)·Table 4(스팟) 동시 parity로 시리즈 선택 검증.
3. **FRED-MD 2019:06 vintage 확보** — 현행 다운로드는 최신 vintage. **완화**: FRED-MD 히스토리컬 vintage 아카이브에서 2019-06 CSV 취득(McCracken 홈페이지 monthly 아카이브 유지); 실패 시 최신 vintage로 돌리고 편차를 "vintage 차이" 절에 명시 (기존 vintage 레이어로 감사 가능).
4. **선택자 정확성** — t-stat 임계 1.65의 자유도 미세변동(p.489 fn.4), ΔR² 1% 컷, min-5 규칙의 동률 처리 등 미세 규약. **완화**: A5 구현 시 ZWW 규약을 옵션 조합으로 표현하고, Fig 2의 "평균 선택개수"(h=1 ≈10개, 장기 t-stat≈35/ΔR²≈25/lasso·ENet≈20, p.497)를 정합성 스모크로 사용.
5. **타깃 변환 신규 코드** — `log_average_value` 추가가 기존 average 경로와 이웃해 회귀 위험. **완화**: h=1에서 LV+`value` 경로와 수치 동일성 테스트 + 기존 `average_*` 골든 불변 테스트.

## 파일/근거
읽음: 논문 전문 추출텍스트(전량)·Table 1-7 원문 확인; NAS `KRW_2025.ipynb`(185셀 중 데이터셀 전수 grep + cell 6/8/9 정독)·`data/` 목록·README; 미러 `feature_engineering/targets.py:29-114`·`shared.py(_average_future_path)`·`metrics.py:288,414,846`·`models/linear.py:571-636`·`feature_engineering/specs.py:176,326-365`. 씀: 본 문서 1건. 게이트: read-only 준수, 실행 없음.
미해결: [UNVERIFIED] EIA 선물 시리즈 정확 명칭/롤; [UNVERIFIED] 2019:06 vintage 입수 가능성; [GAP] 발표지연 lag 적용 변수의 정확 목록(논문 부록 대조 필요 — 구현 시 확인).
