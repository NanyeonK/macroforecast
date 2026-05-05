#!/usr/bin/env python3
"""Build validation artifacts for inferred FRED-SD t-code candidates."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from macroforecast.raw.datasets.shared_csv import parse_fred_csv
from macroforecast.raw.sd_analog_candidates import (
    MAP_VERSION,
    OFFICIAL,
    SOURCE,
    SD_ANALOG_CANDIDATES,
    SdAnalogCandidate,
)
from macroforecast.raw.sd_inferred_tcodes import (
    DEFAULT_RUNTIME_STATUSES,
    SD_INFERRED_TCODE_MAP,
)

DIAGNOSTIC_COLUMNS = (
    "n_overlap",
    "aggregate_corr",
    "pearson_pvalue",
    "spearman_corr",
    "spearman_pvalue",
    "rolling_corr_mean",
    "rolling_corr_min",
    "state_corr_median",
    "state_corr_iqr",
    "state_corr_positive_share",
    "n_state_corr",
    "adf_pass_rate",
    "kpss_pass_rate",
    "sd_aggregate_adf_pass",
    "sd_aggregate_kpss_pass",
    "analog_adf_pass",
    "analog_kpss_pass",
    "acf_distance",
    "low_frequency_ratio",
    "analog_low_frequency_ratio",
    "low_frequency_distance",
    "outlier_rate",
    "analog_outlier_rate",
    "volatility_ratio",
    "missing_rate",
    "n_states_used",
)

REPORT_COLUMNS = (
    "sd_variable",
    "candidate_code",
    "analog_dataset",
    "analog_series",
    "analog_official_code",
    "prior_confidence",
    "score",
    *DIAGNOSTIC_COLUMNS,
)


def _fred_tcode_transform(series: pd.Series, code: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if code == 1:
        return s
    if code == 2:
        return s.diff()
    if code == 3:
        return s.diff().diff()
    if code == 4:
        return np.log(s.where(s > 0))
    if code == 5:
        return np.log(s.where(s > 0)).diff()
    if code == 6:
        return np.log(s.where(s > 0)).diff().diff()
    if code == 7:
        return s.pct_change().diff()
    raise ValueError(f"unsupported t-code {code!r}")


def _acf_values(series: pd.Series, lags: int = 8) -> np.ndarray:
    clean = series.dropna()
    if len(clean) < lags + 5:
        return np.full(lags, np.nan)
    return np.array([clean.autocorr(lag=i) for i in range(1, lags + 1)], dtype=float)


def _acf_distance(left: pd.Series, right: pd.Series) -> float | None:
    left_acf = _acf_values(left)
    right_acf = _acf_values(right)
    valid = np.isfinite(left_acf) & np.isfinite(right_acf)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((left_acf[valid] - right_acf[valid]) ** 2)))


def _low_frequency_ratio(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 24:
        return None
    window = min(12, max(4, len(clean) // 6))
    smooth = clean.rolling(window=window, min_periods=max(3, window // 2)).mean()
    denom = float(clean.var())
    if denom <= 0 or math.isnan(denom):
        return None
    return float(smooth.var() / denom)


def _outlier_rate(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 5:
        return None
    scale = float(clean.std(ddof=0))
    if scale <= 0 or math.isnan(scale):
        return 0.0
    z = (clean - float(clean.mean())) / scale
    return float((z.abs() > 6).mean())


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _corr_stats(joined: pd.DataFrame) -> dict[str, float | None]:
    if len(joined) < 10:
        return {
            "aggregate_corr": None,
            "pearson_pvalue": None,
            "spearman_corr": None,
            "spearman_pvalue": None,
        }

    pearson = _finite_float(joined["sd"].corr(joined["analog"]))
    spearman = _finite_float(joined["sd"].corr(joined["analog"], method="spearman"))
    pearson_pvalue: float | None = None
    spearman_pvalue: float | None = None

    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        return {
            "aggregate_corr": pearson,
            "pearson_pvalue": None,
            "spearman_corr": spearman,
            "spearman_pvalue": None,
        }

    try:
        pearson_pvalue = _finite_float(pearsonr(joined["sd"], joined["analog"]).pvalue)
    except Exception:
        pearson_pvalue = None
    try:
        spearman_pvalue = _finite_float(spearmanr(joined["sd"], joined["analog"]).pvalue)
    except Exception:
        spearman_pvalue = None
    return {
        "aggregate_corr": pearson,
        "pearson_pvalue": pearson_pvalue,
        "spearman_corr": spearman,
        "spearman_pvalue": spearman_pvalue,
    }


def _rolling_corr_stats(joined: pd.DataFrame, analog_dataset: str) -> tuple[float | None, float | None]:
    window = 8 if analog_dataset == "fred_qd" else 24
    if len(joined) < window + 3:
        return None, None
    rolling = joined["sd"].rolling(window=window).corr(joined["analog"]).dropna()
    if rolling.empty:
        return None, None
    return _finite_float(rolling.mean()), _finite_float(rolling.min())


def _state_corr_stats(
    sd_transformed: pd.DataFrame,
    analog_compare: pd.Series,
    analog_dataset: str,
) -> dict[str, float | int | None]:
    corrs: list[float] = []
    for col in sd_transformed.columns:
        state_compare = _comparison_series(sd_transformed[col], analog_dataset)
        joined = pd.concat([state_compare.rename("sd"), analog_compare.rename("analog")], axis=1, join="inner").dropna()
        if len(joined) < 10:
            continue
        corr = _finite_float(joined["sd"].corr(joined["analog"]))
        if corr is not None:
            corrs.append(corr)

    if not corrs:
        return {
            "state_corr_median": None,
            "state_corr_iqr": None,
            "state_corr_positive_share": None,
            "n_state_corr": 0,
        }

    q75, q25 = np.percentile(corrs, [75, 25])
    return {
        "state_corr_median": _finite_float(np.median(corrs)),
        "state_corr_iqr": _finite_float(q75 - q25),
        "state_corr_positive_share": _finite_float(np.mean(np.array(corrs) > 0.0)),
        "n_state_corr": len(corrs),
    }


def _volatility_ratio(left: pd.Series, right: pd.Series) -> float | None:
    left_std = _finite_float(left.dropna().std(ddof=0))
    right_std = _finite_float(right.dropna().std(ddof=0))
    if left_std is None or right_std is None or right_std <= 0:
        return None
    return float(left_std / right_std)


def _adf_kpss(series: pd.Series) -> tuple[bool | None, bool | None]:
    clean = series.dropna()
    if len(clean) < 20 or float(clean.std(ddof=0)) <= 0:
        return None, None
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception:
        return None, None
    adf_pass: bool | None
    kpss_pass: bool | None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_pass = bool(adfuller(clean, autolag="AIC")[1] < 0.05)
    except Exception:
        adf_pass = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_pass = bool(kpss(clean, regression="c", nlags="auto")[1] > 0.05)
    except Exception:
        kpss_pass = None
    return adf_pass, kpss_pass


def _pass_rate(values: Iterable[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(sum(bool(value) for value in usable) / len(usable))


def _stationarity_rates(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    adf_values: list[bool | None] = []
    kpss_values: list[bool | None] = []
    for col in frame.columns:
        adf_pass, kpss_pass = _adf_kpss(frame[col])
        adf_values.append(adf_pass)
        kpss_values.append(kpss_pass)
    return _pass_rate(adf_values), _pass_rate(kpss_values)


def _score(*, corr: float | None, adf_rate: float | None, kpss_rate: float | None, acf_distance: float | None, low_freq_ratio: float | None, outlier_rate: float | None) -> float:
    corr_score = max(float(corr or 0.0), 0.0)
    stationarity_parts = [x for x in (adf_rate, kpss_rate) if x is not None]
    stationarity_score = float(sum(stationarity_parts) / len(stationarity_parts)) if stationarity_parts else 0.0
    acf_score = 0.0 if acf_distance is None else max(0.0, 1.0 - min(float(acf_distance), 1.0))
    low_freq_score = 0.0 if low_freq_ratio is None else max(0.0, 1.0 - min(float(low_freq_ratio), 1.0))
    outlier_score = 0.0 if outlier_rate is None else max(0.0, 1.0 - min(float(outlier_rate) * 10.0, 1.0))
    return float(
        0.35 * corr_score
        + 0.20 * stationarity_score
        + 0.20 * acf_score
        + 0.15 * low_freq_score
        + 0.10 * outlier_score
    )


def _load_sd_sheet(path: Path, sheet: str, sample_start: str | None, sample_end: str | None) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, index_col=0, engine="openpyxl")
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    frame.index.name = "date"
    if sample_start:
        frame = frame.loc[frame.index >= pd.Timestamp(sample_start)]
    if sample_end:
        frame = frame.loc[frame.index <= pd.Timestamp(sample_end)]
    return frame


def _filter_sample(frame: pd.DataFrame, sample_start: str | None, sample_end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if sample_start:
        out = out.loc[out.index >= pd.Timestamp(sample_start)]
    if sample_end:
        out = out.loc[out.index <= pd.Timestamp(sample_end)]
    return out


def _analog_frame(dataset: str, md: pd.DataFrame, qd: pd.DataFrame) -> pd.DataFrame:
    if dataset == "fred_md":
        return md
    if dataset == "fred_qd":
        return qd
    raise ValueError(f"unsupported analog dataset {dataset!r}")


def _analog_tcodes(dataset: str, md_tcodes: dict[str, int], qd_tcodes: dict[str, int]) -> dict[str, int]:
    if dataset == "fred_md":
        return md_tcodes
    if dataset == "fred_qd":
        return qd_tcodes
    raise ValueError(f"unsupported analog dataset {dataset!r}")


def _comparison_series(series: pd.Series, analog_dataset: str) -> pd.Series:
    clean = series.dropna().copy()
    if not isinstance(clean.index, pd.DatetimeIndex):
        return clean
    if analog_dataset == "fred_qd":
        periods = clean.index.to_period("Q")
    else:
        periods = clean.index.to_period("M")
    grouped = clean.groupby(periods).mean()
    grouped.index = grouped.index.astype(str)
    return grouped


def _evaluate_candidate(
    *,
    candidate: SdAnalogCandidate,
    code: int,
    sd_frame: pd.DataFrame,
    md: pd.DataFrame,
    qd: pd.DataFrame,
    md_tcodes: dict[str, int],
    qd_tcodes: dict[str, int],
) -> list[dict[str, object]]:
    sd_transformed = sd_frame.apply(lambda col: _fred_tcode_transform(col, code))
    sd_aggregate = sd_transformed.mean(axis=1, skipna=True)
    adf_rate, kpss_rate = _stationarity_rates(sd_transformed)
    sd_aggregate_adf_pass, sd_aggregate_kpss_pass = _adf_kpss(sd_aggregate)
    sd_low_frequency_ratio = _low_frequency_ratio(sd_aggregate)
    sd_outlier_rate = _outlier_rate(sd_aggregate)
    missing_rate = float(sd_transformed.isna().mean().mean()) if sd_transformed.size else None
    n_states_used = int((sd_transformed.notna().sum(axis=0) >= 20).sum())
    rows: list[dict[str, object]] = []

    if not candidate.candidate_analogs:
        rows.append(
            {
                "sd_variable": candidate.sd_variable,
                "candidate_code": code,
                "analog_dataset": None,
                "analog_series": None,
                "analog_official_code": None,
                "prior_confidence": candidate.prior_confidence,
                "n_overlap": 0,
                "aggregate_corr": None,
                "pearson_pvalue": None,
                "spearman_corr": None,
                "spearman_pvalue": None,
                "rolling_corr_mean": None,
                "rolling_corr_min": None,
                "state_corr_median": None,
                "state_corr_iqr": None,
                "state_corr_positive_share": None,
                "n_state_corr": 0,
                "adf_pass_rate": adf_rate,
                "kpss_pass_rate": kpss_rate,
                "sd_aggregate_adf_pass": sd_aggregate_adf_pass,
                "sd_aggregate_kpss_pass": sd_aggregate_kpss_pass,
                "analog_adf_pass": None,
                "analog_kpss_pass": None,
                "acf_distance": None,
                "low_frequency_ratio": sd_low_frequency_ratio,
                "analog_low_frequency_ratio": None,
                "low_frequency_distance": None,
                "outlier_rate": sd_outlier_rate,
                "analog_outlier_rate": None,
                "volatility_ratio": None,
                "missing_rate": missing_rate,
                "n_states_used": n_states_used,
                "score": 0.0,
                "note": candidate.note,
            }
        )
        return rows

    for analog in candidate.candidate_analogs:
        analog_data = _analog_frame(analog.dataset, md, qd)
        analog_codes = _analog_tcodes(analog.dataset, md_tcodes, qd_tcodes)
        if analog.series not in analog_data.columns:
            rows.append(
                {
                    "sd_variable": candidate.sd_variable,
                    "candidate_code": code,
                    "analog_dataset": analog.dataset,
                    "analog_series": analog.series,
                    "analog_official_code": None,
                    "prior_confidence": candidate.prior_confidence,
                    "n_overlap": 0,
                    "aggregate_corr": None,
                    "pearson_pvalue": None,
                    "spearman_corr": None,
                    "spearman_pvalue": None,
                    "rolling_corr_mean": None,
                    "rolling_corr_min": None,
                    "state_corr_median": None,
                    "state_corr_iqr": None,
                    "state_corr_positive_share": None,
                    "n_state_corr": 0,
                    "adf_pass_rate": adf_rate,
                    "kpss_pass_rate": kpss_rate,
                    "sd_aggregate_adf_pass": sd_aggregate_adf_pass,
                    "sd_aggregate_kpss_pass": sd_aggregate_kpss_pass,
                    "analog_adf_pass": None,
                    "analog_kpss_pass": None,
                    "acf_distance": None,
                    "low_frequency_ratio": sd_low_frequency_ratio,
                    "analog_low_frequency_ratio": None,
                    "low_frequency_distance": None,
                    "outlier_rate": sd_outlier_rate,
                    "analog_outlier_rate": None,
                    "volatility_ratio": None,
                    "missing_rate": missing_rate,
                    "n_states_used": n_states_used,
                    "score": 0.0,
                    "note": f"analog series missing: {analog.reason}",
                }
            )
            continue
        analog_code = int(analog_codes.get(analog.series, 1))
        analog_transformed = _fred_tcode_transform(analog_data[analog.series], analog_code)
        sd_compare = _comparison_series(sd_aggregate, analog.dataset)
        analog_compare = _comparison_series(analog_transformed, analog.dataset)
        joined = pd.concat([sd_compare.rename("sd"), analog_compare.rename("analog")], axis=1, join="inner").dropna()
        n_overlap = int(len(joined))
        corr_diagnostics = _corr_stats(joined)
        corr = corr_diagnostics["aggregate_corr"]
        acf_distance = _acf_distance(joined["sd"], joined["analog"]) if n_overlap >= 20 else None
        rolling_corr_mean, rolling_corr_min = _rolling_corr_stats(joined, analog.dataset)
        state_corr_diagnostics = _state_corr_stats(sd_transformed, analog_compare, analog.dataset)
        analog_adf_pass, analog_kpss_pass = _adf_kpss(analog_compare)
        low_freq_ratio = _low_frequency_ratio(joined["sd"])
        analog_low_freq_ratio = _low_frequency_ratio(joined["analog"])
        low_freq_distance = None
        if low_freq_ratio is not None and analog_low_freq_ratio is not None:
            low_freq_distance = abs(float(low_freq_ratio) - float(analog_low_freq_ratio))
        outlier_rate = _outlier_rate(joined["sd"])
        analog_outlier_rate = _outlier_rate(joined["analog"])
        volatility_ratio = _volatility_ratio(joined["sd"], joined["analog"])
        score = _score(
            corr=corr,
            adf_rate=adf_rate,
            kpss_rate=kpss_rate,
            acf_distance=acf_distance,
            low_freq_ratio=low_freq_ratio,
            outlier_rate=outlier_rate,
        )
        rows.append(
            {
                "sd_variable": candidate.sd_variable,
                "candidate_code": code,
                "analog_dataset": analog.dataset,
                "analog_series": analog.series,
                "analog_official_code": analog_code,
                "prior_confidence": candidate.prior_confidence,
                "n_overlap": n_overlap,
                "aggregate_corr": corr,
                "pearson_pvalue": corr_diagnostics["pearson_pvalue"],
                "spearman_corr": corr_diagnostics["spearman_corr"],
                "spearman_pvalue": corr_diagnostics["spearman_pvalue"],
                "rolling_corr_mean": rolling_corr_mean,
                "rolling_corr_min": rolling_corr_min,
                "state_corr_median": state_corr_diagnostics["state_corr_median"],
                "state_corr_iqr": state_corr_diagnostics["state_corr_iqr"],
                "state_corr_positive_share": state_corr_diagnostics["state_corr_positive_share"],
                "n_state_corr": state_corr_diagnostics["n_state_corr"],
                "adf_pass_rate": adf_rate,
                "kpss_pass_rate": kpss_rate,
                "sd_aggregate_adf_pass": sd_aggregate_adf_pass,
                "sd_aggregate_kpss_pass": sd_aggregate_kpss_pass,
                "analog_adf_pass": analog_adf_pass,
                "analog_kpss_pass": analog_kpss_pass,
                "acf_distance": acf_distance,
                "low_frequency_ratio": low_freq_ratio,
                "analog_low_frequency_ratio": analog_low_freq_ratio,
                "low_frequency_distance": low_freq_distance,
                "outlier_rate": outlier_rate,
                "analog_outlier_rate": analog_outlier_rate,
                "volatility_ratio": volatility_ratio,
                "missing_rate": missing_rate,
                "n_states_used": n_states_used,
                "score": score,
                "note": f"{candidate.note} Analog reason: {analog.reason}",
            }
        )
    return rows


def _clean_json(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    return value


def _selected_map(results: pd.DataFrame) -> dict[str, object]:
    variables: dict[str, object] = {}
    for sd_variable, group in results.groupby("sd_variable", sort=True):
        ranked = group.sort_values(["score", "n_overlap"], ascending=False)
        best = ranked.iloc[0].to_dict()
        reviewed = dict(SD_INFERRED_TCODE_MAP.get(sd_variable, {}))
        review_status = reviewed.get("status", "review_required")
        runtime_eligible = review_status in set(DEFAULT_RUNTIME_STATUSES)
        if not reviewed:
            selected = None if best.get("prior_confidence") == "reject" or not best.get("analog_series") else int(best["candidate_code"])
        elif reviewed.get("code_by_frequency"):
            selected = None
        elif runtime_eligible and reviewed.get("code") is not None:
            selected = int(reviewed["code"])
        else:
            selected = None
        diagnostics: dict[str, object] = {}
        for column in DIAGNOSTIC_COLUMNS:
            value = best.get(column)
            if column in {"n_overlap", "n_state_corr", "n_states_used"}:
                diagnostics[column] = int(value or 0)
            else:
                diagnostics[column] = value
        variables[sd_variable] = {
            "selected_code": selected,
            "status": review_status,
            "official": OFFICIAL,
            "source": SOURCE,
            "best_analog": None
            if not best.get("analog_series")
            else {
                "dataset": best.get("analog_dataset"),
                "series": best.get("analog_series"),
                "official_tcode": None if pd.isna(best.get("analog_official_code")) else int(best.get("analog_official_code")),
            },
            "prior_confidence": best.get("prior_confidence"),
            "score": float(best.get("score", 0.0)),
            "review_status": review_status,
            "runtime_eligible": bool(runtime_eligible),
            "code_by_frequency": reviewed.get("code_by_frequency"),
            "source_frequency": reviewed.get("source_frequency"),
            "review_reason": reviewed.get("reason"),
            "diagnostics": diagnostics,
        }
    return {
        "map_version": MAP_VERSION,
        "official": OFFICIAL,
        "source": SOURCE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variables": _clean_json(variables),
    }


def _best_rows(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["sd_variable", "score", "n_overlap"], ascending=[True, False, False])
        .groupby("sd_variable", sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def _format_report_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return ""
        if value != 0.0 and abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    visible = [column for column in columns if column in frame.columns]
    if not visible:
        return []
    lines = [
        "| " + " | ".join(visible) + " |",
        "| " + " | ".join("---" for _ in visible) + " |",
    ]
    for row in frame[visible].to_dict(orient="records"):
        lines.append("| " + " | ".join(_format_report_cell(row.get(column)) for column in visible) + " |")
    return lines


def _write_report(path: Path, results: pd.DataFrame, selected: dict[str, object]) -> None:
    best_rows = _best_rows(results)
    lines = [
        "# FRED-SD Inferred T-Code Validation",
        "",
        "FRED-SD does not provide official t-codes. This report ranks macroforecast-inferred candidates.",
        "",
        f"- map_version: `{selected['map_version']}`",
        f"- official: `{selected['official']}`",
        f"- source: `{selected['source']}`",
        "- runtime_default: `none` (FRED-SD inferred t-codes are opt-in)",
        f"- runtime_allowed_statuses: `{', '.join(DEFAULT_RUNTIME_STATUSES)}`",
        "",
        "## Top Candidate By Variable",
        "",
        "| SD variable | selected code | analog | confidence | score | n overlap | corr | review status | runtime eligible |",
        "|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    variables = selected["variables"]
    for sd_variable in sorted(variables):
        payload = variables[sd_variable]
        analog = payload["best_analog"]
        analog_text = "" if analog is None else f"{analog['dataset']}:{analog['series']}"
        diagnostics = payload["diagnostics"]
        corr = diagnostics.get("aggregate_corr")
        lines.append(
            "| {sd} | {code} | {analog} | {conf} | {score:.3f} | {n} | {corr} | {status} | {eligible} |".format(
                sd=sd_variable,
                code=_format_report_cell(payload.get("code_by_frequency")) if payload.get("code_by_frequency") else ("" if payload["selected_code"] is None else payload["selected_code"]),
                analog=analog_text,
                conf=payload["prior_confidence"],
                score=float(payload["score"]),
                n=diagnostics.get("n_overlap", 0),
                corr="" if corr is None else f"{float(corr):.3f}",
                status=payload.get("review_status", payload["status"]),
                eligible=payload.get("runtime_eligible"),
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Definitions",
            "",
            "- `aggregate_corr`: Pearson correlation between transformed SD state aggregate and transformed MD/QD analog.",
            "- `pearson_pvalue` and `spearman_pvalue`: optional scipy p-values for aggregate Pearson and Spearman correlation.",
            "- `rolling_corr_mean` and `rolling_corr_min`: rolling aggregate correlation; 24-month window for MD analogs, 8-quarter window for QD analogs.",
            "- `state_corr_*`: cross-state distribution of state-level correlations to the same transformed analog.",
            "- `adf_pass_rate` and `kpss_pass_rate`: state-level stationarity pass rates after the candidate SD transform.",
            "- `sd_aggregate_*` and `analog_*`: aggregate SD and analog stationarity / low-frequency / outlier diagnostics.",
            "- `volatility_ratio`: transformed SD aggregate standard deviation divided by transformed analog standard deviation.",
            "",
            "## Best Candidate Diagnostics",
            "",
            *_markdown_table(best_rows, REPORT_COLUMNS),
            "",
            "## All Candidate Diagnostics",
            "",
            *_markdown_table(results.sort_values(["sd_variable", "candidate_code", "analog_dataset", "analog_series"]), REPORT_COLUMNS),
            "",
            "The same rows are written to `sd_tcode_candidate_results.csv` for spreadsheet review.",
            "",
            "## Candidate Priors",
            "",
            "The candidate priors used for this run are serialized below for audit.",
            "",
            "```json",
            json.dumps([asdict(candidate) for candidate in SD_ANALOG_CANDIDATES], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-workbook", required=True, type=Path)
    parser.add_argument("--md-csv", required=True, type=Path)
    parser.add_argument("--qd-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-start")
    parser.add_argument("--sample-end")
    args = parser.parse_args()

    md, md_tcodes = parse_fred_csv(args.md_csv)
    qd, qd_tcodes = parse_fred_csv(args.qd_csv)
    md = _filter_sample(md, args.sample_start, args.sample_end)
    qd = _filter_sample(qd, args.sample_start, args.sample_end)

    rows: list[dict[str, object]] = []
    for candidate in SD_ANALOG_CANDIDATES:
        sd_frame = _load_sd_sheet(args.sd_workbook, candidate.sd_variable, args.sample_start, args.sample_end)
        for code in candidate.candidate_codes:
            rows.extend(
                _evaluate_candidate(
                    candidate=candidate,
                    code=int(code),
                    sd_frame=sd_frame,
                    md=md,
                    qd=qd,
                    md_tcodes=md_tcodes,
                    qd_tcodes=qd_tcodes,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "sd_tcode_candidate_results.csv", index=False)
    selected = _selected_map(results)
    (args.output_dir / "sd_tcode_selected_map.json").write_text(
        json.dumps(_clean_json(selected), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "sd_tcode_report.md", results, selected)
    print(f"wrote {len(results)} candidate rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
