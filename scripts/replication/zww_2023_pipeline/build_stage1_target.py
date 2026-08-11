#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import macroforecast as mf


RV_SAMPLE_START = pd.Timestamp("1986-06-01")
RV_SAMPLE_END = pd.Timestamp("2023-12-31")
FRED_SAMPLE_START = pd.Timestamp("1985-01-01")
FRED_SAMPLE_END = pd.Timestamp("2018-12-31")
HORIZONS = (1, 3, 6, 12)
PARITY_TOLERANCE = 1e-10
FRED_VINTAGE = "2019-06"
ACOGNO = "ACOGNO"

STALE_STAGE1_FILES = {
    "target.csv",
    "target.parquet",
    "panel.csv",
    "panel.parquet",
    "model_panel.parquet",
    "panel_schema.json",
    "targets_log_average_value.parquet",
}


def date_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_stale_outputs(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in sorted(STALE_STAGE1_FILES):
        path = output_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def strip_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.attrs = {}
    return out


def build_monthly_target(price_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(price_path)
    missing_price_days = int(raw["DCOILWTICO"].isna().sum())

    oil_prc = raw.rename(
        columns={"observation_date": "date", "DCOILWTICO": "prc"}
    ).copy()
    oil_prc.dropna(subset=["prc"], inplace=True)
    oil_prc.index = pd.to_datetime(oil_prc["date"])
    oil_prc.drop(columns=["date"], inplace=True)
    oil_prc["log_ret"] = np.log(oil_prc["prc"]) - np.log(oil_prc["prc"].shift(1))

    monthly_rv = oil_prc["log_ret"].pow(2).groupby(pd.Grouper(freq="M")).sum()
    monthly_rv.name = "RV"
    monthly_lv = np.log(monthly_rv)
    monthly_lv.name = "LV"

    target = pd.concat([monthly_rv, monthly_lv], axis=1)
    target.index.name = "date"
    target = target[(target.index >= RV_SAMPLE_START) & (target.index <= RV_SAMPLE_END)]
    target = target.copy()

    diagnostics = {
        "price_path": str(price_path),
        "raw_daily_rows": int(raw.shape[0]),
        "missing_price_days_dropped": missing_price_days,
        "nonmissing_price_days": int(oil_prc.shape[0]),
        "target_rows": int(target.shape[0]),
        "first_date": date_string(target.index.min()),
        "last_date": date_string(target.index.max()),
        "date_exclusion_rule": (
            "drop rows with missing DCOILWTICO before computing chained daily log returns"
        ),
    }
    return target, diagnostics


def oracle_notebook_target(price_path: Path) -> pd.DataFrame:
    oil_prc = pd.read_csv(price_path)
    oil_prc.rename(
        columns={"observation_date": "date", "DCOILWTICO": "prc"}, inplace=True
    )
    oil_prc.dropna(subset=["prc"], inplace=True)
    oil_prc.index = pd.to_datetime(oil_prc["date"])
    oil_prc.drop(columns=["date"], inplace=True)
    oil_prc["log_ret"] = np.log(oil_prc["prc"]) - np.log(oil_prc["prc"].shift(1))

    monthly_rv = oil_prc["log_ret"].pow(2).groupby(pd.Grouper(freq="M")).sum()
    monthly_rv.name = "RV"
    monthly_lv = np.log(monthly_rv)
    monthly_lv.name = "LV"

    target = pd.concat([monthly_rv, monthly_lv], axis=1)
    target.index.name = "date"
    return target[(target.index >= RV_SAMPLE_START) & (target.index <= RV_SAMPLE_END)].copy()


def rv_oracle_parity(build: pd.DataFrame, oracle: pd.DataFrame) -> dict[str, Any]:
    same_index = build.index.equals(oracle.index)
    aligned = build[["RV", "LV"]].join(
        oracle[["RV", "LV"]], how="inner", lsuffix="_build", rsuffix="_oracle"
    )
    diffs = pd.DataFrame(
        {
            "RV": aligned["RV_build"] - aligned["RV_oracle"],
            "LV": aligned["LV_build"] - aligned["LV_oracle"],
        },
        index=aligned.index,
    ).abs()
    max_abs_diff = float(diffs.to_numpy().max()) if not diffs.empty else float("inf")
    passed = bool(
        same_index
        and build.shape[0] == oracle.shape[0]
        and max_abs_diff <= PARITY_TOLERANCE
    )
    missing_from_build = [date_string(d) for d in oracle.index.difference(build.index)]
    missing_from_oracle = [date_string(d) for d in build.index.difference(oracle.index)]
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "tolerance": PARITY_TOLERANCE,
        "n_months": int(aligned.shape[0]),
        "build_n_months": int(build.shape[0]),
        "oracle_n_months": int(oracle.shape[0]),
        "same_index": bool(same_index),
        "first_date": date_string(aligned.index.min()) if len(aligned) else None,
        "last_date": date_string(aligned.index.max()) if len(aligned) else None,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_by_column": {
            column: float(value) for column, value in diffs.max(axis=0).items()
        },
        "date_diffs": {
            "missing_from_build": missing_from_build,
            "missing_from_oracle": missing_from_oracle,
        },
    }


def month_end_index(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out.index.name = "date"
    return out.sort_index()


def official_fred_md_vintage_source_url(vintage: str, cached_source: str | None) -> str | None:
    if cached_source and cached_source.startswith("http"):
        return cached_source
    entry = cached_source.split("#", 1)[1] if cached_source and "#" in cached_source else f"{vintage}.csv"
    try:
        from macroforecast.data.loaders import _fred_historical_zip_url

        archive_url = _fred_historical_zip_url("fred_md", vintage)
    except Exception:
        return None
    return f"{archive_url}#{entry}"


def build_fred_md_predictors() -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = mf.data.load_fred_md(vintage=FRED_VINTAGE)
    metadata = mf.data.metadata(bundle)
    raw = bundle.panel.copy()
    raw_columns = [str(column) for column in raw.columns]
    acogno_excluded = ACOGNO in raw.columns
    predictor_columns = [column for column in raw_columns if column != ACOGNO]

    tcode_map = {
        column: int(metadata["transform_codes"][column])
        for column in predictor_columns
        if column in metadata.get("transform_codes", {})
    }
    transformed = mf.preprocessing.apply_tcode_transform(raw[predictor_columns], tcode_map)

    lagged_bundle = mf.data.availability_lag(
        mf.data.DataBundle(
            panel=transformed,
            metadata={
                "dataset": "fred_md",
                "vintage": FRED_VINTAGE,
                "transform": "mccracken_ng_tcode",
            },
        ),
        lags=1,
        columns=predictor_columns,
        drop_missing=False,
    )
    lagged = month_end_index(lagged_bundle.panel)
    sample_start = FRED_SAMPLE_START.to_period("M").to_timestamp("M")
    sample_end = FRED_SAMPLE_END.to_period("M").to_timestamp("M")
    sample = lagged.loc[(lagged.index >= sample_start) & (lagged.index <= sample_end)]
    sample = sample[predictor_columns].copy()

    missing_counts = sample.isna().sum().astype(int)
    columns_with_missing = {
        column: int(count) for column, count in missing_counts.items() if int(count) > 0
    }
    artifact = metadata.get("artifact", {})
    cached_source = artifact.get("source_url")
    official_source = official_fred_md_vintage_source_url(FRED_VINTAGE, cached_source)
    local_path = Path(artifact["local_path"]) if artifact.get("local_path") else None
    diagnostics = {
        "loader": "mf.data.load_fred_md(vintage='2019-06')",
        "metadata_vintage": metadata.get("vintage"),
        "metadata_data_through": metadata.get("data_through"),
        "raw_n_variables": int(raw.shape[1]),
        "raw_has_ACOGNO": bool(ACOGNO in raw.columns),
        "acogno_excluded": bool(acogno_excluded),
        "n_predictors": int(sample.shape[1]),
        "expected_zww_predictors_after_acogno_exclusion": 127,
        "count_discrepancy": (
            "The package-returned official 2019-06 vintage has 127 raw fields including "
            "ACOGNO, so excluding ACOGNO leaves 126 predictors. No replacement column was "
            "fabricated."
        )
        if sample.shape[1] != 127
        else None,
        "raw_first_date": date_string(raw.index.min()),
        "raw_last_date": date_string(raw.index.max()),
        "first_date": date_string(sample.index.min()),
        "last_date": date_string(sample.index.max()),
        "n_rows": int(sample.shape[0]),
        "tcode_applied": True,
        "tcode_variable_count": int(len(tcode_map)),
        "publication_lag_applied": True,
        "publication_lag_months": 1,
        "publication_lag_variable_count": int(len(predictor_columns)),
        "publication_lag_policy": (
            "uniform one-month lag applied to all retained FRED-MD predictors; "
            "ZWW main text states many variables are lagged for publication delays "
            "but does not enumerate a narrower set"
        ),
        "publication_lag_variables": predictor_columns,
        "tcode_map": tcode_map,
        "source_url": official_source or cached_source,
        "cached_source_url": cached_source,
        "local_path": str(local_path) if local_path else None,
        "file_sha256": (
            artifact.get("file_sha256")
            or (sha256_file(local_path) if local_path and local_path.exists() else None)
        ),
        "file_size_bytes": artifact.get("file_size_bytes"),
        "cache_hit": artifact.get("cache_hit"),
        "columns_with_missing_values": columns_with_missing,
        "total_missing_values": int(missing_counts.sum()),
        "predictor_columns": predictor_columns,
    }
    return sample, diagnostics


def build_log_average_targets(target: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_panel = mf.data.as_panel(
        target[["RV"]].reset_index(),
        date="date",
        metadata={
            "dataset": "zww_b3_stage1_spot_rv",
            "source_family": "owner_validated_spot_oracle",
            "frequency": "monthly",
        },
    )
    horizon_targets = mf.feature_engineering.direct_target(
        target_panel,
        target="RV",
        horizons=HORIZONS,
        transform="log_average_value",
    )

    checks: dict[str, Any] = {}
    for horizon in HORIZONS:
        column = f"RV_log_average_value_h{horizon}"
        if horizon == 1:
            reference = target["LV"].shift(-1)
            description = "h=1 package target equals ln(RV[t+1])"
        else:
            future_values = pd.concat(
                [target["RV"].shift(-step) for step in range(1, horizon + 1)],
                axis=1,
            )
            reference = np.log(future_values.mean(axis=1, skipna=False))
            description = f"h={horizon} equals log(mean(RV[t+1]..RV[t+{horizon}]))"

        comparison = pd.concat(
            [horizon_targets[column].rename("package"), reference.rename("reference")],
            axis=1,
        ).dropna()
        comparison["abs_diff"] = (comparison["package"] - comparison["reference"]).abs()
        max_abs_diff = (
            float(comparison["abs_diff"].max()) if not comparison.empty else float("inf")
        )

        sample_index: list[pd.Timestamp] = []
        if not comparison.empty:
            candidates = [
                comparison.index[0],
                pd.Timestamp("2000-01-31"),
                comparison.index[-1],
            ]
            for candidate in candidates:
                if candidate in comparison.index and candidate not in sample_index:
                    sample_index.append(candidate)
        sample_records = [
            {
                "origin": date_string(date),
                "package": float(comparison.loc[date, "package"]),
                "reference": float(comparison.loc[date, "reference"]),
                "abs_diff": float(comparison.loc[date, "abs_diff"]),
            }
            for date in sample_index
        ]
        checks[f"h{horizon}"] = {
            "passed": bool(max_abs_diff <= PARITY_TOLERANCE),
            "description": description,
            "finite_origins_checked": int(comparison.shape[0]),
            "max_abs_diff": max_abs_diff,
            "sample_origins": sample_records,
        }

    checks["all_passed"] = bool(all(checks[f"h{h}"]["passed"] for h in HORIZONS))
    checks["tolerance"] = PARITY_TOLERANCE
    return horizon_targets, checks


def build_manifest(
    *,
    price_path: Path,
    target_diag: dict[str, Any],
    parity: dict[str, Any],
    fred_diag: dict[str, Any],
    horizon_checks: dict[str, Any],
    removed_stale_outputs: list[str],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_scope": "ZWW 2023 B3 Stage-1 corrected data build; no package patch; no full run",
        "removed_stale_outputs": removed_stale_outputs,
        "sources": {
            "spot_price_oracle": {
                "path": str(price_path),
                "sha256": sha256_file(price_path),
                "series": "DCOILWTICO",
                "series_type": "WTI spot price",
                "role": "owner-validated RV/LV oracle only",
            },
            "fred_md_2019_06": {
                "loader": fred_diag["loader"],
                "source_url": fred_diag["source_url"],
                "cached_source_url": fred_diag["cached_source_url"],
                "local_path": fred_diag["local_path"],
                "sha256": fred_diag["file_sha256"],
                "file_size_bytes": fred_diag["file_size_bytes"],
                "vintage": fred_diag["metadata_vintage"],
                "data_through": fred_diag["metadata_data_through"],
            },
        },
        "rv_target": {
            **target_diag,
            "oracle_parity": parity,
            "output": "runs/zww_b3_stage1/wti_rv_monthly.csv",
            "market": "spot",
        },
        "fred_md_predictors": fred_diag,
        "multi_horizon_target": {
            "horizons": list(HORIZONS),
            "formula": "LV_{t+1:t+h} = ln(mean(RV[t+1], ..., RV[t+h]))",
            "transform": "log_average_value",
            "checks": horizon_checks,
            "output": "runs/zww_b3_stage1/targets_log_average_value.csv",
        },
        "sample_boundaries": {
            "spot_rv_target": {
                "start": date_string(RV_SAMPLE_START),
                "end": date_string(RV_SAMPLE_END),
            },
            "fred_md_predictors": {
                "start": "1985-01-31",
                "end": "2018-12-31",
            },
        },
        "spot_vs_futures": {
            "this_build": (
                "WTI spot RV from DCOILWTICO, matching ZWW Table 4 spot robustness target"
            ),
            "open_dependency": (
                "ZWW Table 3 main target requires daily WTI futures RV from EIA futures data; "
                "that source is not in the staged archive and was not fabricated"
            ),
        },
        "excluded_sources": {
            "krw_fred_md_csv": "not used",
            "krw_uncertainty_series": "not used",
            "krw_oil_fundamentals": "not used",
        },
        "remaining_package_addition_for_table3": (
            "general AICc lambda-selection unit for lasso/elastic_net in the A2 IC lane"
        ),
    }


def horizon_line(checks: dict[str, Any]) -> str:
    parts = []
    for h in HORIZONS:
        item = checks[f"h{h}"]
        status = "PASS" if item["passed"] else "FAIL"
        parts.append(
            f"h={h} {status} max_abs_diff={item['max_abs_diff']:.17g} "
            f"n={item['finite_origins_checked']}"
        )
    return "; ".join(parts)


def stop_report(summary: dict[str, Any]) -> str:
    parity = summary["rv_oracle_parity"]
    fred = summary["fred_md_2019_06"]
    return "\n".join(
        [
            "RV oracle parity: "
            f"max abs diff={parity['max_abs_diff']:.17g}; "
            f"n months={parity['n_months']}; {parity['status']}.",
            "FRED-MD 2019:06: "
            f"n vars={fred['n_predictors']}; "
            f"ACOGNO-excluded={fred['acogno_excluded']}; "
            "tcode/lag applied=True/True; "
            f"date range={fred['first_date']} to {fred['last_date']}.",
            "Multi-horizon target: " + horizon_line(summary["multi_horizon"]),
            "Spot-vs-futures: this build is WTI spot/Table-4 target; "
            "Table 3 futures RV remains an open data dependency. "
            "Confirmed remaining package addition: general AICc lambda-selection "
            "unit for lasso/elastic_net.",
        ]
    ) + "\n"


def notes_markdown(summary: dict[str, Any]) -> str:
    parity = summary["rv_oracle_parity"]
    fred = summary["fred_md_2019_06"]
    horizons = summary["multi_horizon"]
    discrepancy = fred.get("count_discrepancy")
    discrepancy_text = (
        f"\nCount discrepancy recorded: {discrepancy}\n" if discrepancy else ""
    )
    missing_text = (
        f"\nMissing predictor cells after tcode+lag/sample: {fred['total_missing_values']} "
        f"across {len(fred['columns_with_missing_values'])} columns.\n"
    )
    return f"""# B3 Stage-1 Notes: ZWW 2023 Corrected Data Build

Date: 2026-07-13

Scope: stage-1 data build only. No `macroforecast/**` patch, no Table 3 run, no push.

## 1. RV Oracle Parity

- Status: {parity['status']}.
- Months checked: {parity['n_months']}.
- Date range: {parity['first_date']} to {parity['last_date']}.
- Max absolute difference: {parity['max_abs_diff']:.17g}.
- Missing daily price rows dropped before log returns: {summary['target_diagnostics']['missing_price_days_dropped']}.

Output: `runs/zww_b3_stage1/wti_rv_monthly.csv`.

## 2. FRED-MD 2019:06 Predictors

- Loader: `mf.data.load_fred_md(vintage="2019-06")`.
- Raw loader variables: {fred['raw_n_variables']}; raw `ACOGNO` present: {fred['raw_has_ACOGNO']}.
- `ACOGNO` excluded: {fred['acogno_excluded']}.
- Output predictor variables: {fred['n_predictors']}.
- McCracken-Ng t-code transforms applied: {fred['tcode_applied']} ({fred['tcode_variable_count']} variables).
- One-month publication lag applied: {fred['publication_lag_applied']} ({fred['publication_lag_variable_count']} variables).
- Predictor date range: {fred['first_date']} to {fred['last_date']} ({fred['n_rows']} rows).
- FRED source hash: `{fred['file_sha256']}`.
{discrepancy_text}{missing_text}
Output: `runs/zww_b3_stage1/fred_md_2019_06.csv`.

## 3. Multi-Horizon Target Verification

{horizon_line(horizons)}

Output: `runs/zww_b3_stage1/targets_log_average_value.csv`.

## 4. Spot vs Futures

This build uses WTI spot `DCOILWTICO`, matching ZWW's Table 4 spot robustness target. ZWW's main Table 3 requires daily WTI futures RV from EIA futures data; that source is not in the staged archive and remains an open data dependency.

## 5. Remaining Package Addition

The confirmed Table 3 fix-lane package addition remains a general AICc lambda-selection unit for `lasso`/`elastic_net` in the A2 information-criterion lane. This Stage-1 task did not patch `macroforecast/**`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corrected ZWW 2023 B3 stage-1 spot RV and FRED-MD predictor data."
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        # Repository-local by default. The previous default pointed into a personal
        # ~/second_brain path, which meant a fresh clone silently looked somewhere
        # that does not exist for anyone else (external review, 2026-08-09).
        default=Path(os.environ.get("ZWW_STAGING_ROOT", "data/zww_2023_staging")).expanduser(),
        help="directory holding the staged crude-oil inputs; see replication.yaml "
             "for what belongs in it and where each file comes from",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/zww_b3_stage1"))
    parser.add_argument(
        "--notes-path",
        type=Path,
        default=Path(".dev-notes/b3_stage1_notes.md"),
    )
    parser.add_argument(
        "--stop-report-path",
        type=Path,
        default=Path("qa/codex_last_msg_b3stage1.txt"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    staging_root = args.staging_root.expanduser().resolve()
    data_dir = staging_root / "data"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_stale_outputs = clean_stale_outputs(output_dir)

    price_path = data_dir / "crude_oil_prc.csv"
    target, target_diag = build_monthly_target(price_path)
    oracle = oracle_notebook_target(price_path)
    parity = rv_oracle_parity(target, oracle)
    parity["target_diagnostics"] = target_diag
    write_json(output_dir / "rv_oracle_parity.json", parity)

    if not parity["passed"]:
        report = (
            "RV oracle parity: "
            f"max abs diff={parity['max_abs_diff']:.17g}; "
            f"n months={parity['n_months']}; FAIL.\n"
            f"Date diff: {json.dumps(parity['date_diffs'], sort_keys=True)}\n"
        )
        args.stop_report_path.write_text(report)
        print(report, end="")
        return 1

    target.to_csv(output_dir / "wti_rv_monthly.csv", index=True)

    fred_panel, fred_diag = build_fred_md_predictors()
    fred_panel.to_csv(output_dir / "fred_md_2019_06.csv", index=True)

    horizon_targets, horizon_checks = build_log_average_targets(target)
    strip_attrs(horizon_targets).to_csv(
        output_dir / "targets_log_average_value.csv", index=True
    )
    write_json(output_dir / "multi_horizon_checks.json", horizon_checks)

    manifest = build_manifest(
        price_path=price_path,
        target_diag=target_diag,
        parity=parity,
        fred_diag=fred_diag,
        horizon_checks=horizon_checks,
        removed_stale_outputs=removed_stale_outputs,
    )
    write_json(output_dir / "data_manifest.json", manifest)

    summary = {
        "rv_oracle_parity": {
            "status": parity["status"],
            "passed": parity["passed"],
            "max_abs_diff": parity["max_abs_diff"],
            "n_months": parity["n_months"],
            "first_date": parity["first_date"],
            "last_date": parity["last_date"],
        },
        "target_diagnostics": target_diag,
        "fred_md_2019_06": {
            key: value
            for key, value in fred_diag.items()
            if key
            not in {
                "publication_lag_variables",
                "tcode_map",
                "predictor_columns",
            }
        },
        "multi_horizon": horizon_checks,
        "outputs": {
            "rv": str(output_dir / "wti_rv_monthly.csv"),
            "fred_md": str(output_dir / "fred_md_2019_06.csv"),
            "targets": str(output_dir / "targets_log_average_value.csv"),
            "manifest": str(output_dir / "data_manifest.json"),
            "notes": str(args.notes_path),
            "stop_report": str(args.stop_report_path),
        },
    }
    write_json(output_dir / "stage1_summary.json", summary)

    args.notes_path.parent.mkdir(parents=True, exist_ok=True)
    args.notes_path.write_text(notes_markdown(summary))

    report = stop_report(summary)
    args.stop_report_path.parent.mkdir(parents=True, exist_ok=True)
    args.stop_report_path.write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
