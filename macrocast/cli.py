"""macrocast command-line interface.

Usage
-----
    macrocast --help
    macrocast run experiment.yaml
    macrocast init [--output experiment.yaml]
    macrocast info experiment.yaml

Commands
--------
run     Execute a forecast experiment from a YAML config file.
init    Write a default YAML config template to disk.
info    Print a summary of the resolved config without running.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("macrocast")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a forecast experiment defined in a YAML config file."""
    from macrocast.config import load_config
    from macrocast.pipeline.experiment import ForecastExperiment

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Config error: %s", exc)
        return 1

    logger.info("Experiment: %s", cfg.experiment_id)
    logger.info(
        "Dataset:    %s  (target=%s, vintage=%s)",
        cfg.data.dataset,
        cfg.data.target,
        cfg.data.vintage or "current",
    )
    logger.info("Models:     %d configured", len(cfg.model_specs))
    logger.info("Horizons:   %s", cfg.horizons)
    logger.info("Window:     %s", cfg.window.value)
    logger.info("OOS range:  %s → %s", cfg.oos_start or "auto", cfg.oos_end or "end")

    # Load data
    try:
        panel, target = _load_data(cfg)
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return 1

    output_dir = cfg.output_dir / cfg.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = ForecastExperiment(
        panel=panel,
        target=target,
        horizons=cfg.horizons,
        model_specs=cfg.model_specs,
        feature_spec=cfg.feature_spec,
        window=cfg.window,
        rolling_size=cfg.rolling_size,
        oos_start=cfg.oos_start,
        oos_end=cfg.oos_end,
        n_jobs=cfg.n_jobs,
        experiment_id=cfg.experiment_id,
        output_dir=output_dir,
    )

    logger.info("Running experiment...")
    rs = exp.run()

    logger.info("Done. %d forecast records.", len(rs))

    if args.summary:
        _print_summary(rs)

    return 0


def _load_data(cfg):
    """Load dataset and extract panel + target from MacroFrame."""
    from macrocast.data import load_fred_md, load_fred_qd

    dataset = cfg.data.dataset.lower().replace("-", "_")
    cache_dir = Path(cfg.data.cache_dir).expanduser() if cfg.data.cache_dir else None

    if dataset == "fred_md":
        mf = load_fred_md(vintage=cfg.data.vintage, cache_dir=cache_dir)
    elif dataset == "fred_qd":
        mf = load_fred_qd(vintage=cfg.data.vintage, cache_dir=cache_dir)
    else:
        raise ValueError(
            f"Dataset '{cfg.data.dataset}' not supported via CLI. "
            "Use fred_md or fred_qd.  For fred_sd, load programmatically."
        )

    # Apply stationarity transforms if needed
    if not mf.metadata.is_transformed:
        mf = mf.transform()

    df = mf.data

    if cfg.data.target not in df.columns:
        raise ValueError(
            f"Target '{cfg.data.target}' not found in dataset. "
            f"Available columns: {list(df.columns[:10])} ..."
        )

    target = df[cfg.data.target]
    panel = df.drop(columns=[cfg.data.target])

    # Drop columns with all NaN
    panel = panel.dropna(axis=1, how="all")

    # Restrict to rows where target is observed
    mask = target.notna()
    panel = panel.loc[mask]
    target = target.loc[mask]

    return panel, target


def _print_summary(rs) -> None:
    """Print MSFE table to stdout."""
    try:
        summary = rs.msfe_by_model()
        if summary.empty:
            print("No results to summarise.")
            return
        print("\n--- MSFE Summary ---")
        print(summary.to_string(index=False))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Write the default YAML config template."""
    from macrocast.config import DEFAULT_CONFIG_YAML

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        logger.error("File already exists: %s. Use --force to overwrite.", out_path)
        return 1

    out_path.write_text(DEFAULT_CONFIG_YAML)
    logger.info("Config template written to: %s", out_path)
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def _cmd_info(args: argparse.Namespace) -> int:
    """Print a resolved config summary without running."""
    from macrocast.config import load_config

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Config error: %s", exc)
        return 1

    print(f"Experiment ID:  {cfg.experiment_id}")
    print(f"Output dir:     {cfg.output_dir}")
    print(f"Dataset:        {cfg.data.dataset}")
    print(f"Target:         {cfg.data.target}")
    print(f"Vintage:        {cfg.data.vintage or 'current'}")
    print(f"Horizons:       {cfg.horizons}")
    print(f"Window:         {cfg.window.value}")
    print(f"OOS start:      {cfg.oos_start or 'auto'}")
    print(f"OOS end:        {cfg.oos_end or 'auto'}")
    print(f"n_jobs:         {cfg.n_jobs}")
    print(f"Models ({len(cfg.model_specs)}):")
    for spec in cfg.model_specs:
        print(f"  - {spec.model_id}")
    print("Features:")
    fs = cfg.feature_spec
    print(
        f"  factor_type={fs.factor_type!r}, n_factors={fs.n_factors}, "
        f"n_lags={fs.n_lags}, lookback={fs.lookback}"
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="macrocast",
        description="macrocast — Decomposing ML Forecast Gains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # run
    run_parser = subparsers.add_parser(
        "run", help="Run a forecast experiment from a YAML config."
    )
    run_parser.add_argument(
        "config", metavar="CONFIG.yaml", help="Path to the YAML experiment config file."
    )
    run_parser.add_argument(
        "--summary", action="store_true", help="Print MSFE summary table after the run."
    )

    # init
    init_parser = subparsers.add_parser(
        "init", help="Write a default YAML config template."
    )
    init_parser.add_argument(
        "--output",
        "-o",
        default="experiment.yaml",
        help="Output file path (default: experiment.yaml).",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing file."
    )

    # info
    info_parser = subparsers.add_parser("info", help="Print a resolved config summary.")
    info_parser.add_argument(
        "config", metavar="CONFIG.yaml", help="Path to the YAML config file."
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose)

    dispatch = {
        "run": _cmd_run,
        "init": _cmd_init,
        "info": _cmd_info,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
