"""Generate the models-and-features guide pages from the live package surface.

This writes one overview page plus one detail page per model family, all under a
base directory:

    <base>/model_overview.md          overview (features + choosing guide + index)
    <base>/models/<family>.md         per-family detail tables

The catalog is generated from ``mf.list_model_specs()`` and the feature-step
list from the ``*_step`` builders, so the pages can never drift from the
installed package. Guide model names are checked against the registry at render
time, so the prose cannot reference a model that does not exist.

Usage::

    # Regenerate the committed pages:
    python tools/gen_model_overview.py --out docs/guide

    # CI drift gate (non-zero exit if any committed page is stale):
    python tools/gen_model_overview.py --check docs/guide

This is a standalone script (like ``tools/gen_standalone_docs.py``) and does not
import the ``tools.docgen`` package.
"""
from __future__ import annotations

import argparse
import inspect
import os as _os
import sys
from pathlib import Path

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import macroforecast as mf  # noqa: E402

_HEADER = """# Models and Features

[Back to documentation home](../index.md)

An `Arm` is built from two kinds of block, a set of feature steps and a single
model. This page lists both. It starts with the feature engineering steps that
turn a cleaned panel into model inputs, then helps you choose a model. Each model
family links to a detail page generated from the registry, so the lists always
match the installed version.
"""

_FEATURE_INTRO = """## Feature engineering

Features are organized into five families that recur across the macro
forecasting literature. F is principal-component or sparse factors, X is raw
lagged series, MARX is the moving-average lag cross that is the standard macro
design, MAF is maximum-autocorrelation factors, and Level passes untransformed
level columns through unchanged. The [Features](concepts/features.md) page covers
them in full, and the [Feature Engineering reference](../reference/feature_engineering.md)
lists every step parameter.

You compose these families from the step builders below. Each returns a step you
place in the `feature_steps` list of `feature_spec`, and stateful steps such as
PCA are refit inside every training window.
"""

_MODEL_GUIDE = """## Choosing a model

**Few predictors and a mostly linear signal.** Start from the benchmarks `ar`,
`ols`, and `arima`. They anchor any comparison and are cheap to refit at every
origin.

**Many predictors.** Regularize. `ridge` shrinks all coefficients, `lasso` and
`elastic_net` also select variables, and `adaptive_lasso` and `group_lasso` add
structured selection across feature blocks.

**Latent factor structure.** When series move together, extract common factors
with `far` and `favar`, or use a dynamic factor model from the mixed-frequency
family.

**Nonlinearity and interactions.** The macro forecasting literature finds this is
where the largest gains appear. The workhorses are `random_forest`, `xgboost`,
`lightgbm`, and the macro-adapted `macro_random_forest`.

**Sequence structure.** The neural family includes `lstm` and `gru` for recurrent
dynamics in longer panels.

**Conditional volatility.** For variance forecasting use `garch11`, `egarch`,
`gjr_garch`, and `realized_garch`.
"""

_GUIDE_MODELS = {
    "ar", "ols", "arima", "ridge", "lasso", "elastic_net", "adaptive_lasso",
    "group_lasso", "far", "favar", "random_forest", "xgboost", "lightgbm",
    "macro_random_forest", "lstm", "gru", "garch11", "egarch", "gjr_garch",
    "realized_garch",
}

_FAMILY_ORDER = [
    "linear", "factor", "timeseries", "tree", "support_vector",
    "nonparametric", "neural", "volatility", "mixed_frequency",
    "assemblage", "composite", "spline",
]
_FAMILY_TITLES = {
    "linear": "Linear and regularized",
    "factor": "Factor models",
    "timeseries": "Classical time series",
    "tree": "Tree ensembles",
    "support_vector": "Support vector",
    "nonparametric": "Nonparametric",
    "neural": "Neural networks",
    "volatility": "Volatility and GARCH",
    "mixed_frequency": "Mixed frequency",
    "assemblage": "Assemblage",
    "composite": "Composite",
    "spline": "Spline",
}
# Short curated intro per family. Families without an entry fall back to the
# generic line in build_family_page().
_FAMILY_INTROS = {
    "linear": "Linear and regularized models predict with a weighted sum of "
    "features. They run from ordinary least squares through ridge, lasso, and "
    "elastic net shrinkage to structured and adaptive penalties that also select "
    "variables.",
    "factor": "Factor models summarize many comoving series into a few latent "
    "factors and forecast from those, which suits the strong common movement in "
    "macro panels.",
    "timeseries": "Classical time series models work from the target's own "
    "history, including autoregressions, ARIMA, and exponential smoothing, and "
    "serve as the standard benchmarks.",
    "tree": "Tree ensembles average or boost many decision trees. They capture "
    "nonlinearity and interactions automatically and are the workhorses behind "
    "the largest reported macro forecasting gains.",
    "support_vector": "Support vector regression fits a margin-based predictor "
    "and can use nonlinear kernels for flexible but controlled fits.",
    "nonparametric": "Nonparametric models make few functional-form assumptions "
    "and let the data shape the fit.",
    "neural": "Neural networks learn flexible nonlinear maps, including recurrent "
    "forms for sequence structure in longer panels.",
    "volatility": "Volatility models forecast conditional variance rather than "
    "the level, using the GARCH family and its asymmetric and realized "
    "extensions.",
    "mixed_frequency": "Mixed-frequency models combine series sampled at "
    "different frequencies, for example monthly predictors for a quarterly "
    "target, through MIDAS and dynamic factor designs.",
    "assemblage": "Assemblage models aggregate many component predictions or "
    "ranks into a single forecast.",
    "composite": "Composite models combine several base learners inside one fit.",
    "spline": "Spline models fit smooth nonlinear functions of the predictors "
    "using basis expansions.",
}

_OVERVIEW_FOOTER = """## Notes

Feature steps are passed in the `feature_steps` list of
`mf.feature_engineering.feature_spec(...)`. Model strings are passed as the
`model` argument to `Arm(model=...)` or to `mf.forecasting.run(data, model=...)`.
Full feature-step parameters are on the
[Feature Engineering reference page](../reference/feature_engineering.md), and
model search spaces and presets are on the
[Models reference page](../reference/models.md) and, for fit-time ensembles, the
[Model Ensemble reference page](../reference/model_ensemble.md). The generated
[Model x Forecast Policy Matrix](model_policy_matrix.md) states which forecast
policies are supported for each registered model.
"""


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _extra(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text in ("", "None", "nan"):
        return "none"
    return f"`{text}`"


def _scaling(value: object) -> str:
    return "yes" if bool(value) else "no"


def _preproc(value: object) -> str:
    if value is None:
        return "default"
    if isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value if str(x).strip()]
        return _cell(", ".join(items)) if items else "default"
    text = str(value).strip().strip("()").strip().rstrip(",").strip()
    return _cell(text) if text else "default"


def _step_description(fn: object) -> str:
    doc = (inspect.getdoc(fn) or "").splitlines()
    text = doc[0] if doc else ""
    for prefix in ("Return a reusable ", "Return a target-aware ", "Return a "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    for suffix in (
        " for ``compose_features`` or ``feature_spec``",
        " for ``compose_features``",
        " for ``feature_spec``",
    ):
        text = text.replace(suffix, "")
    text = text.strip().rstrip(".")
    if not text:
        return "Feature step."
    return text[0].upper() + text[1:] + "."


def _feature_steps_table() -> list[str]:
    names = sorted(
        n for n in dir(mf)
        if n.endswith("_step") and not n.startswith("custom")
    )
    rows = ["| Step builder | Description |", "| --- | --- |"]
    for name in names:
        rows.append(f"| `{name}` | {_cell(_step_description(getattr(mf, name)))} |")
    return rows


def _ordered_families(df) -> list[str]:
    families = list(_FAMILY_ORDER)
    for fam in sorted(set(df["family"]) - set(families)):
        families.append(fam)
    return [f for f in families if not df[df["family"] == f].empty]


def build_overview(df) -> str:
    missing = _GUIDE_MODELS - set(df["name"])
    if missing:
        raise ValueError(
            f"Guide references models absent from the registry: {sorted(missing)}"
        )

    parts: list[str] = [_HEADER.rstrip(), ""]
    parts.append(_FEATURE_INTRO.rstrip())
    parts.append("")
    parts.extend(_feature_steps_table())
    parts.append("")
    parts.append(_MODEL_GUIDE.rstrip())
    parts.append("")
    parts.append("## Model families")
    parts.append("")
    parts.append(
        "Each family has a detail page with a per-model table of inputs, optional "
        "dependencies, scaling, recommended preprocessing, and tunable counts."
    )
    parts.append("")
    for fam in _ordered_families(df):
        n = len(df[df["family"] == fam])
        title = _FAMILY_TITLES.get(fam, fam.replace("_", " ").title())
        parts.append(f"- [{title}](models/{fam}.md) — {n} models")
    parts.append("")
    parts.append(_OVERVIEW_FOOTER.rstrip())
    parts.append("")

    parts.append("```{toctree}")
    parts.append(":hidden:")
    parts.append(":maxdepth: 1")
    parts.append("")
    for fam in _ordered_families(df):
        parts.append(f"models/{fam}")
    parts.append("model_policy_matrix")
    parts.append("```")
    parts.append("")
    return "\n".join(parts)


def build_family_page(fam: str, df) -> str:
    sub = df[df["family"] == fam].sort_values("name")
    title = _FAMILY_TITLES.get(fam, fam.replace("_", " ").title())
    intro = _FAMILY_INTROS.get(
        fam, f"The {title.lower()} family groups related model specifications."
    )

    parts: list[str] = [f"# {title}", ""]
    parts.append("[Back to Models and Features](../model_overview.md)")
    parts.append("")
    parts.append(intro)
    parts.append("")
    parts.append(
        "Pass any model string below as `Arm(model=...)`. Extra names an optional "
        "dependency, Scaling flags whether predictors should be standardized, and "
        "Tunable counts the hyperparameters the search space exposes."
    )
    parts.append("")
    parts.append(
        "| Model string | Description | Input | Extra | Scaling | "
        "Recommended preprocessing | Tunable |"
    )
    parts.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in sub.iterrows():
        parts.append(
            f"| `{row['name']}` | {_cell(row['description'])} | "
            f"{_cell(row['input_kind'])} | {_extra(row['requires_extra'])} | "
            f"{_scaling(row['requires_scaling'])} | "
            f"{_preproc(row['recommended_preprocessing'])} | "
            f"{int(row['n_tunable'])} |"
        )
    parts.append("")
    parts.append("## Reference")
    parts.append("")
    parts.append(
        "- [Models reference page](../../reference/models.md) for `ModelSpec`, "
        "`ModelFit`, and fit conventions."
    )
    parts.append("")
    return "\n".join(parts)


def _targets(base: Path, df) -> dict[Path, str]:
    """Map every output path to its rendered content."""
    out: dict[Path, str] = {base / "model_overview.md": build_overview(df)}
    for fam in _ordered_families(df):
        out[base / "models" / f"{fam}.md"] = build_family_page(fam, df)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", help="Base guide directory to write into.")
    group.add_argument(
        "--check", help="Base guide directory to verify; exit 1 if stale."
    )
    args = parser.parse_args(argv)

    df = mf.list_model_specs()

    if args.check:
        base = Path(args.check)
        targets = _targets(base, df)
        stale = []
        for path, content in targets.items():
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != content:
                stale.append(path)
        if stale:
            listing = "\n".join(f"  {p}" for p in stale)
            print(
                f"Model pages out of sync with the package:\n{listing}\n"
                f"Run: python tools/gen_model_overview.py --out {base}",
                file=sys.stderr,
            )
            return 1
        print(f"{len(targets)} model pages in sync with the package.")
        return 0

    base = Path(args.out)
    targets = _targets(base, df)
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(targets)} pages under {base}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
