"""F-P0-1 -- _is_iso_date and _normalize_iso_partial accept partial ISO forms.

Tests:
- YYYY-MM-DD (full) accepted
- YYYY-MM (month-only) accepted, normalized to YYYY-MM-01 / YYYY-MM-<last>
- YYYY (year-only) accepted, normalized to YYYY-01-01 / YYYY-12-31
- "abc" rejected
- Non-string rejected
"""
from __future__ import annotations

import pytest

from macroforecast.core.layers.l1 import _is_iso_date, _normalize_iso_partial


class TestIsIsoDate:
    def test_full_date_accepted(self):
        assert _is_iso_date("1980-01-15") is True

    def test_year_month_accepted(self):
        assert _is_iso_date("1980-01") is True

    def test_year_only_accepted(self):
        assert _is_iso_date("1980") is True

    def test_invalid_string_rejected(self):
        assert _is_iso_date("abc") is False

    def test_empty_string_rejected(self):
        assert _is_iso_date("") is False

    def test_none_rejected(self):
        assert _is_iso_date(None) is False

    def test_integer_rejected(self):
        assert _is_iso_date(1980) is False


class TestNormalizeIsoPartial:
    def test_full_date_passthrough(self):
        result = _normalize_iso_partial("1980-01-15")
        assert result == "1980-01-15"

    def test_year_month_start(self):
        result = _normalize_iso_partial("1980-01", end_of_period=False)
        assert result == "1980-01-01"

    def test_year_month_end_of_period(self):
        result = _normalize_iso_partial("1980-01", end_of_period=True)
        assert result == "1980-01-31"

    def test_year_month_end_feb_leap(self):
        result = _normalize_iso_partial("2000-02", end_of_period=True)
        assert result == "2000-02-29"

    def test_year_month_end_feb_non_leap(self):
        result = _normalize_iso_partial("2001-02", end_of_period=True)
        assert result == "2001-02-28"

    def test_year_only_start(self):
        result = _normalize_iso_partial("1990", end_of_period=False)
        assert result == "1990-01-01"

    def test_year_only_end(self):
        result = _normalize_iso_partial("1990", end_of_period=True)
        assert result == "1990-12-31"

    def test_invalid_returns_none(self):
        assert _normalize_iso_partial("abc") is None

    def test_none_returns_none(self):
        assert _normalize_iso_partial(None) is None


class TestPartialIsoInRecipe:
    """Smoke-tests that sample_start_date / sample_end_date accept partial ISO forms."""

    def test_year_month_accepted_in_recipe(self):
        """A recipe with sample_start_date="1980-01" must not raise at L1 validate."""
        from macroforecast.core.yaml import parse_recipe_yaml
        from macroforecast.core.layers import l1 as l1_layer

        yaml_text = """
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    sample_start_rule: fixed_date
  leaf_config:
    target: y
    sample_start_date: "1980-01"
    custom_panel_inline:
      date: [2020-01-01, 2020-02-01, 2020-03-01]
      y: [1.0, 2.0, 3.0]
"""
        recipe = parse_recipe_yaml(yaml_text)
        raw = recipe.get("1_data", {}) or {}
        report = l1_layer.validate_layer(raw)
        hard_errors = report.hard_errors
        # No hard errors related to sample_start_date format
        date_errors = [e for e in hard_errors if "sample_start_date" in e.message and "ISO" in e.message]
        assert len(date_errors) == 0, f"Unexpected errors: {date_errors}"
