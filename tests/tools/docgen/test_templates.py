"""Pin every starter template builds, validates, and (where it has no
external data dependency) runs end-to-end."""
from __future__ import annotations

import pytest

from tools.docgen import RecipeBuilder, from_template, list_templates


def test_list_templates_returns_five():
    names = list_templates()
    assert len(names) == 5
    assert set(names) == {
        "ridge_baseline",
        "horse_race_md",
        "regime_conditional",
        "fred_md_replication",
        "fred_sd_geographic",
    }


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        from_template("not_a_real_template")


@pytest.mark.parametrize("name", list_templates())
def test_template_returns_recipe_builder(name):
    b = from_template(name)
    assert isinstance(b, RecipeBuilder)
    recipe = b.build()
    # Every template should populate L1 + L4 at minimum.
    for required in ("data", "4_forecasting_model"):
        assert required in recipe, f"{name} missing {required}"


def test_ridge_baseline_runs_end_to_end(tmp_path):
    b = from_template("ridge_baseline")
    result = b.run(output_directory=tmp_path)
    assert len(result.cells) == 1
    assert result.cells[0].succeeded


def test_template_overrides_pass_through():
    b = from_template("ridge_baseline", target="custom_y")
    assert b.build()["data"]["leaf_config"]["target"] == "custom_y"


def test_template_user_can_modify_after_construction(tmp_path):
    """Common workflow: pick template, tweak one field, run."""
    b = from_template("ridge_baseline")
    b.l4.fit("ols", min_train_size=4)
    result = b.run(output_directory=tmp_path)
    assert result.cells[0].succeeded
