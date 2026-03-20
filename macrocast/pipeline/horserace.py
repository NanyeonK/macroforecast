"""Horse race grid runner for CLSS 2021 replication.

HorseRaceGrid runs one ForecastExperiment per FeatureSpec and merges all
results into a single ResultSet.  This enables systematic comparison across
all 15 CLSS 2021 information sets.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path

import pandas as pd

from macrocast.pipeline.components import Window
from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec
from macrocast.pipeline.results import ForecastRecord, ResultSet


class HorseRaceGrid:
    """Run ForecastExperiment for each FeatureSpec and merge results.

    Equivalent to running one ForecastExperiment per FeatureSpec with the same
    model grid, then concatenating all ForecastRecords into a single ResultSet.
    The ``feature_set`` field of each ForecastRecord is set to
    ``FeatureSpec.label``, enabling downstream grouping by information set.

    Parameters
    ----------
    panel : pd.DataFrame
        Stationary-transformed predictor panel, DatetimeIndex.
    target : pd.Series
        Target series, same DatetimeIndex as panel.
    horizons : list of int
        Forecast horizons (periods ahead).
    model_specs : list of ModelSpec
        Model grid applied to every FeatureSpec.
    feature_specs : list of FeatureSpec
        Information set grid.  One ForecastExperiment is run per spec.
    panel_levels : pd.DataFrame or None
        Levels panel for FeatureSpecs with include_levels=True.
    window : Window
        Outer evaluation window strategy.
    rolling_size : int or None
        Required when window=ROLLING.
    oos_start : pd.Timestamp or str or None
        Start of OOS evaluation period.
    oos_end : pd.Timestamp or str or None
        End of OOS evaluation period.
    n_jobs : int
        Parallel workers for the outer loop within each ForecastExperiment.
    experiment_id : str or None
        UUID for this run.  Auto-generated if None.
    output_dir : Path or str or None
        If provided, final merged ResultSet is written to parquet here.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        target: pd.Series,
        horizons: list[int],
        model_specs: list[ModelSpec],
        feature_specs: list[FeatureSpec],
        panel_levels: pd.DataFrame | None = None,
        window: Window = Window.EXPANDING,
        rolling_size: int | None = None,
        oos_start: pd.Timestamp | str | None = None,
        oos_end: pd.Timestamp | str | None = None,
        n_jobs: int = 1,
        experiment_id: str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.panel = panel
        self.target = target
        self.horizons = horizons
        self.model_specs = model_specs
        self.feature_specs = feature_specs
        self.panel_levels = panel_levels
        self.window = window
        self.rolling_size = rolling_size
        self.oos_start = oos_start
        self.oos_end = oos_end
        self.n_jobs = n_jobs
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.output_dir = Path(output_dir) if output_dir else None

    def run(self) -> ResultSet:
        """Run one ForecastExperiment per FeatureSpec; merge into single ResultSet.

        Returns
        -------
        ResultSet
            All forecast records from all FeatureSpec experiments combined.
        """
        result_sets: list[ResultSet] = []
        feature_labels: list[str] = []

        for feat_spec in self.feature_specs:
            exp = ForecastExperiment(
                panel=self.panel,
                target=self.target,
                horizons=self.horizons,
                model_specs=self.model_specs,
                feature_spec=feat_spec,
                panel_levels=self.panel_levels,
                window=self.window,
                rolling_size=self.rolling_size,
                oos_start=self.oos_start,
                oos_end=self.oos_end,
                n_jobs=self.n_jobs,
                experiment_id=f"{self.experiment_id}__{feat_spec.label}",
            )
            rs = exp.run()
            result_sets.append(rs)
            feature_labels.append(feat_spec.label)

        merged = self.merge_result_sets(result_sets, feature_labels)

        if self.output_dir is not None:
            out_path = self.output_dir / f"{self.experiment_id}.parquet"
            merged.to_parquet(out_path)

        return merged

    @staticmethod
    def merge_result_sets(
        result_sets: list[ResultSet],
        feature_labels: list[str],
    ) -> ResultSet:
        """Merge multiple ResultSets into one, tagging each record with its label.

        The ``feature_set`` field of each ForecastRecord is updated to the
        corresponding label if it is empty.

        Parameters
        ----------
        result_sets : list of ResultSet
            Individual experiment results to merge.
        feature_labels : list of str
            Label for each result set (parallel list).

        Returns
        -------
        ResultSet
            Merged collection of all ForecastRecords.

        Raises
        ------
        ValueError
            If result_sets and feature_labels have different lengths.
        """
        if len(result_sets) != len(feature_labels):
            raise ValueError(
                f"result_sets and feature_labels must have the same length, "
                f"got {len(result_sets)} and {len(feature_labels)}."
            )

        merged = ResultSet()
        for rs, label in zip(result_sets, feature_labels):
            for record in rs.records:
                # Tag with label if ForecastExperiment did not set feature_set.
                # Use dataclasses.replace to avoid mutating the source record.
                if not record.feature_set:
                    record = dataclasses.replace(record, feature_set=label)
                merged.add(record)

        return merged
