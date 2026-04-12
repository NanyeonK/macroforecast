from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from macrocast.config import ExperimentConfig, load_config, load_config_from_dict
from macrocast.data import (
    get_data_task_defaults,
    get_dataset_defaults,
    get_target_defaults,
    load_data_task_registry,
    load_dataset_registry,
    load_target_registry,
    validate_data_task_registry,
    validate_dataset_registry,
    validate_target_registry,
)
from macrocast.meta import (
    load_axes_registry,
    load_benchmark_registry,
    load_global_defaults_registry,
    load_preset_registry,
    resolve_meta_config,
    validate_axes_registry,
    validate_benchmark_registry,
    validate_preset_registry,
)
from macrocast.pipeline import get_model_defaults, load_model_registry, validate_model_registry
from macrocast.preprocessing import load_preprocessing_registry, validate_preprocessing_registry

_REQUIRED_COMPILED_META_FIELDS = {
    'dataset','target','horizon','benchmark_id','benchmark_family','benchmark_options','evaluation_scale','target_preprocess_recipe','x_preprocess_recipe','sample_period','oos_period','minimum_train_size','validation_design','outer_window','model_family','tuning_method','hyperparameter_space'
}

@dataclass
class ResolvedExperimentSpec:
    experiment_config: ExperimentConfig
    meta_config: dict[str, Any] = field(default_factory=dict)
    dataset_registry: dict[str, Any] = field(default_factory=dict)
    target_registry: dict[str, Any] = field(default_factory=dict)
    data_task_registry: dict[str, Any] = field(default_factory=dict)
    model_registry: dict[str, Any] = field(default_factory=dict)
    preprocessing_registry: dict[str, Any] = field(default_factory=dict)
    benchmark_registry: dict[str, Any] = field(default_factory=dict)
    axes_registry: dict[str, Any] = field(default_factory=dict)

    def validate_compiled_spec(self) -> 'ResolvedExperimentSpec':
        missing = sorted(_REQUIRED_COMPILED_META_FIELDS - set(self.meta_config))
        if missing:
            raise ValueError(f'compiled experiment spec missing required meta fields: {missing}')
        for key in ['sample_period', 'oos_period', 'benchmark_options']:
            if not isinstance(self.meta_config[key], dict):
                raise ValueError(f'compiled experiment spec field {key} must be a dict')
        if not isinstance(self.meta_config['horizon'], list) or not self.meta_config['horizon']:
            raise ValueError('compiled experiment spec horizon must be a non-empty list')
        return self

    def to_contract_dict(self) -> dict[str, Any]:
        self.validate_compiled_spec()
        return {
            'experiment_id': self.experiment_config.experiment_id,
            'dataset': self.meta_config['dataset'],
            'target': self.meta_config['target'],
            'horizon': self.meta_config['horizon'],
            'benchmark_id': self.meta_config['benchmark_id'],
            'benchmark_family': self.meta_config['benchmark_family'],
            'benchmark_options': self.meta_config['benchmark_options'],
            'evaluation_scale': self.meta_config['evaluation_scale'],
            'target_preprocess_recipe': self.meta_config['target_preprocess_recipe'],
            'x_preprocess_recipe': self.meta_config['x_preprocess_recipe'],
            'sample_period': self.meta_config['sample_period'],
            'oos_period': self.meta_config['oos_period'],
            'minimum_train_size': self.meta_config['minimum_train_size'],
            'validation_design': self.meta_config['validation_design'],
            'outer_window': self.meta_config['outer_window'],
            'model_family': self.meta_config['model_family'],
            'tuning_method': self.meta_config['tuning_method'],
            'hyperparameter_space': self.meta_config['hyperparameter_space'],
        }

_MODEL_CLASS_TO_REGISTRY_ID = {
    'ARModel': 'ar','ARDIModel': 'ardi','RidgeModel': 'ridge','LassoModel': 'lasso','AdaptiveLassoModel': 'adaptive_lasso','GroupLassoModel': 'group_lasso','ElasticNetModel': 'elastic_net','TVPRidgeModel': 'tvp_ridge','BoogingModel': 'booging','BVARModel': 'bvar','KRRModel': 'krr','SVRRBFModel': 'svr_rbf','SVRLinearModel': 'svr_linear','RFModel': 'rf','XGBoostModel': 'xgboost','GBModel': 'gb','NNModel': 'nn','LSTMModel': 'lstm',
}

def _known_axes(axes: dict[str, Any]) -> set[str]:
    classes = axes['axis_classes']
    return set(classes['invariant']) | set(classes['experiment_fixed']) | set(classes['research_sweep']) | set(classes['conditional'])

def _infer_model_registry_id(exp_cfg: ExperimentConfig) -> str | None:
    if not exp_cfg.model_specs:
        return None
    return _MODEL_CLASS_TO_REGISTRY_ID.get(exp_cfg.model_specs[0].model_cls.__name__)

def _validate_meta_registries() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    axes = load_axes_registry(); benchmarks = load_benchmark_registry(); presets = load_preset_registry(); global_defaults = load_global_defaults_registry()
    validate_axes_registry(axes); validate_benchmark_registry(benchmarks); validate_preset_registry(presets, invariant_axes=set(axes['invariant_axes']), known_axes=_known_axes(axes))
    if not isinstance(global_defaults.get('global_defaults'), dict):
        raise ValueError('global_defaults.yaml must contain global_defaults dict')
    return axes, benchmarks, presets, global_defaults

def resolve_experiment_spec(path: str | Path, *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> ResolvedExperimentSpec:
    return _build_resolved_spec(load_config(path), preset_id=preset_id, experiment_overrides=experiment_overrides)

def resolve_experiment_spec_from_dict(raw: dict[str, Any], *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> ResolvedExperimentSpec:
    return _build_resolved_spec(load_config_from_dict(raw), preset_id=preset_id, experiment_overrides=experiment_overrides)

def _build_resolved_spec(exp_cfg: ExperimentConfig, *, preset_id: str | None, experiment_overrides: dict[str, Any] | None) -> ResolvedExperimentSpec:
    axes, benchmarks, presets, global_defaults = _validate_meta_registries()
    dataset_registry = load_dataset_registry(); target_registry = load_target_registry(); data_task_registry = load_data_task_registry(); model_registry = load_model_registry()
    validate_dataset_registry(dataset_registry); validate_target_registry(target_registry); validate_data_task_registry(data_task_registry); validate_model_registry(model_registry)
    preprocessing_registry = load_preprocessing_registry(); validate_preprocessing_registry(preprocessing_registry)
    dataset_defaults = get_dataset_defaults(dataset_registry, exp_cfg.data.dataset)
    target_defaults = get_target_defaults(target_registry, exp_cfg.data.target, dataset_id=exp_cfg.data.dataset)
    task_defaults = get_data_task_defaults(data_task_registry, dataset_defaults.get('task_id'), dataset_id=exp_cfg.data.dataset)
    inferred_model_id = _infer_model_registry_id(exp_cfg)
    model_defaults = get_model_defaults(model_registry, inferred_model_id) if inferred_model_id else {}
    config_derived_overrides = {'dataset': exp_cfg.data.dataset, 'target': exp_cfg.data.target, 'horizon': list(exp_cfg.horizons), 'outer_window': exp_cfg.window.value}
    if exp_cfg.oos_start or exp_cfg.oos_end:
        config_derived_overrides['oos_period'] = {'start': exp_cfg.oos_start, 'end': exp_cfg.oos_end}
    if exp_cfg.rolling_size is not None:
        config_derived_overrides['rolling_window_size'] = exp_cfg.rolling_size
    final_experiment_overrides = dict(config_derived_overrides); final_experiment_overrides.update(experiment_overrides or {})
    meta_config = resolve_meta_config(preset_registry=presets, axes_registry=axes, benchmark_registry=benchmarks, preset_id=preset_id, global_defaults=global_defaults['global_defaults'], dataset_defaults=dataset_defaults, target_defaults={**task_defaults, **target_defaults}, model_defaults=model_defaults, experiment_overrides=final_experiment_overrides)
    meta_config.setdefault('target_preprocess_recipe', meta_config.get('target_preprocess_recipe', 'basic_none'))
    meta_config.setdefault('x_preprocess_recipe', meta_config.get('x_preprocess_recipe', 'basic_none'))
    return ResolvedExperimentSpec(experiment_config=exp_cfg, meta_config=meta_config, dataset_registry=dataset_registry, target_registry=target_registry, data_task_registry=data_task_registry, model_registry=model_registry, preprocessing_registry=preprocessing_registry, benchmark_registry=benchmarks, axes_registry=axes).validate_compiled_spec()
