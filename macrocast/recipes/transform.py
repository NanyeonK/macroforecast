from __future__ import annotations

from typing import Any


def recipe_to_runtime_config(recipe: dict[str, Any]) -> dict[str, Any]:
    path = recipe['taxonomy_path']
    nums = recipe['numeric_params']
    outputs = recipe['outputs']

    model_key = path['model']
    model_map = {
        'random_forest': 'RF',
        'kernel_ridge': 'KRR',
        'elastic_net': 'EN',
        'adaptive_lasso': 'AL',
        'ar': 'AR',
    }
    feature_map = {
        'factors_x': 'X',
    }
    raw = {
        'experiment_id': recipe['recipe_id'],
        'dataset': path['data'],
        'target': path['target'],
        'horizons': nums.get('horizons', [1]),
        'window': path['framework'],
        'oos_start': nums.get('oos_start'),
        'oos_end': nums.get('oos_end'),
        'models': [model_map.get(model_key, model_key)],
        'n_factors': nums.get('n_factors', 4),
        'n_lags': nums.get('n_lags', 2),
        'factor_type': feature_map.get(path['features'], 'X'),
    }
    if 'output_dir' in outputs:
        raw['output_dir'] = outputs['output_dir']
    return raw
