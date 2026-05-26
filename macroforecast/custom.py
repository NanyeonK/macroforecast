"""Backward-compatibility shim for macroforecast.custom.

This module was moved to macroforecast.api.custom in v0.10 Phase 4
restructure. This shim re-exports everything so existing imports continue
to work.
"""
from macroforecast.api.custom import *  # noqa: F401, F403
from macroforecast.api.custom import (  # noqa: F401
    CUSTOM_MODEL_CONTRACT_VERSION,
    CUSTOM_PREPROCESSOR_CONTRACT_VERSION,
    TARGET_TRANSFORMER_CONTRACT_VERSION,
    custom_model,
    register_model,
    register_feature_block,
    register_feature_combiner,
    register_preprocessor,
    register_target_transformer,
    get_custom_model,
    get_custom_feature_block,
    get_custom_feature_combiner,
    get_custom_preprocessor,
    get_custom_target_transformer,
    is_custom_model,
    is_custom_feature_block,
    is_custom_feature_combiner,
    is_custom_preprocessor,
    is_custom_target_transformer,
    list_custom_models,
    list_custom_feature_blocks,
    list_custom_feature_combiners,
    list_custom_preprocessors,
    list_custom_target_transformers,
    clear_custom_models,
    clear_custom_feature_blocks,
    clear_custom_feature_combiners,
    clear_custom_preprocessors,
    clear_custom_target_transformers,
    clear_custom_extensions,
)
