"""macrocast.preprocessing — Data transforms and panel preprocessing."""

from macrocast.preprocessing.missing import (
    classify_missing,
    detect_missing_type,
    em_factor,
    handle_missing,
    prepare_fredmd,
    remove_outliers_iqr,
)
from macrocast.preprocessing.panel import (
    CustomTransform,
    DemeanTransform,
    DropTransform,
    HPFilterTransform,
    PanelTransformer,
    StandardizeTransform,
    WinsorizeTransform,
)
from macrocast.preprocessing.registry import (
    get_target_recipe,
    get_x_recipe,
    load_preprocessing_registry,
    validate_preprocessing_registry,
)
from macrocast.preprocessing.transforms import (
    TransformCode,
    apply_hamilton_filter,
    apply_maf,
    apply_marx,
    apply_pca,
    apply_tcode,
    apply_tcodes,
    apply_x_factors,
    inverse_tcode,
)

__all__ = [
    'TransformCode',
    'apply_tcode',
    'apply_tcodes',
    'inverse_tcode',
    'apply_marx',
    'apply_maf',
    'apply_x_factors',
    'apply_pca',
    'apply_hamilton_filter',
    'detect_missing_type',
    'classify_missing',
    'handle_missing',
    'remove_outliers_iqr',
    'prepare_fredmd',
    'em_factor',
    'get_target_recipe',
    'get_x_recipe',
    'load_preprocessing_registry',
    'validate_preprocessing_registry',
    'PanelTransformer',
    'WinsorizeTransform',
    'DemeanTransform',
    'HPFilterTransform',
    'StandardizeTransform',
    'CustomTransform',
    'DropTransform',
]
