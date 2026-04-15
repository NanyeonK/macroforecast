from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='dataset_source',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fred_md',
            description='fred md source',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fred_qd',
            description='fred qd source',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fred_sd',
            description='fred sd source',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fred_api_custom',
            description='fred api custom',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='bea',
            description='bea',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='bls',
            description='bls',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='census',
            description='census',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='oecd',
            description='oecd',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='imf_ifs',
            description='imf ifs',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='ecb_sdw',
            description='ecb sdw',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='bis',
            description='bis',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='world_bank',
            description='world bank',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='wrds_macro_finance',
            description='wrds macro finance',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='survey_spf',
            description='survey spf',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='blue_chip',
            description='blue chip',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='market_prices',
            description='market prices',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='high_frequency_surprises',
            description='high frequency surprises',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='google_trends',
            description='google trends',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='news_text',
            description='news text',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='climate_series',
            description='climate series',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='satellite_proxy',
            description='satellite proxy',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='custom_csv',
            description='custom csv',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='custom_parquet',
            description='custom parquet',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='custom_sql',
            description='custom sql',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
