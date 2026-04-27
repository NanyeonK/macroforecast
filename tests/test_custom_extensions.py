from __future__ import annotations

from macrocast import (
    CUSTOM_MODEL_CONTRACT_VERSION,
    custom_method_extension_contracts,
    custom_model_contract_metadata,
)


def test_custom_model_contract_metadata_describes_context_and_payloads() -> None:
    metadata = custom_model_contract_metadata()

    assert metadata["contract_version"] == CUSTOM_MODEL_CONTRACT_VERSION
    assert "contract_version" in metadata["required_context_fields"]
    assert "model_name" in metadata["required_context_fields"]
    assert "auxiliary_payloads" in metadata["optional_context_fields"]
    assert set(metadata["required_context_fields"]).issubset(metadata["context_fields"])
    assert set(metadata["optional_context_fields"]).issubset(metadata["context_fields"])
    assert "fred_sd_native_frequency_block_payload_v1" in metadata["auxiliary_payload_contracts"]
    assert "fred_sd_mixed_frequency_model_adapter_v1" in metadata["auxiliary_payload_contracts"]
    assert (
        metadata["auxiliary_payload_conditions"]["fred_sd_native_frequency_block_payload_v1"][
            "context_key"
        ]
        == "auxiliary_payloads.fred_sd_native_frequency_block_payload"
    )
    assert (
        metadata["auxiliary_payload_conditions"]["fred_sd_mixed_frequency_model_adapter_v1"][
            "context_key"
        ]
        == "auxiliary_payloads.fred_sd_mixed_frequency_model_adapter"
    )
    assert metadata["accepted_return"]["scalar"] == "single forecast value"


def test_custom_method_extension_contracts_surface_layer3_payload_contracts() -> None:
    contracts = custom_method_extension_contracts()
    layer3 = contracts["layer3_model"]

    assert layer3["contract_version"] == CUSTOM_MODEL_CONTRACT_VERSION
    assert "required_context_fields" in layer3
    assert "fred_sd_mixed_frequency_model_adapter_v1" in layer3["auxiliary_payload_contracts"]
    assert "routing_notes" in layer3
