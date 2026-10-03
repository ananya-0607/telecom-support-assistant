"""Verify that validation and API schema share the configured vocabulary."""
import pytest
from pydantic import ValidationError
from telecom_support.taxonomy import Classification, TAXONOMY, classification_instructions


def test_classification_schema_uses_taxonomy():
    schema = Classification.model_json_schema()
    for field, registry in [("category", "categories"), ("product", "products"),
                             ("severity", "severities"), ("sentiment", "sentiments")]:
        definition = schema["properties"][field]["$ref"].split("/")[-1]
        assert schema["$defs"][definition]["enum"] == list(TAXONOMY[registry])
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("field", ["category", "product", "severity", "sentiment"])
def test_unknown_classification_label_is_rejected(field):
    values = dict(category="Slow speed", product="Fiber Broadband & Gateway",
                  severity="Medium", sentiment="Neutral")
    values[field] = "unregistered-label"
    with pytest.raises(ValidationError):
        Classification(**values)


def test_prompt_includes_shared_definitions():
    prompt = classification_instructions()
    for registry in ("categories", "products", "severities", "sentiments"):
        for description in TAXONOMY[registry].values():
            assert description in prompt
