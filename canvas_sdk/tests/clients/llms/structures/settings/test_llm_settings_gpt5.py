from canvas_sdk.clients.llms import LlmSettings
from canvas_sdk.clients.llms.structures.settings.llm_settings_gpt5 import LlmSettingsGpt5
from canvas_sdk.tests.conftest import is_dataclass


def test_class() -> None:
    """Test LlmSettingsGpt5 is a dataclass subclass of LlmSettings with correct fields."""
    assert issubclass(LlmSettingsGpt5, LlmSettings)
    assert is_dataclass(
        LlmSettingsGpt5,
        {
            "api_key": str,
            "model": str,
            "reasoning_effort": str,
            "text_verbosity": str,
        },
    )


def test_to_dict() -> None:
    """Test conversion of LlmSettingsGpt5 to dictionary format."""
    tested = LlmSettingsGpt5(
        api_key="theKey",
        model="theModel",
        reasoning_effort="medium",
        text_verbosity="low",
    )
    result = tested.to_dict()
    expected = {
        "model": "theModel",
        "reasoning": {
            "effort": "medium",
        },
        "text": {
            "verbosity": "low",
        },
    }
    assert result == expected
