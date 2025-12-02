from canvas_sdk.clients.llms import LlmSettings
from canvas_sdk.clients.llms.structures.settings.llm_settings_gpt4 import LlmSettingsGpt4
from canvas_sdk.tests.conftest import is_dataclass


def test_class() -> None:
    """Test LlmSettingsGpt4 is a dataclass subclass of LlmSettings with correct fields."""
    assert issubclass(LlmSettingsGpt4, LlmSettings)
    assert is_dataclass(
        LlmSettingsGpt4,
        {
            "api_key": str,
            "model": str,
            "temperature": float,
        },
    )


def test_to_dict() -> None:
    """Test conversion of LlmSettingsGpt4 to dictionary format."""
    tested = LlmSettingsGpt4(
        api_key="theKey",
        model="theModel",
        temperature=2.0,
    )
    result = tested.to_dict()
    expected = {
        "model": "theModel",
        "temperature": 2.0,
    }
    assert result == expected
