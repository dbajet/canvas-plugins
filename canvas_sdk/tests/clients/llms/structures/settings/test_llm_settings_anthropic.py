from canvas_sdk.clients.llms import LlmSettings
from canvas_sdk.clients.llms.structures.settings.llm_settings_anthropic import LlmSettingsAnthropic
from canvas_sdk.tests.conftest import is_dataclass


def test_class() -> None:
    """Test LlmSettingsAnthropic is a dataclass subclass of LlmSettings with correct fields."""
    assert issubclass(LlmSettingsAnthropic, LlmSettings)
    assert is_dataclass(
        LlmSettingsAnthropic,
        {
            "api_key": str,
            "model": str,
            "temperature": float,
            "max_tokens": int,
        },
    )


def test_to_dict() -> None:
    """Test conversion of LlmSettingsAnthropic to dictionary format."""
    tested = LlmSettingsAnthropic(
        api_key="theKey",
        model="theModel",
        temperature=0.78,
        max_tokens=8192,
    )
    result = tested.to_dict()
    expected = {
        "model": "theModel",
        "temperature": 0.78,
        "max_tokens": 8192,
    }
    assert result == expected
