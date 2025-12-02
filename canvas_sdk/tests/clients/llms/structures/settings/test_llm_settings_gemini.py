from canvas_sdk.clients.llms import LlmSettings
from canvas_sdk.clients.llms.structures.settings.llm_settings_gemini import LlmSettingsGemini
from canvas_sdk.tests.conftest import is_dataclass


def test_class() -> None:
    """Test LlmSettingsGemini is a dataclass subclass of LlmSettings with correct fields."""
    assert issubclass(LlmSettingsGemini, LlmSettings)
    assert is_dataclass(
        LlmSettingsGemini,
        {
            "api_key": str,
            "model": str,
            "temperature": float,
        },
    )


def test_to_dict() -> None:
    """Test conversion of LlmSettingsGemini to dictionary format."""
    tested = LlmSettingsGemini(
        api_key="theKey",
        model="theModel",
        temperature=2.0,
    )
    result = tested.to_dict()
    expected = {
        "model": "theModel",
        "generationConfig": {"temperature": 2.0},
    }
    assert result == expected
