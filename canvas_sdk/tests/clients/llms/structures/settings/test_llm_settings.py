from canvas_sdk.clients.llms.structures.settings.llm_settings import LlmSettings
from canvas_sdk.tests.conftest import is_dataclass


def test_class() -> None:
    """Test LlmSettings is a dataclass with correct fields and types."""
    assert is_dataclass(
        LlmSettings,
        {
            "api_key": str,
            "model": str,
        },
    )


def test_to_dict() -> None:
    """Test conversion of LlmSettings to dictionary format excludes API key."""
    tested = LlmSettings(
        api_key="theKey",
        model="theModel",
    )
    result = tested.to_dict()
    expected = {
        "model": "theModel",
    }
    assert result == expected
