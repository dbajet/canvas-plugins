from http import HTTPStatus

from canvas_sdk.clients.llms.structures.llm_response import LlmResponse
from canvas_sdk.clients.llms.structures.llm_tokens import LlmTokens
from canvas_sdk.tests.conftest import is_namedtuple


def test_class() -> None:
    """Test LlmResponse is a NamedTuple with correct fields and types."""
    assert is_namedtuple(
        LlmResponse,
        {
            "code": HTTPStatus,
            "response": str,
            "tokens": LlmTokens,
        },
    )


def test_to_dict() -> None:
    """Test conversion of LlmResponse to dictionary format."""
    tested = LlmResponse(
        code=HTTPStatus.OK,
        response="theResponse",
        tokens=LlmTokens(
            prompt=145,
            generated=475,
        ),
    )
    result = tested.to_dict()
    expected = {
        "code": 200,
        "response": "theResponse",
        "tokens": {
            "generated": 475,
            "prompt": 145,
        },
    }
    assert result == expected
