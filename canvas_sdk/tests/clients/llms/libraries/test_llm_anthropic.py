import base64
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from requests import exceptions, models

from canvas_sdk.clients.llms.constants.file_type import FileType
from canvas_sdk.clients.llms.libraries.llm_anthropic import LlmAnthropic
from canvas_sdk.clients.llms.structures.file_content import FileContent
from canvas_sdk.clients.llms.structures.llm_response import LlmResponse
from canvas_sdk.clients.llms.structures.llm_tokens import LlmTokens
from canvas_sdk.clients.llms.structures.llm_turn import LlmTurn
from canvas_sdk.clients.llms.structures.llm_url_file import LlmFileUrl
from canvas_sdk.clients.llms.structures.settings.llm_settings import LlmSettings


def test_to_dict() -> None:
    """Test conversion of prompts to Anthropic API format."""
    settings = LlmSettings(api_key="test_key", model="test_model")
    tested = LlmAnthropic(settings)

    # Test with system, user, and model prompts
    tested.add_prompt(LlmTurn(role="system", text=["system prompt 1"]))
    tested.add_prompt(LlmTurn(role="user", text=["user message 1"]))
    tested.add_prompt(LlmTurn(role="user", text=["user message 2"]))
    tested.add_prompt(LlmTurn(role="user", text=["user message 3"]))
    tested.add_prompt(LlmTurn(role="model", text=["model response 1"]))
    tested.add_prompt(LlmTurn(role="model", text=["model response 2"]))
    tested.add_prompt(LlmTurn(role="system", text=["system prompt 2"]))
    tested.add_prompt(LlmTurn(role="user", text=["user message 4"]))

    result = tested.to_dict()

    # System and user are both mapped to "user" role and merged
    expected = {
        "messages": [
            {
                "content": [
                    {"text": "system prompt 2", "type": "text"},
                    {"text": "user message 1", "type": "text"},
                    {"text": "user message 2", "type": "text"},
                    {"text": "user message 3", "type": "text"},
                ],
                "role": "user",
            },
            {
                "content": [
                    {"text": "model response 1", "type": "text"},
                    {"text": "model response 2", "type": "text"},
                ],
                "role": "assistant",
            },
            {
                "content": [
                    {"text": "user message 4", "type": "text"},
                ],
                "role": "user",
            },
        ],
        "model": "test_model",
    }

    assert result == expected


@patch.object(LlmAnthropic, "base64_encoded_content_of")
def test_to_dict__with_files(base64_encoded_content_of: MagicMock) -> None:
    """Test conversion of prompts with file attachments to Anthropic API format."""

    def reset_mocks() -> None:
        base64_encoded_content_of.reset_mock()

    settings = LlmSettings(api_key="test_key", model="test_model")

    exp_model = {
        "content": [{"text": "the response", "type": "text"}],
        "role": "assistant",
    }
    exp_user = {
        "content": [
            {
                "text": "the prompt",
                "type": "text",
            },
            {
                "source": {
                    "type": "url",
                    "url": "https://example.com/doc.pdf",
                },
                "type": "document",
            },
            {
                "source": {
                    "type": "url",
                    "url": "https://example.com/pic.jpg",
                },
                "type": "image",
            },
            {
                "source": {
                    "data": "theContent",
                    "media_type": "text/plain",
                    "type": "text",
                },
                "type": "document",
            },
        ],
        "role": "user",
    }

    tests = [
        # no turn
        (
            [],
            {"model": "test_model", "messages": []},
            4,
            [],
        ),
        # model turn
        (
            [LlmTurn(role="model", text=["the response"])],
            {"model": "test_model", "messages": [exp_model]},
            4,
            [],
        ),
        # system turn
        (
            [LlmTurn(role="system", text=["the prompt"])],
            {"model": "test_model", "messages": [exp_user]},
            0,
            [call(LlmFileUrl(url="https://example.com/text.txt", type=FileType.TEXT))],
        ),
        # user turn
        (
            [LlmTurn(role="user", text=["the prompt"])],
            {"model": "test_model", "messages": [exp_user]},
            0,
            [call(LlmFileUrl(url="https://example.com/text.txt", type=FileType.TEXT))],
        ),
    ]
    for prompts, expected, exp_files, exp_calls in tests:
        tested = LlmAnthropic(settings)

        tested.file_urls = [
            LlmFileUrl(url="https://example.com/doc.pdf", type=FileType.PDF),
            LlmFileUrl(url="https://example.com/pic.jpg", type=FileType.IMAGE),
            LlmFileUrl(url="https://example.com/text.txt", type=FileType.TEXT),
            LlmFileUrl(url="https://example.com/some.nop", type="unknown"),  # type: ignore
        ]
        assert len(tested.file_urls) == 4

        for prompt in prompts:
            tested.add_prompt(prompt)

        base64_encoded_content_of.side_effect = [
            FileContent(
                mime_type="theMimeType",
                content=base64.b64encode(b"theContent"),
                size=123,
            )
        ]
        result = tested.to_dict()
        assert result == expected
        assert len(tested.file_urls) == exp_files

        assert base64_encoded_content_of.mock_calls == exp_calls
        reset_mocks()


@patch("canvas_sdk.clients.llms.libraries.llm_anthropic.Http")
def test_request(http: MagicMock) -> None:
    """Test successful API request to Anthropic."""

    def reset_mocks() -> None:
        http.reset_mock()

    settings = LlmSettings(api_key="test_key", model="test_model")
    tested = LlmAnthropic(settings)
    tested.add_prompt(LlmTurn(role="user", text=["test"]))

    # exceptions
    exception_no_response = exceptions.RequestException("Connection error")
    exception_with_response = exceptions.RequestException("Server error")
    exception_with_response.response = models.Response()
    exception_with_response.response.status_code = 404
    exception_with_response.response._content = b"not found"

    tests = [
        # success
        (
            SimpleNamespace(
                status_code=200,
                text="{"
                '"content": [{"text": "response text"}], '
                '"usage": {"input_tokens": 10, "output_tokens": 20}'
                "}",
            ),
            LlmResponse(
                code=HTTPStatus.OK,
                response="response text",
                tokens=LlmTokens(prompt=10, generated=20),
            ),
        ),
        # error
        (
            SimpleNamespace(
                status_code=403,
                text="forbidden",
            ),
            LlmResponse(
                code=HTTPStatus.FORBIDDEN,
                response="forbidden",
                tokens=LlmTokens(prompt=0, generated=0),
            ),
        ),
        # exception -- no response
        (
            exception_no_response,
            LlmResponse(
                code=HTTPStatus.BAD_REQUEST,
                response="Request failed: Connection error",
                tokens=LlmTokens(prompt=0, generated=0),
            ),
        ),
        # exception -- with response
        (
            exception_with_response,
            LlmResponse(
                code=HTTPStatus.NOT_FOUND,
                response="not found",
                tokens=LlmTokens(prompt=0, generated=0),
            ),
        ),
    ]
    for response, expected in tests:
        http.return_value.post.side_effect = [response]

        result = tested.request()
        assert result == expected

        calls = [
            call("https://api.anthropic.com/v1/messages"),
            call().post(
                "",
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": "test_key",
                },
                data="{"
                '"model": "test_model", '
                '"messages": [{'
                '"role": "user", '
                '"content": [{"type": "text", "text": "test"}]'
                "}]"
                "}",
            ),
        ]
        assert http.mock_calls == calls
        reset_mocks()
