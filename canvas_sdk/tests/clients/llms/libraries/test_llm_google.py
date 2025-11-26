import base64
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from requests import exceptions, models

from canvas_sdk.clients.llms.constants.file_type import FileType
from canvas_sdk.clients.llms.libraries.llm_google import LlmGoogle
from canvas_sdk.clients.llms.structures.file_content import FileContent
from canvas_sdk.clients.llms.structures.llm_response import LlmResponse
from canvas_sdk.clients.llms.structures.llm_tokens import LlmTokens
from canvas_sdk.clients.llms.structures.llm_turn import LlmTurn
from canvas_sdk.clients.llms.structures.llm_url_file import LlmFileUrl
from canvas_sdk.clients.llms.structures.settings.llm_settings import LlmSettings


def test_to_dict() -> None:
    """Test conversion of prompts to Google API format."""
    settings = LlmSettings(api_key="test_key", model="test_model")
    tested = LlmGoogle(settings)

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
        "contents": [
            {
                "parts": [
                    {"text": "system prompt 2"},
                    {"text": "user message 1"},
                    {"text": "user message 2"},
                    {"text": "user message 3"},
                ],
                "role": "user",
            },
            {
                "parts": [
                    {"text": "model response 1"},
                    {"text": "model response 2"},
                ],
                "role": "model",
            },
            {
                "parts": [
                    {"text": "user message 4"},
                ],
                "role": "user",
            },
        ],
        "model": "test_model",
    }

    assert result == expected


@patch.object(LlmGoogle, "base64_encoded_content_of")
def test_to_dict_with_files(base64_encoded_content_of: MagicMock) -> None:
    """Test conversion of prompts with file attachments to Google API format."""

    def reset_mocks() -> None:
        base64_encoded_content_of.reset_mock()

    settings = LlmSettings(api_key="test_key", model="test_model")

    exp_model = {
        "parts": [{"text": "the response"}],
        "role": "model",
    }
    exp_user = {
        "parts": [
            {"text": "the prompt"},
            {"inline_data": {"data": "Y29udGVudDE=", "mime_type": "type1"}},
            {"inline_data": {"data": "Y29udGVudDI=", "mime_type": "type2"}},
            {"inline_data": {"data": "Y29udGVudDM=", "mime_type": "type3"}},
            {"inline_data": {"data": "Y29udGVudDY=", "mime_type": "type6"}},
        ],
        "role": "user",
    }
    calls = [
        call(LlmFileUrl(url="https://example.com/doc1.pdf", type=FileType.PDF)),
        call(LlmFileUrl(url="https://example.com/pic1.jpg", type=FileType.IMAGE)),
        call(LlmFileUrl(url="https://example.com/text1.txt", type=FileType.TEXT)),
        call(LlmFileUrl(url="https://example.com/doc2.pdf", type=FileType.PDF)),
        call(LlmFileUrl(url="https://example.com/pic2.jpg", type=FileType.IMAGE)),
        call(LlmFileUrl(url="https://example.com/text2.txt", type=FileType.TEXT)),
    ]

    tests = [
        # no turn
        (
            [],
            {"model": "test_model", "contents": []},
            6,
            [],
        ),
        # model turn
        (
            [LlmTurn(role="model", text=["the response"])],
            {"model": "test_model", "contents": [exp_model]},
            6,
            [],
        ),
        # system turn
        (
            [LlmTurn(role="system", text=["the prompt"])],
            {"model": "test_model", "contents": [exp_user]},
            0,
            calls,
        ),
        # user turn
        (
            [LlmTurn(role="user", text=["the prompt"])],
            {"model": "test_model", "contents": [exp_user]},
            0,
            calls,
        ),
    ]
    for prompts, expected, exp_files, exp_calls in tests:
        tested = LlmGoogle(settings)

        tested.file_urls = [
            LlmFileUrl(url="https://example.com/doc1.pdf", type=FileType.PDF),
            LlmFileUrl(url="https://example.com/pic1.jpg", type=FileType.IMAGE),
            LlmFileUrl(url="https://example.com/text1.txt", type=FileType.TEXT),
            LlmFileUrl(url="https://example.com/doc2.pdf", type=FileType.PDF),
            LlmFileUrl(url="https://example.com/pic2.jpg", type=FileType.IMAGE),
            LlmFileUrl(url="https://example.com/text2.txt", type=FileType.TEXT),
        ]
        assert len(tested.file_urls) == 6

        for prompt in prompts:
            tested.add_prompt(prompt)

        base64_encoded_content_of.side_effect = [
            FileContent(
                mime_type="type1", content=base64.b64encode(b"content1"), size=4 * 1024 * 1024
            ),
            FileContent(
                mime_type="type2", content=base64.b64encode(b"content2"), size=3 * 1024 * 1024
            ),
            FileContent(
                mime_type="type3", content=base64.b64encode(b"content3"), size=2 * 1024 * 1024
            ),
            FileContent(
                mime_type="type4", content=base64.b64encode(b"content4"), size=2 * 1024 * 1024
            ),
            FileContent(
                mime_type="type5", content=base64.b64encode(b"content5"), size=2 * 1024 * 1024
            ),
            FileContent(
                mime_type="type6", content=base64.b64encode(b"content6"), size=1 * 1024 * 1024 - 1
            ),
        ]
        result = tested.to_dict()
        assert result == expected
        assert len(tested.file_urls) == exp_files

        assert base64_encoded_content_of.mock_calls == exp_calls
        reset_mocks()


@patch("canvas_sdk.clients.llms.libraries.llm_google.Http")
def test_request(http: MagicMock) -> None:
    """Test successful API request to Google."""

    def reset_mocks() -> None:
        http.reset_mock()

    settings = LlmSettings(api_key="test_key", model="test_model")
    tested = LlmGoogle(settings)
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
                '"candidates": [{"content": {"parts": [{"text": "response text"}]}}], '
                '"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15, "thoughtsTokenCount": 5}'
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
                status_code=429,
                text="Rate limit exceeded",
            ),
            LlmResponse(
                code=HTTPStatus.TOO_MANY_REQUESTS,
                response="Rate limit exceeded",
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
            call(
                "https://generativelanguage.googleapis.com/"
                "v1beta/"
                "test_model:generateContent?key=test_key"
            ),
            call().post(
                "",
                headers={
                    "Content-Type": "application/json",
                },
                data="{"
                '"model": "test_model", '
                '"contents": [{'
                '"role": "user", '
                '"parts": [{"text": "test"}]'
                "}]"
                "}",
            ),
        ]
        assert http.mock_calls == calls
        reset_mocks()
