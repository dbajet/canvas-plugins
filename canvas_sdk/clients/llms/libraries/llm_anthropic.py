import base64
import json
from http import HTTPStatus

from requests import exceptions, models

from canvas_sdk.clients.llms.constants.file_type import FileType
from canvas_sdk.clients.llms.libraries.llm_base import LlmBase
from canvas_sdk.clients.llms.structures.llm_response import LlmResponse
from canvas_sdk.clients.llms.structures.llm_tokens import LlmTokens
from canvas_sdk.utils.http import Http


class LlmAnthropic(LlmBase):
    """Anthropic Claude LLM API client.

    Implements the LlmBase interface for Anthropic's Claude API.
    """

    def to_dict(self) -> dict:
        """Convert prompts and add the necessary information to Anthropic API request format.

        Returns:
            Dictionary formatted for Anthropic API with messages array.
        """
        messages: list[dict] = []

        roles = {
            self.ROLE_SYSTEM: "user",
            self.ROLE_USER: "user",
            self.ROLE_MODEL: "assistant",
        }
        for prompt in self.prompts:
            role = roles[prompt.role]
            part = {"type": "text", "text": "\n".join(prompt.text)}
            # contiguous parts for the same role are merged
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].append(part)
            else:
                messages.append({"role": role, "content": [part]})

        # if there are files and the last message has the user's role
        if self.file_urls and messages and messages[-1]["role"] == roles[self.ROLE_USER]:
            while self.file_urls and (file_url := self.file_urls.pop(0)):
                item = {}
                if file_url.type == FileType.PDF:
                    item = {
                        "type": "document",
                        "source": {
                            "type": "url",
                            "url": file_url.url,
                        },
                    }
                elif file_url.type == FileType.IMAGE:
                    item = {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": file_url.url,
                        },
                    }
                elif file_url.type == FileType.TEXT:
                    file_content = self.base64_encoded_content_of(file_url)
                    item = {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": base64.standard_b64decode(file_content.content).decode("utf-8"),
                        },
                    }
                if item:
                    messages[-1]["content"].append(item)

        return self.settings.to_dict() | {
            "messages": messages,
        }

    def request(self) -> LlmResponse:
        """Make a request to the Anthropic Claude API.

        Returns:
            Response containing status code, generated text, and token usage.
        """
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.settings.api_key,
        }
        data = json.dumps(self.to_dict())

        tokens = LlmTokens(prompt=0, generated=0)
        try:
            request = Http(url).post("", headers=headers, data=data)
            code = request.status_code
            response = request.text
            if code == HTTPStatus.OK.value:
                content = json.loads(request.text)
                response = content.get("content", [{}])[0].get("text", "")
                usage = content.get("usage", {})
                tokens = LlmTokens(
                    prompt=usage.get("input_tokens") or 0,
                    generated=usage.get("output_tokens") or 0,
                )
        except exceptions.RequestException as e:
            code = HTTPStatus.BAD_REQUEST
            response = f"Request failed: {e}"
            if hasattr(e, "response") and isinstance(e.response, models.Response):
                code = e.response.status_code
                response = e.response.text

        return LlmResponse(
            code=HTTPStatus(code),
            response=response,
            tokens=tokens,
        )


__exports__ = ("LlmAnthropic",)
