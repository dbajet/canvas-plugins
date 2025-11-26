import json
from http import HTTPStatus

from requests import exceptions, models

from canvas_sdk.clients.llms.libraries.llm_base import LlmBase
from canvas_sdk.clients.llms.structures.llm_response import LlmResponse
from canvas_sdk.clients.llms.structures.llm_tokens import LlmTokens
from canvas_sdk.utils.http import Http


class LlmGoogle(LlmBase):
    """Google Gemini LLM API client.

    Implements the LlmBase interface for Google's Generative Language API.
    """

    def to_dict(self) -> dict:
        """Convert prompts and add the necessary information to Google API request format.

        Returns:
            Dictionary formatted for Google API with contents array.
        """
        contents: list[dict] = []
        roles = {
            self.ROLE_SYSTEM: "user",
            self.ROLE_USER: "user",
            self.ROLE_MODEL: "model",
        }
        for prompt in self.prompts:
            role = roles[prompt.role]
            part = {"text": "\n".join(prompt.text)}
            # contiguous parts for the same role are merged
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": role, "parts": [part]})

        # if there are files and the last message has the user's role
        if self.file_urls and contents and contents[-1]["role"] == roles[self.ROLE_USER]:
            size_sum = 0
            max_size = (
                10 * 1024 * 1024
            )  # 10MB - arbitrary limit to prevent high latency (hard limit for Gemini 2.0 is 500MB)
            while self.file_urls and (file_url := self.file_urls.pop(0)):
                file_content = self.base64_encoded_content_of(file_url)
                if (
                    file_content is not None
                    and file_content.size > 0
                    and size_sum + file_content.size < max_size
                ):
                    size_sum += file_content.size
                    contents[-1]["parts"].append(
                        {
                            "inline_data": {
                                "mime_type": file_content.mime_type,
                                "data": file_content.content.decode("utf-8"),
                            },
                        }
                    )
        return self.settings.to_dict() | {
            "contents": contents,
        }

    def request(self) -> LlmResponse:
        """Make a request to the Google Gemini API.

        Returns:
            Response containing status code, generated text, and token usage.
        """
        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/"
            f"{self.settings.model}:generateContent?key={self.settings.api_key}"
        )
        headers = {"Content-Type": "application/json"}
        data = json.dumps(self.to_dict())

        tokens = LlmTokens(prompt=0, generated=0)
        try:
            request = Http(url).post("", headers=headers, data=data)
            code = request.status_code
            response = request.text
            if code == HTTPStatus.OK.value:
                content = json.loads(request.text)
                response = (
                    content.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                usage = content.get("usageMetadata", {})
                tokens = LlmTokens(
                    prompt=usage.get("promptTokenCount") or 0,
                    generated=(usage.get("candidatesTokenCount") or 0)
                    + (usage.get("thoughtsTokenCount") or 0),
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


__exports__ = ("LlmGoogle",)
