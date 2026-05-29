from google.genai import Client as GeminiClient
from openai import OpenAI
from anthropic import Anthropic
from src.core.models import TaskType


class LLMRouter:
    def __init__(
        self,
        provider: str = "gemini",
        gemini_api_key: str = "",
        openai_api_key: str = "",
        anthropic_api_key: str = "",
    ):
        self.provider = provider.lower()

        if self.provider == "gemini":
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when using Gemini")
            self._client = GeminiClient(api_key=gemini_api_key)
        elif self.provider == "openai":
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI")
            self._client = OpenAI(api_key=openai_api_key)
        elif self.provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic")
            self._client = Anthropic(api_key=anthropic_api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'openai', or 'anthropic'")

    def complete(self, task_type: TaskType, system: str, user: str) -> str:
        if self.provider == "gemini":
            return self._gemini_complete(system, user)
        elif self.provider == "openai":
            return self._openai_complete(system, user)
        elif self.provider == "anthropic":
            return self._anthropic_complete(system, user)

    def _gemini_complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        response = self._client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return response.text

    def _openai_complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    def _anthropic_complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
