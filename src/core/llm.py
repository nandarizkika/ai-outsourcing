from anthropic import Anthropic
from openai import OpenAI
from src.core.models import TaskType


class LLMRouter:
    def __init__(self, anthropic_api_key: str, openai_api_key: str):
        self._anthropic = Anthropic(api_key=anthropic_api_key)
        self._openai = OpenAI(api_key=openai_api_key)

    def complete(self, task_type: TaskType, system: str, user: str) -> str:
        if task_type == TaskType.REASONING:
            return self._claude(system, user, model="claude-sonnet-4-6")
        elif task_type == TaskType.TOOL:
            return self._gpt(system, user, model="gpt-4o")
        else:
            return self._claude(system, user, model="claude-haiku-4-5-20251001")

    def _claude(self, system: str, user: str, model: str) -> str:
        msg = self._anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def _gpt(self, system: str, user: str, model: str) -> str:
        resp = self._openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
