from datetime import datetime

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from src.core.models import Channel, ClientConfig, ClarificationState, Request, Response
from src.orchestrator.orchestrator import Orchestrator


class SlackChannel:
    def __init__(
        self,
        bot_token: str,
        signing_secret: str,
        bot_user_id: str,
        orchestrator: Orchestrator,
        client_configs: dict[str, ClientConfig],
    ):
        self._bot_user_id = bot_user_id
        self._orchestrator = orchestrator
        self._client_configs = client_configs
        self._pending: dict[str, ClarificationState] = {}

        self._app = App(token=bot_token, signing_secret=signing_secret, token_verification_enabled=False)
        self._app.event("app_mention")(self._handle_mention)
        self._handler = SlackRequestHandler(self._app)

    def _handle_mention(self, event: dict, say) -> None:
        team_id = event.get("team")
        config = self._client_configs.get(team_id)
        if not config:
            self._on_unconfigured_workspace(say, event.get("thread_ts") or event.get("ts"))
            return

        thread_ts = event.get("thread_ts") or event.get("ts")
        request = self._build_request(event, team_id)

        pending = self._pending.get(thread_ts)
        if pending and not pending.is_resolved:
            pending.answers_received.append(request.text)

        result = self._orchestrator.process(request, config, pending)
        self._handle_result(result, event, say)

    def _build_request(self, event: dict, team_id: str) -> Request:
        text = event.get("text", "").replace(f"<@{self._bot_user_id}>", "").strip()
        config = self._client_configs.get(team_id)
        return Request(
            channel=Channel.SLACK,
            sender_id=event["user"],
            sender_name=event["user"],
            text=text,
            thread_id=event.get("thread_ts") or event.get("ts"),
            timestamp=datetime.utcnow().isoformat(),
            client_id=config.client_id if config else "unknown",
        )

    def _handle_result(self, result, event: dict, say) -> None:
        thread_ts = event.get("thread_ts") or event.get("ts")

        if isinstance(result, ClarificationState):
            self._pending[thread_ts] = result
            questions = "\n".join(f"• {q}" for q in result.questions_asked[-2:])
            say(text=f"Quick question before I run this:\n{questions}", thread_ts=thread_ts)
            return

        if thread_ts in self._pending:
            del self._pending[thread_ts]

        say(text=result.text, thread_ts=thread_ts)

        for chart_png in result.charts:
            self._app.client.files_upload_v2(
                channel=event["channel"],
                file=chart_png,
                filename="analysis.png",
                thread_ts=thread_ts,
            )

    def _on_unconfigured_workspace(self, say, thread_ts: str) -> None:
        say(
            text="I'm not configured for this workspace yet. Please contact your account manager.",
            thread_ts=thread_ts,
        )

    def get_handler(self) -> SlackRequestHandler:
        return self._handler
