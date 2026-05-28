import asyncio
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
        token_verification_enabled: bool = True,
        ticketing_service=None,
    ):
        self._bot_user_id = bot_user_id
        self._orchestrator = orchestrator
        self._client_configs = client_configs
        self._pending: dict[str, ClarificationState] = {}
        self._ticketing_service = ticketing_service

        self._app = App(
            token=bot_token,
            signing_secret=signing_secret,
            token_verification_enabled=token_verification_enabled,
        )
        self._app.event("app_mention")(self._handle_mention)
        self._app.event("message")(self._handle_message)
        self._handler = SlackRequestHandler(self._app)

    def _handle_mention(self, event: dict, say) -> None:
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"[Slack] ===== MENTION HANDLER CALLED =====")
        logger.info(f"[Slack] Event type: {event.get('type')}, Event keys: {event.keys()}")

        team_id = event.get("team")
        logger.info(f"[Slack] team_id={team_id}, user={event.get('user')}, text={event.get('text', '')}")

        config = self._client_configs.get(team_id)
        logger.info(f"[Slack] Looking for config with team_id={team_id}, available keys: {list(self._client_configs.keys())}")

        if not config:
            logger.warning(f"[Slack] No config found for team_id={team_id}!")
            self._on_unconfigured_workspace(say, event.get("thread_ts") or event.get("ts"))
            return

        logger.info(f"[Slack] Found config: client_id={config.client_id}")
        thread_ts = event.get("thread_ts") or event.get("ts")

        try:
            request = self._build_request(event, config)
            logger.info(f"[Slack] Built request: {request.text}")

            pending = self._pending.get(thread_ts)
            active_clarification = pending if (pending and not pending.is_resolved) else None
            if active_clarification:
                active_clarification.answers_received.append(request.text)

            ticket = None
            if self._ticketing_service is not None:
                try:
                    ticket = self._ticketing_service.create_for_request(request)
                except Exception as e:
                    logger.warning(f"[Slack] Failed to create ticket: {str(e)}")

            logger.info(f"[Slack] Calling orchestrator.process()...")
            result = asyncio.run(self._orchestrator.process(request, config, active_clarification))
            logger.info(f"[Slack] Orchestrator returned: {type(result).__name__}")

            self._handle_result(result, event, say, ticket=ticket)
            logger.info(f"[Slack] Response sent successfully")
        except Exception as e:
            logger.error(f"[Slack] EXCEPTION in handler: {str(e)}", exc_info=True)
            try:
                say(text=f"Sorry, I encountered an error: {str(e)}", thread_ts=thread_ts)
            except Exception as say_error:
                logger.error(f"[Slack] Failed to send error message: {str(say_error)}")

    def _handle_message(self, event: dict, say) -> None:
        import logging
        logger = logging.getLogger(__name__)

        if event.get("bot_id"):
            return

        thread_ts = event.get("thread_ts")
        if not thread_ts:
            return

        pending = self._pending.get(thread_ts)
        if not pending or pending.is_resolved:
            return

        logger.info(f"[Slack] ===== MESSAGE HANDLER: PROCESSING CLARIFICATION RESPONSE =====")
        logger.info(f"[Slack] Thread: {thread_ts}, User: {event.get('user')}, Text: {event.get('text', '')}")

        team_id = event.get("team")
        config = self._client_configs.get(team_id)

        if not config:
            logger.warning(f"[Slack] No config found for team_id={team_id} in message handler")
            return

        try:
            request = self._build_request(event, config)
            logger.info(f"[Slack] Built request from clarification response: {request.text}")

            pending.answers_received.append(request.text)

            logger.info(f"[Slack] Calling orchestrator.process() with clarification state...")
            result = asyncio.run(self._orchestrator.process(request, config, pending))
            logger.info(f"[Slack] Orchestrator returned: {type(result).__name__}")

            self._handle_result(result, event, say)
            logger.info(f"[Slack] Clarification response processed successfully")
        except Exception as e:
            logger.error(f"[Slack] EXCEPTION in clarification handler: {str(e)}", exc_info=True)
            try:
                say(text=f"Sorry, I encountered an error processing your response: {str(e)}", thread_ts=thread_ts)
            except Exception as say_error:
                logger.error(f"[Slack] Failed to send error message: {str(say_error)}")

    def _build_request(self, event: dict, config: ClientConfig) -> Request:
        text = event.get("text", "").replace(f"<@{self._bot_user_id}>", "").strip()
        return Request(
            channel=Channel.SLACK,
            sender_id=event["user"],
            sender_name=event["user"],
            text=text,
            thread_id=event.get("thread_ts") or event.get("ts"),
            timestamp=datetime.utcnow().isoformat(),
            client_id=config.client_id,
        )

    def _handle_result(self, result, event: dict, say, ticket=None) -> None:
        import logging
        from src.formatters.slack_formatter import SlackFormatter
        logger = logging.getLogger(__name__)
        thread_ts = event.get("thread_ts") or event.get("ts")

        if isinstance(result, ClarificationState):
            self._pending[thread_ts] = result
            questions = "\n".join(f"{q}" for q in result.questions_asked)
            say(text=questions, thread_ts=thread_ts)
            return

        if thread_ts in self._pending:
            del self._pending[thread_ts]

        request = None
        if isinstance(result, dict) and "request" in result:
            request = result["request"]

        reply_text = result.text
        if ticket is not None:
            reply_text = f"[{ticket.key}] {reply_text}"

        say(text=reply_text, thread_ts=thread_ts)

        for chart_png in result.charts:
            try:
                self._app.client.files_upload_v2(
                    channel=event["channel"],
                    file=chart_png,
                    filename="analysis.png",
                    thread_ts=thread_ts,
                )
            except Exception as e:
                logger.warning(f"[Slack] Failed to upload chart: {str(e)}")

        if result.report_markdown:
            try:
                say(text=result.report_markdown, thread_ts=thread_ts)
            except Exception as e:
                logger.warning(f"[Slack] Failed to send report: {str(e)}")

        if result.deck_pptx:
            try:
                self._app.client.files_upload_v2(
                    channel=event["channel"],
                    file=result.deck_pptx,
                    filename="presentation.pptx",
                    thread_ts=thread_ts,
                )
            except Exception as e:
                logger.warning(f"[Slack] Failed to upload presentation: {str(e)}")

    def _on_unconfigured_workspace(self, say, thread_ts: str) -> None:
        say(
            text="I'm not configured for this workspace yet. Please contact your account manager.",
            thread_ts=thread_ts,
        )

    def get_handler(self) -> SlackRequestHandler:
        return self._handler

    def send_message(self, channel: str, text: str) -> None:
        self._app.client.chat_postMessage(channel=channel, text=text)
