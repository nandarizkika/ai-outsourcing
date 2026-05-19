# tests/conftest.py
import os
from unittest.mock import patch, MagicMock

# Set required env vars before any imports happen
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-signing-secret")

# Patch the Slack WebClient auth.test call so main.py can be imported without
# a real Slack connection. We mock only the network call, not the whole App.
_auth_patch = patch(
    "slack_sdk.web.client.WebClient.api_call",
    return_value={"ok": True, "bot_id": "B000", "user_id": "U000"},
)
_auth_patch.start()
