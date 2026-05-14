# AI Data Analyst — Phase 2: Email Channel & Jira Self-Ticketing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Email channel (IMAP polling + SMTP replies with chart attachments) and Jira self-ticketing so every request from Slack or Email is auto-logged as a Jira ticket and the ticket key is referenced in replies.

**Architecture:** `TicketingService` wraps `JiraClient` (httpx → Jira Cloud REST API v3) to create a Jira ticket from any `Request`. `EmailChannel` polls IMAP, normalizes emails into `Request` objects, routes through the Orchestrator, and replies via SMTP with text + PNG chart attachments. `SlackChannel` gains an optional `ticketing_service` parameter and includes the ticket key in every reply. All new settings default to empty strings so Phase 1 still runs without Jira or email configured.

**Tech Stack:** Python 3.11, httpx>=0.27 (Jira Cloud REST API v3), imaplib + smtplib (Python stdlib), existing Orchestrator/SQLAgent/ChartAgent stack, pydantic-settings v2

---

## File Structure

**New files:**
- `src/ticketing/__init__.py` — empty
- `src/ticketing/jira_client.py` — JiraClient: POST /rest/api/3/issue via httpx Basic Auth
- `src/ticketing/ticketing_service.py` — TicketingService: derives summary + description from Request, calls JiraClient
- `src/channels/email_channel.py` — EmailChannel: IMAP4_SSL polling, Request normalization, SMTP replies with chart attachments
- `tests/ticketing/__init__.py` — empty
- `tests/ticketing/test_jira_client.py` — 3 tests (create_ticket, long summary truncation, HTTP error)
- `tests/ticketing/test_ticketing_service.py` — 2 tests (creates ticket, truncates long request)
- `tests/channels/test_email_channel.py` — 4 tests (build_request, SMTP call, chart attachment, process_pending)
- `tests/test_integration_phase2.py` — 2 tests (email end-to-end, Slack + Jira ticketing)

**Modified files:**
- `src/core/models.py` — add `TicketRef` model; add `ticket: Optional[TicketRef] = None` to `Response`
- `src/core/config.py` — add optional Jira + email settings (all default to empty/0)
- `pyproject.toml` — add `httpx>=0.27` to dependencies
- `src/channels/slack.py` — add optional `ticketing_service` param, call it in `_handle_mention`, include key in replies
- `main.py` — conditionally wire JiraClient + TicketingService + EmailChannel + `/email/poll` endpoint

---

## Task 1: TicketRef Model + Response.ticket

**Files:**
- Modify: `src/core/models.py`
- Create: `tests/core/test_models.py` (create `tests/core/__init__.py` if missing)

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_models.py`:

```python
# tests/core/test_models.py
from src.core.models import TicketRef, Response


def test_ticket_ref_has_key_url_summary():
    ref = TicketRef(key="AI-42", url="https://test.atlassian.net/browse/AI-42", summary="Show sales")
    assert ref.key == "AI-42"
    assert "AI-42" in ref.url
    assert ref.summary == "Show sales"


def test_response_ticket_defaults_to_none():
    r = Response(request_id="r1", text="Analysis")
    assert r.ticket is None


def test_response_accepts_ticket_ref():
    ref = TicketRef(key="AI-1", url="https://x.atlassian.net/browse/AI-1", summary="test")
    r = Response(request_id="r1", text="Analysis", ticket=ref)
    assert r.ticket.key == "AI-1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/core/test_models.py -v
```

Expected: `ImportError: cannot import name 'TicketRef' from 'src.core.models'`

- [ ] **Step 3: Write implementation**

In `src/core/models.py`, add the `TicketRef` class after `AgentResult` and before `Response`, and add `ticket` field to `Response`:

```python
class TicketRef(BaseModel):
    key: str      # e.g. "AI-42"
    url: str      # e.g. "https://mycompany.atlassian.net/browse/AI-42"
    summary: str


class Response(BaseModel):
    request_id: str
    text: str
    charts: list[bytes] = []
    assumptions: list[str] = []
    ticket: Optional[TicketRef] = None
```

The full updated `src/core/models.py`:

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Channel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    JIRA = "jira"


class SkillModule(str, Enum):
    SQL_QUERYING = "sql_querying"
    DATA_VISUALIZATION = "data_visualization"
    REPORT_GENERATION = "report_generation"
    SCHEDULED_REPORTING = "scheduled_reporting"
    HARD_RULE_ANOMALY = "hard_rule_anomaly"
    SPREADSHEET_ANALYSIS = "spreadsheet_analysis"
    LOOKER_INTEGRATION = "looker_integration"
    STATISTICAL_ANOMALY = "statistical_anomaly"
    MACHINE_LEARNING = "machine_learning"
    PRESENTATION_BUILDING = "presentation_building"


class Tier(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"


class TaskType(str, Enum):
    REASONING = "reasoning"
    TOOL = "tool"
    SIMPLE = "simple"


class Request(BaseModel):
    channel: Channel
    sender_id: str
    sender_name: str
    text: str
    thread_id: Optional[str] = None
    timestamp: str
    client_id: str


class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[SkillModule]
    account_mode: str
    active_channels: list[Channel]


class ClarificationState(BaseModel):
    original_request: Request
    rounds: int = 0
    questions_asked: list[str] = []
    answers_received: list[str] = []
    is_resolved: bool = False
    assumptions: list[str] = []


class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: Optional[dict] = None
    chart_png: Optional[bytes] = None
    error: Optional[str] = None


class TicketRef(BaseModel):
    key: str
    url: str
    summary: str


class Response(BaseModel):
    request_id: str
    text: str
    charts: list[bytes] = []
    assumptions: list[str] = []
    ticket: Optional[TicketRef] = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/core/test_models.py -v
```

Expected: `3 passed`

Full suite — confirm no regressions:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `54 passed` (all Phase 1 tests still pass)

- [ ] **Step 5: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add src/core/models.py tests/core/test_models.py
git commit -m "feat: TicketRef model and Response.ticket field"
```

---

## Task 2: httpx Dependency + JiraClient

**Files:**
- Modify: `pyproject.toml`
- Create: `src/ticketing/__init__.py`
- Create: `src/ticketing/jira_client.py`
- Create: `tests/ticketing/__init__.py`
- Create: `tests/ticketing/test_jira_client.py`

- [ ] **Step 1: Add httpx to pyproject.toml and install**

In `pyproject.toml`, add `"httpx>=0.27",` to the end of the `dependencies` list:

```toml
dependencies = [
    "anthropic>=0.30.0",
    "openai>=1.30.0",
    "chromadb>=0.5.0",
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "slack-bolt>=1.18",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pdfplumber>=0.10",
    "python-docx>=1.0",
    "openpyxl>=3.1",
    "python-dotenv>=1.0",
    "httpx>=0.27",
]
```

Install it:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pip install "httpx>=0.27"
```

Expected: `Successfully installed httpx-...` (or `already satisfied`)

- [ ] **Step 2: Write the failing test**

Create `tests/ticketing/__init__.py` (empty) and `tests/ticketing/test_jira_client.py`:

```python
# tests/ticketing/test_jira_client.py
import pytest
from unittest.mock import MagicMock, patch
from src.ticketing.jira_client import JiraClient
from src.core.models import TicketRef


def make_client() -> JiraClient:
    return JiraClient(
        jira_url="https://test.atlassian.net",
        email="ai@test.com",
        api_token="fake-token",
        project_key="AI",
    )


def test_create_ticket_returns_ticket_ref():
    client = make_client()
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "AI-42", "id": "10001"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.post", return_value=mock_response):
        result = client.create_ticket("Show sales data", "Full request text")

    assert isinstance(result, TicketRef)
    assert result.key == "AI-42"
    assert "AI-42" in result.url
    assert "test.atlassian.net" in result.url
    assert result.summary == "Show sales data"


def test_create_ticket_truncates_long_summary():
    client = make_client()
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "AI-43"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.post", return_value=mock_response) as mock_post:
        client.create_ticket("a" * 300, "description")

    payload = mock_post.call_args[1]["json"]
    assert len(payload["fields"]["summary"]) <= 255


def test_create_ticket_raises_on_http_error():
    client = make_client()
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(Exception, match="401"):
            client.create_ticket("Test", "Test")
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/ticketing/test_jira_client.py -v
```

Expected: `ImportError: No module named 'src.ticketing.jira_client'`

- [ ] **Step 4: Write implementation**

Create `src/ticketing/__init__.py` (empty) and `src/ticketing/jira_client.py`:

```python
# src/ticketing/jira_client.py
import base64

import httpx

from src.core.models import TicketRef


class JiraClient:
    def __init__(self, jira_url: str, email: str, api_token: str, project_key: str):
        self._base_url = jira_url.rstrip("/")
        self._project_key = project_key
        credentials = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def create_ticket(self, summary: str, description: str) -> TicketRef:
        payload = {
            "fields": {
                "project": {"key": self._project_key},
                "summary": summary[:255],
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                },
                "issuetype": {"name": "Task"},
            }
        }
        response = httpx.post(
            f"{self._base_url}/rest/api/3/issue",
            json=payload,
            headers=self._headers,
            timeout=10.0,
        )
        response.raise_for_status()
        key = response.json()["key"]
        return TicketRef(
            key=key,
            url=f"{self._base_url}/browse/{key}",
            summary=summary,
        )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/ticketing/test_jira_client.py -v
```

Expected: `3 passed`

Full suite:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `57 passed`

- [ ] **Step 6: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add pyproject.toml src/ticketing/__init__.py src/ticketing/jira_client.py tests/ticketing/__init__.py tests/ticketing/test_jira_client.py
git commit -m "feat: Jira client — create tickets via Jira Cloud REST API v3"
```

---

## Task 3: TicketingService

**Files:**
- Create: `src/ticketing/ticketing_service.py`
- Create: `tests/ticketing/test_ticketing_service.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ticketing/test_ticketing_service.py`:

```python
# tests/ticketing/test_ticketing_service.py
from datetime import datetime
from unittest.mock import MagicMock
from src.ticketing.ticketing_service import TicketingService
from src.core.models import Request, Channel, TicketRef


def make_request(text: str = "Show total sales by region") -> Request:
    return Request(
        channel=Channel.SLACK,
        sender_id="U123",
        sender_name="Budi",
        text=text,
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-1",
    )


def test_creates_ticket_from_request():
    jira = MagicMock()
    jira.create_ticket.return_value = TicketRef(
        key="AI-10",
        url="https://test.atlassian.net/browse/AI-10",
        summary="Show total sales by region",
    )
    service = TicketingService(jira_client=jira)

    ref = service.create_for_request(make_request())

    assert ref.key == "AI-10"
    jira.create_ticket.assert_called_once()
    call_kwargs = jira.create_ticket.call_args[1]
    assert "summary" in call_kwargs
    assert "description" in call_kwargs
    assert "Budi" in call_kwargs["description"]


def test_truncates_long_request_text_for_summary():
    jira = MagicMock()
    jira.create_ticket.return_value = TicketRef(
        key="AI-11", url="https://x/browse/AI-11", summary="x" * 100
    )
    service = TicketingService(jira_client=jira)

    service.create_for_request(make_request("x" * 200))

    call_kwargs = jira.create_ticket.call_args[1]
    assert len(call_kwargs["summary"]) <= 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/ticketing/test_ticketing_service.py -v
```

Expected: `ImportError: No module named 'src.ticketing.ticketing_service'`

- [ ] **Step 3: Write implementation**

Create `src/ticketing/ticketing_service.py`:

```python
# src/ticketing/ticketing_service.py
from src.core.models import Request, TicketRef
from src.ticketing.jira_client import JiraClient


class TicketingService:
    def __init__(self, jira_client: JiraClient):
        self._jira = jira_client

    def create_for_request(self, request: Request) -> TicketRef:
        summary = request.text[:100]
        description = (
            f"Requested by: {request.sender_name} ({request.sender_id})\n"
            f"Channel: {request.channel.value}\n"
            f"Client: {request.client_id}\n\n"
            f"Full request:\n{request.text}"
        )
        return self._jira.create_ticket(summary=summary, description=description)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/ticketing/test_ticketing_service.py -v
```

Expected: `2 passed`

Full suite:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `59 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add src/ticketing/ticketing_service.py tests/ticketing/test_ticketing_service.py
git commit -m "feat: ticketing service — derives Jira ticket from Request"
```

---

## Task 4: EmailChannel

**Files:**
- Create: `src/channels/email_channel.py`
- Create: `tests/channels/test_email_channel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/channels/test_email_channel.py`:

```python
# tests/channels/test_email_channel.py
import email.mime.text
from unittest.mock import MagicMock, patch

from src.channels.email_channel import EmailChannel
from src.core.models import Channel, ClientConfig, Tier, Response, ClarificationState
from src.skill_modules import SkillModuleRegistry


def make_config() -> ClientConfig:
    return ClientConfig(
        client_id="client-1", name="Test Corp", tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor", active_channels=[Channel.EMAIL],
    )


def make_channel(orchestrator=None) -> EmailChannel:
    if orchestrator is None:
        orchestrator = MagicMock()
        orchestrator.process.return_value = Response(request_id="r1", text="Here is the analysis.")
    return EmailChannel(
        imap_host="imap.test.com",
        imap_user="ai@test.com",
        imap_password="password",
        smtp_host="smtp.test.com",
        smtp_port=587,
        orchestrator=orchestrator,
        config=make_config(),
    )


def make_email_msg(body: str = "Show total sales by region", sender: str = "budi@corp.com"):
    msg = email.mime.text.MIMEText(body)
    msg["From"] = sender
    msg["To"] = "ai@test.com"
    msg["Subject"] = "Analysis Request"
    msg["Message-ID"] = "<test-001@corp.com>"
    return msg


def test_build_request_extracts_body_and_sets_channel():
    channel = make_channel()
    msg = make_email_msg("Show total sales by region", "budi@corp.com")

    request = channel._build_request(msg)

    assert request.channel == Channel.EMAIL
    assert request.client_id == "client-1"
    assert "Show total sales" in request.text
    assert request.sender_id == "budi@corp.com"
    assert request.thread_id == "<test-001@corp.com>"


def test_send_reply_calls_smtp_with_correct_host():
    channel = make_channel()

    with patch("smtplib.SMTP") as mock_smtp_class:
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        channel._send_reply(
            to="budi@corp.com",
            subject="Re: Analysis",
            in_reply_to="<test-001@corp.com>",
            text="Here is the result.",
            charts=[],
        )

    mock_smtp_class.assert_called_once_with("smtp.test.com", 587)
    mock_server.sendmail.assert_called_once()


def test_send_reply_attaches_png_chart():
    channel = make_channel()
    sent_messages = []

    with patch("smtplib.SMTP") as mock_smtp_class:
        mock_server = MagicMock()
        mock_server.sendmail.side_effect = lambda f, t, m: sent_messages.append(m)
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        channel._send_reply(
            to="budi@corp.com",
            subject="Re: Analysis",
            in_reply_to="",
            text="Results:",
            charts=[b"\x89PNG\r\nfake"],
        )

    assert len(sent_messages) == 1
    assert "chart_1.png" in sent_messages[0]


def test_process_pending_calls_orchestrator_and_sends_reply(mocker):
    orchestrator = MagicMock()
    orchestrator.process.return_value = Response(request_id="r1", text="Sales analysis complete.")
    channel = make_channel(orchestrator)

    msg = make_email_msg("Show total sales")
    mocker.patch.object(channel, "_fetch_unseen", return_value=[("1", msg)])
    mock_send = mocker.patch.object(channel, "_send_reply")

    processed = channel.process_pending()

    assert processed == ["1"]
    orchestrator.process.assert_called_once()
    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args[1]
    assert "Sales analysis" in call_kwargs["text"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/channels/test_email_channel.py -v
```

Expected: `ImportError: cannot import name 'EmailChannel'`

- [ ] **Step 3: Write implementation**

Create `src/channels/email_channel.py`:

```python
# src/channels/email_channel.py
import email
import email.mime.application
import email.mime.multipart
import email.mime.text
import imaplib
import smtplib
from datetime import datetime, timezone
from email.header import decode_header

from src.core.models import Channel, ClientConfig, ClarificationState, Request, Response
from src.orchestrator.orchestrator import Orchestrator


class EmailChannel:
    def __init__(
        self,
        imap_host: str,
        imap_user: str,
        imap_password: str,
        smtp_host: str,
        smtp_port: int,
        orchestrator: Orchestrator,
        config: ClientConfig,
        ticketing_service=None,
    ):
        self._imap_host = imap_host
        self._imap_user = imap_user
        self._imap_password = imap_password
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._orchestrator = orchestrator
        self._config = config
        self._ticketing = ticketing_service

    def process_pending(self) -> list[str]:
        messages = self._fetch_unseen()
        processed = []
        for msg_id, msg in messages:
            try:
                request = self._build_request(msg)

                ticket = None
                if self._ticketing:
                    try:
                        ticket = self._ticketing.create_for_request(request)
                    except Exception:
                        pass

                result = self._orchestrator.process(request, self._config)

                if isinstance(result, ClarificationState):
                    questions = "\n".join(f"- {q}" for q in result.questions_asked[-2:])
                    text = f"Before I run this analysis, I need to clarify:\n\n{questions}"
                    charts: list[bytes] = []
                else:
                    text = result.text
                    if ticket:
                        text = f"[{ticket.key}] {result.text}"
                    charts = result.charts

                from_addr = msg.get("From", "")
                subject = msg.get("Subject", "Analysis Results")
                in_reply_to = msg.get("Message-ID", "")
                self._send_reply(
                    to=from_addr,
                    subject=f"Re: {subject}",
                    in_reply_to=in_reply_to,
                    text=text,
                    charts=charts,
                )
                processed.append(msg_id)
            except Exception:
                pass
        return processed

    def _fetch_unseen(self) -> list[tuple[str, email.message.Message]]:
        messages = []
        with imaplib.IMAP4_SSL(self._imap_host) as conn:
            conn.login(self._imap_user, self._imap_password)
            conn.select("INBOX")
            _, data = conn.search(None, "UNSEEN")
            for msg_id in data[0].split():
                _, msg_data = conn.fetch(msg_id, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                conn.store(msg_id, "+FLAGS", "\\Seen")
                messages.append((msg_id.decode(), msg))
        return messages

    def _build_request(self, msg: email.message.Message) -> Request:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")

        from_header = msg.get("From", "")
        decoded = decode_header(from_header)[0][0]
        sender_name = decoded.decode() if isinstance(decoded, bytes) else (decoded or from_header)

        return Request(
            channel=Channel.EMAIL,
            sender_id=from_header,
            sender_name=sender_name,
            text=body.strip(),
            thread_id=msg.get("Message-ID"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            client_id=self._config.client_id,
        )

    def _send_reply(
        self,
        to: str,
        subject: str,
        in_reply_to: str,
        text: str,
        charts: list[bytes],
    ) -> None:
        outer = email.mime.multipart.MIMEMultipart()
        outer["From"] = self._imap_user
        outer["To"] = to
        outer["Subject"] = subject
        if in_reply_to:
            outer["In-Reply-To"] = in_reply_to
            outer["References"] = in_reply_to
        outer.attach(email.mime.text.MIMEText(text, "plain"))
        for i, png in enumerate(charts):
            part = email.mime.application.MIMEApplication(png, Name=f"chart_{i + 1}.png")
            part["Content-Disposition"] = f'attachment; filename="chart_{i + 1}.png"'
            outer.attach(part)
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._imap_user, self._imap_password)
            server.sendmail(self._imap_user, to, outer.as_string())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/channels/test_email_channel.py -v
```

Expected: `4 passed`

Full suite:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `63 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add src/channels/email_channel.py tests/channels/test_email_channel.py
git commit -m "feat: email channel — IMAP polling and SMTP replies with chart attachments"
```

---

## Task 5: SlackChannel Self-Ticketing

**Files:**
- Modify: `src/channels/slack.py`
- Modify: `tests/channels/test_slack.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/channels/test_slack.py` (after the existing 3 tests):

```python
def test_ticket_key_included_in_slack_reply():
    from src.ticketing.ticketing_service import TicketingService
    from src.core.models import TicketRef, Response

    orchestrator = MagicMock()
    orchestrator.process.return_value = Response(request_id="r1", text="Here is the analysis.")

    ticketing = MagicMock(spec=TicketingService)
    ticketing.create_for_request.return_value = TicketRef(
        key="AI-42",
        url="https://test.atlassian.net/browse/AI-42",
        summary="Show sales",
    )

    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
        token_verification_enabled=False,
        ticketing_service=ticketing,
    )
    say = MagicMock()
    event = {
        "user": "U123",
        "text": "<@BOTID> show sales",
        "ts": "123.456",
        "channel": "C001",
        "team": "T001",
    }
    channel._handle_mention(event, say)

    ticketing.create_for_request.assert_called_once()
    call_text = say.call_args[1]["text"]
    assert "AI-42" in call_text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/channels/test_slack.py::test_ticket_key_included_in_slack_reply -v
```

Expected: `FAILED — TypeError: SlackChannel.__init__() got an unexpected keyword argument 'ticketing_service'`

- [ ] **Step 3: Write implementation**

Replace the full `src/channels/slack.py` with:

```python
# src/channels/slack.py
from __future__ import annotations

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
        self._ticketing = ticketing_service
        self._pending: dict[str, ClarificationState] = {}

        self._app = App(
            token=bot_token,
            signing_secret=signing_secret,
            token_verification_enabled=token_verification_enabled,
        )
        self._app.event("app_mention")(self._handle_mention)
        self._handler = SlackRequestHandler(self._app)

    def _handle_mention(self, event: dict, say) -> None:
        team_id = event.get("team")
        config = self._client_configs.get(team_id)
        if not config:
            self._on_unconfigured_workspace(say, event.get("thread_ts") or event.get("ts"))
            return

        thread_ts = event.get("thread_ts") or event.get("ts")
        request = self._build_request(event, config)

        ticket = None
        if self._ticketing:
            try:
                ticket = self._ticketing.create_for_request(request)
            except Exception:
                pass

        pending = self._pending.get(thread_ts)
        active_clarification = pending if (pending and not pending.is_resolved) else None
        if active_clarification:
            active_clarification.answers_received.append(request.text)

        result = self._orchestrator.process(request, config, active_clarification)
        self._handle_result(result, event, say, ticket=ticket)

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
        thread_ts = event.get("thread_ts") or event.get("ts")

        if isinstance(result, ClarificationState):
            self._pending[thread_ts] = result
            questions = "\n".join(f"• {q}" for q in result.questions_asked[-2:])
            ticket_ref = f" [{ticket.key}]" if ticket else ""
            say(text=f"Quick question before I run this{ticket_ref}:\n{questions}", thread_ts=thread_ts)
            return

        if thread_ts in self._pending:
            del self._pending[thread_ts]

        text = result.text
        if ticket:
            text = f"[{ticket.key}] {result.text}"
        say(text=text, thread_ts=thread_ts)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/channels/test_slack.py -v
```

Expected: `4 passed` (3 existing + 1 new)

Full suite:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `64 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add src/channels/slack.py tests/channels/test_slack.py
git commit -m "feat: slack channel self-ticketing — optional Jira ticket key on every mention"
```

---

## Task 6: Config Updates + main.py Wiring

**Files:**
- Modify: `src/core/config.py`
- Modify: `main.py`
- Modify: `tests/core/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_config.py`:

```python
def test_jira_and_email_settings_have_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "c")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "d")

    s = Settings()
    assert s.jira_url == ""
    assert s.jira_email == ""
    assert s.jira_api_token == ""
    assert s.jira_project_key == "AI"
    assert s.email_imap_host == ""
    assert s.email_imap_user == ""
    assert s.email_imap_password == ""
    assert s.email_smtp_host == ""
    assert s.email_smtp_port == 587
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/core/test_config.py::test_jira_and_email_settings_have_defaults -v
```

Expected: `FAILED — AttributeError: 'Settings' object has no attribute 'jira_url'`

- [ ] **Step 3: Update Settings**

Replace `src/core/config.py`:

```python
# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Required
    anthropic_api_key: str
    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str

    # ChromaDB
    chromadb_persist_dir: str = ".chromadb"

    # Jira (optional — leave empty to disable ticketing)
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "AI"

    # Email (optional — leave empty to disable email channel)
    email_imap_host: str = ""
    email_imap_user: str = ""
    email_imap_password: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587

    model_config = SettingsConfigDict(env_file=".env")
```

- [ ] **Step 4: Update main.py**

Replace `main.py`:

```python
# main.py
from fastapi import FastAPI, Request
from src.core.config import Settings
from src.core.llm import LLMRouter
from src.knowledge.vector_store import VectorStore
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.channels.slack import SlackChannel

settings = Settings()

llm = LLMRouter(
    anthropic_api_key=settings.anthropic_api_key,
    openai_api_key=settings.openai_api_key,
)
store = VectorStore(
    persist_dir=settings.chromadb_persist_dir,
    openai_api_key=settings.openai_api_key,
)
retriever = KnowledgeRetriever(store=store)
clarifier = ClarificationChecker(llm=llm)
chart_agent = ChartAgent()

client_configs = {}  # workspace_team_id -> ClientConfig; loaded at onboarding

orchestrator = Orchestrator(
    llm=llm,
    retriever=retriever,
    clarifier=clarifier,
    sql_agent=None,   # injected per-client at runtime
    chart_agent=chart_agent,
)

# Jira ticketing — enabled when jira_url and jira_api_token are set
ticketing_service = None
if settings.jira_url and settings.jira_api_token:
    from src.ticketing.jira_client import JiraClient
    from src.ticketing.ticketing_service import TicketingService
    ticketing_service = TicketingService(
        jira_client=JiraClient(
            jira_url=settings.jira_url,
            email=settings.jira_email,
            api_token=settings.jira_api_token,
            project_key=settings.jira_project_key,
        )
    )

SLACK_BOT_USER_ID = "REPLACE_WITH_BOT_USER_ID"

slack = SlackChannel(
    bot_token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    bot_user_id=SLACK_BOT_USER_ID,
    orchestrator=orchestrator,
    client_configs=client_configs,
    ticketing_service=ticketing_service,
)

# Email channel — enabled when email_imap_host is set and at least one client is configured
email_channel = None
if settings.email_imap_host and client_configs:
    from src.channels.email_channel import EmailChannel
    _email_config = next(iter(client_configs.values()))
    email_channel = EmailChannel(
        imap_host=settings.email_imap_host,
        imap_user=settings.email_imap_user,
        imap_password=settings.email_imap_password,
        smtp_host=settings.email_smtp_host,
        smtp_port=settings.email_smtp_port,
        orchestrator=orchestrator,
        config=_email_config,
        ticketing_service=ticketing_service,
    )

app = FastAPI()


@app.post("/slack/events")
async def slack_events(req: Request):
    return await slack.get_handler().handle(req)


@app.post("/email/poll")
def poll_email():
    if not email_channel:
        return {"status": "email not configured"}
    processed = email_channel.process_pending()
    return {"processed": len(processed), "message_ids": processed}


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/core/test_config.py -v
```

Expected: `3 passed` (2 existing + 1 new)

Full suite:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `65 passed`

- [ ] **Step 6: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add src/core/config.py main.py tests/core/test_config.py
git commit -m "feat: config and main.py wiring for Jira ticketing and email channel"
```

---

## Task 7: Phase 2 Integration Test

**Files:**
- Create: `tests/test_integration_phase2.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration_phase2.py
"""
Phase 2 integration tests:
- Full email flow: fetch email → Orchestrator → SMTP reply with chart attached
- Slack @mention with Jira ticketing → ticket key appears in Slack reply
"""
import email.mime.text
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.core.models import Channel, ClientConfig, Tier, Response, TicketRef
from src.core.llm import LLMRouter
from src.knowledge.retriever import KnowledgeRetriever
from src.connectors.sql import SQLConnector
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.channels.email_channel import EmailChannel
from src.channels.slack import SlackChannel
from src.ticketing.jira_client import JiraClient
from src.ticketing.ticketing_service import TicketingService
from src.skill_modules import SkillModuleRegistry


def make_config() -> ClientConfig:
    return ClientConfig(
        client_id="client-1", name="Test Corp", tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor", active_channels=[Channel.SLACK, Channel.EMAIL],
    )


@pytest.fixture
def real_orchestrator(mocker):
    llm = MagicMock(spec=LLMRouter)
    llm.complete.side_effect = [
        '{"is_clear": true, "questions": []}',      # clarifier
        '{"sql": true, "chart": true}',              # plan
        "SELECT region, amount FROM sales",          # SQL agent generates SQL
        "Total sales: Jakarta 1M, Surabaya 2M",     # response text
    ]

    store = MagicMock()
    store.search.return_value = []
    store.add.return_value = None
    retriever = KnowledgeRetriever(store=store)

    connector = MagicMock(spec=SQLConnector)
    connector.get_schema.return_value = {
        "sales": [{"name": "region", "type": "TEXT"}, {"name": "amount", "type": "REAL"}]
    }
    connector.execute.return_value = pd.DataFrame([
        {"region": "Jakarta", "amount": 1_000_000},
        {"region": "Surabaya", "amount": 2_000_000},
    ])

    sql_agent = SQLAgent(llm=llm, connector=connector, store=store)
    chart_agent = ChartAgent()
    clarifier = ClarificationChecker(llm=llm)
    return Orchestrator(
        llm=llm, retriever=retriever, clarifier=clarifier,
        sql_agent=sql_agent, chart_agent=chart_agent,
    )


def test_email_flow_sends_reply_with_analysis_and_chart(real_orchestrator, mocker):
    channel = EmailChannel(
        imap_host="imap.test.com",
        imap_user="ai@test.com",
        imap_password="pass",
        smtp_host="smtp.test.com",
        smtp_port=587,
        orchestrator=real_orchestrator,
        config=make_config(),
    )

    msg = email.mime.text.MIMEText("Show total sales by region")
    msg["From"] = "budi@corp.com"
    msg["Message-ID"] = "<001@corp.com>"
    msg["Subject"] = "Analysis Request"

    mocker.patch.object(channel, "_fetch_unseen", return_value=[("1", msg)])
    sent_replies = []
    mocker.patch.object(channel, "_send_reply", side_effect=lambda **kw: sent_replies.append(kw))

    processed = channel.process_pending()

    assert len(processed) == 1
    assert len(sent_replies) == 1
    assert "Jakarta" in sent_replies[0]["text"] or "sales" in sent_replies[0]["text"].lower()
    assert len(sent_replies[0]["charts"]) == 1
    assert sent_replies[0]["charts"][0][:4] == b"\x89PNG"


def test_slack_mention_creates_jira_ticket_and_key_appears_in_reply():
    orch = MagicMock()
    orch.process.return_value = Response(request_id="r1", text="Here is the analysis.")

    jira = MagicMock(spec=JiraClient)
    jira.create_ticket.return_value = TicketRef(
        key="AI-99",
        url="https://test.atlassian.net/browse/AI-99",
        summary="show total sales",
    )
    ticketing = TicketingService(jira_client=jira)

    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake",
        bot_user_id="BOTID",
        orchestrator=orch,
        client_configs={"T001": make_config()},
        token_verification_enabled=False,
        ticketing_service=ticketing,
    )
    say = MagicMock()
    event = {
        "user": "U123", "text": "<@BOTID> show total sales",
        "ts": "123.456", "channel": "C001", "team": "T001",
    }
    channel._handle_mention(event, say)

    jira.create_ticket.assert_called_once()
    call_kwargs = jira.create_ticket.call_args[1]
    assert "show total sales" in call_kwargs["summary"].lower()

    reply_text = say.call_args[1]["text"]
    assert "AI-99" in reply_text
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest tests/test_integration_phase2.py -v
```

Expected: `2 passed`

Full suite:

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent && .venv/bin/pytest --tb=short 2>&1 | tail -5
```

Expected: `67 passed`

- [ ] **Step 3: Commit**

```bash
cd /Users/nandarizkika/Documents/Project/ai_talent
git add tests/test_integration_phase2.py
git commit -m "test: Phase 2 integration — email flow with chart, Slack with Jira self-ticketing"
```

---

## Spec Coverage Self-Review

Design spec coverage for Phase 2:

| Requirement | Task |
|---|---|
| Email channel (IMAP + SMTP) | Task 4 (EmailChannel) |
| Chart attachments in email replies | Task 4 (_send_reply with MIMEApplication) |
| Jira self-ticketing from Slack | Task 3 + Task 5 (SlackChannel) |
| Jira self-ticketing from Email | Task 3 + Task 4 (EmailChannel) |
| Ticket key referenced in reply | Task 5 (`[AI-42] {text}` prefix) |
| Title derived from request text | Task 3 (TicketingService `summary = request.text[:100]`) |
| Linked to originating thread | Task 4 (In-Reply-To SMTP header from Message-ID) |
| Optional — won't break if not configured | Task 6 (conditional wiring, empty string defaults) |
| Clarification questions via email | Task 4 (process_pending handles ClarificationState) |

**Out of scope for Phase 2 (Phase 3):**
- WhatsApp channel (requires Facebook Business API approval)
- Scheduled jobs (APScheduler or Celery beat)
- ML Agent (Prophet forecasting, scikit-learn classification)
- Deck Agent (python-pptx / Google Slides API)
- Anomaly detection agents
