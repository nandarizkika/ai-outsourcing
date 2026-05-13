import email as email_lib
import imaplib
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Optional

from src.core.models import Request, Channel, ClientConfig


class EmailChannel:
    def __init__(
        self,
        imap_host: str,
        imap_user: str,
        imap_password: str,
        smtp_host: str,
        smtp_port: int,
        orchestrator,
        client_configs: dict[str, ClientConfig],
        ticketing_service=None,
    ):
        self._imap_host = imap_host
        self._imap_user = imap_user
        self._imap_password = imap_password
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._orchestrator = orchestrator
        self._client_configs = client_configs
        self._ticketing = ticketing_service

    def poll_once(self, client_id: str) -> None:
        config = self._client_configs.get(client_id)
        if config is None:
            return
        with imaplib.IMAP4_SSL(self._imap_host) as imap:
            imap.login(self._imap_user, self._imap_password)
            imap.select("INBOX")
            _, data = imap.search(None, "UNSEEN")
            msg_ids = data[0].split() if data[0] else []
            for msg_id in msg_ids:
                _, msg_data = imap.fetch(msg_id, "(RFC822)")
                raw = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw)
                subject = msg.get("Subject", "")
                from_addr = msg.get("From", "")
                message_id = msg.get("Message-ID", "")
                body = self._extract_body(msg)
                text = f"{subject}\n\n{body}".strip() if subject else body
                request = Request(
                    channel=Channel.EMAIL,
                    sender_id=from_addr,
                    sender_name=from_addr.split("@")[0],
                    text=text,
                    thread_id=message_id or None,
                    timestamp=datetime.utcnow().isoformat(),
                    client_id=client_id,
                )
                ticket = None
                if self._ticketing:
                    try:
                        ticket = self._ticketing.create_for_request(request)
                    except Exception:
                        pass
                result = self._orchestrator.process(request, config)
                self._send_reply(from_addr, message_id, result, ticket)

    def _extract_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", errors="replace")
        return ""

    def _send_reply(self, to_addr: str, in_reply_to: str, result, ticket=None) -> None:
        reply = MIMEMultipart("mixed")
        reply["From"] = self._imap_user
        reply["To"] = to_addr
        text = result.text
        if ticket:
            text = f"[{ticket.key}] {result.text}"
        reply["Subject"] = f"Re: Analysis{f' [{ticket.key}]' if ticket else ''}"
        reply.attach(MIMEText(text, "plain"))
        for i, png in enumerate(result.charts or []):
            img = MIMEImage(png, name=f"chart_{i + 1}.png")
            img.add_header("Content-Disposition", "attachment", filename=f"chart_{i + 1}.png")
            reply.attach(img)
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as smtp:
            smtp.sendmail(self._imap_user, to_addr, reply.as_bytes())
