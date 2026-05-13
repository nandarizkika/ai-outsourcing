from src.core.models import Request, TicketRef
from src.ticketing.jira_client import JiraClient

_MAX_SUMMARY = 255


class TicketingService:
    def __init__(self, jira_client: JiraClient):
        self._client = jira_client

    def create_for_request(self, request: Request) -> TicketRef:
        summary = request.text[:_MAX_SUMMARY]
        description = (
            f"Request from {request.channel.value} by {request.client_id}"
            f" (sender: {request.sender_name}):\n\n{request.text}"
        )
        return self._client.create_ticket(summary=summary, description=description)
