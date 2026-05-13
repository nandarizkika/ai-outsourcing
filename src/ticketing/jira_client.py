import httpx
from src.core.models import TicketRef

_MAX_SUMMARY_LEN = 255


class JiraClient:
    def __init__(self, jira_url: str, email: str, api_token: str, project_key: str):
        self._url = jira_url.rstrip("/")
        self._auth = (email, api_token)
        self._project_key = project_key

    def create_ticket(self, summary: str, description: str) -> TicketRef:
        summary = summary[:_MAX_SUMMARY_LEN]
        payload = {
            "fields": {
                "project": {"key": self._project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}],
                },
                "issuetype": {"name": "Task"},
            }
        }
        resp = httpx.post(
            f"{self._url}/rest/api/3/issue",
            json=payload,
            auth=self._auth,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        key = data["key"]
        return TicketRef(
            key=key,
            url=f"{self._url}/browse/{key}",
            summary=summary,
        )
