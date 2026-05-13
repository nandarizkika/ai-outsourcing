import logging
from datetime import datetime, timezone
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.core.models import Channel, Request, Response, ScheduledJob

_logger = logging.getLogger(__name__)


class JobScheduler:
    def __init__(self, orchestrator, client_configs: dict):
        self._orchestrator = orchestrator
        self._client_configs = client_configs
        self._jobs: dict[str, ScheduledJob] = {}
        self._callbacks: dict[str, Callable[[Response], None]] = {}
        self._scheduler = BackgroundScheduler()

    def add_job(self, job: ScheduledJob, on_complete: Callable[[Response], None]) -> None:
        self._jobs[job.job_id] = job
        self._callbacks[job.job_id] = on_complete
        self._scheduler.add_job(
            self._run_job,
            trigger=CronTrigger.from_crontab(job.cron_expression),
            args=[job.job_id],
            id=job.job_id,
            replace_existing=True,
        )

    def remove_job(self, job_id: str) -> None:
        if job_id in self._jobs:
            if self._scheduler.get_job(job_id):
                self._scheduler.remove_job(job_id)
            del self._jobs[job_id]
            self._callbacks.pop(job_id, None)

    def _run_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        config = self._client_configs.get(job.client_id)
        if config is None:
            _logger.warning("No config for client_id=%s in job %s", job.client_id, job_id)
            return
        request = Request(
            channel=Channel.JIRA,
            sender_id="scheduler",
            sender_name="Scheduler",
            text=job.request_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            client_id=job.client_id,
        )
        try:
            result = self._orchestrator.process(request, config)
            job.last_run = datetime.now(timezone.utc).isoformat()
            if hasattr(result, "text"):
                job.last_result_summary = result.text[:500]
                callback = self._callbacks.get(job_id)
                if callback:
                    callback(result)
        except Exception as exc:
            _logger.error("Scheduled job %s failed: %s", job_id, exc)

    def list_jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    def start(self) -> None:
        self._scheduler.start()

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
