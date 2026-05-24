from unittest.mock import AsyncMock, MagicMock
import pytest

from src.jobs.scheduler import JobScheduler
from src.core.models import ScheduledJob, Channel, Response, ClientConfig, Tier


def _make_scheduler():
    mock_orchestrator = MagicMock()
    mock_orchestrator.process = AsyncMock(return_value=Response(
        request_id="r1", text="Revenue grew 10%", charts=[]
    ))
    client_configs = {
        "client1": ClientConfig(
            client_id="client1",
            name="Client One",
            tier=Tier.BASIC,
            enabled_skills=[],
            account_mode="vendor",
            active_channels=[Channel.SLACK],
        )
    }
    scheduler = JobScheduler(
        orchestrator=mock_orchestrator,
        client_configs=client_configs,
    )
    return scheduler, mock_orchestrator


def _make_job(cron="0 9 * * 1"):
    return ScheduledJob(
        job_id="job1",
        client_id="client1",
        description="Weekly revenue",
        request_text="Show weekly revenue",
        cron_expression=cron,
        delivery_channel=Channel.SLACK,
        delivery_destination="C_GENERAL",
    )


def test_add_and_list_job():
    scheduler, _ = _make_scheduler()
    job = _make_job()
    scheduler.add_job(job, on_complete=lambda r: None)
    jobs = scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "job1"


def test_remove_job():
    scheduler, _ = _make_scheduler()
    job = _make_job()
    scheduler.add_job(job, on_complete=lambda r: None)
    scheduler.remove_job("job1")
    assert scheduler.list_jobs() == []


def test_run_job_calls_orchestrator_and_callback():
    scheduler, mock_orchestrator = _make_scheduler()
    callback = MagicMock()
    job = _make_job()
    scheduler.add_job(job, on_complete=callback)
    scheduler._run_job("job1")
    mock_orchestrator.process.assert_called_once()
    callback.assert_called_once()
    response_arg = callback.call_args[0][0]
    assert response_arg.text == "Revenue grew 10%"


def test_run_job_updates_last_run():
    scheduler, _ = _make_scheduler()
    job = _make_job()
    scheduler.add_job(job, on_complete=lambda r: None)
    scheduler._run_job("job1")
    updated = scheduler.list_jobs()[0]
    assert updated.last_run is not None
    assert updated.last_result_summary == "Revenue grew 10%"
