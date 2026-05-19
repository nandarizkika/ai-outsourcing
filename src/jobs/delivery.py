from src.core.models import Channel, ScheduledJob


def deliver(job: ScheduledJob, response, slack_channel=None, email_channel=None) -> None:
    if not hasattr(response, "text"):
        return
    content_md = getattr(response, "report_markdown", None) or response.text
    content_html = getattr(response, "report_html", None) or f"<pre>{response.text}</pre>"
    if job.delivery_channel == Channel.SLACK and slack_channel is not None:
        slack_channel.send_message(job.delivery_destination, content_md)
    elif job.delivery_channel == Channel.EMAIL and email_channel is not None:
        email_channel.send_email(
            to_addr=job.delivery_destination,
            subject=job.description,
            html_body=content_html,
        )
