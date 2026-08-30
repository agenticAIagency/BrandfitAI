from celery import Celery
from celery.schedules import crontab

from brandfit_core.config import get_settings

settings = get_settings()
celery_app = Celery("brandfit_core", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    beat_schedule={
        "weekly-persona-refresh": {
            "task": "brandfit.weekly_persona_refresh",
            "schedule": crontab(
                minute=0,
                hour=2,
                day_of_week=settings.persona_refresh_day,
            ),
        }
    },
)
celery_app.autodiscover_tasks(["brandfit_core.workers"])
