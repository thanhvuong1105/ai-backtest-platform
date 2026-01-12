"""
Celery Application Configuration

Configures Celery with Redis as broker and result backend.
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery app
celery = Celery(
    "ai_backtest_platform",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.services.tasks"]
)

# Celery configuration
celery.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing
    task_default_queue="default",

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", 2)),

    # Task time limits
    task_soft_time_limit=int(os.getenv("JOB_TIMEOUT_SEC", 1800)),  # 30 min soft limit
    task_time_limit=int(os.getenv("JOB_TIMEOUT_SEC", 1800)) + 60,  # 31 min hard limit

    # Result settings
    result_expires=86400,  # Results expire after 24 hours

    # Broker settings for Redis
    broker_connection_retry_on_startup=True,

    # Task acknowledgment
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,  # Requeue task if worker dies
)

# Optional: Configure task routes
celery.conf.task_routes = {
    "app.services.tasks.run_backtest_task": {"queue": "default"},
    "app.services.tasks.optimize_task": {"queue": "default"},
    "app.services.tasks.ai_agent_task": {"queue": "default"},
    "app.services.tasks.chart_data_task": {"queue": "default"},
}


if __name__ == "__main__":
    celery.start()
