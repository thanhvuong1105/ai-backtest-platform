"""
Celery Application Configuration

Configures Celery with Redis as broker and result backend.
Optimized for: 8 vCPU, 32GB RAM, 20+ Workers
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SERVER SPECIFICATIONS
# =============================================================================
VCPU_COUNT = int(os.getenv("CPU_COUNT", 8))
TOTAL_RAM_GB = int(os.getenv("TOTAL_RAM_GB", 32))
MIN_WORKERS = int(os.getenv("MIN_WORKERS", 20))

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

# =============================================================================
# OPTIMIZED CELERY CONFIGURATION
# For 8 vCPU, 32GB RAM, minimum 20 workers
# =============================================================================
celery.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing
    task_default_queue="default",

    # ==========================================================================
    # WORKER SETTINGS - OPTIMIZED FOR 20+ WORKERS
    # ==========================================================================
    # Prefetch 2 tasks per worker for better throughput
    worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH", 2)),

    # Concurrency: 20 workers minimum for I/O-bound tasks
    # Workers are NOT limited by CPU when tasks are I/O-bound
    worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", MIN_WORKERS)),

    # Pool type: prefork for CPU-bound, gevent/eventlet for I/O-bound
    worker_pool=os.getenv("CELERY_POOL", "prefork"),

    # ==========================================================================
    # MEMORY MANAGEMENT
    # ==========================================================================
    # Max memory per worker: 512MB (allows 20 workers in 10GB)
    worker_max_memory_per_child=int(os.getenv("WORKER_MAX_MEMORY_KB", 512_000)),

    # Restart worker after 100 tasks to prevent memory leaks
    worker_max_tasks_per_child=int(os.getenv("WORKER_MAX_TASKS", 100)),

    # ==========================================================================
    # TASK TIME LIMITS
    # ==========================================================================
    task_soft_time_limit=int(os.getenv("JOB_TIMEOUT_SEC", 1800)),  # 30 min soft limit
    task_time_limit=int(os.getenv("JOB_TIMEOUT_SEC", 1800)) + 60,  # 31 min hard limit

    # ==========================================================================
    # RESULT SETTINGS
    # ==========================================================================
    result_expires=86400,  # Results expire after 24 hours
    result_extended=True,  # Store task metadata

    # ==========================================================================
    # BROKER SETTINGS FOR REDIS
    # ==========================================================================
    broker_connection_retry_on_startup=True,
    broker_pool_limit=int(os.getenv("BROKER_POOL_LIMIT", 20)),

    # ==========================================================================
    # TASK ACKNOWLEDGMENT
    # ==========================================================================
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,  # Requeue task if worker dies

    # ==========================================================================
    # PERFORMANCE OPTIMIZATIONS
    # ==========================================================================
    # Disable rate limiting for max throughput
    worker_disable_rate_limits=True,

    # Send task-sent event for monitoring
    task_send_sent_event=True,

    # Optimize for throughput
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
)

# =============================================================================
# TASK ROUTES - OPTIMIZED QUEUES
# =============================================================================
celery.conf.task_routes = {
    # Backtest tasks - high concurrency for I/O-bound operations
    "app.services.tasks.run_backtest_task": {
        "queue": "backtest",
    },
    # Optimization tasks - moderate concurrency
    "app.services.tasks.optimize_task": {
        "queue": "optimization",
    },
    # AI Agent tasks - high concurrency
    "app.services.tasks.ai_agent_task": {
        "queue": "ai_agent",
    },
    # Quant Brain tasks - CPU-intensive, limited by vCPU
    "app.services.tasks.quant_brain_task": {
        "queue": "quant_brain",
    },
    # Chart data tasks - I/O-bound
    "app.services.tasks.chart_data_task": {
        "queue": "default",
    },
}

# =============================================================================
# QUEUE CONCURRENCY SETTINGS
# =============================================================================
# Use different concurrency per queue based on task type
QUEUE_CONCURRENCY = {
    "default": MIN_WORKERS,
    "backtest": MIN_WORKERS,
    "optimization": VCPU_COUNT,  # CPU-bound
    "ai_agent": MIN_WORKERS,
    "quant_brain": MIN_WORKERS,
}


if __name__ == "__main__":
    celery.start()
