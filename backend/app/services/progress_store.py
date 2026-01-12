"""
Redis-based progress and result storage for Celery jobs.

Provides helpers to:
- Store/retrieve job progress
- Store/retrieve job results
- Publish progress updates via Redis pub/sub
- Check cancel flags
"""

import os
import json
from typing import Optional, Dict, Any
import redis
from dotenv import load_dotenv

load_dotenv()

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Redis client - lazy initialization
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Get Redis client with lazy initialization"""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# Key prefixes
PROGRESS_KEY_PREFIX = "job:progress:"
RESULT_KEY_PREFIX = "job:result:"
CANCEL_KEY_PREFIX = "job:cancel:"
PROGRESS_CHANNEL_PREFIX = "job:progress:channel:"

# Default TTL for keys (24 hours)
DEFAULT_TTL = 86400


def set_progress(job_id: str, progress_data: Dict[str, Any], ttl: int = DEFAULT_TTL) -> None:
    """
    Store job progress in Redis.

    Args:
        job_id: The job ID
        progress_data: Dict containing progress info (progress, total, status, error)
        ttl: Time-to-live in seconds
    """
    r = get_redis()
    key = f"{PROGRESS_KEY_PREFIX}{job_id}"
    r.setex(key, ttl, json.dumps(progress_data))


def get_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job progress from Redis.

    Args:
        job_id: The job ID

    Returns:
        Progress data dict or None if not found
    """
    r = get_redis()
    key = f"{PROGRESS_KEY_PREFIX}{job_id}"
    data = r.get(key)
    if data:
        return json.loads(data)
    return None


def set_result(job_id: str, result_data: Dict[str, Any], ttl: int = DEFAULT_TTL) -> None:
    """
    Store final job result in Redis.

    Args:
        job_id: The job ID
        result_data: The final result data
        ttl: Time-to-live in seconds
    """
    r = get_redis()
    key = f"{RESULT_KEY_PREFIX}{job_id}"
    r.setex(key, ttl, json.dumps(result_data))


def get_result(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job result from Redis.

    Args:
        job_id: The job ID

    Returns:
        Result data dict or None if not found
    """
    r = get_redis()
    key = f"{RESULT_KEY_PREFIX}{job_id}"
    data = r.get(key)
    if data:
        return json.loads(data)
    return None


def publish_progress(job_id: str, progress_data: Dict[str, Any]) -> None:
    """
    Publish progress update to Redis pub/sub channel.
    Also updates the progress key.

    Args:
        job_id: The job ID
        progress_data: Progress data to publish
    """
    r = get_redis()
    channel = f"{PROGRESS_CHANNEL_PREFIX}{job_id}"

    # Store progress
    set_progress(job_id, progress_data)

    # Publish to channel for SSE subscribers
    r.publish(channel, json.dumps(progress_data))


def set_cancel_flag(job_id: str, ttl: int = 3600) -> None:
    """
    Set cancel flag for a job.

    Args:
        job_id: The job ID
        ttl: Time-to-live in seconds
    """
    r = get_redis()
    key = f"{CANCEL_KEY_PREFIX}{job_id}"
    r.setex(key, ttl, "1")


def check_cancel_flag(job_id: str) -> bool:
    """
    Check if a job has been cancelled.

    Args:
        job_id: The job ID

    Returns:
        True if job is cancelled, False otherwise
    """
    r = get_redis()
    key = f"{CANCEL_KEY_PREFIX}{job_id}"
    return r.exists(key) > 0


def clear_cancel_flag(job_id: str) -> None:
    """
    Clear cancel flag for a job.

    Args:
        job_id: The job ID
    """
    r = get_redis()
    key = f"{CANCEL_KEY_PREFIX}{job_id}"
    r.delete(key)


def subscribe_progress(job_id: str):
    """
    Subscribe to progress updates for a job.
    Returns a pubsub object that can be iterated.

    Args:
        job_id: The job ID

    Returns:
        Redis pubsub subscription
    """
    r = get_redis()
    pubsub = r.pubsub()
    channel = f"{PROGRESS_CHANNEL_PREFIX}{job_id}"
    pubsub.subscribe(channel)
    return pubsub


def cleanup_job(job_id: str) -> None:
    """
    Clean up all Redis keys for a job.

    Args:
        job_id: The job ID
    """
    r = get_redis()
    r.delete(
        f"{PROGRESS_KEY_PREFIX}{job_id}",
        f"{RESULT_KEY_PREFIX}{job_id}",
        f"{CANCEL_KEY_PREFIX}{job_id}"
    )
