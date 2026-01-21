"""
Server Optimization Configuration for Quant Brain
Optimized for: 8 vCPU, 32GB RAM, 20+ Workers

This configuration maximizes server utilization while maintaining stability.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

# =============================================================================
# SERVER SPECIFICATIONS
# =============================================================================

VCPU_COUNT = int(os.getenv("CPU_COUNT", 8))
TOTAL_RAM_GB = int(os.getenv("TOTAL_RAM_GB", 32))
MIN_WORKERS = int(os.getenv("MIN_WORKERS", 20))


@dataclass
class MemoryConfig:
    """Memory allocation configuration"""
    max_usage_gb: float = 28.0  # Leave 4GB for system
    per_worker_mb: int = 512
    data_cache_gb: float = 12.0
    model_cache_gb: float = 6.0
    worker_memory_gb: float = 10.0  # 500MB × 20 workers


@dataclass
class WorkerConfig:
    """Worker pool configuration"""
    # Async workers for I/O operations (data fetching, API calls)
    async_workers: int = 32

    # Thread pool for concurrent I/O tasks
    thread_workers: int = 24

    # Process pool for CPU-intensive tasks (limited by vCPU)
    process_workers: int = 8

    # Minimum guaranteed workers
    min_workers: int = 20

    # Maximum workers
    max_workers: int = 32


@dataclass
class BatchConfig:
    """Batch processing configuration"""
    size: int = 50_000  # Rows per batch
    concurrent_batches: int = 20


@dataclass
class DatabaseConfig:
    """Database connection pool configuration"""
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800


@dataclass
class GeneticAlgorithmConfig:
    """Genetic Algorithm optimization settings"""
    population_size: int = 100  # Reduced for faster iterations
    parallel_evaluations: int = 20
    generations: int = 10
    elite_count: int = 5
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8


@dataclass
class BacktestConfig:
    """Backtest engine configuration"""
    workers: int = 20
    batch_size: int = 50_000
    memory_limit_gb: float = 28.0
    timeout_seconds: int = 300
    max_concurrent: int = 20


@dataclass
class CeleryConfig:
    """Celery task queue configuration"""
    worker_concurrency: int = 20
    worker_prefetch_multiplier: int = 2
    worker_max_memory_per_child: int = 512_000  # 512MB in KB
    worker_max_tasks_per_child: int = 100
    task_time_limit: int = 600  # 10 minutes
    task_soft_time_limit: int = 540  # 9 minutes


@dataclass
class ServerConfig:
    """Main server configuration"""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ga: GeneticAlgorithmConfig = field(default_factory=GeneticAlgorithmConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    celery: CeleryConfig = field(default_factory=CeleryConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'vcpu_count': VCPU_COUNT,
            'total_ram_gb': TOTAL_RAM_GB,
            'min_workers': MIN_WORKERS,
            'memory': {
                'max_usage_gb': self.memory.max_usage_gb,
                'per_worker_mb': self.memory.per_worker_mb,
                'data_cache_gb': self.memory.data_cache_gb,
                'model_cache_gb': self.memory.model_cache_gb,
            },
            'workers': {
                'async_workers': self.workers.async_workers,
                'thread_workers': self.workers.thread_workers,
                'process_workers': self.workers.process_workers,
                'min_workers': self.workers.min_workers,
                'max_workers': self.workers.max_workers,
            },
            'batch': {
                'size': self.batch.size,
                'concurrent_batches': self.batch.concurrent_batches,
            },
            'ga': {
                'population_size': self.ga.population_size,
                'parallel_evaluations': self.ga.parallel_evaluations,
                'generations': self.ga.generations,
            },
        }


# =============================================================================
# ENVIRONMENT-BASED CONFIGURATION
# =============================================================================

def get_config_from_env() -> ServerConfig:
    """Load configuration from environment variables"""
    config = ServerConfig()

    # Override from environment
    if os.getenv("MIN_WORKERS"):
        config.workers.min_workers = int(os.getenv("MIN_WORKERS"))
    if os.getenv("MAX_WORKERS"):
        config.workers.max_workers = int(os.getenv("MAX_WORKERS"))
    if os.getenv("THREAD_WORKERS"):
        config.workers.thread_workers = int(os.getenv("THREAD_WORKERS"))
    if os.getenv("PROCESS_WORKERS"):
        config.workers.process_workers = int(os.getenv("PROCESS_WORKERS"))
    if os.getenv("MAX_MEMORY_GB"):
        config.memory.max_usage_gb = float(os.getenv("MAX_MEMORY_GB"))
    if os.getenv("GA_POPULATION_SIZE"):
        config.ga.population_size = int(os.getenv("GA_POPULATION_SIZE"))
    if os.getenv("GA_GENERATIONS"):
        config.ga.generations = int(os.getenv("GA_GENERATIONS"))
    if os.getenv("GA_PARALLEL_EVALUATIONS"):
        config.ga.parallel_evaluations = int(os.getenv("GA_PARALLEL_EVALUATIONS"))
    if os.getenv("CELERY_CONCURRENCY"):
        config.celery.worker_concurrency = int(os.getenv("CELERY_CONCURRENCY"))

    return config


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

# Default configuration optimized for 8 vCPU, 32GB RAM
SERVER_CONFIG = get_config_from_env()


# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================

PERFORMANCE_TARGETS = {
    'concurrent_backtests': 20,
    'api_throughput_rps': 500,
    'ga_parallel_evaluations': 20,
    'memory_utilization_pct': 85,
    'cpu_utilization_pct': 90,
}


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def get_worker_count() -> int:
    """Get optimal worker count based on server specs"""
    return max(MIN_WORKERS, SERVER_CONFIG.workers.min_workers)


def get_thread_pool_size() -> int:
    """Get thread pool size for I/O operations"""
    return SERVER_CONFIG.workers.thread_workers


def get_process_pool_size() -> int:
    """Get process pool size for CPU operations"""
    return min(VCPU_COUNT, SERVER_CONFIG.workers.process_workers)


def get_memory_limit_bytes() -> int:
    """Get memory limit in bytes"""
    return int(SERVER_CONFIG.memory.max_usage_gb * 1024 * 1024 * 1024)


def get_ga_config() -> GeneticAlgorithmConfig:
    """Get GA configuration"""
    return SERVER_CONFIG.ga


def get_celery_config() -> CeleryConfig:
    """Get Celery configuration"""
    return SERVER_CONFIG.celery
