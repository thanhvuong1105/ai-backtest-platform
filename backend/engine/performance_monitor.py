# engine/performance_monitor.py
"""
Performance Monitor - System resource detection and real-time monitoring

Part of Quant Brain Auto-Tuning System:
1. Startup Detection: CPU, RAM, Docker cgroup limits
2. Runtime Monitoring: Resource usage every 1-2 seconds
3. Metrics Collection: throughput, latency, queue depth
"""

import os
import time
import platform
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════

@dataclass
class SystemResources:
    """Detected system resources at startup."""
    # CPU
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_frequency_mhz: float = 0.0
    cpu_architecture: str = ""

    # Memory
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # Docker/cgroup limits (if applicable)
    is_docker: bool = False
    docker_cpu_quota: float = 0.0  # Number of CPUs allocated
    docker_memory_limit_gb: float = 0.0

    # Effective resources (considering Docker limits)
    effective_cpus: float = 0.0
    effective_memory_gb: float = 0.0


@dataclass
class RuntimeMetrics:
    """Real-time performance metrics."""
    timestamp: float = 0.0

    # CPU
    cpu_percent: float = 0.0
    cpu_load_1m: float = 0.0
    cpu_load_5m: float = 0.0
    cpu_load_15m: float = 0.0

    # Memory
    ram_used_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_percent: float = 0.0
    swap_used_gb: float = 0.0
    swap_percent: float = 0.0

    # Quant Brain specific
    generations_completed: int = 0
    throughput_gen_per_sec: float = 0.0
    avg_time_per_gen_sec: float = 0.0
    queue_depth: int = 0

    # Derived warnings
    memory_pressure: bool = False  # RAM > 85%
    cpu_saturation: bool = False   # CPU > 95% but throughput flat
    swap_active: bool = False      # Swap > 0


@dataclass
class TuningState:
    """Current tuning state and parameters."""
    # Tunable parameters (runtime-adjustable)
    batch_size: int = 10
    chunk_size: int = 100_000
    queue_depth_limit: int = 100
    max_workers: int = 6

    # Cooldown tracking
    last_adjustment_time: float = 0.0
    adjustment_count: int = 0

    # History for analysis
    adjustment_history: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# SYSTEM DETECTION
# ═══════════════════════════════════════════════════════

def detect_system_resources() -> SystemResources:
    """
    Detect system resources at startup.

    Detects:
    - CPU cores/threads, architecture, frequency
    - RAM total/available
    - Docker cgroup limits (if running in container)

    Returns:
        SystemResources object with detected values
    """
    resources = SystemResources()

    # ─────────────────────────────────────────────────────
    # CPU Detection
    # ─────────────────────────────────────────────────────
    resources.cpu_cores = os.cpu_count() or 1
    resources.cpu_threads = resources.cpu_cores  # Same on most systems
    resources.cpu_architecture = platform.machine()

    # Try to get frequency
    try:
        import psutil
        freq_info = psutil.cpu_freq()
        if freq_info:
            resources.cpu_frequency_mhz = freq_info.current or 0
    except (ImportError, Exception):
        pass

    # ─────────────────────────────────────────────────────
    # Memory Detection
    # ─────────────────────────────────────────────────────
    try:
        import psutil
        mem = psutil.virtual_memory()
        resources.ram_total_gb = mem.total / (1024 ** 3)
        resources.ram_available_gb = mem.available / (1024 ** 3)
    except ImportError:
        # Fallback: read from /proc/meminfo (Linux)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = dict(
                    (i.split()[0].rstrip(':'), int(i.split()[1]))
                    for i in f.readlines() if len(i.split()) >= 2
                )
            resources.ram_total_gb = meminfo.get('MemTotal', 0) / (1024 ** 2)
            resources.ram_available_gb = meminfo.get('MemAvailable', 0) / (1024 ** 2)
        except Exception:
            resources.ram_total_gb = 8.0  # Default assumption
            resources.ram_available_gb = 4.0

    # ─────────────────────────────────────────────────────
    # Docker/cgroup Detection
    # ─────────────────────────────────────────────────────
    resources.is_docker = _detect_docker_environment()

    if resources.is_docker:
        # Read cgroup CPU quota
        cpu_quota = _read_cgroup_cpu_quota()
        if cpu_quota > 0:
            resources.docker_cpu_quota = cpu_quota

        # Read cgroup memory limit
        mem_limit = _read_cgroup_memory_limit()
        if mem_limit > 0:
            resources.docker_memory_limit_gb = mem_limit / (1024 ** 3)

    # ─────────────────────────────────────────────────────
    # Effective Resources (considering Docker limits)
    # ─────────────────────────────────────────────────────
    if resources.is_docker and resources.docker_cpu_quota > 0:
        resources.effective_cpus = resources.docker_cpu_quota
    else:
        resources.effective_cpus = resources.cpu_cores

    if resources.is_docker and resources.docker_memory_limit_gb > 0:
        resources.effective_memory_gb = resources.docker_memory_limit_gb
    else:
        resources.effective_memory_gb = resources.ram_total_gb

    # Log detection results
    _log_system_detection(resources)

    return resources


def _detect_docker_environment() -> bool:
    """Check if running inside Docker container."""
    # Method 1: Check /.dockerenv
    if os.path.exists('/.dockerenv'):
        return True

    # Method 2: Check cgroup
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'kubepods' in content:
                return True
    except Exception:
        pass

    # Method 3: Check environment variable
    if os.getenv('DOCKER_CONTAINER', '').lower() == 'true':
        return True

    return False


def _read_cgroup_cpu_quota() -> float:
    """Read CPU quota from cgroup (v1 or v2)."""
    # cgroup v2
    try:
        with open('/sys/fs/cgroup/cpu.max', 'r') as f:
            content = f.read().strip()
            if content != 'max':
                parts = content.split()
                if len(parts) >= 2:
                    quota = int(parts[0])
                    period = int(parts[1])
                    return quota / period
    except Exception:
        pass

    # cgroup v1
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
            period = int(f.read().strip())

        if quota > 0 and period > 0:
            return quota / period
    except Exception:
        pass

    return 0.0


def _read_cgroup_memory_limit() -> int:
    """Read memory limit from cgroup (v1 or v2)."""
    # cgroup v2
    try:
        with open('/sys/fs/cgroup/memory.max', 'r') as f:
            content = f.read().strip()
            if content != 'max':
                return int(content)
    except Exception:
        pass

    # cgroup v1
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit = int(f.read().strip())
            # Check if it's not the max value (unlimited)
            if limit < 9223372036854771712:
                return limit
    except Exception:
        pass

    return 0


def _log_system_detection(resources: SystemResources):
    """Log detected system resources."""
    logger.info("=" * 60)
    logger.info("QUANT BRAIN - Performance Monitor")
    logger.info("=" * 60)
    logger.info(f"CPU: {resources.cpu_cores} cores, {resources.cpu_architecture}")
    if resources.cpu_frequency_mhz > 0:
        logger.info(f"CPU Frequency: {resources.cpu_frequency_mhz:.0f} MHz")
    logger.info(f"RAM: {resources.ram_total_gb:.1f} GB total, {resources.ram_available_gb:.1f} GB available")

    if resources.is_docker:
        logger.info("-" * 60)
        logger.info("Docker Container Detected:")
        if resources.docker_cpu_quota > 0:
            logger.info(f"  CPU Quota: {resources.docker_cpu_quota:.1f} CPUs")
        if resources.docker_memory_limit_gb > 0:
            logger.info(f"  Memory Limit: {resources.docker_memory_limit_gb:.1f} GB")

    logger.info("-" * 60)
    logger.info(f"Effective Resources:")
    logger.info(f"  CPUs: {resources.effective_cpus:.1f}")
    logger.info(f"  Memory: {resources.effective_memory_gb:.1f} GB")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════
# RUNTIME MONITORING
# ═══════════════════════════════════════════════════════

class PerformanceMonitor:
    """
    Real-time performance monitor for Quant Brain.

    Features:
    - Monitors CPU, RAM, swap every 1-2 seconds
    - Tracks throughput (generations/sec)
    - Calculates average time per generation
    - Detects memory pressure and CPU saturation
    """

    def __init__(
        self,
        resources: SystemResources = None,
        sample_interval: float = 1.5,
        history_size: int = 60
    ):
        """
        Initialize performance monitor.

        Args:
            resources: Detected system resources (or detect automatically)
            sample_interval: Seconds between samples (default: 1.5)
            history_size: Number of samples to keep in history
        """
        self.resources = resources or detect_system_resources()
        self.sample_interval = sample_interval
        self.history_size = history_size

        # Metrics history (circular buffer)
        self.metrics_history: deque = deque(maxlen=history_size)

        # Generation tracking
        self._gen_start_time: float = 0.0
        self._gen_times: deque = deque(maxlen=100)  # Last 100 gen times
        self._total_generations: int = 0
        self._session_start_time: float = time.time()

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Callbacks
        self._on_metrics_callback: Optional[Callable[[RuntimeMetrics], None]] = None
        self._on_warning_callback: Optional[Callable[[str, RuntimeMetrics], None]] = None

        # Current metrics
        self.current_metrics: RuntimeMetrics = RuntimeMetrics()

        # Tuning state (shared with auto-tuner)
        self.tuning_state: TuningState = TuningState()

    def start(self):
        """Start background monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
        logger.info("Performance Monitor started")

    def stop(self):
        """Stop monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=3.0)
        logger.info("Performance Monitor stopped")

    def _monitor_loop(self):
        """Background loop to collect metrics."""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)

                # Check for warnings
                self._check_warnings(metrics)

                # Callback
                if self._on_metrics_callback:
                    self._on_metrics_callback(metrics)

            except Exception as e:
                logger.warning(f"Metrics collection failed: {e}")

            self._stop_event.wait(self.sample_interval)

    def _collect_metrics(self) -> RuntimeMetrics:
        """Collect current runtime metrics."""
        metrics = RuntimeMetrics(timestamp=time.time())

        try:
            import psutil

            # CPU
            metrics.cpu_percent = psutil.cpu_percent(interval=None)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            metrics.cpu_load_1m = load_avg[0]
            metrics.cpu_load_5m = load_avg[1]
            metrics.cpu_load_15m = load_avg[2]

            # Memory
            mem = psutil.virtual_memory()
            metrics.ram_used_gb = (mem.total - mem.available) / (1024 ** 3)
            metrics.ram_available_gb = mem.available / (1024 ** 3)
            metrics.ram_percent = mem.percent

            # Swap
            swap = psutil.swap_memory()
            metrics.swap_used_gb = swap.used / (1024 ** 3)
            metrics.swap_percent = swap.percent

        except ImportError:
            # Fallback without psutil
            metrics.cpu_percent = self._get_cpu_percent_fallback()
            metrics.ram_percent = self._get_ram_percent_fallback()

        # Quant Brain specific metrics
        metrics.generations_completed = self._total_generations
        metrics.throughput_gen_per_sec = self._calculate_throughput()
        metrics.avg_time_per_gen_sec = self._calculate_avg_gen_time()
        metrics.queue_depth = self.tuning_state.queue_depth_limit

        # Derived warnings
        metrics.memory_pressure = metrics.ram_percent > 85
        metrics.swap_active = metrics.swap_used_gb > 0.1
        metrics.cpu_saturation = (
            metrics.cpu_percent > 95 and
            len(self.metrics_history) > 5 and
            self._is_throughput_flat()
        )

        return metrics

    def _get_cpu_percent_fallback(self) -> float:
        """Fallback CPU usage from /proc/stat."""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                parts = line.split()
                if parts[0] == 'cpu':
                    idle = float(parts[4])
                    total = sum(float(p) for p in parts[1:])
                    return 100 * (1 - idle / total)
        except Exception:
            pass
        return 0.0

    def _get_ram_percent_fallback(self) -> float:
        """Fallback RAM usage from /proc/meminfo."""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            meminfo = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1])

            total = meminfo.get('MemTotal', 1)
            available = meminfo.get('MemAvailable', total)
            return 100 * (1 - available / total)
        except Exception:
            pass
        return 0.0

    def _calculate_throughput(self) -> float:
        """Calculate generations per second."""
        elapsed = time.time() - self._session_start_time
        if elapsed > 0:
            return self._total_generations / elapsed
        return 0.0

    def _calculate_avg_gen_time(self) -> float:
        """Calculate average time per generation."""
        if self._gen_times:
            return sum(self._gen_times) / len(self._gen_times)
        return 0.0

    def _is_throughput_flat(self) -> bool:
        """Check if throughput is flat despite high CPU."""
        if len(self.metrics_history) < 10:
            return False

        recent = list(self.metrics_history)[-10:]
        throughputs = [m.throughput_gen_per_sec for m in recent]

        if not throughputs or throughputs[0] == 0:
            return False

        # Check if throughput hasn't improved much
        avg_early = sum(throughputs[:5]) / 5
        avg_late = sum(throughputs[5:]) / 5

        # Flat if less than 5% improvement
        return avg_late <= avg_early * 1.05

    def _check_warnings(self, metrics: RuntimeMetrics):
        """Check for warning conditions and log/callback."""
        warnings = []

        if metrics.memory_pressure:
            warnings.append(
                f"MEMORY PRESSURE: RAM {metrics.ram_percent:.1f}% "
                f"(available: {metrics.ram_available_gb:.1f} GB)"
            )

        if metrics.swap_active:
            warnings.append(
                f"SWAP ACTIVE: {metrics.swap_used_gb:.2f} GB used "
                f"({metrics.swap_percent:.1f}%)"
            )

        if metrics.cpu_saturation:
            warnings.append(
                f"CPU SATURATION: {metrics.cpu_percent:.1f}% but throughput flat "
                f"({metrics.throughput_gen_per_sec:.2f} gen/s)"
            )

        for warning in warnings:
            logger.warning(f"[PERF] {warning}")
            if self._on_warning_callback:
                self._on_warning_callback(warning, metrics)

    # ─────────────────────────────────────────────────────
    # Generation Tracking
    # ─────────────────────────────────────────────────────

    def generation_started(self):
        """Call when a generation starts."""
        self._gen_start_time = time.time()

    def generation_completed(self):
        """Call when a generation completes."""
        if self._gen_start_time > 0:
            gen_time = time.time() - self._gen_start_time
            self._gen_times.append(gen_time)
            self._gen_start_time = 0
        self._total_generations += 1

    def reset_session(self):
        """Reset session counters."""
        self._total_generations = 0
        self._gen_times.clear()
        self._session_start_time = time.time()

    # ─────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────

    def on_metrics(self, callback: Callable[[RuntimeMetrics], None]):
        """Register callback for metrics updates."""
        self._on_metrics_callback = callback

    def on_warning(self, callback: Callable[[str, RuntimeMetrics], None]):
        """Register callback for warning conditions."""
        self._on_warning_callback = callback

    # ─────────────────────────────────────────────────────
    # Getters
    # ─────────────────────────────────────────────────────

    def get_current_metrics(self) -> RuntimeMetrics:
        """Get most recent metrics."""
        return self.current_metrics

    def get_metrics_history(self, n: int = None) -> List[RuntimeMetrics]:
        """Get metrics history (last n samples or all)."""
        if n is None:
            return list(self.metrics_history)
        return list(self.metrics_history)[-n:]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary for logging/reporting."""
        m = self.current_metrics
        return {
            "timestamp": datetime.fromtimestamp(m.timestamp).isoformat() if m.timestamp else None,
            "cpu": {
                "percent": round(m.cpu_percent, 1),
                "load_1m": round(m.cpu_load_1m, 2),
            },
            "memory": {
                "used_gb": round(m.ram_used_gb, 2),
                "available_gb": round(m.ram_available_gb, 2),
                "percent": round(m.ram_percent, 1),
                "swap_gb": round(m.swap_used_gb, 2),
            },
            "quant_brain": {
                "generations": m.generations_completed,
                "throughput_gen_sec": round(m.throughput_gen_per_sec, 3),
                "avg_time_per_gen_sec": round(m.avg_time_per_gen_sec, 3),
            },
            "warnings": {
                "memory_pressure": m.memory_pressure,
                "cpu_saturation": m.cpu_saturation,
                "swap_active": m.swap_active,
            },
            "tuning": {
                "batch_size": self.tuning_state.batch_size,
                "max_workers": self.tuning_state.max_workers,
                "adjustments": self.tuning_state.adjustment_count,
            }
        }


# ═══════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════

_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create singleton PerformanceMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance


def start_monitoring():
    """Start the performance monitor."""
    monitor = get_performance_monitor()
    monitor.start()


def stop_monitoring():
    """Stop the performance monitor."""
    global _monitor_instance
    if _monitor_instance is not None:
        _monitor_instance.stop()


# ═══════════════════════════════════════════════════════
# CLI FOR TESTING
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("\n=== Performance Monitor Test ===\n")

    # Detect resources
    resources = detect_system_resources()

    # Start monitor
    monitor = PerformanceMonitor(resources)

    def on_metrics(metrics: RuntimeMetrics):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"CPU: {metrics.cpu_percent:.1f}% | "
              f"RAM: {metrics.ram_percent:.1f}% ({metrics.ram_available_gb:.1f}GB free) | "
              f"Gen/s: {metrics.throughput_gen_per_sec:.2f}")

    def on_warning(warning: str, metrics: RuntimeMetrics):
        print(f"[WARNING] {warning}")

    monitor.on_metrics(on_metrics)
    monitor.on_warning(on_warning)

    monitor.start()

    # Simulate some work
    print("\nMonitoring for 10 seconds...\n")
    try:
        for i in range(10):
            time.sleep(1)
            # Simulate generation
            monitor.generation_started()
            time.sleep(0.1)  # Simulated work
            monitor.generation_completed()
    except KeyboardInterrupt:
        pass

    monitor.stop()

    print("\n=== Summary ===")
    import json
    print(json.dumps(monitor.get_summary(), indent=2))
