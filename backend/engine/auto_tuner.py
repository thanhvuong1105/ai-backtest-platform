# engine/auto_tuner.py
"""
Auto-Tuner - Closed-loop auto-tuning engine for Quant Brain

Implements the tuning cycle:
    Detect → Monitor → Decide → Apply → Verify

Key Features:
1. Safety-first: Prioritize preventing OOM over performance
2. Cooldown periods: 10-20 seconds between adjustments
3. Constraint-aware: Never changes population/generations
4. Detailed logging: Before/after throughput, reasons

Tuning Rules:
- RAM > 85% or swap increase → reduce batch/chunk/queue 20-40%
- CPU < 85% and RAM > 25% free → increase concurrency/batch 10-20%
- CPU = 100% but throughput flat → reduce workers 10-20%
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .performance_monitor import (
    PerformanceMonitor,
    RuntimeMetrics,
    TuningState,
    SystemResources,
    detect_system_resources,
    get_performance_monitor
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

# Thresholds
MEMORY_HIGH_THRESHOLD = float(os.getenv("TUNING_MEMORY_HIGH_PCT", "85"))
MEMORY_SAFE_THRESHOLD = float(os.getenv("TUNING_MEMORY_SAFE_PCT", "75"))
MEMORY_FREE_THRESHOLD = float(os.getenv("TUNING_MEMORY_FREE_PCT", "25"))  # % free to consider scaling up
CPU_HIGH_THRESHOLD = float(os.getenv("TUNING_CPU_HIGH_PCT", "95"))
CPU_NORMAL_THRESHOLD = float(os.getenv("TUNING_CPU_NORMAL_PCT", "85"))

# Cooldown
COOLDOWN_SECONDS = float(os.getenv("TUNING_COOLDOWN_SEC", "15"))
MIN_SAMPLES_FOR_DECISION = int(os.getenv("TUNING_MIN_SAMPLES", "5"))

# Adjustment factors
REDUCE_FACTOR_SMALL = 0.8   # Reduce by 20%
REDUCE_FACTOR_LARGE = 0.6   # Reduce by 40%
INCREASE_FACTOR = 1.15      # Increase by 15%

# Parameter limits (env vars can override)
MIN_BATCH_SIZE = int(os.getenv("TUNING_MIN_BATCH", "5"))
MAX_BATCH_SIZE = int(os.getenv("TUNING_MAX_BATCH", "50"))
MIN_WORKERS = int(os.getenv("TUNING_MIN_WORKERS", "2"))
MAX_WORKERS = int(os.getenv("TUNING_MAX_WORKERS", str(os.cpu_count() or 8)))
MIN_CHUNK_SIZE = int(os.getenv("TUNING_MIN_CHUNK", "50000"))
MAX_CHUNK_SIZE = int(os.getenv("TUNING_MAX_CHUNK", "500000"))
MIN_QUEUE_DEPTH = int(os.getenv("TUNING_MIN_QUEUE", "25"))
MAX_QUEUE_DEPTH = int(os.getenv("TUNING_MAX_QUEUE", "200"))


# ═══════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════

class TuningAction(Enum):
    """Possible tuning actions."""
    NONE = "none"
    REDUCE_BATCH = "reduce_batch"
    REDUCE_WORKERS = "reduce_workers"
    REDUCE_CHUNK = "reduce_chunk"
    REDUCE_QUEUE = "reduce_queue"
    INCREASE_BATCH = "increase_batch"
    INCREASE_WORKERS = "increase_workers"
    INCREASE_CHUNK = "increase_chunk"
    EMERGENCY_REDUCE = "emergency_reduce"


class TuningReason(Enum):
    """Reasons for tuning decisions."""
    MEMORY_PRESSURE = "memory_pressure"
    SWAP_ACTIVE = "swap_active"
    CPU_SATURATION = "cpu_saturation"
    RESOURCES_AVAILABLE = "resources_available"
    THROUGHPUT_DEGRADED = "throughput_degraded"
    EMERGENCY_OOM_RISK = "emergency_oom_risk"


# ═══════════════════════════════════════════════════════
# TUNING DECISION RECORD
# ═══════════════════════════════════════════════════════

@dataclass
class TuningDecision:
    """Record of a tuning decision."""
    timestamp: float
    action: TuningAction
    reason: TuningReason
    parameter: str
    old_value: Any
    new_value: Any
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]] = None
    success: Optional[bool] = None


# ═══════════════════════════════════════════════════════
# AUTO-TUNER
# ═══════════════════════════════════════════════════════

class AutoTuner:
    """
    Closed-loop auto-tuning engine for Quant Brain.

    Implements safety-first tuning with:
    - OOM prevention as highest priority
    - Cooldown periods between adjustments
    - Detailed logging for diagnostics
    """

    def __init__(
        self,
        monitor: PerformanceMonitor = None,
        enabled: bool = True,
        cooldown_seconds: float = COOLDOWN_SECONDS
    ):
        """
        Initialize auto-tuner.

        Args:
            monitor: PerformanceMonitor instance (or use global)
            enabled: Whether auto-tuning is enabled
            cooldown_seconds: Seconds between adjustments
        """
        self.monitor = monitor or get_performance_monitor()
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds

        # Decision history
        self.decisions: List[TuningDecision] = []

        # Baseline metrics (captured at start)
        self.baseline_throughput: float = 0.0
        self.baseline_metrics: Optional[RuntimeMetrics] = None

        # Last swap value (for detecting increase)
        self._last_swap_gb: float = 0.0

        # Log startup
        if enabled:
            logger.info("=" * 60)
            logger.info("AUTO-TUNER ENABLED")
            logger.info(f"  Cooldown: {cooldown_seconds}s")
            logger.info(f"  Memory high threshold: {MEMORY_HIGH_THRESHOLD}%")
            logger.info(f"  Memory free threshold: {MEMORY_FREE_THRESHOLD}%")
            logger.info(f"  CPU saturation threshold: {CPU_HIGH_THRESHOLD}%")
            logger.info("=" * 60)
        else:
            logger.info("Auto-tuner disabled")

    # ─────────────────────────────────────────────────────
    # MAIN TUNING CYCLE
    # ─────────────────────────────────────────────────────

    def check_and_tune(self) -> Optional[TuningDecision]:
        """
        Main tuning cycle: Detect → Monitor → Decide → Apply → Verify

        This should be called periodically (e.g., every few seconds or
        after each generation batch).

        Returns:
            TuningDecision if an adjustment was made, None otherwise
        """
        if not self.enabled:
            return None

        # Check cooldown
        state = self.monitor.tuning_state
        if time.time() - state.last_adjustment_time < self.cooldown_seconds:
            return None

        # Get current metrics
        metrics = self.monitor.get_current_metrics()
        history = self.monitor.get_metrics_history(MIN_SAMPLES_FOR_DECISION)

        if len(history) < MIN_SAMPLES_FOR_DECISION:
            return None

        # Store baseline on first check
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
            self.baseline_throughput = metrics.throughput_gen_per_sec

        # DETECT & DECIDE
        action, reason = self._decide_action(metrics, history)

        if action == TuningAction.NONE:
            return None

        # APPLY
        decision = self._apply_action(action, reason, metrics)

        # Record decision
        if decision:
            self.decisions.append(decision)
            state.last_adjustment_time = time.time()
            state.adjustment_count += 1
            state.adjustment_history.append({
                "timestamp": decision.timestamp,
                "action": decision.action.value,
                "reason": decision.reason.value,
                "parameter": decision.parameter,
                "old_value": decision.old_value,
                "new_value": decision.new_value,
            })

        return decision

    def _decide_action(
        self,
        metrics: RuntimeMetrics,
        history: List[RuntimeMetrics]
    ) -> Tuple[TuningAction, Optional[TuningReason]]:
        """
        Decide what tuning action to take based on current metrics.

        Priority order (safety first):
        1. Emergency: RAM > 90% → aggressive reduce
        2. Memory pressure: RAM > 85% → reduce batch/chunk
        3. Swap active → reduce queue
        4. CPU saturation + flat throughput → reduce workers
        5. Resources available → increase (cautiously)
        """
        state = self.monitor.tuning_state

        # ─────────────────────────────────────────────────
        # 1. EMERGENCY: RAM > 90% - Aggressive reduction
        # ─────────────────────────────────────────────────
        if metrics.ram_percent > 90:
            logger.warning(f"[TUNER] EMERGENCY: RAM at {metrics.ram_percent:.1f}%!")
            return TuningAction.EMERGENCY_REDUCE, TuningReason.EMERGENCY_OOM_RISK

        # ─────────────────────────────────────────────────
        # 2. MEMORY PRESSURE: RAM > 85%
        # ─────────────────────────────────────────────────
        if metrics.memory_pressure:
            # Check if we can still reduce
            if state.batch_size > MIN_BATCH_SIZE:
                return TuningAction.REDUCE_BATCH, TuningReason.MEMORY_PRESSURE
            elif state.chunk_size > MIN_CHUNK_SIZE:
                return TuningAction.REDUCE_CHUNK, TuningReason.MEMORY_PRESSURE
            elif state.max_workers > MIN_WORKERS:
                return TuningAction.REDUCE_WORKERS, TuningReason.MEMORY_PRESSURE

        # ─────────────────────────────────────────────────
        # 3. SWAP INCREASING
        # ─────────────────────────────────────────────────
        if metrics.swap_active:
            swap_increased = metrics.swap_used_gb > self._last_swap_gb + 0.05
            self._last_swap_gb = metrics.swap_used_gb

            if swap_increased and state.queue_depth_limit > MIN_QUEUE_DEPTH:
                return TuningAction.REDUCE_QUEUE, TuningReason.SWAP_ACTIVE

        # ─────────────────────────────────────────────────
        # 4. CPU SATURATION: CPU > 95% but throughput flat
        # ─────────────────────────────────────────────────
        if metrics.cpu_saturation:
            if state.max_workers > MIN_WORKERS:
                return TuningAction.REDUCE_WORKERS, TuningReason.CPU_SATURATION

        # ─────────────────────────────────────────────────
        # 5. RESOURCES AVAILABLE: CPU < 85% and RAM > 25% free
        # ─────────────────────────────────────────────────
        ram_free_pct = 100 - metrics.ram_percent
        if metrics.cpu_percent < CPU_NORMAL_THRESHOLD and ram_free_pct > MEMORY_FREE_THRESHOLD:
            # Only increase if throughput is not degrading
            if not self._is_throughput_degrading(history):
                if state.batch_size < MAX_BATCH_SIZE:
                    return TuningAction.INCREASE_BATCH, TuningReason.RESOURCES_AVAILABLE
                elif state.max_workers < MAX_WORKERS:
                    return TuningAction.INCREASE_WORKERS, TuningReason.RESOURCES_AVAILABLE

        return TuningAction.NONE, None

    def _apply_action(
        self,
        action: TuningAction,
        reason: TuningReason,
        metrics: RuntimeMetrics
    ) -> Optional[TuningDecision]:
        """
        Apply the decided tuning action.

        Returns:
            TuningDecision with details of the change
        """
        state = self.monitor.tuning_state
        decision = TuningDecision(
            timestamp=time.time(),
            action=action,
            reason=reason,
            parameter="",
            old_value=None,
            new_value=None,
            metrics_before={
                "cpu_percent": metrics.cpu_percent,
                "ram_percent": metrics.ram_percent,
                "throughput": metrics.throughput_gen_per_sec,
                "avg_gen_time": metrics.avg_time_per_gen_sec,
            }
        )

        if action == TuningAction.EMERGENCY_REDUCE:
            # Aggressive reduction of all parameters
            old_batch = state.batch_size
            old_workers = state.max_workers
            old_chunk = state.chunk_size

            state.batch_size = max(MIN_BATCH_SIZE, int(state.batch_size * REDUCE_FACTOR_LARGE))
            state.max_workers = max(MIN_WORKERS, int(state.max_workers * REDUCE_FACTOR_LARGE))
            state.chunk_size = max(MIN_CHUNK_SIZE, int(state.chunk_size * REDUCE_FACTOR_LARGE))

            decision.parameter = "batch+workers+chunk"
            decision.old_value = f"batch={old_batch}, workers={old_workers}, chunk={old_chunk}"
            decision.new_value = f"batch={state.batch_size}, workers={state.max_workers}, chunk={state.chunk_size}"

            self._log_adjustment(decision, "EMERGENCY")

        elif action == TuningAction.REDUCE_BATCH:
            old_value = state.batch_size
            state.batch_size = max(MIN_BATCH_SIZE, int(state.batch_size * REDUCE_FACTOR_SMALL))
            decision.parameter = "batch_size"
            decision.old_value = old_value
            decision.new_value = state.batch_size
            self._log_adjustment(decision)

        elif action == TuningAction.INCREASE_BATCH:
            old_value = state.batch_size
            state.batch_size = min(MAX_BATCH_SIZE, int(state.batch_size * INCREASE_FACTOR))
            decision.parameter = "batch_size"
            decision.old_value = old_value
            decision.new_value = state.batch_size
            self._log_adjustment(decision)

        elif action == TuningAction.REDUCE_WORKERS:
            old_value = state.max_workers
            state.max_workers = max(MIN_WORKERS, int(state.max_workers * REDUCE_FACTOR_SMALL))
            decision.parameter = "max_workers"
            decision.old_value = old_value
            decision.new_value = state.max_workers
            self._log_adjustment(decision)

        elif action == TuningAction.INCREASE_WORKERS:
            old_value = state.max_workers
            state.max_workers = min(MAX_WORKERS, int(state.max_workers * INCREASE_FACTOR))
            decision.parameter = "max_workers"
            decision.old_value = old_value
            decision.new_value = state.max_workers
            self._log_adjustment(decision)

        elif action == TuningAction.REDUCE_CHUNK:
            old_value = state.chunk_size
            state.chunk_size = max(MIN_CHUNK_SIZE, int(state.chunk_size * REDUCE_FACTOR_SMALL))
            decision.parameter = "chunk_size"
            decision.old_value = old_value
            decision.new_value = state.chunk_size
            self._log_adjustment(decision)

        elif action == TuningAction.REDUCE_QUEUE:
            old_value = state.queue_depth_limit
            state.queue_depth_limit = max(MIN_QUEUE_DEPTH, int(state.queue_depth_limit * REDUCE_FACTOR_SMALL))
            decision.parameter = "queue_depth_limit"
            decision.old_value = old_value
            decision.new_value = state.queue_depth_limit
            self._log_adjustment(decision)

        else:
            return None

        return decision

    def _is_throughput_degrading(self, history: List[RuntimeMetrics]) -> bool:
        """Check if throughput is degrading over recent history."""
        if len(history) < 5:
            return False

        throughputs = [m.throughput_gen_per_sec for m in history]
        early_avg = sum(throughputs[:len(throughputs)//2]) / (len(throughputs)//2)
        late_avg = sum(throughputs[len(throughputs)//2:]) / (len(throughputs) - len(throughputs)//2)

        # Degrading if late is < 90% of early
        return late_avg < early_avg * 0.9

    def _log_adjustment(self, decision: TuningDecision, prefix: str = "TUNING"):
        """Log a tuning adjustment with details."""
        logger.info("-" * 60)
        logger.info(f"[{prefix}] {decision.action.value.upper()}")
        logger.info(f"  Reason: {decision.reason.value}")
        logger.info(f"  Parameter: {decision.parameter}")
        logger.info(f"  Change: {decision.old_value} → {decision.new_value}")
        logger.info(f"  Metrics before:")
        logger.info(f"    CPU: {decision.metrics_before['cpu_percent']:.1f}%")
        logger.info(f"    RAM: {decision.metrics_before['ram_percent']:.1f}%")
        logger.info(f"    Throughput: {decision.metrics_before['throughput']:.2f} gen/s")
        logger.info("-" * 60)

    # ─────────────────────────────────────────────────────
    # VERIFICATION
    # ─────────────────────────────────────────────────────

    def verify_last_adjustment(self) -> Optional[bool]:
        """
        Verify if the last adjustment was successful.

        Should be called after cooldown period to assess impact.

        Returns:
            True if improvement, False if regression, None if inconclusive
        """
        if not self.decisions:
            return None

        last_decision = self.decisions[-1]
        current_metrics = self.monitor.get_current_metrics()

        # Get metrics after
        metrics_after = {
            "cpu_percent": current_metrics.cpu_percent,
            "ram_percent": current_metrics.ram_percent,
            "throughput": current_metrics.throughput_gen_per_sec,
            "avg_gen_time": current_metrics.avg_time_per_gen_sec,
        }
        last_decision.metrics_after = metrics_after

        # Assess based on action type
        before = last_decision.metrics_before
        after = metrics_after

        if last_decision.reason in [
            TuningReason.MEMORY_PRESSURE,
            TuningReason.SWAP_ACTIVE,
            TuningReason.EMERGENCY_OOM_RISK
        ]:
            # Success if RAM decreased
            success = after["ram_percent"] < before["ram_percent"]
        elif last_decision.reason == TuningReason.CPU_SATURATION:
            # Success if throughput improved or CPU decreased
            success = (
                after["throughput"] > before["throughput"] * 1.05 or
                after["cpu_percent"] < before["cpu_percent"]
            )
        elif last_decision.reason == TuningReason.RESOURCES_AVAILABLE:
            # Success if throughput improved without memory issues
            success = (
                after["throughput"] > before["throughput"] and
                after["ram_percent"] < MEMORY_HIGH_THRESHOLD
            )
        else:
            success = None

        last_decision.success = success

        # Log verification
        self._log_verification(last_decision)

        return success

    def _log_verification(self, decision: TuningDecision):
        """Log verification result."""
        status = "SUCCESS" if decision.success else "NEEDS_ATTENTION" if decision.success is False else "INCONCLUSIVE"

        logger.info("-" * 60)
        logger.info(f"[VERIFY] {decision.action.value.upper()}: {status}")
        if decision.metrics_after:
            logger.info(f"  Metrics after:")
            logger.info(f"    CPU: {decision.metrics_after['cpu_percent']:.1f}% (was {decision.metrics_before['cpu_percent']:.1f}%)")
            logger.info(f"    RAM: {decision.metrics_after['ram_percent']:.1f}% (was {decision.metrics_before['ram_percent']:.1f}%)")
            logger.info(f"    Throughput: {decision.metrics_after['throughput']:.2f} gen/s (was {decision.metrics_before['throughput']:.2f})")
        logger.info("-" * 60)

    # ─────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────

    def get_current_tuning_params(self) -> Dict[str, Any]:
        """Get current tuning parameters."""
        state = self.monitor.tuning_state
        return {
            "batch_size": state.batch_size,
            "max_workers": state.max_workers,
            "chunk_size": state.chunk_size,
            "queue_depth_limit": state.queue_depth_limit,
        }

    def set_initial_params(
        self,
        batch_size: int = None,
        max_workers: int = None,
        chunk_size: int = None,
        queue_depth_limit: int = None
    ):
        """Set initial tuning parameters."""
        state = self.monitor.tuning_state

        if batch_size is not None:
            state.batch_size = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, batch_size))
        if max_workers is not None:
            state.max_workers = max(MIN_WORKERS, min(MAX_WORKERS, max_workers))
        if chunk_size is not None:
            state.chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, chunk_size))
        if queue_depth_limit is not None:
            state.queue_depth_limit = max(MIN_QUEUE_DEPTH, min(MAX_QUEUE_DEPTH, queue_depth_limit))

        logger.info(f"Initial tuning params: batch={state.batch_size}, workers={state.max_workers}, "
                    f"chunk={state.chunk_size}, queue={state.queue_depth_limit}")

    def get_summary(self) -> Dict[str, Any]:
        """Get tuning summary for logging/reporting."""
        state = self.monitor.tuning_state

        return {
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
            "total_adjustments": state.adjustment_count,
            "current_params": self.get_current_tuning_params(),
            "thresholds": {
                "memory_high_pct": MEMORY_HIGH_THRESHOLD,
                "memory_free_pct": MEMORY_FREE_THRESHOLD,
                "cpu_saturation_pct": CPU_HIGH_THRESHOLD,
            },
            "limits": {
                "batch_size": {"min": MIN_BATCH_SIZE, "max": MAX_BATCH_SIZE},
                "max_workers": {"min": MIN_WORKERS, "max": MAX_WORKERS},
                "chunk_size": {"min": MIN_CHUNK_SIZE, "max": MAX_CHUNK_SIZE},
                "queue_depth": {"min": MIN_QUEUE_DEPTH, "max": MAX_QUEUE_DEPTH},
            },
            "recent_decisions": [
                {
                    "timestamp": datetime.fromtimestamp(d.timestamp).isoformat(),
                    "action": d.action.value,
                    "reason": d.reason.value,
                    "parameter": d.parameter,
                    "change": f"{d.old_value} → {d.new_value}",
                    "success": d.success,
                }
                for d in self.decisions[-5:]
            ],
        }

    def reset(self):
        """Reset tuner state."""
        self.decisions.clear()
        self.baseline_throughput = 0.0
        self.baseline_metrics = None
        self._last_swap_gb = 0.0

        state = self.monitor.tuning_state
        state.adjustment_count = 0
        state.last_adjustment_time = 0.0
        state.adjustment_history.clear()

        logger.info("Auto-tuner reset")


# ═══════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════

_tuner_instance: Optional[AutoTuner] = None


def get_auto_tuner() -> AutoTuner:
    """Get or create singleton AutoTuner instance."""
    global _tuner_instance
    if _tuner_instance is None:
        enabled = os.getenv("AUTO_TUNING_ENABLED", "true").lower() == "true"
        _tuner_instance = AutoTuner(enabled=enabled)
    return _tuner_instance


def check_and_tune() -> Optional[TuningDecision]:
    """Convenience function to run tuning check."""
    return get_auto_tuner().check_and_tune()


# ═══════════════════════════════════════════════════════
# CONTEXT MANAGER FOR INTEGRATION
# ═══════════════════════════════════════════════════════

class AutoTuningContext:
    """
    Context manager for auto-tuning during optimization.

    Usage:
        with AutoTuningContext() as tuner:
            for generation in generations:
                tuner.generation_start()
                # ... run generation ...
                tuner.generation_end()
                tuner.check()  # Check for tuning adjustments
    """

    def __init__(self, enabled: bool = True, initial_params: Dict = None):
        self.enabled = enabled
        self.initial_params = initial_params or {}
        self.tuner: Optional[AutoTuner] = None
        self.monitor: Optional[PerformanceMonitor] = None

    def __enter__(self):
        if not self.enabled:
            return self

        self.monitor = get_performance_monitor()
        self.tuner = get_auto_tuner()

        # Set initial params
        if self.initial_params:
            self.tuner.set_initial_params(**self.initial_params)

        # Start monitoring
        self.monitor.start()
        self.monitor.reset_session()

        logger.info("Auto-tuning context started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor:
            self.monitor.stop()

        if self.tuner:
            # Log final summary
            summary = self.tuner.get_summary()
            logger.info("=" * 60)
            logger.info("AUTO-TUNING SESSION SUMMARY")
            logger.info(f"  Total adjustments: {summary['total_adjustments']}")
            logger.info(f"  Final params: {summary['current_params']}")
            logger.info("=" * 60)

        return False

    def generation_start(self):
        """Mark generation start."""
        if self.monitor:
            self.monitor.generation_started()

    def generation_end(self):
        """Mark generation end."""
        if self.monitor:
            self.monitor.generation_completed()

    def check(self) -> Optional[TuningDecision]:
        """Check and apply tuning if needed."""
        if self.tuner:
            return self.tuner.check_and_tune()
        return None

    def get_params(self) -> Dict[str, Any]:
        """Get current tuning parameters."""
        if self.tuner:
            return self.tuner.get_current_tuning_params()
        return {
            "batch_size": 10,
            "max_workers": 6,
            "chunk_size": 100_000,
            "queue_depth_limit": 100,
        }


# ═══════════════════════════════════════════════════════
# CLI FOR TESTING
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    print("\n=== Auto-Tuner Test ===\n")

    # Initialize
    monitor = get_performance_monitor()
    tuner = AutoTuner(monitor=monitor, enabled=True, cooldown_seconds=5)

    # Set initial params
    tuner.set_initial_params(
        batch_size=20,
        max_workers=6,
        chunk_size=100_000,
        queue_depth_limit=100
    )

    # Start monitoring
    monitor.start()

    print("\nSimulating optimization loop for 30 seconds...\n")

    try:
        for i in range(30):
            # Simulate generation
            monitor.generation_started()
            time.sleep(0.2)  # Simulated work
            monitor.generation_completed()

            # Check for tuning
            decision = tuner.check_and_tune()
            if decision:
                print(f"[{i}] Tuning applied: {decision.action.value}")

            # Print status every 5 iterations
            if i % 5 == 0:
                metrics = monitor.get_current_metrics()
                params = tuner.get_current_tuning_params()
                print(f"[{i}] CPU: {metrics.cpu_percent:.1f}%, RAM: {metrics.ram_percent:.1f}%, "
                      f"Gen/s: {metrics.throughput_gen_per_sec:.2f}, Batch: {params['batch_size']}")

            time.sleep(0.8)

    except KeyboardInterrupt:
        pass

    monitor.stop()

    print("\n=== Final Summary ===")
    print(json.dumps(tuner.get_summary(), indent=2, default=str))
