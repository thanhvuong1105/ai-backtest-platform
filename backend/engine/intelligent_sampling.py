# engine/intelligent_sampling.py
"""
Intelligent Parameter Sampling for Quant AI Brain

Combines 4 strategies for smarter parameter generation:
1. Logical Constraints (Hard Rules) - Reject invalid combinations
2. Correlated Sampling (Soft Rules) - Sample related params together
3. Template-Based Sampling (Archetypes) - Use trading style templates
4. Bayesian-Guided Sampling - Learn from past results

This replaces pure random sampling in Phase 2 optimization.
"""

import random
import copy
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 1: LOGICAL CONSTRAINTS (HARD RULES)
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintValidator:
    """
    Validates genome parameters against logical trading constraints.

    Constraints:
    1. Entry vs SL Sensitivity - SL should respond appropriately to Entry
    2. Risk-Reward Logic - R:R must be reasonable given SL width
    3. RSI vs MA Length - MA must be shorter than RSI period
    4. Long vs Short Symmetry - Long/Short params should be balanced
    5. TP vs Entry - TP should respond quickly enough
    """

    # Constraint coefficients - MAXIMUM RELAXED for extreme diversity
    SL_PERIOD_MIN_RATIO = 0.1   # ULTRA RELAXED: allow very fast SL
    SL_PERIOD_MAX_RATIO = 10.0  # ULTRA RELAXED: allow very slow SL
    SL_MULT_MIN_RATIO = 0.2     # ULTRA RELAXED: allow very tight SL
    SL_MULT_MAX_RATIO = 15.0    # ULTRA RELAXED: allow very wide SL
    RR_MIN_RATIO = 0.02         # ULTRA RELAXED: allow very low R:R
    RR_MAX_RATIO = 5.0          # ULTRA RELAXED: allow very high R:R
    LONG_SHORT_MAX_DIFF = 25.0  # ULTRA RELAXED: allow extreme asymmetry
    TP_PERIOD_MAX_RATIO = 8.0   # ULTRA RELAXED: allow very slow TP

    @classmethod
    def validate(cls, genome: Dict) -> Tuple[bool, List[str]]:
        """
        Validate genome against all constraints.

        Args:
            genome: Genome dict with entry, sl_long, sl_short, tp_dual_long, etc.

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Extract Entry params
        entry = genome.get("entry", {})
        st_atrPeriod = entry.get("st_atrPeriod", 10)
        st_mult = entry.get("st_mult", 2.0)
        rsi_length = entry.get("rsi_length", 14)
        rsi_ma_length = entry.get("rsi_ma_length", 6)

        # ═══════════════════════════════════════════════════════════════
        # CONSTRAINT 3: RSI vs MA Length (always check)
        # ═══════════════════════════════════════════════════════════════
        if rsi_ma_length >= rsi_length:
            violations.append(
                f"RSI MA Length ({rsi_ma_length}) must be < RSI Length ({rsi_length})"
            )

        # Check if this is combined strategy (has sl_long/sl_short)
        is_combined = "sl_long" in genome

        if is_combined:
            # Combined strategy: validate Long and Short separately
            violations.extend(cls._validate_side(genome, "long", st_atrPeriod, st_mult))
            violations.extend(cls._validate_side(genome, "short", st_atrPeriod, st_mult))

            # ═══════════════════════════════════════════════════════════════
            # CONSTRAINT 4: Long vs Short Symmetry
            # ═══════════════════════════════════════════════════════════════
            sl_long_mult = genome.get("sl_long", {}).get("st_mult", 3.0)
            sl_short_mult = genome.get("sl_short", {}).get("st_mult", 3.0)

            if abs(sl_long_mult - sl_short_mult) > cls.LONG_SHORT_MAX_DIFF:
                violations.append(
                    f"Long/Short SL Mult too different ({sl_long_mult} vs {sl_short_mult}), "
                    f"max diff = {cls.LONG_SHORT_MAX_DIFF}"
                )
        else:
            # Standard strategy: validate single SL/TP
            violations.extend(cls._validate_standard(genome, st_atrPeriod, st_mult))

        is_valid = len(violations) == 0
        return is_valid, violations

    @classmethod
    def _validate_side(cls, genome: Dict, side: str, entry_period: int, entry_mult: float) -> List[str]:
        """Validate constraints for one side (long or short)."""
        violations = []

        sl_block = genome.get(f"sl_{side}", {})
        tp_dual_block = genome.get(f"tp_dual_{side}", {})
        tp_rsi_block = genome.get(f"tp_rsi_{side}", {})

        sl_period = sl_block.get("st_atrPeriod", 10)
        sl_mult = sl_block.get("st_mult", 3.0)
        tp_dual_period = tp_dual_block.get("st_atrPeriod", 10)
        tp_rsi_period = tp_rsi_block.get("st_atrPeriod", 10)
        rr_dual = tp_dual_block.get("rr_mult", 1.0)
        rr_rsi = tp_rsi_block.get("rr_mult", 1.0)

        # ═══════════════════════════════════════════════════════════════
        # CONSTRAINT 1: Entry vs SL Sensitivity
        # ═══════════════════════════════════════════════════════════════
        min_sl_period = entry_period * cls.SL_PERIOD_MIN_RATIO
        max_sl_period = entry_period * cls.SL_PERIOD_MAX_RATIO

        if sl_period < min_sl_period:
            violations.append(
                f"[{side.upper()}] SL Period ({sl_period}) too fast vs Entry ({entry_period}), "
                f"min = {min_sl_period:.1f}"
            )
        if sl_period > max_sl_period:
            violations.append(
                f"[{side.upper()}] SL Period ({sl_period}) too slow vs Entry ({entry_period}), "
                f"max = {max_sl_period:.1f}"
            )

        min_sl_mult = entry_mult * cls.SL_MULT_MIN_RATIO
        max_sl_mult = entry_mult * cls.SL_MULT_MAX_RATIO

        if sl_mult < min_sl_mult:
            violations.append(
                f"[{side.upper()}] SL Mult ({sl_mult}) too tight vs Entry ({entry_mult}), "
                f"min = {min_sl_mult:.1f}"
            )
        if sl_mult > max_sl_mult:
            violations.append(
                f"[{side.upper()}] SL Mult ({sl_mult}) too wide vs Entry ({entry_mult}), "
                f"max = {max_sl_mult:.1f}"
            )

        # ═══════════════════════════════════════════════════════════════
        # CONSTRAINT 2: Risk-Reward Logic
        # ═══════════════════════════════════════════════════════════════
        rr_min = sl_mult * cls.RR_MIN_RATIO
        rr_max = sl_mult * cls.RR_MAX_RATIO

        if rr_dual < rr_min:
            violations.append(
                f"[{side.upper()}] Dual R:R ({rr_dual}) too low for SL Mult ({sl_mult}), "
                f"min = {rr_min:.2f}"
            )
        if rr_rsi < rr_min:
            violations.append(
                f"[{side.upper()}] RSI R:R ({rr_rsi}) too low for SL Mult ({sl_mult}), "
                f"min = {rr_min:.2f}"
            )

        # ═══════════════════════════════════════════════════════════════
        # CONSTRAINT 5: TP vs Entry
        # ═══════════════════════════════════════════════════════════════
        max_tp_period = entry_period * cls.TP_PERIOD_MAX_RATIO

        if tp_dual_period > max_tp_period:
            violations.append(
                f"[{side.upper()}] TP Dual Period ({tp_dual_period}) too slow vs Entry ({entry_period}), "
                f"max = {max_tp_period:.1f}"
            )
        if tp_rsi_period > max_tp_period:
            violations.append(
                f"[{side.upper()}] TP RSI Period ({tp_rsi_period}) too slow vs Entry ({entry_period}), "
                f"max = {max_tp_period:.1f}"
            )

        return violations

    @classmethod
    def _validate_standard(cls, genome: Dict, entry_period: int, entry_mult: float) -> List[str]:
        """Validate constraints for standard (non-combined) strategy."""
        violations = []

        sl = genome.get("sl", {})
        tp_dual = genome.get("tp_dual", {})
        tp_rsi = genome.get("tp_rsi", {})

        sl_period = sl.get("st_atrPeriod", 10)
        sl_mult = sl.get("st_mult", 3.0)
        rr_dual = tp_dual.get("rr_mult", 1.0)
        rr_rsi = tp_rsi.get("rr_mult", 1.0)

        # Constraint 1: Entry vs SL
        min_sl_period = entry_period * cls.SL_PERIOD_MIN_RATIO
        max_sl_period = entry_period * cls.SL_PERIOD_MAX_RATIO

        if sl_period < min_sl_period or sl_period > max_sl_period:
            violations.append(
                f"SL Period ({sl_period}) outside valid range [{min_sl_period:.1f}, {max_sl_period:.1f}]"
            )

        # Constraint 2: R:R Logic
        rr_min = sl_mult * cls.RR_MIN_RATIO
        if rr_dual < rr_min or rr_rsi < rr_min:
            violations.append(
                f"R:R ({rr_dual}, {rr_rsi}) too low for SL Mult ({sl_mult}), min = {rr_min:.2f}"
            )

        return violations


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 2: CORRELATED SAMPLING (SOFT RULES)
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelatedSampler:
    """
    Sample parameters in correlated groups based on trading logic.

    Groups:
    1. Entry Sensitivity: (st_atrPeriod, st_mult, rf_period) - how sensitive to trends
    2. Risk Profile: (sl_mult, rr_mult) - risk tolerance
    3. Speed Profile: (all periods) - fast vs slow indicators
    """

    # Sensitivity levels - MAXIMUM WIDE RANGES for extreme diversity
    SENSITIVITY_LEVELS = {
        "fast": {
            "st_atrPeriod": (1, 30),        # ULTRA WIDE: 1-30
            "st_mult": (0.5, 8.0),          # ULTRA WIDE: covers most range
            "rf_period": (1, 150),          # ULTRA WIDE: 1-150
            "rf_mult": (1.0, 10.0),         # ULTRA WIDE
        },
        "medium": {
            "st_atrPeriod": (1, 50),        # ULTRA WIDE: full range
            "st_mult": (0.5, 10.0),         # ULTRA WIDE: full range
            "rf_period": (1, 200),          # ULTRA WIDE: full range
            "rf_mult": (1.0, 10.0),         # ULTRA WIDE
        },
        "slow": {
            "st_atrPeriod": (10, 50),       # ULTRA WIDE: 10-50
            "st_mult": (2.0, 10.0),         # ULTRA WIDE
            "rf_period": (50, 200),         # ULTRA WIDE
            "rf_mult": (2.0, 10.0),         # ULTRA WIDE
        },
    }

    # Risk profiles - MAXIMUM WIDE RANGES for extreme diversity
    RISK_PROFILES = {
        "aggressive": {
            "sl_mult_range": (0.5, 10.0),   # ULTRA WIDE: very tight to medium
            "rr_mult_range": (1.0, 5.0),    # ULTRA WIDE
        },
        "balanced": {
            "sl_mult_range": (1.0, 15.0),   # ULTRA WIDE: full range
            "rr_mult_range": (0.5, 5.0),    # ULTRA WIDE: full range
        },
        "conservative": {
            "sl_mult_range": (3.0, 20.0),   # ULTRA WIDE: medium to very wide
            "rr_mult_range": (0.2, 3.0),    # ULTRA WIDE
        },
    }

    @classmethod
    def sample(cls, bounds: Dict, sensitivity: str = None, risk: str = None) -> Dict:
        """
        Sample a genome using correlated groups.

        Args:
            bounds: Parameter bounds from UI
            sensitivity: 'fast', 'medium', 'slow', or None (random)
            risk: 'aggressive', 'balanced', 'conservative', or None (random)

        Returns:
            Sampled genome
        """
        # Choose sensitivity and risk if not specified
        if sensitivity is None:
            sensitivity = random.choice(["fast", "medium", "slow"])
        if risk is None:
            risk = random.choice(["aggressive", "balanced", "conservative"])

        sens_params = cls.SENSITIVITY_LEVELS[sensitivity]
        risk_params = cls.RISK_PROFILES[risk]

        # Sample Entry with sensitivity-constrained ranges
        entry = cls._sample_entry(bounds, sens_params)

        # Sample SL/TP with risk-constrained ranges
        sl_long = cls._sample_sl(bounds, "sl_long", risk_params, entry)
        sl_short = cls._sample_sl(bounds, "sl_short", risk_params, entry)

        tp_dual_long = cls._sample_tp(bounds, "tp_dual_long", risk_params, sl_long)
        tp_dual_short = cls._sample_tp(bounds, "tp_dual_short", risk_params, sl_short)
        tp_rsi_long = cls._sample_tp(bounds, "tp_rsi_long", risk_params, sl_long)
        tp_rsi_short = cls._sample_tp(bounds, "tp_rsi_short", risk_params, sl_short)

        genome = {
            "entry": entry,
            "sl_long": sl_long,
            "sl_short": sl_short,
            "tp_dual_long": tp_dual_long,
            "tp_dual_short": tp_dual_short,
            "tp_rsi_long": tp_rsi_long,
            "tp_rsi_short": tp_rsi_short,
            "mode": {
                "enableLong": True,
                "enableShort": True,
                "showDualFlip": True,
                "showRSI": True,
            },
            "_meta": {
                "sensitivity": sensitivity,
                "risk": risk,
                "sampling_method": "correlated",
            }
        }

        return genome

    @classmethod
    def _sample_entry(cls, bounds: Dict, sens_params: Dict) -> Dict:
        """Sample entry params with sensitivity constraints."""
        entry_bounds = bounds.get("entry", {})

        def constrained_sample(param: str, sens_range: Tuple, is_int: bool = False):
            """Sample within intersection of UI bounds and sensitivity range."""
            ui_min, ui_max = entry_bounds.get(param, sens_range)
            # Intersect with sensitivity range
            final_min = max(ui_min, sens_range[0])
            final_max = min(ui_max, sens_range[1])
            # Ensure valid range
            if final_min > final_max:
                final_min, final_max = sens_range

            if is_int:
                return random.randint(int(final_min), int(final_max))
            return round(random.uniform(final_min, final_max), 2)

        # RSI length and MA length need special handling
        rsi_length = constrained_sample("rsi_length", (5, 20), is_int=True)
        rsi_ma_bounds = entry_bounds.get("rsi_ma_length", (2, 15))
        rsi_ma_max = int(min(rsi_length - 2, int(rsi_ma_bounds[1])))
        rsi_ma_min = int(rsi_ma_bounds[0]) if isinstance(rsi_ma_bounds[0], (int, float)) else 2
        rsi_ma_length = random.randint(max(2, rsi_ma_min), max(2, rsi_ma_max))

        return {
            "st_atrPeriod": constrained_sample("st_atrPeriod", sens_params["st_atrPeriod"], is_int=True),
            "st_src": "hl2",
            "st_mult": constrained_sample("st_mult", sens_params["st_mult"]),
            "st_useATR": True,
            "rf_src": "close",
            "rf_period": constrained_sample("rf_period", sens_params["rf_period"], is_int=True),
            "rf_mult": constrained_sample("rf_mult", sens_params["rf_mult"]),
            "rsi_length": rsi_length,
            "rsi_ma_length": rsi_ma_length,
        }

    @classmethod
    def _sample_sl(cls, bounds: Dict, block: str, risk_params: Dict, entry: Dict) -> Dict:
        """Sample SL params correlated with entry and risk profile."""
        block_bounds = bounds.get(block, {})
        entry_period = entry.get("st_atrPeriod", 10)
        entry_mult = entry.get("st_mult", 2.0)

        # SL Period should be relative to Entry Period (constraint 1)
        sl_period_min = int(entry_period * ConstraintValidator.SL_PERIOD_MIN_RATIO)
        sl_period_max = int(entry_period * ConstraintValidator.SL_PERIOD_MAX_RATIO)

        # Intersect with UI bounds
        ui_period_min, ui_period_max = block_bounds.get("st_atrPeriod", (1, 50))
        sl_period_min = max(sl_period_min, int(ui_period_min))
        sl_period_max = min(sl_period_max, int(ui_period_max))
        if sl_period_min > sl_period_max:
            sl_period_min, sl_period_max = int(ui_period_min), int(ui_period_max)

        # SL Mult from risk profile, but also constrained by Entry
        sl_mult_range = risk_params["sl_mult_range"]
        sl_mult_min = max(sl_mult_range[0], entry_mult * ConstraintValidator.SL_MULT_MIN_RATIO)
        sl_mult_max = min(sl_mult_range[1], entry_mult * ConstraintValidator.SL_MULT_MAX_RATIO)

        # Intersect with UI bounds
        ui_mult_min, ui_mult_max = block_bounds.get("st_mult", (1.0, 20.0))
        sl_mult_min = max(sl_mult_min, float(ui_mult_min))
        sl_mult_max = min(sl_mult_max, float(ui_mult_max))
        if sl_mult_min > sl_mult_max:
            sl_mult_min, sl_mult_max = float(ui_mult_min), float(ui_mult_max)

        return {
            "st_atrPeriod": random.randint(int(sl_period_min), int(sl_period_max)),
            "st_src": "hl2",
            "st_mult": round(random.uniform(sl_mult_min, sl_mult_max), 2),
            "st_useATR": True,
            "rf_period": random.randint(*[int(x) for x in block_bounds.get("rf_period", (1, 200))]),
            "rf_mult": round(random.uniform(*[float(x) for x in block_bounds.get("rf_mult", (1.0, 20.0))]), 2),
        }

    @classmethod
    def _sample_tp(cls, bounds: Dict, block: str, risk_params: Dict, sl: Dict) -> Dict:
        """Sample TP params correlated with SL and risk profile."""
        block_bounds = bounds.get(block, {})
        sl_mult = sl.get("st_mult", 3.0)

        # R:R from risk profile, but constrained by SL mult (constraint 2)
        rr_range = risk_params["rr_mult_range"]
        rr_min = max(rr_range[0], sl_mult * ConstraintValidator.RR_MIN_RATIO)
        rr_max = min(rr_range[1], sl_mult * ConstraintValidator.RR_MAX_RATIO)

        # Intersect with UI bounds
        ui_rr_min, ui_rr_max = block_bounds.get("rr_mult", (0.1, 5.0))
        rr_min = max(rr_min, float(ui_rr_min))
        rr_max = min(rr_max, float(ui_rr_max))
        if rr_min > rr_max:
            rr_min, rr_max = float(ui_rr_min), float(ui_rr_max)

        # Get st_atrPeriod bounds with proper type conversion
        st_period_bounds = block_bounds.get("st_atrPeriod", (1, 50))
        st_mult_bounds = block_bounds.get("st_mult", (1.0, 20.0))

        return {
            "st_atrPeriod": random.randint(int(st_period_bounds[0]), int(st_period_bounds[1])),
            "st_mult": round(random.uniform(float(st_mult_bounds[0]), float(st_mult_bounds[1])), 2),
            "rr_mult": round(random.uniform(rr_min, rr_max), 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 3: TEMPLATE-BASED SAMPLING (ARCHETYPES)
# ═══════════════════════════════════════════════════════════════════════════════

class ArchetypeSampler:
    """
    Sample genomes based on predefined trading style archetypes.

    Archetypes:
    1. Scalper - Fast entries, tight SL, quick TP
    2. Swing Trader - Medium speed, balanced risk
    3. Trend Follower - Slow entries, wide SL, high R:R
    4. Mean Reverter - Fast RSI, tight SL, moderate R:R
    """

    # DISTINCT RANGES - each archetype has unique characteristics for cross-mixing
    ARCHETYPES = {
        "scalper": {
            "description": "Fast entries, tight stops, quick profits",
            "entry": {
                "st_atrPeriod": (1, 15),       # FAST: quick response
                "st_mult": (0.5, 3.0),         # LOW: sensitive to small moves
                "rf_period": (1, 60),          # SHORT: fast trend detection
                "rf_mult": (1.0, 4.0),         # LOW: tight filter
                "rsi_length": (5, 12),         # SHORT: fast RSI
                "rsi_ma_length": (2, 6),       # SHORT: fast MA
            },
            "sl": {
                "st_atrPeriod": (1, 15),       # FAST: quick SL response
                "st_mult": (1.0, 5.0),         # TIGHT: small SL
                "rf_period": (1, 80),          # SHORT
                "rf_mult": (1.0, 6.0),         # LOW
            },
            "tp": {
                "st_atrPeriod": (1, 15),       # FAST: quick TP
                "st_mult": (0.5, 4.0),         # TIGHT
                "rr_mult": (1.5, 5.0),         # HIGH R:R: small risk, big reward
            },
        },
        "swing_trader": {
            "description": "Medium speed, balanced risk-reward",
            "entry": {
                "st_atrPeriod": (10, 30),      # MEDIUM: balanced response
                "st_mult": (2.0, 6.0),         # MEDIUM: balanced sensitivity
                "rf_period": (50, 150),        # MEDIUM: swing detection
                "rf_mult": (2.0, 7.0),         # MEDIUM
                "rsi_length": (10, 16),        # MEDIUM RSI
                "rsi_ma_length": (4, 9),       # MEDIUM MA
            },
            "sl": {
                "st_atrPeriod": (8, 35),       # MEDIUM SL
                "st_mult": (3.0, 10.0),        # MEDIUM: balanced SL
                "rf_period": (40, 160),        # MEDIUM
                "rf_mult": (3.0, 12.0),        # MEDIUM
            },
            "tp": {
                "st_atrPeriod": (8, 30),       # MEDIUM TP
                "st_mult": (2.0, 8.0),         # MEDIUM
                "rr_mult": (1.0, 3.5),         # BALANCED R:R
            },
        },
        "trend_follower": {
            "description": "Slow entries, ride trends, high R:R",
            "entry": {
                "st_atrPeriod": (20, 50),      # SLOW: wait for strong trends
                "st_mult": (4.0, 10.0),        # HIGH: filter noise
                "rf_period": (100, 200),       # LONG: major trend detection
                "rf_mult": (4.0, 10.0),        # HIGH: strong filter
                "rsi_length": (14, 20),        # LONG RSI
                "rsi_ma_length": (6, 12),      # LONG MA
            },
            "sl": {
                "st_atrPeriod": (15, 50),      # SLOW SL: give room
                "st_mult": (6.0, 20.0),        # WIDE: let trends breathe
                "rf_period": (80, 200),        # LONG
                "rf_mult": (6.0, 20.0),        # HIGH
            },
            "tp": {
                "st_atrPeriod": (15, 50),      # SLOW TP: ride trends
                "st_mult": (4.0, 15.0),        # WIDE
                "rr_mult": (2.0, 5.0),         # HIGH R:R: big wins
            },
        },
        "mean_reverter": {
            "description": "Fast RSI signals, tight risk, moderate reward",
            "entry": {
                "st_atrPeriod": (3, 20),       # FAST-MEDIUM: catch reversals
                "st_mult": (1.0, 5.0),         # LOW-MEDIUM: sensitive
                "rf_period": (20, 100),        # SHORT-MEDIUM
                "rf_mult": (1.5, 6.0),         # LOW-MEDIUM
                "rsi_length": (5, 14),         # SHORT: fast overbought/oversold
                "rsi_ma_length": (2, 7),       # SHORT MA
            },
            "sl": {
                "st_atrPeriod": (3, 25),       # FAST-MEDIUM SL
                "st_mult": (2.0, 8.0),         # TIGHT-MEDIUM: limit losses
                "rf_period": (20, 120),        # SHORT-MEDIUM
                "rf_mult": (2.0, 10.0),        # LOW-MEDIUM
            },
            "tp": {
                "st_atrPeriod": (3, 20),       # FAST-MEDIUM TP
                "st_mult": (1.5, 6.0),         # TIGHT-MEDIUM
                "rr_mult": (0.8, 3.0),         # MODERATE R:R: higher win rate
            },
        },
    }

    @classmethod
    def sample(cls, bounds: Dict, archetype: str = None, cross_mix: bool = True) -> Dict:
        """
        Sample a genome from archetype templates.

        Args:
            bounds: Parameter bounds from UI
            archetype: 'scalper', 'swing_trader', 'trend_follower', 'mean_reverter', or None (random)
            cross_mix: If True, randomly mix Entry/SL/TP from DIFFERENT archetypes for diversity

        Returns:
            Sampled genome
        """
        archetypes_list = list(cls.ARCHETYPES.keys())

        # Decide whether to use cross-mixing (50% chance if cross_mix=True)
        use_cross_mix = cross_mix and random.random() < 0.5

        if use_cross_mix:
            # CROSS-ARCHETYPE SAMPLING: Mix Entry/SL/TP from different archetypes
            entry_arch = random.choice(archetypes_list)
            sl_arch = random.choice(archetypes_list)
            tp_arch = random.choice(archetypes_list)

            entry_template = cls.ARCHETYPES[entry_arch]["entry"]
            sl_template = cls.ARCHETYPES[sl_arch]["sl"]
            tp_template = cls.ARCHETYPES[tp_arch]["tp"]

            entry = cls._sample_block(bounds.get("entry", {}), entry_template, is_entry=True)
            sl_long = cls._sample_block(bounds.get("sl_long", {}), sl_template)
            sl_short = cls._sample_block(bounds.get("sl_short", {}), sl_template)
            tp_dual_long = cls._sample_block(bounds.get("tp_dual_long", {}), tp_template)
            tp_dual_short = cls._sample_block(bounds.get("tp_dual_short", {}), tp_template)
            tp_rsi_long = cls._sample_block(bounds.get("tp_rsi_long", {}), tp_template)
            tp_rsi_short = cls._sample_block(bounds.get("tp_rsi_short", {}), tp_template)

            meta_archetype = f"cross:{entry_arch[:2]}/{sl_arch[:2]}/{tp_arch[:2]}"
        else:
            # SINGLE ARCHETYPE: Use one archetype for all blocks
            if archetype is None:
                archetype = random.choice(archetypes_list)

            template = cls.ARCHETYPES[archetype]

            entry = cls._sample_block(bounds.get("entry", {}), template["entry"], is_entry=True)
            sl_long = cls._sample_block(bounds.get("sl_long", {}), template["sl"])
            sl_short = cls._sample_block(bounds.get("sl_short", {}), template["sl"])
            tp_dual_long = cls._sample_block(bounds.get("tp_dual_long", {}), template["tp"])
            tp_dual_short = cls._sample_block(bounds.get("tp_dual_short", {}), template["tp"])
            tp_rsi_long = cls._sample_block(bounds.get("tp_rsi_long", {}), template["tp"])
            tp_rsi_short = cls._sample_block(bounds.get("tp_rsi_short", {}), template["tp"])

            meta_archetype = archetype

        genome = {
            "entry": entry,
            "sl_long": sl_long,
            "sl_short": sl_short,
            "tp_dual_long": tp_dual_long,
            "tp_dual_short": tp_dual_short,
            "tp_rsi_long": tp_rsi_long,
            "tp_rsi_short": tp_rsi_short,
            "mode": {
                "enableLong": True,
                "enableShort": True,
                "showDualFlip": True,
                "showRSI": True,
            },
            "_meta": {
                "archetype": meta_archetype,
                "sampling_method": "archetype",
                "cross_mixed": use_cross_mix,
            }
        }

        return genome

    @classmethod
    def _sample_block(cls, ui_bounds: Dict, template_ranges: Dict, is_entry: bool = False) -> Dict:
        """Sample a block with template ranges intersected with UI bounds."""
        result = {}

        for param, template_range in template_ranges.items():
            ui_range = ui_bounds.get(param, template_range)

            # Intersect ranges
            final_min = max(template_range[0], ui_range[0])
            final_max = min(template_range[1], ui_range[1])

            if final_min > final_max:
                final_min, final_max = template_range

            # Determine if int or float
            is_int = param in ["st_atrPeriod", "rf_period", "rsi_length", "rsi_ma_length"]

            if is_int:
                result[param] = random.randint(int(final_min), int(final_max))
            else:
                result[param] = round(random.uniform(final_min, final_max), 2)

        # Add fixed params
        if is_entry:
            result["st_src"] = "hl2"
            result["st_useATR"] = True
            result["rf_src"] = "close"
            # Ensure RSI MA < RSI Length
            if "rsi_length" in result and "rsi_ma_length" in result:
                if result["rsi_ma_length"] >= result["rsi_length"]:
                    result["rsi_ma_length"] = max(2, result["rsi_length"] - 2)
        else:
            if "st_atrPeriod" in result:
                result["st_src"] = "hl2"
                result["st_useATR"] = True

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 4: BAYESIAN-GUIDED SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianSampler:
    """
    Learn from past results to guide future sampling.

    Tracks:
    - Parameter ranges that produced high scores
    - Successful archetypes/sensitivity combinations
    - Constraint violations that frequently occur
    """

    def __init__(self):
        self.history = []  # (genome, score) pairs
        self.param_stats = defaultdict(lambda: {"values": [], "scores": []})
        self.archetype_scores = defaultdict(list)
        self.sensitivity_scores = defaultdict(list)
        self.risk_scores = defaultdict(list)

    def record(self, genome: Dict, score: float):
        """Record a genome and its score for learning."""
        self.history.append((copy.deepcopy(genome), score))

        # Track meta info if available
        meta = genome.get("_meta", {})
        if "archetype" in meta:
            self.archetype_scores[meta["archetype"]].append(score)
        if "sensitivity" in meta:
            self.sensitivity_scores[meta["sensitivity"]].append(score)
        if "risk" in meta:
            self.risk_scores[meta["risk"]].append(score)

        # Track individual parameters
        self._record_params(genome.get("entry", {}), "entry", score)
        for block in ["sl_long", "sl_short", "tp_dual_long", "tp_dual_short", "tp_rsi_long", "tp_rsi_short"]:
            if block in genome:
                self._record_params(genome[block], block, score)

    def _record_params(self, block_data: Dict, block_name: str, score: float):
        """Record parameter values with their scores."""
        for param, value in block_data.items():
            if isinstance(value, (int, float)):
                key = f"{block_name}.{param}"
                self.param_stats[key]["values"].append(value)
                self.param_stats[key]["scores"].append(score)

    def get_promising_ranges(self, percentile: float = 0.2) -> Dict:
        """
        Get parameter ranges from top-performing genomes.

        Args:
            percentile: Top percentage to consider (0.2 = top 20%)

        Returns:
            Dict of block -> param -> (min, max) from high performers
        """
        if len(self.history) < 10:
            return {}

        # Get score threshold for top performers
        scores = [s for _, s in self.history]
        threshold = sorted(scores, reverse=True)[int(len(scores) * percentile)]

        # Collect values from high performers
        high_performers = [(g, s) for g, s in self.history if s >= threshold]

        promising_ranges = defaultdict(dict)

        for key, stats in self.param_stats.items():
            values = stats["values"]
            param_scores = stats["scores"]

            # Filter to high-scoring values
            high_values = [v for v, s in zip(values, param_scores) if s >= threshold]

            if len(high_values) >= 3:
                block, param = key.split(".", 1)
                promising_ranges[block][param] = (min(high_values), max(high_values))

        return dict(promising_ranges)

    def get_best_archetype(self) -> Optional[str]:
        """Get the archetype with highest average score."""
        if not self.archetype_scores:
            return None

        avg_scores = {
            arch: sum(scores) / len(scores)
            for arch, scores in self.archetype_scores.items()
            if len(scores) >= 3
        }

        if not avg_scores:
            return None

        return max(avg_scores, key=avg_scores.get)

    def get_best_sensitivity(self) -> Optional[str]:
        """Get the sensitivity level with highest average score."""
        if not self.sensitivity_scores:
            return None

        avg_scores = {
            sens: sum(scores) / len(scores)
            for sens, scores in self.sensitivity_scores.items()
            if len(scores) >= 3
        }

        if not avg_scores:
            return None

        return max(avg_scores, key=avg_scores.get)

    def get_best_risk(self) -> Optional[str]:
        """Get the risk profile with highest average score."""
        if not self.risk_scores:
            return None

        avg_scores = {
            risk: sum(scores) / len(scores)
            for risk, scores in self.risk_scores.items()
            if len(scores) >= 3
        }

        if not avg_scores:
            return None

        return max(avg_scores, key=avg_scores.get)


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT SAMPLER (COMBINES ALL 4 IDEAS)
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentSampler:
    """
    Main sampler that combines all 4 ideas:
    1. Logical Constraints - Validate and reject invalid genomes
    2. Correlated Sampling - Sample related params together
    3. Archetype Templates - Use predefined trading styles
    4. Bayesian Learning - Learn from past results

    Sampling strategy mix:
    - 30% Archetype-based (exploitation of known good styles)
    - 30% Correlated sampling (structured exploration)
    - 20% Bayesian-guided (learn from history)
    - 20% Pure random (exploration)
    """

    # BALANCED WEIGHTS with cross-archetype mixing for diversity
    STRATEGY_WEIGHTS = {
        "archetype": 0.35,      # INCREASED: now supports cross-mixing (Entry/SL/TP from different archetypes)
        "correlated": 0.20,     # Correlated sampling with sensitivity/risk profiles
        "bayesian": 0.10,       # Learn from history
        "random": 0.35,         # Pure random for exploration
    }

    def __init__(self, bounds: Dict):
        """
        Initialize sampler with parameter bounds.

        Args:
            bounds: Parameter bounds from UI/config
        """
        self.bounds = bounds
        self.bayesian = BayesianSampler()
        self.samples_generated = 0
        self.samples_rejected = 0
        self.strategy_counts = defaultdict(int)

    def sample(self, max_attempts: int = 50) -> Dict:
        """
        Generate a single valid genome using intelligent sampling.

        Args:
            max_attempts: Maximum attempts before falling back to random

        Returns:
            Valid genome dict
        """
        for attempt in range(max_attempts):
            # Choose sampling strategy
            strategy = self._choose_strategy()
            self.strategy_counts[strategy] += 1

            # Generate genome
            if strategy == "archetype":
                genome = ArchetypeSampler.sample(self.bounds)
            elif strategy == "correlated":
                genome = CorrelatedSampler.sample(self.bounds)
            elif strategy == "bayesian":
                genome = self._bayesian_sample()
            else:  # random
                genome = self._random_sample()

            # Validate against constraints
            is_valid, violations = ConstraintValidator.validate(genome)

            if is_valid:
                self.samples_generated += 1
                return genome
            else:
                self.samples_rejected += 1
                if attempt < 3:
                    logger.debug(f"Rejected sample (attempt {attempt}): {violations[:2]}")

        # Fallback: return last sample with warning
        logger.warning(f"Could not find valid genome after {max_attempts} attempts, using fallback")
        self.samples_generated += 1
        return self._safe_fallback()

    def sample_batch(self, n: int, max_attempts_per_sample: int = 50) -> List[Dict]:
        """
        Generate a batch of valid genomes.

        Args:
            n: Number of genomes to generate
            max_attempts_per_sample: Max attempts per genome

        Returns:
            List of valid genomes
        """
        genomes = []
        for _ in range(n):
            genome = self.sample(max_attempts_per_sample)
            genomes.append(genome)
        return genomes

    def record_result(self, genome: Dict, score: float):
        """
        Record a genome result for Bayesian learning.

        Args:
            genome: Genome that was tested
            score: Fitness score achieved
        """
        self.bayesian.record(genome, score)

    def get_stats(self) -> Dict:
        """Get sampling statistics."""
        total = self.samples_generated + self.samples_rejected
        return {
            "total_attempts": total,
            "generated": self.samples_generated,
            "rejected": self.samples_rejected,
            "rejection_rate": self.samples_rejected / total if total > 0 else 0,
            "strategy_distribution": dict(self.strategy_counts),
            "bayesian_history_size": len(self.bayesian.history),
            "best_archetype": self.bayesian.get_best_archetype(),
            "best_sensitivity": self.bayesian.get_best_sensitivity(),
            "best_risk": self.bayesian.get_best_risk(),
        }

    def _choose_strategy(self) -> str:
        """Choose sampling strategy based on weights and learning."""
        # Adjust weights based on Bayesian learning
        weights = dict(self.STRATEGY_WEIGHTS)

        # If we have enough history, increase Bayesian weight
        if len(self.bayesian.history) >= 20:
            weights["bayesian"] = 0.35
            weights["random"] = 0.10
            weights["archetype"] = 0.28
            weights["correlated"] = 0.27

        # Weighted random choice
        strategies = list(weights.keys())
        probs = [weights[s] for s in strategies]
        return random.choices(strategies, weights=probs, k=1)[0]

    def _bayesian_sample(self) -> Dict:
        """Sample using Bayesian-guided ranges."""
        # Get promising ranges from history
        promising = self.bayesian.get_promising_ranges()

        if not promising:
            # Fall back to correlated sampling
            return CorrelatedSampler.sample(self.bounds)

        # Merge promising ranges with bounds
        merged_bounds = copy.deepcopy(self.bounds)
        for block, params in promising.items():
            if block not in merged_bounds:
                merged_bounds[block] = {}
            for param, (pmin, pmax) in params.items():
                # Narrow the range to promising region
                if param in merged_bounds[block]:
                    ui_min, ui_max = merged_bounds[block][param]
                    merged_bounds[block][param] = (
                        max(pmin * 0.9, ui_min),  # Allow 10% margin
                        min(pmax * 1.1, ui_max)
                    )

        # Use best archetype if known
        best_arch = self.bayesian.get_best_archetype()
        if best_arch and random.random() < 0.6:
            return ArchetypeSampler.sample(merged_bounds, archetype=best_arch)

        # Use best sensitivity/risk if known
        best_sens = self.bayesian.get_best_sensitivity()
        best_risk = self.bayesian.get_best_risk()
        return CorrelatedSampler.sample(merged_bounds, sensitivity=best_sens, risk=best_risk)

    def _random_sample(self) -> Dict:
        """Pure random sample within bounds (for exploration)."""
        from .coherence_validator import PARAM_BOUNDS_COMBINED

        def sample_block(block_name: str) -> Dict:
            block_bounds = self.bounds.get(block_name, PARAM_BOUNDS_COMBINED.get(block_name, {}))
            result = {}

            for param, (min_val, max_val) in block_bounds.items():
                is_int = param in ["st_atrPeriod", "rf_period", "rsi_length", "rsi_ma_length"]
                if is_int:
                    result[param] = random.randint(int(min_val), int(max_val))
                else:
                    result[param] = round(random.uniform(min_val, max_val), 2)

            return result

        entry = sample_block("entry")
        entry["st_src"] = "hl2"
        entry["st_useATR"] = True
        entry["rf_src"] = "close"

        # Ensure RSI MA < RSI Length
        if entry.get("rsi_ma_length", 6) >= entry.get("rsi_length", 14):
            entry["rsi_ma_length"] = max(2, entry["rsi_length"] - 2)

        sl_long = sample_block("sl_long")
        sl_long["st_src"] = "hl2"
        sl_long["st_useATR"] = True

        sl_short = sample_block("sl_short")
        sl_short["st_src"] = "hl2"
        sl_short["st_useATR"] = True

        return {
            "entry": entry,
            "sl_long": sl_long,
            "sl_short": sl_short,
            "tp_dual_long": sample_block("tp_dual_long"),
            "tp_dual_short": sample_block("tp_dual_short"),
            "tp_rsi_long": sample_block("tp_rsi_long"),
            "tp_rsi_short": sample_block("tp_rsi_short"),
            "mode": {
                "enableLong": True,
                "enableShort": True,
                "showDualFlip": True,
                "showRSI": True,
            },
            "_meta": {
                "sampling_method": "random",
            }
        }

    def _safe_fallback(self) -> Dict:
        """Generate a safe fallback genome that satisfies all constraints."""
        # Use swing_trader archetype as it's the most balanced
        genome = ArchetypeSampler.sample(self.bounds, archetype="swing_trader")

        # Force constraint satisfaction
        entry = genome.get("entry", {})
        entry_period = entry.get("st_atrPeriod", 15)
        entry_mult = entry.get("st_mult", 3.0)

        for side in ["long", "short"]:
            sl = genome.get(f"sl_{side}", {})
            # Ensure SL period is in valid range
            sl["st_atrPeriod"] = max(
                int(entry_period * 0.5),
                min(int(entry_period * 3.0), sl.get("st_atrPeriod", entry_period))
            )
            # Ensure SL mult is in valid range
            sl["st_mult"] = max(
                entry_mult * 1.0,
                min(entry_mult * 5.0, sl.get("st_mult", entry_mult * 1.5))
            )

            # Ensure R:R is valid for both TP blocks
            for tp_block in [f"tp_dual_{side}", f"tp_rsi_{side}"]:
                tp = genome.get(tp_block, {})
                sl_mult = sl.get("st_mult", 3.0)
                rr_min = sl_mult * 0.1
                tp["rr_mult"] = max(rr_min, tp.get("rr_mult", 1.5))

        # Ensure RSI MA < RSI Length
        if entry.get("rsi_ma_length", 6) >= entry.get("rsi_length", 14):
            entry["rsi_ma_length"] = max(2, entry["rsi_length"] - 2)

        genome["_meta"] = {"sampling_method": "fallback"}
        return genome


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_intelligent_sampler(bounds: Dict) -> IntelligentSampler:
    """Create an IntelligentSampler instance with given bounds."""
    return IntelligentSampler(bounds)


def sample_intelligent_genome(bounds: Dict, max_attempts: int = 50) -> Dict:
    """Quick function to sample a single intelligent genome."""
    sampler = IntelligentSampler(bounds)
    return sampler.sample(max_attempts)


def validate_genome_constraints(genome: Dict) -> Tuple[bool, List[str]]:
    """Validate a genome against logical constraints."""
    return ConstraintValidator.validate(genome)
