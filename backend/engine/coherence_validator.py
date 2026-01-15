# engine/coherence_validator.py
"""
Coherence Validator for Quant AI Brain

Validates genome configurations against trading logic rules.
Rejects invalid genomes BEFORE backtesting to save compute.
"""

from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# COHERENCE RULES
# ═══════════════════════════════════════════════════════

COHERENCE_RULES = [
    # Entry genome coherence
    {
        "name": "rf_period_vs_st_atr",
        "rule": lambda g: g["entry"]["rf_period"] >= g["entry"]["st_atrPeriod"] * 5,
        "message": "RF_Period should be >= ST_ATR × 5 for proper signal timing"
    },
    {
        "name": "rsi_ma_vs_rsi_length",
        "rule": lambda g: g["entry"]["rsi_ma_length"] <= g["entry"]["rsi_length"],
        "message": "RSI MA length must be <= RSI length"
    },
    {
        "name": "st_mult_range",
        "rule": lambda g: 0.5 <= g["entry"]["st_mult"] <= 5.0,
        "message": "ST multiplier should be between 0.5 and 5.0"
    },
    {
        "name": "rf_mult_range",
        "rule": lambda g: 1.0 <= g["entry"]["rf_mult"] <= 10.0,
        "message": "RF multiplier should be between 1.0 and 10.0"
    },

    # SL genome coherence
    {
        "name": "sl_st_mult_vs_entry_st_mult",
        "rule": lambda g: g["sl"]["st_mult"] >= g["entry"]["st_mult"],
        "message": "SL SuperTrend mult should be >= Entry SuperTrend mult (wider SL)"
    },
    {
        "name": "sl_rf_mult_vs_entry_rf_mult",
        "rule": lambda g: g["sl"]["rf_mult"] >= g["entry"]["rf_mult"],
        "message": "SL Range Filter mult should be >= Entry RF mult (wider SL)"
    },

    # TP genome coherence
    {
        "name": "tp_dual_rr_range",
        "rule": lambda g: 0.5 <= g["tp_dual"]["rr_mult"] <= 5.0,
        "message": "Dual Flip RR should be between 0.5 and 5.0"
    },
    {
        "name": "tp_rsi_rr_range",
        "rule": lambda g: 0.5 <= g["tp_rsi"]["rr_mult"] <= 5.0,
        "message": "RSI RR should be between 0.5 and 5.0"
    },

    # Mode coherence
    {
        "name": "at_least_one_mode",
        "rule": lambda g: g["mode"]["showDualFlip"] or g["mode"]["showRSI"],
        "message": "At least one entry mode must be enabled"
    },

    # Cross-genome coherence
    {
        "name": "tp_st_vs_sl_st",
        "rule": lambda g: g["tp_dual"]["st_atrPeriod"] <= g["sl"]["st_atrPeriod"] * 2,
        "message": "TP ATR period should not be too large compared to SL"
    },
]


# ═══════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════

def validate_genome(genome: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a genome against all coherence rules.

    Args:
        genome: Structured genome dictionary with entry, sl, tp_dual, tp_rsi, mode

    Returns:
        (is_valid, list_of_violations)
    """
    violations = []

    # Ensure genome has required structure
    required_blocks = ["entry", "sl", "tp_dual", "tp_rsi", "mode"]
    for block in required_blocks:
        if block not in genome:
            violations.append(f"Missing genome block: {block}")
            return False, violations

    # Check each rule
    for rule_def in COHERENCE_RULES:
        try:
            if not rule_def["rule"](genome):
                violations.append(f"[{rule_def['name']}] {rule_def['message']}")
        except (KeyError, TypeError) as e:
            violations.append(f"[{rule_def['name']}] Rule check failed: {e}")

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_genome_batch(genomes: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate a batch of genomes.

    Returns:
        (valid_genomes, invalid_genomes_with_reasons)
    """
    valid = []
    invalid = []

    for genome in genomes:
        is_valid, violations = validate_genome(genome)
        if is_valid:
            valid.append(genome)
        else:
            invalid.append({
                "genome": genome,
                "violations": violations
            })

    logger.info(f"Coherence validation: {len(valid)} valid, {len(invalid)} invalid")
    return valid, invalid


# ═══════════════════════════════════════════════════════
# GENOME REPAIR
# ═══════════════════════════════════════════════════════

def repair_genome(genome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to repair a genome to make it coherent.

    Applies fixes for common violations.

    Args:
        genome: Potentially invalid genome

    Returns:
        Repaired genome (may still be invalid if unfixable)
    """
    repaired = {
        "entry": dict(genome.get("entry", {})),
        "sl": dict(genome.get("sl", {})),
        "tp_dual": dict(genome.get("tp_dual", {})),
        "tp_rsi": dict(genome.get("tp_rsi", {})),
        "mode": dict(genome.get("mode", {})),
    }

    # Fix: RSI MA <= RSI length
    if repaired["entry"].get("rsi_ma_length", 6) > repaired["entry"].get("rsi_length", 14):
        repaired["entry"]["rsi_ma_length"] = repaired["entry"]["rsi_length"] - 2

    # Fix: SL mult >= Entry mult
    if repaired["sl"].get("st_mult", 4) < repaired["entry"].get("st_mult", 2):
        repaired["sl"]["st_mult"] = repaired["entry"]["st_mult"] + 2

    if repaired["sl"].get("rf_mult", 7) < repaired["entry"].get("rf_mult", 3):
        repaired["sl"]["rf_mult"] = repaired["entry"]["rf_mult"] + 3

    # Fix: RF period >= ST ATR * 5
    min_rf = repaired["entry"].get("st_atrPeriod", 10) * 5
    if repaired["entry"].get("rf_period", 100) < min_rf:
        repaired["entry"]["rf_period"] = min_rf

    # Fix: At least one mode enabled
    if not repaired["mode"].get("showDualFlip") and not repaired["mode"].get("showRSI"):
        repaired["mode"]["showDualFlip"] = True

    # Fix: RR ranges
    for tp_block in ["tp_dual", "tp_rsi"]:
        rr = repaired[tp_block].get("rr_mult", 1.3)
        repaired[tp_block]["rr_mult"] = max(0.5, min(5.0, rr))

    # Fix: ST mult range
    st_mult = repaired["entry"].get("st_mult", 2.0)
    repaired["entry"]["st_mult"] = max(0.5, min(5.0, st_mult))

    # Fix: RF mult range
    rf_mult = repaired["entry"].get("rf_mult", 3.0)
    repaired["entry"]["rf_mult"] = max(1.0, min(10.0, rf_mult))

    return repaired


def repair_and_validate(genome: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Attempt to repair a genome and validate the result.

    Args:
        genome: Input genome

    Returns:
        (is_valid, repaired_genome, remaining_violations)
    """
    repaired = repair_genome(genome)
    is_valid, violations = validate_genome(repaired)
    return is_valid, repaired, violations


# ═══════════════════════════════════════════════════════
# PARAM BOUNDS
# ═══════════════════════════════════════════════════════

# Valid ranges for each parameter
PARAM_BOUNDS = {
    "entry": {
        "st_atrPeriod": (1, 20),
        "st_mult": (0.5, 5.0),
        "rf_period": (30, 200),
        "rf_mult": (1.0, 10.0),
        "rsi_length": (5, 30),
        "rsi_ma_length": (2, 20),
    },
    "sl": {
        "st_atrPeriod": (1, 20),
        "st_mult": (2.0, 10.0),
        "rf_period": (30, 200),
        "rf_mult": (3.0, 15.0),
    },
    "tp_dual": {
        "st_atrPeriod": (1, 20),
        "st_mult": (0.5, 5.0),
        "rr_mult": (0.5, 5.0),
    },
    "tp_rsi": {
        "st_atrPeriod": (1, 20),
        "st_mult": (0.5, 5.0),
        "rr_mult": (0.5, 5.0),
    },
}


def clamp_to_bounds(genome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clamp all genome parameters to valid bounds.

    Args:
        genome: Input genome

    Returns:
        Genome with all params clamped to bounds
    """
    clamped = {
        "entry": dict(genome.get("entry", {})),
        "sl": dict(genome.get("sl", {})),
        "tp_dual": dict(genome.get("tp_dual", {})),
        "tp_rsi": dict(genome.get("tp_rsi", {})),
        "mode": dict(genome.get("mode", {})),
    }

    for block, bounds in PARAM_BOUNDS.items():
        for param, (min_val, max_val) in bounds.items():
            if param in clamped[block]:
                val = clamped[block][param]
                clamped[block][param] = max(min_val, min(max_val, val))

    return clamped


# ═══════════════════════════════════════════════════════
# CUSTOM PARAM BOUNDS FROM DASHBOARD
# ═══════════════════════════════════════════════════════

def extract_param_bounds_from_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Extract parameter bounds from Dashboard config.

    Dashboard format:
    {
        "st_atrPeriod": {"start": 8, "end": 14, "step": 2},
        "sl_st_atrPeriod": {"start": 10, "end": 10, "step": 0},
        ...
    }

    Returns bounds dict:
    {
        "entry": {
            "st_atrPeriod": (8, 14),
            "st_mult": (1.5, 3.0),
            ...
        },
        "sl": {
            "st_atrPeriod": (10, 10),
            ...
        },
        ...
    }

    Args:
        cfg: Dashboard config with parameter ranges

    Returns:
        Bounds dict compatible with genome_optimizer
    """
    bounds = {
        "entry": {},
        "sl": {},
        "tp_dual": {},
        "tp_rsi": {},
    }

    # Mapping: Dashboard param name → (block, genome param name)
    param_mapping = {
        # Entry
        "st_atrPeriod": ("entry", "st_atrPeriod"),
        "st_mult": ("entry", "st_mult"),
        "rf_period": ("entry", "rf_period"),
        "rf_mult": ("entry", "rf_mult"),
        "rsi_length": ("entry", "rsi_length"),
        "rsi_ma_length": ("entry", "rsi_ma_length"),

        # Stop Loss
        "sl_st_atrPeriod": ("sl", "st_atrPeriod"),
        "sl_st_mult": ("sl", "st_mult"),
        "sl_rf_period": ("sl", "rf_period"),
        "sl_rf_mult": ("sl", "rf_mult"),

        # TP Dual Flip
        "tp_dual_st_atrPeriod": ("tp_dual", "st_atrPeriod"),
        "tp_dual_st_mult": ("tp_dual", "st_mult"),
        "tp_dual_rr_mult": ("tp_dual", "rr_mult"),

        # TP RSI
        "tp_rsi_st_atrPeriod": ("tp_rsi", "st_atrPeriod"),
        "tp_rsi_st_mult": ("tp_rsi", "st_mult"),
        "tp_rsi_rr_mult": ("tp_rsi", "rr_mult"),
    }

    # Extract bounds from config (check both cfg.paramBounds and top-level cfg)
    param_bounds_source = cfg.get("paramBounds", cfg)  # Try paramBounds first, fallback to cfg

    for dashboard_param, (block, genome_param) in param_mapping.items():
        if dashboard_param in param_bounds_source and isinstance(param_bounds_source[dashboard_param], dict):
            param_config = param_bounds_source[dashboard_param]
            start = param_config.get("start")
            end = param_config.get("end")

            if start is not None and end is not None:
                # Use min/max to handle reversed ranges
                min_val = min(float(start), float(end))
                max_val = max(float(start), float(end))

                # Expand bounds slightly to allow mutation beyond range
                range_size = max_val - min_val
                expansion = max(range_size * 0.2, 1.0)  # 20% expansion or minimum 1.0

                min_val = max(0.5, min_val - expansion)  # Don't go below 0.5
                max_val = max_val + expansion

                bounds[block][genome_param] = (min_val, max_val)

    # Merge with default bounds for missing params
    for block in bounds:
        if block in PARAM_BOUNDS:
            for param, default_bound in PARAM_BOUNDS[block].items():
                if param not in bounds[block]:
                    bounds[block][param] = default_bound

    logger.info(f"Extracted custom param bounds from config: {bounds}")
    return bounds


def get_param_bounds() -> Dict[str, Any]:
    """Get parameter bounds for UI/documentation."""
    return PARAM_BOUNDS


# ═══════════════════════════════════════════════════════
# AUTO-EXPAND BOUNDS FROM MEMORY
# ═══════════════════════════════════════════════════════

def expand_bounds_from_memory(
    memory_genomes: List[Dict[str, Any]],
    user_bounds: Dict[str, Any] = None,
    expansion_margin: float = 0.1
) -> Dict[str, Any]:
    """
    Automatically expand bounds to include good genomes from memory.

    When memory contains genomes with high scores that fall outside
    user-defined bounds, this function expands the bounds to include them.

    Args:
        memory_genomes: List of genome records from ParamMemory
            Each record should have 'genome' and optionally 'results.score'
        user_bounds: Optional user-defined bounds. If None, uses PARAM_BOUNDS
        expansion_margin: Additional margin to add beyond the memory values (default 10%)

    Returns:
        Expanded bounds dictionary with same structure as PARAM_BOUNDS
    """
    import copy

    # Start with user bounds or default bounds
    expanded = copy.deepcopy(user_bounds) if user_bounds else copy.deepcopy(PARAM_BOUNDS)

    if not memory_genomes:
        return expanded

    # Track which params were expanded for logging
    expansions = []

    # Iterate through memory genomes
    for record in memory_genomes:
        genome = record.get("genome", record)  # Support both record format and raw genome
        score = record.get("results", {}).get("score", 0) if isinstance(record.get("results"), dict) else 0

        # Only expand for reasonably good genomes (score > 0)
        if score <= 0 and "results" in record:
            continue

        # Check each block and param
        for block in ["entry", "sl", "tp_dual", "tp_rsi"]:
            if block not in genome or block not in expanded:
                continue

            block_data = genome[block]
            block_bounds = expanded[block]

            for param, value in block_data.items():
                if param not in block_bounds:
                    continue

                # Skip non-numeric values
                if not isinstance(value, (int, float)):
                    continue

                current_min, current_max = block_bounds[param]
                param_range = current_max - current_min
                margin = param_range * expansion_margin

                # Expand min if needed
                if value < current_min:
                    new_min = value - margin
                    # Ensure new_min doesn't go below 0 for most params
                    new_min = max(0, new_min) if param not in ["rr_mult"] else max(0.1, new_min)
                    expanded[block][param] = (new_min, current_max)
                    expansions.append(f"{block}.{param} min: {current_min:.2f} → {new_min:.2f}")

                # Expand max if needed
                if value > current_max:
                    new_max = value + margin
                    expanded[block][param] = (expanded[block][param][0], new_max)
                    expansions.append(f"{block}.{param} max: {current_max:.2f} → {new_max:.2f}")

    if expansions:
        logger.info(f"Auto-expanded bounds from memory genomes: {expansions}")

    return expanded


def merge_bounds(
    base_bounds: Dict[str, Any],
    user_ranges: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge user-specified ranges with base bounds.

    User ranges can be:
    - List [min, max]: Use as bounds
    - Single value: Use as both min and max
    - None: Use base bounds

    Args:
        base_bounds: Base PARAM_BOUNDS
        user_ranges: User-specified ranges from config

    Returns:
        Merged bounds
    """
    import copy
    merged = copy.deepcopy(base_bounds)

    if not user_ranges:
        return merged

    # Map from flat param names to genome structure
    param_mapping = {
        "st_atrPeriod": ("entry", "st_atrPeriod"),
        "st_mult": ("entry", "st_mult"),
        "rf_period": ("entry", "rf_period"),
        "rf_mult": ("entry", "rf_mult"),
        "rsi_length": ("entry", "rsi_length"),
        "rsi_ma_length": ("entry", "rsi_ma_length"),
        "sl_st_atrPeriod": ("sl", "st_atrPeriod"),
        "sl_st_mult": ("sl", "st_mult"),
        "sl_rf_period": ("sl", "rf_period"),
        "sl_rf_mult": ("sl", "rf_mult"),
        "tp_dual_st_atrPeriod": ("tp_dual", "st_atrPeriod"),
        "tp_dual_st_mult": ("tp_dual", "st_mult"),
        "tp_dual_rr_mult": ("tp_dual", "rr_mult"),
        "tp_rsi_st_atrPeriod": ("tp_rsi", "st_atrPeriod"),
        "tp_rsi_st_mult": ("tp_rsi", "st_mult"),
        "tp_rsi_rr_mult": ("tp_rsi", "rr_mult"),
    }

    for param_name, values in user_ranges.items():
        if param_name not in param_mapping:
            continue

        block, param = param_mapping[param_name]

        if isinstance(values, list) and len(values) >= 2:
            # List of values - use min/max as bounds
            merged[block][param] = (min(values), max(values))
        elif isinstance(values, (int, float)):
            # Single value - narrow bounds
            merged[block][param] = (values, values)

    return merged


def get_effective_bounds(
    user_config: Dict[str, Any],
    memory_genomes: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get effective bounds considering user config and memory.

    Priority:
    1. Start with default PARAM_BOUNDS
    2. Apply user-specified ranges (narrow or expand)
    3. Auto-expand to include good memory genomes

    Args:
        user_config: User configuration with strategy.params
        memory_genomes: Optional list of genomes from memory

    Returns:
        Effective bounds to use for optimization
    """
    # Start with default bounds
    bounds = get_param_bounds()

    # Apply user ranges if specified
    user_params = user_config.get("strategy", {}).get("params", {})
    if user_params:
        bounds = merge_bounds(bounds, user_params)

    # Expand from memory if available
    if memory_genomes:
        bounds = expand_bounds_from_memory(memory_genomes, bounds)

    return bounds


# ═══════════════════════════════════════════════════════
# QUICK CHECKS
# ═══════════════════════════════════════════════════════

def is_valid_quick(genome: Dict[str, Any]) -> bool:
    """
    Quick validation check (no detailed violation list).

    Args:
        genome: Genome to check

    Returns:
        True if genome is valid
    """
    is_valid, _ = validate_genome(genome)
    return is_valid


def count_violations(genome: Dict[str, Any]) -> int:
    """
    Count number of rule violations.

    Args:
        genome: Genome to check

    Returns:
        Number of violations
    """
    _, violations = validate_genome(genome)
    return len(violations)
