from itertools import product


def build_runs(cfg):
    """
    Build all run configs from Level 6 search space
    Supports both array params (for optimization) and scalar params (fixed values)
    """

    symbols = cfg["symbols"]
    timeframes = cfg["timeframes"]

    strategy = cfg["strategy"]
    strat_type = strategy["type"]
    param_space = strategy["params"]

    windows = cfg.get("windows", [None])

    runs = []

    # Separate array params (to iterate) from scalar params (fixed)
    array_keys = []
    array_values = []
    scalar_params = {}

    for k, v in param_space.items():
        if isinstance(v, list):
            array_keys.append(k)
            array_values.append(v)
        else:
            # Boolean, int, float, string - fixed value
            scalar_params[k] = v

    # If no array params, create single combo with empty tuple
    if not array_values:
        array_values = [()]
        array_keys = []

    for symbol, tf, combo, window in product(
        symbols,
        timeframes,
        product(*array_values) if array_values and array_values != [()] else [()],
        windows
    ):
        # Build params: combine array combo with scalar params
        params = dict(zip(array_keys, combo)) if array_keys else {}
        params.update(scalar_params)

        run_cfg = {
            "meta": {
                "symbols": [symbol],
                "timeframe": tf
            },
            "strategy": {
                "type": strat_type,
                "params": params
            },
            "window": window,
            "costs": cfg.get("costs", {}),
            "range": cfg.get("range"),
            "properties": cfg.get("properties", {}),
        }

        runs.append(run_cfg)

    return runs
