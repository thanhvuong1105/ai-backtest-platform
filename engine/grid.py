import itertools


def generate_param_grid(search_space: dict):
    """
    search_space = {
      "emaFast": [8, 10, 12],
      "emaSlow": [20, 25]
    }
    """
    keys = list(search_space.keys())
    values = list(search_space.values())

    combos = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        combos.append(params)

    return combos
