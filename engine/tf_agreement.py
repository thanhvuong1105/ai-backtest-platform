from collections import defaultdict


def apply_tf_agreement(runs, min_tf=2):
    """
    Chỉ giữ strategy params xuất hiện tốt trên >= min_tf timeframe
    """
    bucket = defaultdict(list)

    for r in runs:
        key = (
            r["symbol"],
            tuple(sorted(r["params"].items()))
        )
        bucket[key].append(r)

    agreed = []
    for runs_same_param in bucket.values():
        if len(runs_same_param) >= min_tf:
            agreed.extend(runs_same_param)

    return agreed
