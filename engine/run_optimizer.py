# engine/run_optimizer.py
import sys
import json

from engine.optimizer import optimize

if __name__ == "__main__":
    cfg = json.load(sys.stdin)
    result = optimize(cfg)
    print(json.dumps(result))
