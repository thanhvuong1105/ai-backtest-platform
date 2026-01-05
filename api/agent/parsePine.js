// api/agent/parsePine.js

export function parsePineStrategy(pineCode) {
  // MOCK logic – sau này AI thật sẽ thay thế
  return {
    meta: {
      name: "Parsed Pine Strategy",
      timeframe: "30m",
      symbols: ["BTCUSDT", "ETHUSDT"]
    },
    entry: {
      long: "mock_long_condition",
      short: "mock_short_condition"
    },
    exit: {
      type: "rr",
      rr: 2
    }
  };
}
