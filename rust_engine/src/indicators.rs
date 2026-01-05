// rust_engine/src/indicators.rs
//! Technical indicators implemented in Rust
//! All indicators match Pine Script / TradingView behavior exactly

/// Calculate EMA (Exponential Moving Average)
/// Matches pandas ewm(span=length, adjust=False).mean()
pub fn calc_ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];
    let multiplier = 2.0 / (period as f64 + 1.0);

    // First value is the data itself (ewm with adjust=False)
    result[0] = data[0];

    for i in 1..data.len() {
        result[i] = data[i] * multiplier + result[i - 1] * (1.0 - multiplier);
    }

    result
}

/// Calculate RMA (Wilder's Moving Average / SMMA)
/// Used by RSI and ATR in Pine Script
pub fn calc_rma(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];
    let alpha = 1.0 / period as f64;

    // First `period` values: use SMA for initialization
    if data.len() >= period {
        let sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        // Subsequent values use RMA formula
        for i in period..data.len() {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
    }

    result
}

/// Calculate SMA (Simple Moving Average)
pub fn calc_sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; data.len()];

    for i in (period - 1)..data.len() {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        result[i] = sum / period as f64;
    }

    result
}

/// Calculate True Range
pub fn calc_true_range(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
    let n = highs.len();
    let mut tr = vec![0.0; n];

    if n == 0 {
        return tr;
    }

    tr[0] = highs[0] - lows[0];

    for i in 1..n {
        let hl = highs[i] - lows[i];
        let hc = (highs[i] - closes[i - 1]).abs();
        let lc = (lows[i] - closes[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    tr
}

/// Calculate ATR (Average True Range)
/// Uses RMA (Wilder's smoothing) by default, like TradingView
pub fn calc_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let tr = calc_true_range(highs, lows, closes);
    calc_rma(&tr, period)
}

/// Calculate RSI (Relative Strength Index)
/// Matches Pine Script ta.rsi() exactly
pub fn calc_rsi(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];
    let mut ups = vec![0.0; n];
    let mut downs = vec![0.0; n];

    // Calculate price changes
    for i in 1..n {
        let change = data[i] - data[i - 1];
        ups[i] = change.max(0.0);
        downs[i] = (-change).max(0.0);
    }

    // Calculate RMA of ups and downs
    let up_rma = calc_rma(&ups[1..].to_vec(), period);
    let down_rma = calc_rma(&downs[1..].to_vec(), period);

    // Calculate RSI
    for i in 0..up_rma.len() {
        let up = up_rma[i];
        let down = down_rma[i];

        if down.is_nan() || up.is_nan() {
            continue;
        }

        result[i + 1] = if down == 0.0 {
            100.0
        } else if up == 0.0 {
            0.0
        } else {
            100.0 - (100.0 / (1.0 + up / down))
        };
    }

    result
}

/// SuperTrend indicator
/// Returns (supertrend_values, trend_direction)
/// trend_direction: 1 = bullish, -1 = bearish
pub fn calc_supertrend(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
    multiplier: f64,
    use_atr: bool,
) -> (Vec<f64>, Vec<i32>) {
    let n = highs.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let tr = calc_true_range(highs, lows, closes);
    let atr = if use_atr {
        calc_rma(&tr, period)
    } else {
        calc_sma(&tr, period)
    };

    let mut st_up = vec![f64::NAN; n];
    let mut st_dn = vec![f64::NAN; n];
    let mut trend = vec![1i32; n];
    let mut supertrend = vec![f64::NAN; n];

    // hl2 source
    let src: Vec<f64> = highs.iter().zip(lows.iter())
        .map(|(h, l)| (h + l) / 2.0)
        .collect();

    for i in 0..n {
        if atr[i].is_nan() {
            continue;
        }

        let up = src[i] - multiplier * atr[i];
        let dn = src[i] + multiplier * atr[i];

        // Smoothing logic matching Pine Script
        if i > 0 && !st_up[i - 1].is_nan() {
            let up_prev = st_up[i - 1];
            let dn_prev = st_dn[i - 1];

            st_up[i] = if closes[i - 1] > up_prev {
                up.max(up_prev)
            } else {
                up
            };

            st_dn[i] = if closes[i - 1] < dn_prev {
                dn.min(dn_prev)
            } else {
                dn
            };

            // Trend direction
            let prev_trend = trend[i - 1];
            if prev_trend == -1 && closes[i] > dn_prev {
                trend[i] = 1;
            } else if prev_trend == 1 && closes[i] < up_prev {
                trend[i] = -1;
            } else {
                trend[i] = prev_trend;
            }
        } else {
            st_up[i] = up;
            st_dn[i] = dn;
            trend[i] = 1;
        }

        supertrend[i] = if trend[i] == 1 { st_up[i] } else { st_dn[i] };
    }

    (supertrend, trend)
}

/// Range Filter indicator
/// Returns (filter_value, high_band, low_band, direction)
/// direction: 1 = up, -1 = down, 0 = flat
pub fn calc_range_filter(
    src: &[f64],
    period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>) {
    let n = src.len();
    if n == 0 {
        return (vec![], vec![], vec![], vec![]);
    }

    // Calculate smoothed range: EMA(EMA(|change|, period), period*2-1) * multiplier
    let mut abs_diff = vec![0.0; n];
    for i in 1..n {
        abs_diff[i] = (src[i] - src[i - 1]).abs();
    }

    let ema1 = calc_ema(&abs_diff, period);
    let ema2 = calc_ema(&ema1, period * 2 - 1);

    let rng: Vec<f64> = ema2.iter().map(|x| x * multiplier).collect();

    let mut filt = vec![f64::NAN; n];
    let mut hband = vec![f64::NAN; n];
    let mut lband = vec![f64::NAN; n];
    let mut direction = vec![0i32; n];

    for i in 0..n {
        if rng[i].is_nan() {
            continue;
        }

        let prev_filt = if i > 0 && !filt[i - 1].is_nan() {
            filt[i - 1]
        } else {
            src[i]
        };

        // Filter calculation matching Pine Script
        filt[i] = if src[i] > prev_filt {
            (src[i] - rng[i]).max(prev_filt)
        } else {
            (src[i] + rng[i]).min(prev_filt)
        };

        hband[i] = filt[i] + rng[i];
        lband[i] = filt[i] - rng[i];

        // Direction
        if i > 0 && !filt[i - 1].is_nan() {
            if filt[i] > filt[i - 1] {
                direction[i] = 1;
            } else if filt[i] < filt[i - 1] {
                direction[i] = -1;
            } else {
                direction[i] = direction[i - 1];
            }
        }
    }

    (filt, hband, lband, direction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = calc_ema(&data, 3);
        assert!((ema[4] - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.5, 43.5, 44.0, 44.5, 44.0, 44.5, 44.0, 44.0, 43.5,
                       44.0, 44.5, 44.5, 45.0, 45.0, 45.5, 46.0, 45.5, 46.0, 46.5];
        let rsi = calc_rsi(&data, 14);
        // RSI should be between 0 and 100
        for val in rsi.iter().skip(14) {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0);
            }
        }
    }
}
