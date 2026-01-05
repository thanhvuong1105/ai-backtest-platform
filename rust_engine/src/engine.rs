// rust_engine/src/engine.rs
//! Core backtest engine implemented in Rust
//! Matches Python engine/backtest_engine.py logic exactly

use std::collections::HashMap;
use crate::{BacktestConfig, BacktestResult, Trade, EquityPoint, Summary};
use crate::strategies::{create_strategy, Strategy};

/// Warmup bars for indicator calculation (matches Python INDICATOR_WARMUP_BARS)
const INDICATOR_WARMUP_BARS: usize = 500;

/// Convert timestamp to UTC+7 Ho Chi Minh timezone string
fn to_hcm_time(time_str: &str) -> String {
    // For now, just pass through the time string
    // Python handles timezone conversion at data loading
    time_str.to_string()
}

/// Parse date string to comparable format (YYYY-MM-DD)
fn parse_date(date_str: &str) -> Option<String> {
    // Extract YYYY-MM-DD part
    if date_str.len() >= 10 {
        Some(date_str[..10].to_string())
    } else {
        None
    }
}

/// Check if time is >= from_date
fn is_after_or_equal(time: &str, from_date: &str) -> bool {
    let time_date = &time[..10.min(time.len())];
    time_date >= from_date
}

/// Check if time is <= to_date
fn is_before_or_equal(time: &str, to_date: &str) -> bool {
    let time_date = &time[..10.min(time.len())];
    time_date <= to_date
}

/// Calculate tick size from price data
fn calc_tick_size(closes: &[f64]) -> f64 {
    if closes.is_empty() {
        return 1.0;
    }

    let sample = closes[0];
    let s = format!("{:.10}", sample);
    let s = s.trim_end_matches('0').trim_end_matches('.');

    if let Some(pos) = s.find('.') {
        let decimals = s.len() - pos - 1;
        10f64.powi(-(decimals as i32))
    } else {
        1.0
    }
}

/// Run a single backtest
pub fn run_backtest_internal(
    config: &BacktestConfig,
    times: &[String],
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) -> BacktestResult {
    let n = times.len();

    // Empty result helper
    let empty_result = || BacktestResult {
        meta: create_meta(config),
        summary: empty_summary(config.initial_equity),
        equity_curve: vec![],
        trades: vec![],
    };

    if n == 0 {
        return empty_result();
    }

    // Calculate tick size
    let tick_size = calc_tick_size(closes);

    // Find execution range indices
    let (start_idx, end_idx, execution_start_idx) = find_execution_range(
        times,
        &config.from_date,
        &config.to_date,
    );

    if start_idx >= n {
        return empty_result();
    }

    // Slice data for indicator calculation (with warmup buffer)
    let times_slice = &times[start_idx..=end_idx.min(n - 1)];
    let opens_slice = &opens[start_idx..=end_idx.min(n - 1)];
    let highs_slice = &highs[start_idx..=end_idx.min(n - 1)];
    let lows_slice = &lows[start_idx..=end_idx.min(n - 1)];
    let closes_slice = &closes[start_idx..=end_idx.min(n - 1)];

    // Create and prepare strategy
    let mut strategy = create_strategy(&config.strategy_type, &config.strategy_params);
    strategy.prepare(opens_slice, highs_slice, lows_slice, closes_slice);

    // Run backtest loop
    let (trades, equity_curve) = run_backtest_loop(
        config,
        &*strategy,
        times_slice,
        opens_slice,
        closes_slice,
        execution_start_idx - start_idx,
        tick_size,
    );

    // Calculate metrics
    let summary = calculate_metrics(&trades, &equity_curve, config.initial_equity);

    BacktestResult {
        meta: create_meta(config),
        summary,
        equity_curve,
        trades,
    }
}

/// Find execution range indices based on date range
fn find_execution_range(
    times: &[String],
    from_date: &Option<String>,
    to_date: &Option<String>,
) -> (usize, usize, usize) {
    let n = times.len();

    // Filter by to_date first
    let end_idx = if let Some(ref to_d) = to_date {
        times.iter().rposition(|t| is_before_or_equal(t, to_d)).unwrap_or(n - 1)
    } else {
        n - 1
    };

    // Find execution start (from_date)
    let execution_start_idx = if let Some(ref from_d) = from_date {
        times.iter().position(|t| is_after_or_equal(t, from_d)).unwrap_or(0)
    } else {
        0
    };

    // Calculate start index with warmup buffer
    let start_idx = execution_start_idx.saturating_sub(INDICATOR_WARMUP_BARS);

    (start_idx, end_idx, execution_start_idx)
}

/// Main backtest loop
fn run_backtest_loop(
    config: &BacktestConfig,
    strategy: &dyn Strategy,
    times: &[String],
    opens: &[f64],
    closes: &[f64],
    execution_start_idx: usize,
    tick_size: f64,
) -> (Vec<Trade>, Vec<EquityPoint>) {
    let n = times.len();
    let mut equity = config.initial_equity;
    let mut trades: Vec<Trade> = vec![];
    let mut equity_curve: Vec<EquityPoint> = vec![];
    let mut open_trades: Vec<Trade> = vec![];
    let mut pending_entry = false;

    let fee_rate = config.fee_rate;
    let slippage_ticks = config.slippage_ticks;
    let order_type = &config.order_type;
    let order_value = config.order_value;
    let pyramiding = config.pyramiding;
    let compound = config.compound;
    let initial_equity = config.initial_equity;

    for i in 0..n {
        let time = &times[i];
        let time_hcm = to_hcm_time(time);
        let price_open = opens[i];
        let price_close = closes[i];

        let in_execution_range = i >= execution_start_idx;

        // Reset pending_entry when entering execution range
        if i == execution_start_idx {
            pending_entry = false;
        }

        // Execute pending entry at Open of current bar
        if pending_entry && open_trades.len() < pyramiding && in_execution_range {
            let slip = slippage_ticks * tick_size;
            let entry_price = price_open + slip;

            // Calculate position size
            let current_exposure: f64 = open_trades.iter()
                .map(|t| t.entry_price * t.size)
                .sum();

            let base_equity = if compound {
                equity - current_exposure
            } else {
                (initial_equity - current_exposure).max(0.0)
            };

            let position_size = match order_type.as_str() {
                "percent" => ((order_value / 100.0) * base_equity / entry_price).max(0.0),
                "usdt" | "fixed" => (order_value / entry_price).max(0.0),
                _ => ((order_value / 100.0) * base_equity / entry_price).max(0.0),
            };

            let entry_fee = entry_price * fee_rate * position_size;
            equity -= entry_fee;

            let new_trade = Trade {
                side: "Long".to_string(),
                entry_type: strategy.entry_type().to_string(),
                entry_time: time_hcm.clone(),
                entry_price: round6(entry_price),
                entry_fee: round6(entry_fee),
                size: position_size,
                notional: round6(entry_price * position_size),
                exit_time: None,
                exit_price: None,
                exit_fee: None,
                pnl: None,
                pnl_pct: None,
            };
            open_trades.push(new_trade);
        }

        pending_entry = false;

        // Check exit at Close of current bar
        if !open_trades.is_empty() && strategy.check_exit(i) {
            // Close first trade (FIFO)
            let mut trade = open_trades.remove(0);
            let slip = slippage_ticks * tick_size;
            let exit_price = price_close - slip;
            let pos_size = trade.size;

            let gross_pnl = (exit_price - trade.entry_price) * pos_size;
            let exit_fee = exit_price * fee_rate * pos_size;
            let pnl = gross_pnl - exit_fee;
            let notional = trade.entry_price * pos_size;
            let pnl_pct = if notional > 0.0 {
                (pnl / notional) * 100.0
            } else {
                0.0
            };

            equity += pnl;

            trade.exit_time = Some(time_hcm.clone());
            trade.exit_price = Some(round6(exit_price));
            trade.exit_fee = Some(round6(exit_fee));
            trade.pnl = Some(round6(pnl));
            trade.pnl_pct = Some(round6(pnl_pct));
            trade.notional = round6(notional);

            trades.push(trade);
        }

        // Check entry signal (will execute at Open of next bar)
        if in_execution_range
            && strategy.check_entry(i)
            && open_trades.len() < pyramiding
            && !pending_entry
        {
            pending_entry = true;
        }

        // Record equity curve for bars in execution range
        if in_execution_range {
            equity_curve.push(EquityPoint {
                time: time_hcm,
                equity: round6(equity),
            });
        }
    }

    // Close remaining open trades at last price
    if !open_trades.is_empty() && !closes.is_empty() {
        let slip = slippage_ticks * tick_size;
        let exit_price = closes[n - 1] - slip;
        let time_hcm = to_hcm_time(&times[n - 1]);

        while let Some(mut trade) = open_trades.pop() {
            let pos_size = trade.size;
            let gross_pnl = (exit_price - trade.entry_price) * pos_size;
            let exit_fee = exit_price * fee_rate * pos_size;
            let pnl = gross_pnl - exit_fee;
            let notional = trade.entry_price * pos_size;
            let pnl_pct = if notional > 0.0 {
                (pnl / notional) * 100.0
            } else {
                0.0
            };

            equity += pnl;

            trade.exit_time = Some(time_hcm.clone());
            trade.exit_price = Some(round6(exit_price));
            trade.exit_fee = Some(round6(exit_fee));
            trade.pnl = Some(round6(pnl));
            trade.pnl_pct = Some(round6(pnl_pct));
            trade.notional = round6(notional);

            trades.push(trade);
        }
    }

    (trades, equity_curve)
}

/// Calculate performance metrics
fn calculate_metrics(trades: &[Trade], equity_curve: &[EquityPoint], initial_equity: f64) -> Summary {
    if trades.is_empty() {
        return empty_summary(initial_equity);
    }

    let pnls: Vec<f64> = trades.iter()
        .filter_map(|t| t.pnl)
        .collect();

    let wins: Vec<f64> = pnls.iter().filter(|&&p| p > 0.0).copied().collect();
    let losses: Vec<f64> = pnls.iter().filter(|&&p| p < 0.0).copied().collect();

    let total_trades = pnls.len();
    let win_trades = wins.len();
    let loss_trades = losses.len();

    let gross_profit: f64 = wins.iter().sum();
    let gross_loss: f64 = losses.iter().map(|x| x.abs()).sum();

    let final_equity = equity_curve.last()
        .map(|e| e.equity)
        .unwrap_or(initial_equity);

    // Drawdown calculation
    let mut peak = initial_equity;
    let mut max_dd = 0.0f64;

    for point in equity_curve {
        peak = peak.max(point.equity);
        let dd = point.equity - peak;
        max_dd = max_dd.min(dd);
    }

    let avg_win = if win_trades > 0 {
        gross_profit / win_trades as f64
    } else {
        0.0
    };

    let avg_loss = if loss_trades > 0 {
        losses.iter().sum::<f64>() / loss_trades as f64
    } else {
        0.0
    };

    let expectancy = if total_trades > 0 {
        let win_rate = win_trades as f64 / total_trades as f64;
        let loss_rate = loss_trades as f64 / total_trades as f64;
        win_rate * avg_win + loss_rate * avg_loss
    } else {
        0.0
    };

    let raw_pf = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        1e9
    };
    let profit_factor = raw_pf.min(1e6);

    Summary {
        initial_equity,
        final_equity,
        net_profit: final_equity - initial_equity,
        net_profit_pct: (final_equity - initial_equity) / initial_equity * 100.0,
        total_trades,
        win_trades,
        loss_trades,
        winrate: if total_trades > 0 {
            win_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        },
        profit_factor,
        avg_trade: if total_trades > 0 {
            pnls.iter().sum::<f64>() / total_trades as f64
        } else {
            0.0
        },
        avg_win,
        avg_loss,
        expectancy,
        max_drawdown: max_dd,
        max_drawdown_pct: if initial_equity > 0.0 {
            max_dd.abs() / initial_equity * 100.0
        } else {
            0.0
        },
    }
}

/// Create empty summary
fn empty_summary(initial_equity: f64) -> Summary {
    Summary {
        initial_equity,
        final_equity: initial_equity,
        net_profit: 0.0,
        net_profit_pct: 0.0,
        total_trades: 0,
        win_trades: 0,
        loss_trades: 0,
        winrate: 0.0,
        profit_factor: 0.0,
        avg_trade: 0.0,
        avg_win: 0.0,
        avg_loss: 0.0,
        expectancy: 0.0,
        max_drawdown: 0.0,
        max_drawdown_pct: 0.0,
    }
}

/// Create meta dictionary
fn create_meta(config: &BacktestConfig) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    meta.insert("strategyId".to_string(), format!("{}_v1", config.strategy_type));
    meta.insert("strategyName".to_string(), config.strategy_type.clone());
    meta.insert("symbol".to_string(), config.symbol.clone());
    meta.insert("timeframe".to_string(), config.timeframe.clone());
    meta
}

/// Round to 6 decimal places
fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}
