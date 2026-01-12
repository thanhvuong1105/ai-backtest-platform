// rust_engine/src/strategies.rs
//! Trading strategies implemented in Rust
//! Each strategy must match Python implementation exactly

use std::collections::HashMap;
use crate::indicators::*;

/// Strategy trait - all strategies implement this
pub trait Strategy {
    /// Initialize and calculate indicators
    fn prepare(&mut self, opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]);

    /// Check for entry signal at bar index i
    /// Returns true if should enter position
    fn check_entry(&self, i: usize) -> bool;

    /// Check for exit signal at bar index i
    /// Returns true if should exit position
    fn check_exit(&self, i: usize) -> bool;

    /// Get entry type string for logging
    fn entry_type(&self) -> &str;
}

/// EMA Cross Strategy
/// Entry: Fast EMA crosses above Slow EMA
/// Exit: Fast EMA crosses below Slow EMA
pub struct EMACrossStrategy {
    ema_fast_period: usize,
    ema_slow_period: usize,
    ema_fast: Vec<f64>,
    ema_slow: Vec<f64>,
}

impl EMACrossStrategy {
    pub fn new(params: &HashMap<String, f64>) -> Self {
        let ema_fast_period = params.get("emaFast").copied().unwrap_or(10.0) as usize;
        let ema_slow_period = params.get("emaSlow").copied().unwrap_or(30.0) as usize;

        Self {
            ema_fast_period,
            ema_slow_period,
            ema_fast: vec![],
            ema_slow: vec![],
        }
    }
}

impl Strategy for EMACrossStrategy {
    fn prepare(&mut self, _opens: &[f64], _highs: &[f64], _lows: &[f64], closes: &[f64]) {
        self.ema_fast = calc_ema(closes, self.ema_fast_period);
        self.ema_slow = calc_ema(closes, self.ema_slow_period);
    }

    fn check_entry(&self, i: usize) -> bool {
        if i == 0 {
            return false;
        }

        let fast_prev = self.ema_fast[i - 1];
        let slow_prev = self.ema_slow[i - 1];
        let fast_curr = self.ema_fast[i];
        let slow_curr = self.ema_slow[i];

        // Crossover: fast was below or equal, now above
        fast_prev <= slow_prev && fast_curr > slow_curr
    }

    fn check_exit(&self, i: usize) -> bool {
        if i == 0 {
            return false;
        }

        let fast_prev = self.ema_fast[i - 1];
        let slow_prev = self.ema_slow[i - 1];
        let fast_curr = self.ema_fast[i];
        let slow_curr = self.ema_slow[i];

        // Crossunder: fast was above or equal, now below
        fast_prev >= slow_prev && fast_curr < slow_curr
    }

    fn entry_type(&self) -> &str {
        "Long"
    }
}

/// Create strategy from type name and params
pub fn create_strategy(strategy_type: &str, params: &HashMap<String, f64>) -> Box<dyn Strategy + Send> {
    match strategy_type {
        "ema_cross" => Box::new(EMACrossStrategy::new(params)),
        // Add more strategies here
        _ => panic!("Unknown strategy type: {}", strategy_type),
    }
}
