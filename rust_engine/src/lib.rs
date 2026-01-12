// rust_engine/src/lib.rs
//! High-performance backtest engine implemented in Rust
//! Provides 50-100x speedup over Python implementation
//!
//! Strategies supported:
//! - EMA Cross
//! - RF + ST + RSI (future)

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyReadonlyArray1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod indicators;
mod strategies;
mod engine;

use indicators::*;
use strategies::*;
use engine::*;

/// Trade result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub side: String,
    pub entry_type: String,
    pub entry_time: String,
    pub entry_price: f64,
    pub entry_fee: f64,
    pub size: f64,
    pub notional: f64,
    pub exit_time: Option<String>,
    pub exit_price: Option<f64>,
    pub exit_fee: Option<f64>,
    pub pnl: Option<f64>,
    pub pnl_pct: Option<f64>,
}

/// Equity curve point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub time: String,
    pub equity: f64,
}

/// Backtest summary metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub initial_equity: f64,
    pub final_equity: f64,
    pub net_profit: f64,
    pub net_profit_pct: f64,
    pub total_trades: usize,
    pub win_trades: usize,
    pub loss_trades: usize,
    pub winrate: f64,
    pub profit_factor: f64,
    pub avg_trade: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub expectancy: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
}

/// Backtest result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub meta: HashMap<String, String>,
    pub summary: Summary,
    pub equity_curve: Vec<EquityPoint>,
    pub trades: Vec<Trade>,
}

/// Main Python module
#[pymodule]
fn backtest_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch_backtests_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ema_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_rsi_py, m)?)?;
    Ok(())
}

/// Run a single backtest from Python
///
/// Args:
///     config: Dict with strategy configuration
///     times: numpy array of timestamps (as strings)
///     opens: numpy array of open prices
///     highs: numpy array of high prices
///     lows: numpy array of low prices
///     closes: numpy array of close prices
///
/// Returns:
///     Dict with backtest results
#[pyfunction]
fn run_backtest_py(
    py: Python,
    config: &PyDict,
    times: Vec<String>,
    opens: PyReadonlyArray1<f64>,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let opens = opens.as_slice()?;
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;

    // Parse config
    let config = parse_config(py, config)?;

    // Run backtest
    let result = run_backtest_internal(&config, &times, opens, highs, lows, closes);

    // Convert result to Python dict
    result_to_pydict(py, &result)
}

/// Run multiple backtests in parallel (batch mode)
///
/// Args:
///     configs: List of config dicts
///     times: numpy array of timestamps
///     opens, highs, lows, closes: numpy arrays
///
/// Returns:
///     List of backtest results
#[pyfunction]
fn run_batch_backtests_py(
    py: Python,
    configs: &PyList,
    times: Vec<String>,
    opens: PyReadonlyArray1<f64>,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    use rayon::prelude::*;

    let opens = opens.as_slice()?;
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;

    // Parse all configs
    let parsed_configs: Vec<BacktestConfig> = configs
        .iter()
        .map(|c| parse_config(py, c.downcast::<PyDict>().unwrap()).unwrap())
        .collect();

    // Run backtests in parallel using rayon
    let results: Vec<BacktestResult> = parsed_configs
        .par_iter()
        .map(|config| {
            run_backtest_internal(config, &times, opens, highs, lows, closes)
        })
        .collect();

    // Convert results to Python list
    let py_list = PyList::empty(py);
    for result in results {
        py_list.append(result_to_pydict(py, &result)?)?;
    }

    Ok(py_list.into())
}

/// Calculate EMA (exposed to Python for testing)
#[pyfunction]
fn calc_ema_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let data = data.as_slice()?;
    let result = calc_ema(data, period);
    Ok(PyArray1::from_vec(py, result))
}

/// Calculate RSI (exposed to Python for testing)
#[pyfunction]
fn calc_rsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let data = data.as_slice()?;
    let result = calc_rsi(data, period);
    Ok(PyArray1::from_vec(py, result))
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub symbol: String,
    pub timeframe: String,
    pub strategy_type: String,
    pub strategy_params: HashMap<String, f64>,
    pub initial_equity: f64,
    pub order_type: String,
    pub order_value: f64,
    pub fee_rate: f64,
    pub slippage_ticks: f64,
    pub tick_size: f64,
    pub pyramiding: usize,
    pub compound: bool,
    pub from_date: Option<String>,
    pub to_date: Option<String>,
}

/// Parse Python config dict into BacktestConfig
fn parse_config(py: Python, dict: &PyDict) -> PyResult<BacktestConfig> {
    // Meta
    let meta = dict.get_item("meta")?.unwrap().downcast::<PyDict>()?;
    let symbols: Vec<String> = meta.get_item("symbols")?.unwrap().extract()?;
    let symbol = symbols.first().cloned().unwrap_or_default();
    let timeframe: String = meta.get_item("timeframe")?.unwrap().extract()?;

    // Strategy
    let strategy = dict.get_item("strategy")?.unwrap().downcast::<PyDict>()?;
    let strategy_type: String = strategy.get_item("type")?.unwrap().extract()?;
    let params_dict = strategy.get_item("params")?.unwrap().downcast::<PyDict>()?;

    let mut strategy_params = HashMap::new();
    for (k, v) in params_dict.iter() {
        let key: String = k.extract()?;
        let value: f64 = v.extract()?;
        strategy_params.insert(key, value);
    }

    // Initial equity
    let initial_equity: f64 = dict.get_item("initial_equity")?
        .map(|v| v.extract().unwrap_or(10000.0))
        .unwrap_or(10000.0);

    // Properties
    let props = dict.get_item("properties")?
        .map(|v| v.downcast::<PyDict>().ok())
        .flatten();

    let (order_type, order_value) = if let Some(props) = props {
        if let Some(order_size) = props.get_item("orderSize")? {
            let os = order_size.downcast::<PyDict>()?;
            let otype: String = os.get_item("type")?
                .map(|v| v.extract().unwrap_or("percent".to_string()))
                .unwrap_or("percent".to_string());
            let ovalue: f64 = os.get_item("value")?
                .map(|v| v.extract().unwrap_or(100.0))
                .unwrap_or(100.0);
            (otype, ovalue)
        } else {
            ("percent".to_string(), 100.0)
        }
    } else {
        ("percent".to_string(), 100.0)
    };

    // Costs
    let costs = dict.get_item("costs")?
        .map(|v| v.downcast::<PyDict>().ok())
        .flatten();

    let fee_rate = costs
        .and_then(|c| c.get_item("fee").ok().flatten())
        .map(|v| v.extract().unwrap_or(0.0))
        .unwrap_or(0.0);

    let slippage_ticks = costs
        .and_then(|c| c.get_item("slippage").ok().flatten())
        .map(|v| v.extract().unwrap_or(0.0))
        .unwrap_or(0.0);

    // Range
    let range = dict.get_item("range")?
        .map(|v| v.downcast::<PyDict>().ok())
        .flatten();

    let from_date = range
        .and_then(|r| r.get_item("from").ok().flatten())
        .map(|v| v.extract().ok())
        .flatten();

    let to_date = range
        .and_then(|r| r.get_item("to").ok().flatten())
        .map(|v| v.extract().ok())
        .flatten();

    Ok(BacktestConfig {
        symbol,
        timeframe,
        strategy_type,
        strategy_params,
        initial_equity,
        order_type,
        order_value,
        fee_rate,
        slippage_ticks,
        tick_size: 0.01,  // Will be calculated from data
        pyramiding: 1,
        compound: true,
        from_date,
        to_date,
    })
}

/// Convert BacktestResult to Python dict
fn result_to_pydict(py: Python, result: &BacktestResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    // Meta
    let meta = PyDict::new(py);
    for (k, v) in &result.meta {
        meta.set_item(k, v)?;
    }
    dict.set_item("meta", meta)?;

    // Summary
    let summary = PyDict::new(py);
    summary.set_item("initialEquity", result.summary.initial_equity)?;
    summary.set_item("finalEquity", result.summary.final_equity)?;
    summary.set_item("netProfit", result.summary.net_profit)?;
    summary.set_item("netProfitPct", result.summary.net_profit_pct)?;
    summary.set_item("totalTrades", result.summary.total_trades)?;
    summary.set_item("winTrades", result.summary.win_trades)?;
    summary.set_item("lossTrades", result.summary.loss_trades)?;
    summary.set_item("winrate", result.summary.winrate)?;
    summary.set_item("profitFactor", result.summary.profit_factor)?;
    summary.set_item("avgTrade", result.summary.avg_trade)?;
    summary.set_item("avgWin", result.summary.avg_win)?;
    summary.set_item("avgLoss", result.summary.avg_loss)?;
    summary.set_item("expectancy", result.summary.expectancy)?;
    summary.set_item("maxDrawdown", result.summary.max_drawdown)?;
    summary.set_item("maxDrawdownPct", result.summary.max_drawdown_pct)?;
    dict.set_item("summary", summary)?;

    // Equity curve
    let eq_list = PyList::empty(py);
    for point in &result.equity_curve {
        let p = PyDict::new(py);
        p.set_item("time", &point.time)?;
        p.set_item("equity", point.equity)?;
        eq_list.append(p)?;
    }
    dict.set_item("equityCurve", eq_list)?;

    // Trades
    let trades_list = PyList::empty(py);
    for trade in &result.trades {
        let t = PyDict::new(py);
        t.set_item("side", &trade.side)?;
        t.set_item("entry_type", &trade.entry_type)?;
        t.set_item("entry_time", &trade.entry_time)?;
        t.set_item("entry_price", trade.entry_price)?;
        t.set_item("entry_fee", trade.entry_fee)?;
        t.set_item("size", trade.size)?;
        t.set_item("notional", trade.notional)?;
        if let Some(ref exit_time) = trade.exit_time {
            t.set_item("exit_time", exit_time)?;
        }
        if let Some(exit_price) = trade.exit_price {
            t.set_item("exit_price", exit_price)?;
        }
        if let Some(exit_fee) = trade.exit_fee {
            t.set_item("exit_fee", exit_fee)?;
        }
        if let Some(pnl) = trade.pnl {
            t.set_item("pnl", pnl)?;
        }
        if let Some(pnl_pct) = trade.pnl_pct {
            t.set_item("pnl_pct", pnl_pct)?;
        }
        trades_list.append(t)?;
    }
    dict.set_item("trades", trades_list)?;

    Ok(dict.into())
}
