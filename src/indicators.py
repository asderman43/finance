import numpy as np
import pandas as pd

def calc_macd(closing: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ma(closing, fast, "ema")
    ema_slow = calc_ma(closing, slow, "ema")
    
    macd_line = ema_fast - ema_slow
    
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }
    
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    return (series - mean) / std

def calc_ma(closing: pd.Series, n=50, type="sma"):
    match type:
        case "sma":
            windows  = closing.rolling(n)
            moving_averages = windows.mean()
            
            return moving_averages
        
        case "ema":
            moving_averages = closing.ewm(span=n, adjust=False).mean()

            return moving_averages
        case _:
            raise Exception(f"Type must be ema, or sma! Unkown type: {type}")

def calc_log_rt(closing: pd.Series):
    return np.log(closing / closing.shift(1))

def calc_rvol(log_rt: pd.Series, n=30):
    """
        Calc realized volatility from std of log return
    """
    return log_rt.rolling(n).std() * np.sqrt(n)

def calc_rsi(closing: pd.Series, period=14):
    # overbought(>70) vs underbought(<30)
    # only in trading ranges (not momentum)
    change = closing.diff()
    
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    avg_loss = avg_loss if avg_loss != 0 else 1.0
    
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    
    return rsi

def calc_adx(high, low, close, period=14):
    """Average Directional Index (Wilder).

    Returns a DataFrame with plus_di, minus_di, and adx. ADX measures
    trend *strength* regardless of direction (0-100); +DI/-DI give
    direction. Uses Wilder's smoothing (EMA with alpha = 1/period).
    """
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})