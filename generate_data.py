#!/usr/bin/env python3
"""
Generate data.json for the momentum dashboard
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

def fetch_stock_data(ticker):
    """Fetch and analyze data for a single ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        info = stock.info
        
        # Current price
        hist_full = stock.history(period="1d")
        current_price = hist_full['Close'].iloc[-1] if not hist_full.empty else np.nan
        
        # 52 week range
        week_52_high = info.get('fiftyTwoWeekHigh', np.nan)
        week_52_low = info.get('fiftyTwoWeekLow', np.nan)
        
        # Percentage changes
        changes = {}
        if len(hist) >= 2:
            changes['daily'] = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
        if len(hist) >= 5:
            changes['weekly'] = (hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100
        if len(hist) >= 21:
            changes['monthly'] = (hist['Close'].iloc[-1] - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100
        
        # YTD
        today = datetime.today()
        year_start = datetime(today.year, 1, 1)
        hist_ytd = stock.history(start=year_start, end=today)
        changes['ytd'] = (hist['Close'].iloc[-1] - hist_ytd['Close'].iloc[0]) / hist_ytd['Close'].iloc[0] * 100 if not hist_ytd.empty else np.nan
        
        # Yearly
        if len(hist) >= 252:
            changes['yearly'] = (hist['Close'].iloc[-1] - hist['Close'].iloc[-252]) / hist['Close'].iloc[-252] * 100
        else:
            changes['yearly'] = np.nan
        
        # Technical indicators
        # RSI
        def compute_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        rsi = compute_rsi(hist['Close'], 14).iloc[-1] if len(hist) >= 14 else np.nan
        
        # MACD
        def compute_macd(series, fast=12, slow=26, signal=9):
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            hist = macd - signal_line
            return macd, signal_line, hist
        
        if len(hist) >= 26:
            macd_line, signal_line, macd_hist = compute_macd(hist['Close'])
            macd_val = macd_line.iloc[-1]
            macd_signal_val = signal_line.iloc[-1]
            macd_hist_val = macd_hist.iloc[-1]
        else:
            macd_val = macd_signal_val = macd_hist_val = np.nan
        
        # Moving averages
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else np.nan
        ma_100 = hist['Close'].rolling(window=100).mean().iloc[-1] if len(hist) >= 100 else np.nan
        ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else np.nan
        
        # Volume trend
        if len(hist) >= 40:
            vol_recent = hist['Volume'].iloc[-20:].mean()
            vol_prior = hist['Volume'].iloc[-40:-20].mean()
            vol_trend = (vol_recent - vol_prior) / vol_prior * 100 if vol_prior != 0 else np.nan
        else:
            vol_trend = np.nan
        
        # Support/Resistance
        recent_low = hist['Low'].rolling(window=20).min().iloc[-1] if len(hist) >= 20 else np.nan
        recent_high = hist['High'].rolling(window=20).max().iloc[-1] if len(hist) >= 20 else np.nan
        
        # Determine action and factors
        action = "Watch"
        action_class = "action-watch"
        rsi_val = rsi if not pd.isna(rsi) else 50
        monthly_change = changes.get('monthly', 0) if not pd.isna(changes.get('monthly')) else 0
        macd_hist = macd_hist_val if not pd.isna(macd_hist_val) else 0
        
        if rsi_val < 30 and monthly_change < -10:
            action = "Accumulate (Oversold)"
            action_class = "action-buy"
        elif rsi_val > 70 and monthly_change > 10:
            action = "Consider Profit Taking (Overbought)"
            action_class = "action-sell"
        elif macd_hist > 0 and monthly_change > 5:
            action = "Buy (Positive Momentum)"
            action_class = "action-buy"
        elif macd_hist < 0 and monthly_change < -5:
            action = "Sell / Avoid (Negative Momentum)"
            action_class = "action-sell"
        
        # Bullish/Bearish factors
        bullish = []
        bearish = []
        if rsi_val < 30:
            bullish.append("RSI oversold (<30)")
        elif rsi_val > 70:
            bearish.append("RSI overbought (>70)")
        if monthly_change < -15:
            bullish.append("Significant monthly decline (>15% down)")
        elif monthly_change > 15:
            bearish.append("Strong monthly advance (>15% up)")
        if macd_hist > 0:
            bullish.append("MACD histogram positive")
        elif macd_hist < 0:
            bearish.append("MACD histogram negative")
        if not pd.isna(vol_trend) and vol_trend > 20:
            bullish.append("Volume trending up (>20%)")
        elif not pd.isna(vol_trend) and vol_trend < -20:
            bearish.append("Volume trending down (>20%)")
        if not pd.isna(ma_200) and current_price > ma_200 * 1.1:
            bullish.append("Price > 200-day MA by 10%+")
        elif not pd.isna(ma_200) and current_price < ma_200 * 0.9:
            bearish.append("Price < 200-day MA by 10%+")
        
        # Entry/exit levels
        entry_low = entry_high = stop_loss = target1 = target2 = rr = None
        if not pd.isna(current_price) and not pd.isna(recent_low):
            entry_low = max(recent_low * 0.98, current_price * 0.95)
            entry_high = min(recent_high * 1.02, current_price * 1.05)
            stop_loss = recent_low * 0.95
            risk = current_price - stop_loss
            target1 = current_price + risk
            target2 = current_price + 2 * risk
            if risk > 0 and target1 != current_price:
                rr = risk / (target1 - current_price)
        
        return {
            'ticker': ticker,
            'current_price': float(current_price) if not pd.isna(current_price) else None,
            'week_52_high': float(week_52_high) if not pd.isna(week_52_high) else None,
            'week_52_low': float(week_52_low) if not pd.isna(week_52_low) else None,
            'changes': {k: float(v) if not pd.isna(v) else None for k, v in changes.items()},
            'rsi': float(rsi) if not pd.isna(rsi) else None,
            'macd': float(macd_val) if not pd.isna(macd_val) else None,
            'macd_signal': float(macd_signal_val) if not pd.isna(macd_signal_val) else None,
            'macd_hist': float(macd_hist_val) if not pd.isna(macd_hist_val) else None,
            'ma_50': float(ma_50) if not pd.isna(ma_50) else None,
            'ma_100': float(ma_100) if not pd.isna(ma_100) else None,
            'ma_200': float(ma_200) if not pd.isna(ma_200) else None,
            'vol_trend': float(vol_trend) if not pd.isna(vol_trend) else None,
            'recent_low': float(recent_low) if not pd.isna(recent_low) else None,
            'recent_high': float(recent_high) if not pd.isna(recent_high) else None,
            'action': action,
            'action_class': action_class,
            'bullish': bullish,
            'bearish': bearish,
            'entry_low': float(entry_low) if entry_low is not None else None,
            'entry_high': float(entry_high) if entry_high is not None else None,
            'stop_loss': float(stop_loss) if stop_loss is not None else None,
            'target1': float(target1) if target1 is not None else None,
            'target2': float(target2) if target2 is not None else None,
            'risk_reward': float(rr) if rr is not None else None,
            'info': info  # Keep for any extra info if needed
        }
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}

def main():
    # Curated list based on Sam's preferences
    tickers = [
        "F", "T", "PFE", "SOFI", "NVDA", "PLTR", "TSLA", "RKLB", "NVO", "CRSP",
        "JEPI", "JEPQ", "SCHD", "CEW", "XLK", "XLE", "XLF", "XLRE", "GLD", "SLV"
    ]
    
    print(f"Analyzing {len(tickers)} tickers...")
    results = []
    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = fetch_stock_data(ticker)
        results.append(data)
    
    # Save to file
    with open('data.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete! Data saved as data.json")

if __name__ == "__main__":
    main()