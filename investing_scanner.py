# investing_scanner.py — סורק עם ספקים: Polygon/FMP/AlphaVantage + yfinance fallback
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from data_providers import DataHub

# ---------- אינדיקטורים ----------
def _rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna().astype(float)
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])

def _sma(series: pd.Series, window: int) -> float:
    s = series.dropna().astype(float)
    if len(s) < window:
        return float("nan")
    return float(s.rolling(window).mean().iloc[-1])

@dataclass
class ScanRow:
    Ticker: str
    Profile: str
    Score: float
    Last: float
    RSI14: float
    GapToMA200: float
    MA50: float
    MA200: float
    MarketCap: Optional[float]
    BeatRatio: Optional[float]
    AvgVolume: Optional[float]
    Sector: Optional[str]
    Industry: Optional[str]
    Profitable: Optional[bool]
    Cash: Optional[float]
    Debt: Optional[float]
    DebtCash: Optional[float]
    OCF_TTM: Optional[float]
    FCF_TTM: Optional[float]

def fetch_all(ticker: str, hub: DataHub) -> Dict[str, Optional[float]]:
    close = hub.get_price_history(ticker, days=400)
    last = float(close.iloc[-1])
    ma50 = _sma(close, 50)
    ma200 = _sma(close, 200)
    gap_to_ma200 = (last - ma200) / ma200 * 100.0 if ma200 and not math.isnan(ma200) and ma200 != 0 else float("nan")
    rsi14 = _rsi(close, 14)

    f = hub.get_core_fundamentals(ticker)
    cash = f.get("Cash")
    debt = f.get("Debt")
    dc = None
    if cash not in (None, 0) and debt is not None:
        try:
            dc = float(debt) / float(cash)
        except Exception:
            dc = None

    ocf = f.get("OCF_TTM")
    capex = f.get("Capex_TTM")
    fcf = None
    if ocf is not None and capex is not None:
        fcf = float(ocf) - abs(float(capex))

    sur = f.get("Surprises") or []
    beat_ratio = None
    if sur:
        beat_ratio = float(sum(1 for x in sur if x > 0)) / float(len(sur))

    # רווחיות: נשתמש בקירוב — FCF חיובי או העדר נתון = None
    profitable = None
    if fcf is not None:
        profitable = bool(fcf > 0)

    return dict(
        Last=last, MA50=ma50, MA200=ma200, GapToMA200=gap_to_ma200, RSI14=rsi14,
        MarketCap=f.get("MarketCap"), AvgVolume=f.get("AvgVolume"),
        Sector=f.get("Sector"), Industry=f.get("Industry"),
        Cash=cash, Debt=debt, DebtCash=dc, OCF_TTM=ocf, FCF_TTM=fcf,
        Profitable=profitable, BeatRatio=beat_ratio
    )

def score_row(row: Dict[str, Optional[float]]) -> float:
    score = 0.0
    gap = row.get("GapToMA200")
    if gap is not None and not math.isnan(gap):
        score += max(-10.0, min(0.0, gap)) / 10.0 * 0.35
    rsi = row.get("RSI14")
    if rsi is not None and not math.isnan(rsi):
        score += (1.0 - min(1.0, abs(rsi - 50.0) / 50.0)) * 0.20
    dc = row.get("DebtCash")
    if dc is not None and not math.isnan(dc):
        score += (1.0 - min(1.0, dc / 3.0)) * 0.20
    fcf = row.get("FCF_TTM")
    if fcf is not None and not math.isnan(fcf):
        score += (1.0 if fcf > 0 else 0.0) * 0.15
    prof = row.get("Profitable")
    if prof is not None:
        score += (0.10 if prof else 0.0)
    return round(float(score), 4)

def scan_universe(
    tickers: List[str], profile: str = "V1",
    debt_cash_max: Optional[float] = None, must_profitable: bool = False,
    limit: Optional[int] = 100
) -> pd.DataFrame:
    hub = DataHub()
    rows: List[ScanRow] = []
    for t in tickers:
        try:
            d = fetch_all(t, hub)
        except Exception:
            continue
        if debt_cash_max is not None and d.get("DebtCash") is not None:
            if d["DebtCash"] > float(debt_cash_max):
                continue
        if must_profitable and d.get("Profitable") is not None and not d["Profitable"]:
            continue
        d["Score"] = score_row(d)
        rows.append(ScanRow(
            Ticker=t, Profile=profile, Score=d["Score"],
            Last=d["Last"], RSI14=d["RSI14"], GapToMA200=d["GapToMA200"],
            MA50=d["MA50"], MA200=d["MA200"], MarketCap=d["MarketCap"],
            BeatRatio=d["BeatRatio"], AvgVolume=d["AvgVolume"], Sector=d["Sector"],
            Industry=d["Industry"], Profitable=d["Profitable"], Cash=d["Cash"],
            Debt=d["Debt"], DebtCash=d["DebtCash"], OCF_TTM=d["OCF_TTM"], FCF_TTM=d["FCF_TTM"],
        ))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df
    df = df.sort_values("Score", ascending=False)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)
