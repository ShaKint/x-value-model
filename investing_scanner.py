# investing_scanner.py
# סורק מניות אמיתי על בסיס yfinance + חישובי אינדיקטורים (RSI, MA, GapToMA200, Cash/Debt, OCF/FCF TTM)
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ---------- עזרי חישוב ----------
def _rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna().astype(float)
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])

def _sma(series: pd.Series, window: int) -> float:
    s = series.dropna().astype(float)
    if len(s) < window:
        return float("nan")
    return float(s.rolling(window).mean().iloc[-1])

def _get_first_match(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for key in candidates:
        k = key.lower()
        for idx_l, idx in idx_lower.items():
            if k in idx_l:
                # קח את הערך האחרון הזמין (עמודה אחרונה)
                val = df.loc[idx].dropna()
                if not val.empty:
                    return float(val.iloc[0])
    return None


# ---------- תוצאת סריקה ----------
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


# ---------- שליפת נתונים וחשבונים ----------
def fetch_fundamentals(ticker: str) -> Dict[str, Optional[float]]:
    t = yf.Ticker(ticker)

    # מחירי היסטוריה ל־MA/RSI/Gap
    hist = t.history(period="400d", interval="1d")
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}")
    close = hist["Close"].dropna()
    last = float(close.iloc[-1])
    ma50 = _sma(close, 50)
    ma200 = _sma(close, 200)
    gap_to_ma200 = float("nan")
    if ma200 and not math.isnan(ma200) and ma200 != 0:
        gap_to_ma200 = (last - ma200) / ma200 * 100.0
    rsi14 = _rsi(close, 14)

    # fast_info: שווי שוק, מחזור ממוצע
    fi = getattr(t, "fast_info", None) or {}
    market_cap = getattr(fi, "market_cap", None) if hasattr(fi, "market_cap") else fi.get("market_cap")
    avg_vol = getattr(fi, "three_month_average_volume", None) if hasattr(fi, "three_month_average_volume") else fi.get("three_month_average_volume")

    # תחום/תעשייה
    try:
        info = t.get_info()  # בגרסאות חדשות במקום .info
    except Exception:
        info = {}
    sector = info.get("sector")
    industry = info.get("industry")

    # מאזנים ותזרים
    try:
        bs_q = t.quarterly_balance_sheet
    except Exception:
        bs_q = pd.DataFrame()

    cash = _get_first_match(bs_q, ["Cash And Cash Equivalents", "Cash And Short Term Investments", "Cash"])
    total_debt = _get_first_match(bs_q, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])

    debt_cash = None
    if cash is not None and cash != 0 and total_debt is not None:
        debt_cash = total_debt / cash

    # תזרים מזומנים TTM (חיבור 4 רבעונים)
    try:
        cf_q = t.quarterly_cashflow
    except Exception:
        cf_q = pd.DataFrame()

    def sum4(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
        if df is None or df.empty:
            return None
        idx_lower = {str(i).lower(): i for i in df.index}
        for key in candidates:
            k = key.lower()
            for idx_l, idx in idx_lower.items():
                if k in idx_l:
                    vals = df.loc[idx].dropna().astype(float)
                    if len(vals) == 0:
                        continue
                    return float(vals.iloc[:4].sum())
        return None

    ocf_ttm = sum4(cf_q, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex_ttm = sum4(cf_q, ["Capital Expenditures", "Investments In Property Plant And Equipment"])
    fcf_ttm = None
    if ocf_ttm is not None and capex_ttm is not None:
        fcf_ttm = ocf_ttm - abs(capex_ttm)

    # רווחיות (אמת אם סך 4 רבעונים של net income חיובי)
    try:
        inc_q = t.quarterly_income_stmt
    except Exception:
        inc_q = pd.DataFrame()
    net_income_ttm = sum4(inc_q, ["Net Income", "Net Income Applicable To Common Shares"])
    profitable = None
    if net_income_ttm is not None:
        profitable = bool(net_income_ttm > 0)

    # BeatRatio (כמה מתוך 6–8 הדוחות האחרונים היו עם הפתעה חיובית)
    beat_ratio = None
    try:
        ed = t.get_earnings_dates(limit=8)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            # yfinance מחזיר עמודות כמו EPS Surprise או Surprise(%)
            cols = [c for c in ed.columns if "Surprise" in str(c)]
            if cols:
                s = ed[cols[0]].dropna()
                if len(s) > 0:
                    wins = (s.astype(float) > 0).sum()
                    beat_ratio = float(wins) / float(len(s))
    except Exception:
        pass

    return dict(
        Last=last,
        MA50=ma50,
        MA200=ma200,
        GapToMA200=gap_to_ma200,
        RSI14=rsi14,
        MarketCap=market_cap,
        AvgVolume=avg_vol,
        Sector=sector,
        Industry=industry,
        Cash=cash,
        Debt=total_debt,
        DebtCash=debt_cash,
        OCF_TTM=ocf_ttm,
        FCF_TTM=fcf_ttm,
        Profitable=profitable,
        BeatRatio=beat_ratio,
    )


def score_row(row: Dict[str, Optional[float]]) -> float:
    """ניקוד פשוט: Gap שלילי (קרוב ל-MA200), RSI סביב 40–60, יחס חוב/מזומן נמוך, FCF חיובי, רווחיות."""
    score = 0.0

    gap = row.get("GapToMA200")
    if gap is not None and not math.isnan(gap):
        # ככל שהמחיר מעט מתחת ל-MA200 עדיף (לא קיצוני)
        score += max(-10.0, min(0.0, gap)) / 10.0 * 0.35  # בין 0 ל-0.35

    rsi = row.get("RSI14")
    if rsi is not None and not math.isnan(rsi):
        # מיטבי סביב 50 → מרחק מ-50
        score += (1.0 - min(1.0, abs(rsi - 50.0) / 50.0)) * 0.20

    dc = row.get("DebtCash")
    if dc is not None and not math.isnan(dc):
        # נמוך יותר טוב; מדלל מעל 3
        score += (1.0 - min(1.0, dc / 3.0)) * 0.20

    fcf = row.get("FCF_TTM")
    if fcf is not None and not math.isnan(fcf):
        score += (1.0 if fcf > 0 else 0.0) * 0.15

    prof = row.get("Profitable")
    if prof is not None:
        score += (0.10 if prof else 0.0)

    return round(float(score), 4)


def scan_universe(
    tickers: List[str],
    profile: str = "V1",
    debt_cash_max: Optional[float] = None,
    must_profitable: bool = False,
    limit: Optional[int] = 100,
) -> pd.DataFrame:
    rows: List[ScanRow] = []
    for t in tickers:
        try:
            d = fetch_fundamentals(t)
        except Exception:
            continue

        if debt_cash_max is not None and d.get("DebtCash") is not None:
            if d["DebtCash"] > debt_cash_max:
                continue

        if must_profitable and d.get("Profitable") is not None:
            if not d["Profitable"]:
                continue

        d["Score"] = score_row(d)

        rows.append(
            ScanRow(
                Ticker=t,
                Profile=profile,
                Score=d["Score"],
                Last=d["Last"],
                RSI14=d["RSI14"],
                GapToMA200=d["GapToMA200"],
                MA50=d["MA50"],
                MA200=d["MA200"],
                MarketCap=d["MarketCap"],
                BeatRatio=d["BeatRatio"],
                AvgVolume=d["AvgVolume"],
                Sector=d["Sector"],
                Industry=d["Industry"],
                Profitable=d["Profitable"],
                Cash=d["Cash"],
                Debt=d["Debt"],
                DebtCash=d["DebtCash"],
                OCF_TTM=d["OCF_TTM"],
                FCF_TTM=d["FCF_TTM"],
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df
    df = df.sort_values("Score", ascending=False)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)
