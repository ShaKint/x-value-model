# data_providers.py — עטיפות פשוטות ל-Polygon / FMP / AlphaVantage עם fallback
from __future__ import annotations
import os, time, datetime as dt, math, requests
import pandas as pd

TIMEOUT = 20

def _get_secret(name: str):
    # קורא קודם מ-st.secrets אם יש (כשנרוץ בענן), אחרת מה-ENV
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

def _json(url, params=None):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

class DataHub:
    def __init__(self):
        self.fmp = _get_secret("FMP_API_KEY")
        self.alpha = _get_secret("ALPHAVANTAGE_API_KEY")
        self.polygon = _get_secret("POLYGON_API_KEY")

    # -------- Price History --------
    def get_price_history(self, ticker: str, days: int = 400) -> pd.Series:
        # 1) Polygon (אם יש מפתח)
        if self.polygon:
            try:
                end = dt.date.today()
                start = end - dt.timedelta(days=days*2)  # באפר
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
                js = _json(url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.polygon})
                rows = js.get("results", [])
                if rows:
                    df = pd.DataFrame(rows)
                    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
                    df.set_index("t", inplace=True)
                    close = df["c"].astype(float).rename("Close")
                    return close.tail(days)
            except Exception:
                pass

        # 2) Alpha Vantage
        if self.alpha:
            try:
                url = "https://www.alphavantage.co/query"
                js = _json(url, {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": "full",
                    "apikey": self.alpha
                })
                ts = js.get("Time Series (Daily)", {})
                if ts:
                    df = pd.DataFrame(ts).T
                    df.index = pd.to_datetime(df.index)
                    close = df["5. adjusted close"].astype(float).sort_index().rename("Close")
                    return close.tail(days)
            except Exception:
                pass

        # 3) yfinance כגיבוי אחרון
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=f"{max(days, 250)}d", interval="1d")
            if not hist.empty:
                return hist["Close"].dropna().tail(days)
        except Exception:
            pass

        raise RuntimeError(f"no price history for {ticker}")

    # -------- Fundamentals --------
    def get_core_fundamentals(self, ticker: str) -> dict:
        """cash, debt, ocf_ttm, capex_ttm, sector, industry, marketCap, avgVolume, earnings surprises list"""
        out = dict(Cash=None, Debt=None, OCF_TTM=None, Capex_TTM=None, Sector=None,
                   Industry=None, MarketCap=None, AvgVolume=None, Surprises=[])

        # עדיפות: FMP
        if self.fmp:
            try:
                q = _json(f"https://financialmodelingprep.com/api/v3/quote/{ticker}",
                          {"apikey": self.fmp})
                if isinstance(q, list) and q:
                    q0 = q[0]
                    out["MarketCap"] = q0.get("marketCap")
                    out["AvgVolume"] = q0.get("avgVolume")
                prof = _json(f"https://financialmodelingprep.com/api/v3/profile/{ticker}",
                             {"apikey": self.fmp})
                if isinstance(prof, list) and prof:
                    p0 = prof[0]
                    out["Sector"] = p0.get("sector")
                    out["Industry"] = p0.get("industry")

                bs = _json(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}",
                           {"period": "quarter", "limit": 4, "apikey": self.fmp})
                if isinstance(bs, list) and bs:
                    # מפתחי FMP לפעמים שונים בין חברות → ננסה כמה אפשרויות
                    cands_cash = ["cashAndShortTermInvestments", "cashAndCashEquivalents", "cashAndCashEquivalentsIncludingRestrictedCash"]
                    cands_debt = ["totalDebt", "longTermDebt", "netDebt"]
                    cash, debt = None, None
                    for c in cands_cash:
                        if c in bs[0]:
                            cash = sum([x.get(c) for x in bs[:4] if x.get(c) is not None])
                            break
                    for c in cands_debt:
                        if c in bs[0]:
                            debt = sum([x.get(c) for x in bs[:4] if x.get(c) is not None])
                            break
                    out["Cash"] = cash
                    out["Debt"] = debt

                cf = _json(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}",
                           {"period": "quarter", "limit": 4, "apikey": self.fmp})
                if isinstance(cf, list) and cf:
                    ocf_keys = ["netCashProvidedByOperatingActivities", "operatingCashFlow"]
                    capex_keys = ["capitalExpenditure"]
                    ocf, capex = None, None
                    for k in ocf_keys:
                        if k in cf[0]:
                            ocf = sum([x.get(k) or 0 for x in cf[:4]])
                            break
                    for k in capex_keys:
                        if k in cf[0]:
                            capex = sum([x.get(k) or 0 for x in cf[:4]])
                            break
                    out["OCF_TTM"] = ocf
                    out["Capex_TTM"] = capex

                ec = _json(f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}",
                           {"limit": 8, "apikey": self.fmp})
                if isinstance(ec, list) and ec:
                    sur = []
                    for row in ec[:8]:
                        a = row.get("epsActual")
                        e = row.get("epsEstimate")
                        if a is not None and e is not None:
                            sur.append(a - e)
                    out["Surprises"] = sur
            except Exception:
                pass

        return out
