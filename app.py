# app.py — UI + חיבור לסורק אמיתי
import os, hashlib, time
from datetime import datetime
import pandas as pd
import streamlit as st
import yaml

from investing_scanner import scan_universe

st.set_page_config(page_title="Stock Scanner", layout="wide")

# ---------- עזרי קובץ/קש ----------
def file_sig(path: str):
    with open(path, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    mtime = os.path.getmtime(path)
    return sha, mtime

@st.cache_data(show_spinner=False)
def load_yaml(path: str, sig: tuple):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.isfile(path)
    except Exception:
        return False

def human_time(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

# ---------- סיידבר: קבצים ----------
st.sidebar.header("קבצי הגדרות")
default_model = "models/real_value_x_2025.yaml"
fallback_model = "0d27c238-7461-40fb-b1e9-aef81517644d.yaml"

model_path = st.sidebar.text_input(
    "נתיב קובץ מודל (YAML)",
    value=default_model if exists(default_model) else (fallback_model if exists(fallback_model) else "")
)

profiles_path = st.sidebar.text_input("נתיב קובץ פרופילים (YAML, אופציונלי)", value="")

col_refresh, col_clear = st.sidebar.columns(2)
if col_clear.button("ניקוי Cache"):
    st.cache_data.clear()
    st.sidebar.success("נוקה cache")
if col_refresh.button("רענון"):
    st.experimental_rerun()

# ---------- טעינת מודל להצגת חיווי (לא משתמשים בו עדיין לניקוד) ----------
if not model_path or not exists(model_path):
    st.error("לא נמצא קובץ מודל. הגדר נתיב תקין בסיידבר.")
    st.stop()
m_sig = file_sig(model_path)
model_cfg = load_yaml(model_path, m_sig)
st.caption(f"מודל: `{os.path.abspath(model_path)}` | mtime: {human_time(m_sig[1])} | sha256[:8]: {m_sig[0][:8]}")

# ---------- מסננים ----------
st.title("תוצאות (מדורגות לפי המודל)")
mode = st.radio("מסנן", ["כל", "חייב איתנות", "להוציא חלשות (אם זוהו)"], horizontal=True)
c1, c2 = st.columns(2)
runway_min = c1.number_input("Runway (חודשים) מינימלי", min_value=0.0, value=18.0, step=1.0)  # כרגע איננו מחשבים Runway
debt_cash_max = c2.number_input("Debt/Cash מקס׳", min_value=0.0, value=1.0, step=0.1)

# ---------- טעינת יוניברס ----------
# אם יש data/universe.csv עם עמודת Ticker — נשתמש בו; אחרת יוניברס ברירת מחדל קטן.
universe_path = "data/universe.csv"
if os.path.exists(universe_path):
    uni = pd.read_csv(universe_path)
    tickers = [str(x).strip().upper() for x in uni["Ticker"].dropna().tolist()]
else:
    tickers = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
               "REGN","LII","MA","SNPS","HUBB","ROP","MLM","BRK-B","POOL"]

must_profitable = True if mode == "חייב איתנות" else False

if st.button("Run Scan"):
    with st.spinner("מריץ סריקה אמיתית..."):
        df = scan_universe(
            tickers=tickers,
            profile="V1",
            debt_cash_max=float(debt_cash_max) if debt_cash_max > 0 else None,
            must_profitable=must_profitable,
            limit=100,
        )
        if df.empty:
            st.warning("לא התקבלו תוצאות (בדוק/י יוניברס או מסננים קפדניים מדי).")
        else:
            st.success(f"התקבלו {len(df)} תוצאות.")
            st.dataframe(df, use_container_width=True)
            st.download_button("הורד CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="scan_results.csv", mime="text/csv")
else:
    st.info("לחץ/י 'Run Scan' כדי להפעיל סריקה אמיתית על היוניברס.")
