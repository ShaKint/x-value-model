# streamlit_app.py
# Real Value+ X™ — Smart Scanner UI (FMP) + Model-driven ranking + Thematic/Sector/Industry filters
# + Undervaluation range + Financial Strength + Profile classification + PIE charts

import os
import time
import math
import pandas as pd
import streamlit as st

# ===== Try importing from investing_scanner.py =====
try:
    from investing_scanner import (
        FMPClient,
        compute_context,
        render_report_markdown,
    )
    try:
        from investing_scanner import (
            model_fields_from_yaml, parse_model_spec, score_candidates_by_model,
            passes_constraints
        )
    except Exception:
        # Minimal fallbacks if advanced helpers not found
        def model_fields_from_yaml(model_yaml_path: str):
            import yaml
            with open(model_yaml_path, 'r', encoding='utf-8') as f:
                model = yaml.safe_load(f)
            keymap = {
                'מחיר נוכחי': 'Last','שווי שוק': 'MarketCap','RSI14': 'RSI14',
                'GapToMA200': 'GapToMA200','MA50': 'MA50','MA200': 'MA200','BeatRatio': 'BeatRatio',
            }
            fields = []
            for ch in model.get('chapters', []):
                for it in ch.get('items', []):
                    key = keymap.get(it.get('id') or it.get('label') or '')
                    if key and key not in fields:
                        fields.append(key)
            return fields or ['GapToMA200', 'RSI14', 'MarketCap', 'BeatRatio']

        def parse_model_spec(model_yaml_path: str):
            fields = model_fields_from_yaml(model_yaml_path)
            return fields, {}, []

        def score_candidates_by_model(ctx_list, fields, weights=None):
            weights = weights or {}
            cols = {f: [] for f in fields}
            for c in ctx_list:
                for f in fields:
                    v = c.get(f)
                    if isinstance(v, (int, float)) and not math.isnan(v):
                        cols[f].append(float(v))
            stats = {f: {'min': (min(vs) if vs else 0.0), 'max': (max(vs) if vs else 0.0)} for f, vs in cols.items()}

            def norm_up(x, f):
                a, b = stats[f]['min'], stats[f]['max']
                if b - a == 0: return 0.5
                return (x - a) / (b - a)
            def norm_down(x, f): return 1.0 - norm_up(x, f)

            scored = []
            for c in ctx_list:
                score = 0.0; wsum = 0.0
                for f in fields:
                    v = c.get(f)
                    if not isinstance(v, (int, float)) or math.isnan(v):
                        continue
                    w = float(weights.get(f, 1.0))
                    if f in ('MarketCap','BeatRatio'): part = norm_up(v, f)
                    elif f == 'RSI14':                part = norm_down(v, f)
                    elif f == 'GapToMA200':           part = norm_down(abs(v), f)
                    else:                              part = norm_up(v, f)
                    score += w*part; wsum += w
                c['Score'] = (score / wsum) if wsum else float('nan')
                scored.append(c)
            scored.sort(key=lambda x: (float('-inf') if math.isnan(x.get('Score', float('nan'))) else -x['Score']))
            return scored

        def passes_constraints(ctx, constraints):
            if not constraints: return True
            for field, mn, mx in constraints:
                v = ctx.get(field)
                if not isinstance(v, (int, float)) or math.isnan(v):
                    return False
                if mn is not None and v < mn: return False
                if mx is not None and v > mx: return False
            return True

except Exception as e:
    st.error(f"לא מצליח לייבא את investing_scanner.py: {e}")
    st.stop()

# ===== Helpers =====
def get_universe(fmp: FMPClient, universe: str) -> list[str]:
    uni = (universe or 'sp500').lower()
    if uni == 'sp500':
        data = fmp._get('/sp500_constituent')
        return [x['symbol'] for x in data if x.get('symbol')]
    if uni == 'nasdaq100':
        data = fmp._get('/nasdaq_constituent')
        return [x['symbol'] for x in data if x.get('symbol')]
    return []  # custom CSV handled via upload

def classify_profile(ctx: dict) -> str:
    """Assign best-fit profile (V1/E1/P1/S1/C1) — heuristic based on your spec."""
    g   = ctx.get("GapToMA200")
    rsi = ctx.get("RSI14")
    mc  = ctx.get("MarketCap") or 0
    br  = ctx.get("BeatRatio") or 0
    sec = (ctx.get("Sector") or "").lower()
    eps_pos = ctx.get("Profitable")

    # C1 — Core Legacy
    if mc >= 50e9:
        return "C1"
    # S1 — Strategic Compounders
    if (mc >= 10e9) and (eps_pos is True) and (br >= 50) and (g is not None and g <= 35):
        return "S1"
    # P1 — Breakout / Momentum with Fundamentals
    if (g is not None and 0 <= g <= 20) and (rsi is not None and 40 <= rsi <= 70):
        return "P1"
    # E1 — Emerging / Catalyst
    hot_keywords = ["ai","semi","semiconductor","defense","space","quantum","cyber","data center"]
    if any(k in sec for k in hot_keywords) or br >= 60:
        return "E1"
    # V1 — Deep Value
    if (g is not None and g <= -20) or (br >= 50 and mc <= 5e9):
        return "V1"
    return "V1"

def augment_financial_strength(fmp: FMPClient, ctx: dict, ticker: str) -> dict:
    """
    Adds Cash, Debt, DebtCash, OCF_TTM, FCF_TTM, MonthlyBurn, RunwayMonths, FinancialStrength.
    Uses FMP balance-sheet-statement & cash-flow-statement (quarter, last 4).
    """
    try:
        bs = fmp._get(f"/balance-sheet-statement/{ticker}", params={"period":"quarter","limit":4})
    except Exception:
        bs = []
    try:
        cf = fmp._get(f"/cash-flow-statement/{ticker}", params={"period":"quarter","limit":4})
    except Exception:
        cf = []

    # Cash / Debt
    cash = debt = None
    if isinstance(bs, list) and bs:
        b0 = bs[0]
        cash = (b0.get('cashAndCashEquivalents') or 0) + (b0.get('shortTermInvestments') or 0)
        if b0.get('totalDebt') is not None:
            debt = b0.get('totalDebt')
        else:
            debt = (b0.get('shortTermDebt') or 0) + (b0.get('longTermDebt') or 0)
    ctx['Cash'] = float(cash or 0.0)
    ctx['Debt'] = float(debt or 0.0)
    ctx['DebtCash'] = (ctx['Debt'] / ctx['Cash']) if ctx['Cash'] > 0 else float('inf')

    # OCF/FCF TTM
    ocf_ttm = fcf_ttm = monthly_burn = runway_m = None
    if isinstance(cf, list) and cf:
        ocf_ttm = 0.0
        capex_ttm = 0.0
        for x in cf[:4]:
            ocf_ttm += float(x.get('netCashProvidedByOperatingActivities') or x.get('operatingCashFlow') or 0.0)
            capex_ttm += float(x.get('capitalExpenditure') or 0.0)
        fcf_ttm = ocf_ttm - capex_ttm
        monthly_burn = max(0.0, -ocf_ttm / 12.0)
        runway_m = (ctx['Cash'] / monthly_burn) if monthly_burn and monthly_burn > 0 else float('inf')

    ctx['OCF_TTM'] = None if ocf_ttm is None else float(ocf_ttm)
    ctx['FCF_TTM'] = None if fcf_ttm is None else float(fcf_ttm)
    ctx['MonthlyBurn'] = None if monthly_burn is None else float(monthly_burn)
    ctx['RunwayMonths'] = None if runway_m is None else (float(runway_m) if runway_m != float('inf') else 10_000.0)

    # Flag — financial strength (thresholds applied later in UI)
    # default thresholds (can be overridden from UI): runway>=18 months AND (Debt/Cash<=1.0 or NetCash>=0)
    net_cash = ctx['Cash'] - ctx['Debt']
    strong = False
    if ctx['OCF_TTM'] is not None and ctx['OCF_TTM'] >= 0:
        strong = True
    elif ctx['RunwayMonths'] is not None and ctx['RunwayMonths'] >= 18:
        strong = True
    # leverage check
    if strong:
        strong = (ctx['DebtCash'] <= 1.0) or (net_cash >= 0)
    ctx['FinancialStrength'] = strong
    return ctx

# === Thematic definitions (as per your spec) ===
THEMATIC = {
    "AI/Edge": ["ai","edge","inference","accelerator","ml","machine learning"],
    "Data Center/Connectivity": ["data center","ethernet","optic","interconnect","connect","network","switch"],
    "Power/Storage": ["power","battery","storage","charger","invert","energy"],
    "Auto/ADAS": ["auto","automotive","adas","ev","lidar","sensor","autonomous"],
    "Defense/Space": ["defense","aerospace","space","nasa","dod","satellite","missile"],
    "CleanTech": ["solar","wind","renewable","cleantech","green","emission"],
}

# ===== UI =====
st.set_page_config(page_title="Real Value+ X — Smart Scanner", layout="wide")
st.title("🔎 Real Value+ X™ — Smart Scanner (FMP)")

# Top controls
col1, col2, col3 = st.columns(3)
with col1:
    default_key = os.getenv("FMP_API_KEY") or ""
    fmp_key = st.text_input("FMP API Key", value=default_key, type="password")
with col2:
    model_path = st.text_input("Model YAML", value="models/real_value_x_2025.yaml")
with col3:
    universe_choice = st.selectbox("Universe", ["sp500", "nasdaq100", "custom CSV"])

uploaded_csv = None
if universe_choice == "custom CSV":
    uploaded_file = st.file_uploader("Upload CSV with 'Ticker' column", type=["csv"])
    if uploaded_file is not None:
        uploaded_csv = pd.read_csv(uploaded_file)
        st.write("דוגמה מהקובץ:", uploaded_csv.head())

st.divider()
st.subheader("סוג מניה, סקטור/תמה, תעשייה ותמחור")

style_choice = st.radio("סוג מניה", ["הכל", "Value (Undervalued)", "Momentum"], horizontal=True)

SECTORS = [
    "Technology","Communication Services","Consumer Discretionary","Consumer Staples",
    "Financials","Health Care","Industrials","Energy","Materials","Real Estate","Utilities","Information Technology"
]
sector_choices = st.multiselect("Sectors (אופציונלי)", SECTORS, default=[])

thematic_sel = st.multiselect("סקטורים/תמות (כפי שהגדרת)", list(THEMATIC.keys()), default=[])

COMMON_INDUSTRIES = [
    "Semiconductors", "Software—Infrastructure", "Software—Application", "Consumer Electronics",
    "Aerospace & Defense", "Communication Equipment", "Internet Content & Information",
    "Auto Manufacturers", "Medical Devices", "Biotechnology", "Banks—Diversified",
    "Oil & Gas Integrated", "Utilities—Regulated Electric", "Specialty Retail",
]
industry_choices = st.multiselect("Industries (אופציונלי)", COMMON_INDUSTRIES, default=[])
industry_keywords_raw = st.text_input("מילות מפתח ל-Industry (מופרדות בפסיק)", value="", placeholder="semi, software, telecom")
industry_keywords = [w.strip().lower() for w in industry_keywords_raw.split(",") if w.strip()]

# Undervaluation controls
use_under_range = st.checkbox("סנן לפי תמחור חסר (טווח % מול MA200)", value=(style_choice=="Value (Undervalued)"))
colu1, colu2 = st.columns(2)
with colu1:
    under_lo = st.number_input("טווח תמחור חסר — מ-%", value=-30.0, step=1.0, help="למשל -30")
with colu2:
    under_hi = st.number_input("טווח תמחור חסר — עד-%", value=-15.0, step=1.0, help="למשל -15 (פחות שלילי)")
# Profitability
profit_choice = st.radio("רווחיות", ["הכל", "רווחית (EPS>0)", "לא רווחית (EPS≤0)"], horizontal=True)

st.subheader("מסננים כלליים")
fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
with fcol1: min_cap = st.number_input("Min MarketCap ($)", value=300_000_000.0, step=100_000_000.0, format="%.0f")
with fcol2: max_rsi = st.number_input("Max RSI(14)", value=70.0, step=1.0)
with fcol3: max_gap = st.number_input("Max Gap to MA200 (%)", value=25.0, step=1.0)
with fcol4: min_vol = st.number_input("Min Avg Volume", value=300_000.0, step=50_000.0, format="%.0f")
with fcol5: limit = st.number_input("Limit", value=50, step=10)

st.subheader("איתנות פיננסית")
ff1, ff2, ff3 = st.columns(3)
with ff1:
    fin_mode = st.radio("מסנן", ["הכל", "חייב איתנות", "להוציא חלשות (אם זוהו)"], horizontal=False)
with ff2:
    min_runway = st.number_input("Runway מינימלי (חודשים)", value=18.0, step=3.0)
with ff3:
    max_debtcash = st.number_input("Debt/Cash מקסימלי", value=1.0, step=0.1, format="%.2f")

run_scan = st.button("🚀 Run Scan")

# ===== ACTION =====
if run_scan:
    if not fmp_key:
        st.error("חסר FMP_API_KEY.")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"לא נמצא קובץ מודל: {model_path}")
        st.stop()

    # Prepare clients & universe
    fmp = FMPClient(api_key=fmp_key)
    if universe_choice == "custom CSV":
        if uploaded_csv is None:
            st.error("בחרת custom CSV — צריך להעלות קובץ.")
            st.stop()
        if 'Ticker' in uploaded_csv.columns:
            universe = [str(t).strip().upper() for t in uploaded_csv['Ticker'].dropna().tolist()]
        else:
            first = uploaded_csv.columns[0]
            universe = [str(t).strip().upper() for t in uploaded_csv[first].dropna().tolist()]
    else:
        try:
            universe = get_universe(fmp, universe_choice)
        except Exception as e:
            st.warning(f"שגיאה בשאיבת Universe ({universe_choice}), נעבור לפולבק קצר: {e}")
            universe = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','AVGO','NFLX','COST','ADBE','AMD','QCOM','ORCL','INTC']

    # Model fields/weights/constraints
    fields, weights, constraints = parse_model_spec(model_path)

    rows = []
    progress = st.progress(0.0, text="אוסף נתונים...")
    total = len(universe)
    for i, t in enumerate(universe, 1):
        try:
            ctx = compute_context(fmp, t)  # price/MA/RSI/Gap/MarketCap/BeatRatio/AvgVolume

            # enrich with quote for sector/industry/eps
            try:
                q = fmp.quote(t)
            except Exception:
                q = {}
            ctx['Sector'] = q.get('sector') or q.get('industry') or 'N/A'
            ctx['Industry'] = q.get('industry') or 'N/A'
            eps = q.get('eps')
            ctx['Profitable'] = (float(eps) > 0) if isinstance(eps, (int, float)) else None

            # add financial strength metrics
            ctx = augment_financial_strength(fmp, ctx, t)

            # model constraints
            if not passes_constraints(ctx, constraints):
                pass
            else:
                ok = True

                # sectors
                if sector_choices:
                    ok &= ctx['Sector'] in sector_choices

                # thematic tags (match in Sector+Industry strings)
                if thematic_sel:
                    s_lower = (str(ctx['Sector']) + " " + str(ctx['Industry'])).lower()
                    ok &= any(any(kw in s_lower for kw in THEMATIC[tag]) for tag in thematic_sel)

                # industries
                if industry_choices:
                    ok &= ctx['Industry'] in industry_choices
                if industry_keywords:
                    ind_lower = str(ctx['Industry']).lower()
                    ok &= any(kw in ind_lower for kw in industry_keywords)

                # style
                g = ctx.get('GapToMA200')
                last = ctx.get('Last')
                ma50 = ctx.get('MA50')

                if style_choice == "Value (Undervalued)":
                    if g is None or math.isnan(g):
                        ok = False
                    else:
                        # if range filter on — enforce between under_lo and under_hi (both negative)
                        if use_under_range:
                            lo = min(under_lo, under_hi)
                            hi = max(under_lo, under_hi)
                            ok &= (lo <= g <= hi)
                        else:
                            ok &= (g <= -float(0))  # allow any undervaluation
                elif style_choice == "Momentum":
                    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [g, last, ma50]):
                        ok = False
                    else:
                        ok &= (g >= 0) and (last >= ma50)
                else:
                    # 'הכל' — אם ביקשת טווח undervaluation, החיל אותו
                    if use_under_range and g is not None and not math.isnan(g):
                        lo = min(under_lo, under_hi)
                        hi = max(under_lo, under_hi)
                        ok &= (lo <= g <= hi)

                # profitability
                if profit_choice == "רווחית (EPS>0)":
                    ok &= (ctx['Profitable'] is True)
                elif profit_choice == "לא רווחית (EPS≤0)":
                    ok &= (ctx['Profitable'] is False)

                # general filters
                if min_cap and (ctx.get('MarketCap') or 0) < min_cap: ok = False
                if max_rsi and not math.isnan(ctx.get('RSI14', float('nan'))) and ctx['RSI14'] > max_rsi: ok = False
                if max_gap and not math.isnan(ctx.get('GapToMA200', float('nan'))) and ctx['GapToMA200'] > max_gap: ok = False
                if min_vol and (ctx.get('AvgVolume') or 0) < min_vol: ok = False

                # financial strength filter with thresholds
                # recompute flag with chosen thresholds
                net_cash = (ctx['Cash'] - ctx['Debt'])
                strong = False
                if ctx['OCF_TTM'] is not None and ctx['OCF_TTM'] >= 0:
                    strong = True
                elif ctx['RunwayMonths'] is not None and ctx['RunwayMonths'] >= float(min_runway):
                    strong = True
                if strong:
                    strong = (ctx['DebtCash'] <= float(max_debtcash)) or (net_cash >= 0)
                ctx['FinancialStrength'] = strong

                if fin_mode == "חייב איתנות":
                    ok &= (ctx['FinancialStrength'] is True)
                elif fin_mode == "להוציא חלשות (אם זוהו)":
                    # אם זוהתה חלשה -> להוציא; אם חסר מידע -> להשאיר
                    if ctx['FinancialStrength'] is False:
                        ok = False

                if ok:
                    ctx['Profile'] = classify_profile(ctx)
                    rows.append(ctx)
        except Exception:
            pass

        if i % 5 == 0:
            progress.progress(min(i/total, 1.0), text=f"אוסף נתונים... {i}/{total}")
        if len(rows) >= max(int(limit)*4, 200):
            break
        time.sleep(0.05)

    progress.progress(1.0, text="חישוב ודירוג...")

    scored = score_candidates_by_model(rows, fields, weights)
    top = scored[:int(limit)]
    df = pd.DataFrame(top)

    # Arrange columns
    preferred_cols = ['Ticker','Profile','Score','Last','RSI14','GapToMA200','MA50','MA200',
                      'MarketCap','BeatRatio','AvgVolume',
                      'Sector','Industry','Profitable',
                      'Cash','Debt','DebtCash','OCF_TTM','FCF_TTM','MonthlyBurn','RunwayMonths','FinancialStrength']
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df.loc[:, cols] if not df.empty else df

    st.subheader("תוצאות (מדורגות לפי המודל)")
    st.dataframe(df, use_container_width=True)

    # CSV Download
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ הורד CSV", data=csv_bytes, file_name="model_candidates.csv", mime="text/csv")

    # PIE charts
    if not df.empty:
        import plotly.express as px
        st.divider()
        st.subheader("📊 פילוחים מהירים")

        colA, colB = st.columns(2)
        with colA:
            sec_counts = df['Sector'].value_counts(dropna=False).reset_index()
            sec_counts.columns = ['Sector','Count']
            fig_sec = px.pie(sec_counts, values='Count', names='Sector', title="Sector Breakdown", hole=0.3)
            st.plotly_chart(fig_sec, use_container_width=True)

        with colB:
            prof_counts = df['Profile'].value_counts(dropna=False).reset_index()
            prof_counts.columns = ['Profile','Count']
            fig_prof = px.pie(prof_counts, values='Count', names='Profile', title="Profile Breakdown", hole=0.3)
            st.plotly_chart(fig_prof, use_container_width=True)

    # Analyze
    if not df.empty:
        st.divider()
        st.subheader("ניתוח עומק (Analyze) למניה שנבחרה")
        ticker = st.selectbox("בחר טיקר לניתוח", df['Ticker'].tolist())
        out_md = st.text_input("Output path", value=f"out/{ticker}_report.md")
        go = st.button("📝 הפק דוח Analyze")
        if go:
            # Prefer reusing context from scan results
            ctx = None
            for c in top:
                if c.get('Ticker') == ticker:
                    ctx = c
                    break
            if ctx is None:
                fmp2 = FMPClient(api_key=fmp_key)
                ctx = compute_context(fmp2, ticker)
                ctx = augment_financial_strength(fmp2, ctx, ticker)
            md = render_report_markdown(model_path, ctx)
            os.makedirs(os.path.dirname(out_md), exist_ok=True)
            with open(out_md, 'w', encoding='utf-8') as f:
                f.write(md)
            st.success(f"נוצר דוח: {out_md}")
            st.markdown(md)
