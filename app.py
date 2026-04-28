import streamlit as st
import pandas as pd
from data_fetcher import fetch_company_data
from analyzer import analyze_financials, generate_advisory
from rag_engine import build_rag_context, generate_advice

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI — Financial Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

/* ── Root ─────────────────────────────────────────────────────────────── */
html, body, .stApp {
    background: #080c14;
    color: #e2e8f0;
    font-family: 'Sora', sans-serif;
}

/* ── Streamlit chrome overrides ────────────────────────────────────────── */
.block-container          { padding: 2rem 3rem 4rem; max-width: 1100px; }
.stButton > button        { border-radius: 8px; font-family: 'Sora', sans-serif; }
.stTextInput input        { background: #0e1420; border: 1px solid #1e2d45;
                            border-radius: 8px; color: #e2e8f0;
                            font-family: 'IBM Plex Mono', monospace; }
.stTextInput input:focus  { border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,.25); }
div[data-testid="stTabs"] button { font-family: 'Sora', sans-serif; }

/* ── Heading bar ───────────────────────────────────────────────────────── */
.fin-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 2rem;
    border-bottom: 1px solid #1a2540;
    margin-bottom: 2rem;
}
.fin-logo { font-size: 28px; }
.fin-title { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
.fin-sub   { font-size: 13px; color: #4a6080; font-family: 'IBM Plex Mono', monospace; }

/* ── Cards ─────────────────────────────────────────────────────────────── */
.card {
    background: #0e1420;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-label {
    font-size: 11px; font-weight: 600; letter-spacing: .08em;
    color: #4a6080; text-transform: uppercase; margin-bottom: 4px;
}
.card-value {
    font-size: 22px; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Metric grid ────────────────────────────────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
.metric-box  {
    background: #0b1019; border: 1px solid #1a2540;
    border-radius: 10px; padding: 16px 18px;
}
.metric-box .lbl { font-size: 11px; color: #4a6080; letter-spacing: .06em; text-transform: uppercase; }
.metric-box .val { font-size: 18px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; margin-top: 4px; }
.metric-box .ctx { font-size: 10px; color: #2d4a6a; margin-top: 2px; }

/* ── Recommendation banner ─────────────────────────────────────────────── */
.rec-banner {
    border-radius: 12px; padding: 20px; margin: 20px 0;
    display: flex; align-items: center; justify-content: space-between;
}
.rec-label { font-size: 13px; color: rgba(255,255,255,.6); }
.rec-text  { font-size: 30px; font-weight: 700; letter-spacing: 2px; }
.conf-pill {
    font-size: 12px; padding: 4px 14px; border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    background: rgba(255,255,255,.12); color: rgba(255,255,255,.8);
}

/* ── Advisory blocks ────────────────────────────────────────────────────── */
.adv-block { border-left: 3px solid; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 12px; }
.adv-block.summary { border-color: #3b82f6; background: rgba(59,130,246,.06); }
.adv-block.action  { border-color: #10b981; background: rgba(16,185,129,.06); }
.adv-block.avoid   { border-color: #f59e0b; background: rgba(245,158,11,.06); }
.adv-block.conf    { border-color: #8b5cf6; background: rgba(139,92,246,.06); }
.adv-icon  { font-size: 16px; margin-right: 8px; }
.adv-title { font-size: 11px; font-weight: 600; letter-spacing: .08em;
             text-transform: uppercase; color: #4a6080; margin-bottom: 6px; }
.adv-body  { font-size: 14px; color: #c4d0e0; line-height: 1.6; }

/* ── Portfolio table ────────────────────────────────────────────────────── */
.port-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 0; border-bottom: 1px solid #1a2540;
    font-family: 'IBM Plex Mono', monospace; font-size: 14px;
}
.port-ticker { color: #60a5fa; font-weight: 600; }
.port-qty    { color: #94a3b8; }

/* ── Search history pills ─────────────────────────────────────────────── */
.hist-pill {
    display: inline-block; background: #0e1420; border: 1px solid #1a2540;
    border-radius: 999px; padding: 4px 14px; margin: 4px;
    font-size: 12px; font-family: 'IBM Plex Mono', monospace;
    color: #60a5fa;
}

/* ── Company name badge ─────────────────────────────────────────────────── */
.company-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0e1420; border: 1px solid #1a2540; border-radius: 8px;
    padding: 8px 16px; margin-bottom: 20px;
}
.company-name   { font-size: 16px; font-weight: 600; }
.company-sector { font-size: 12px; color: #4a6080; font-family: 'IBM Plex Mono', monospace; }

/* ── Analyze button ─────────────────────────────────────────────────────── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white; border: none; font-weight: 600;
    padding: 10px 28px; border-radius: 8px;
    transition: opacity .2s;
}
div[data-testid="stButton"] button[kind="primary"]:hover { opacity: .85; }

/* ── Chart overrides ────────────────────────────────────────────────────── */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_large(v):
    if v is None:
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def _fmt(v, prefix="", suffix=""):
    if v is None:
        return "N/A"
    return f"{prefix}{v}{suffix}"

def _pct(v):
    if v is None:
        return "N/A"
    return f"{round(v * 100, 1)}%"

def _color_val(v, good_above=None, bad_below=None):
    """Return an HTML-colored value string."""
    if v is None:
        return '<span style="color:#4a6080;">N/A</span>'
    disp = str(v)
    if good_above is not None and v >= good_above:
        return f'<span style="color:#4ade80;">{disp}</span>'
    if bad_below is not None and v < bad_below:
        return f'<span style="color:#f87171;">{disp}</span>'
    return f'<span style="color:#e2e8f0;">{disp}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fin-header">
    <span class="fin-logo">📈</span>
    <div>
        <div class="fin-title">FinSight AI</div>
        <div class="fin-sub">equity analysis · fundamental scoring · AI advisory</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT ROW
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 1, 2])

with col1:
    ticker_input = st.text_input(
        "Ticker symbol",
        value="AAPL",
        placeholder="e.g. AAPL · MSFT · RELIANCE.NS",
        label_visibility="collapsed",
    )

with col2:
    analyze_btn = st.button("Analyze →", type="primary", use_container_width=True)

with col3:
    add_to_portfolio = st.checkbox("Add result to portfolio", value=False)

# Recent searches
if st.session_state.history:
    pills = "".join(
        f'<span class="hist-pill">{t}</span>'
        for t in st.session_state.history[-8:]
    )
    st.markdown(f'<div style="margin-bottom:1rem;">{pills}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Fetching data for {ticker} …"):
        try:
            data     = fetch_company_data(ticker)
            analysis = analyze_financials(data)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # Track history & portfolio
    if ticker not in st.session_state.history:
        st.session_state.history.append(ticker)
    if add_to_portfolio:
        if not any(p["ticker"] == ticker for p in st.session_state.portfolio):
            st.session_state.portfolio.append({
                "ticker": ticker,
                "name":   analysis.get("company_name", ""),
                "rec":    analysis["recommendation"],
            })

    currency = analysis.get("currency", "USD")

    # ── Company badge ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="company-badge">
        <span class="company-name">{analysis["company_name"]}</span>
        <span class="company-sector">
            {analysis.get("sector", "N/A")} · {ticker} · {currency}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric grid ──────────────────────────────────────────────────────────
    margin_pct = _pct(analysis["profit_margin"])
    roe_pct    = _pct(analysis["roe"])

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="lbl">Market Cap</div>
            <div class="val">{_fmt_large(analysis["market_cap"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Revenue</div>
            <div class="val">{_fmt_large(analysis["revenue"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Net Profit</div>
            <div class="val">{_fmt_large(analysis["profit"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Profit Margin</div>
            <div class="val">{margin_pct}</div>
            <div class="ctx">sector threshold: {round(analysis["margin_good_thresh"]*100,1)}%</div>
        </div>
        <div class="metric-box">
            <div class="lbl">ROE</div>
            <div class="val">{roe_pct}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">EPS</div>
            <div class="val">{_fmt(analysis["eps"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Debt / Equity</div>
            <div class="val">{_fmt(analysis["debt_to_equity"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Current Ratio</div>
            <div class="val">{_fmt(analysis["current_ratio"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">P/E Ratio</div>
            <div class="val">{_fmt(analysis["pe_ratio"])}</div>
            <div class="ctx">sector high: {analysis["pe_high_thresh"]}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Price / Book</div>
            <div class="val">{_fmt(analysis["price_to_book"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">EV / EBITDA</div>
            <div class="val">{_fmt(analysis["ev_ebitda"])}</div>
        </div>
        <div class="metric-box">
            <div class="lbl">Operating CF</div>
            <div class="val">{_fmt_large(analysis["operating_cf"])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendation banner ────────────────────────────────────────────────
    rec     = analysis["recommendation"]
    conf    = analysis["confidence"]
    risk    = analysis["risk_score"]
    colors  = {"BUY": "#052e16, #166534", "HOLD": "#422006, #92400e", "SELL": "#450a0a, #991b1b"}
    bg      = colors.get(rec, "#0e1420, #1a2540")
    txt_col = {"BUY": "#4ade80", "HOLD": "#fbbf24", "SELL": "#f87171"}[rec]

    st.markdown(f"""
    <div class="rec-banner" style="background: linear-gradient(135deg, {bg}); border: 1px solid {txt_col}33;">
        <div>
            <div class="rec-label">Recommendation</div>
            <div class="rec-text" style="color:{txt_col};">{rec}</div>
        </div>
        <div style="text-align:center;">
            <div class="rec-label">Signals</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; margin-top:4px;">
                <span style="color:#4ade80;">▲ {analysis["buy_signals"]} bullish</span>
                &nbsp;&nbsp;
                <span style="color:#f87171;">▼ {analysis["sell_signals"]} bearish</span>
            </div>
        </div>
        <div style="text-align:right;">
            <div class="rec-label">Risk Score</div>
            <div style="font-size:24px; font-weight:700; font-family:'IBM Plex Mono', monospace; color:{txt_col};">
                {risk:.2f}<span style="font-size:14px;color:{txt_col}88;">/1.00</span>
            </div>
        </div>
        <span class="conf-pill">{conf} confidence</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊  Advisory", "📈  Price Chart", "🤖  AI Insight", "💼  Portfolio"])

    # ── ADVISORY ─────────────────────────────────────────────────────────────
    with tab1:
        advisory = generate_advisory(analysis)
        st.markdown(f"""
        <div class="adv-block summary">
            <div class="adv-title">📌  Analysis</div>
            <div class="adv-body">{advisory["summary"]}</div>
        </div>
        <div class="adv-block action">
            <div class="adv-title">✅  What to Do</div>
            <div class="adv-body">{advisory["action"]}</div>
        </div>
        <div class="adv-block avoid">
            <div class="adv-title">⚠️  What to Avoid</div>
            <div class="adv-body">{advisory["avoid"]}</div>
        </div>
        <div class="adv-block conf">
            <div class="adv-title">🕐  Investment Horizon</div>
            <div class="adv-body">{advisory["horizon"]}</div>
        </div>
        <div class="adv-block conf" style="border-color:#8b5cf6;">
            <div class="adv-title">📡  Signal Confidence</div>
            <div class="adv-body">{advisory["confidence_note"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── PRICE CHART ──────────────────────────────────────────────────────────
    with tab2:
        history = data.get("history")
        if history is not None:
            try:
                import pandas as pd

                close  = history["Close"].dropna()
                volume = history["Volume"].dropna() if "Volume" in history.columns else None

                # Moving averages
                ma50  = close.rolling(50).mean()
                ma200 = close.rolling(200).mean()

                # Build a combined DataFrame for st.line_chart
                chart_df = pd.DataFrame({
                    "Close Price": close,
                    "50-day MA":   ma50,
                    "200-day MA":  ma200,
                })

                st.markdown(
                    f"<div style='font-size:13px;color:#4a6080;margin-bottom:6px;'>"
                    f"📈 {ticker} — 1 Year Price History</div>",
                    unsafe_allow_html=True,
                )
                st.line_chart(chart_df, height=360)

                # Volume bar chart
                if volume is not None:
                    st.markdown(
                        "<div style='font-size:12px;color:#4a6080;margin-top:16px;margin-bottom:4px;'>"
                        "Volume</div>",
                        unsafe_allow_html=True,
                    )
                    st.bar_chart(volume, height=140)

                # Price stats
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current",   f"{close.iloc[-1]:.2f}")
                c2.metric("52W High",  f"{close.max():.2f}")
                c3.metric("52W Low",   f"{close.min():.2f}")
                pct_chg = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
                c4.metric("1Y Return", f"{pct_chg:+.1f}%")

            except Exception as e:
                st.warning(f"Could not render price chart: {e}")
        else:
            st.info("No price history available for this ticker.")

    # ── AI INSIGHT ────────────────────────────────────────────────────────────
    with tab3:
        with st.spinner("Building context & querying Gemini …"):
            # Build rich multi-document RAG context
            rag   = build_rag_context(analysis, data)
            query = (
                f"Should I invest in {ticker} ({analysis.get('company_name', '')})? "
                "Provide a structured analysis covering decision, key reasons, action plan, and risks."
            )
            context_docs = rag.query(query, k=5)
            advice = generate_advice(context_docs, query)

        st.markdown("""
        <div class="card">
            <div class="card-label">🤖 Gemini AI Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(advice)

    # ── PORTFOLIO ─────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Watchlist / Portfolio")

        if not st.session_state.portfolio:
            st.info("No stocks tracked yet. Tick 'Add result to portfolio' before analyzing.")
        else:
            for i, stock in enumerate(st.session_state.portfolio):
                rec_col = {"BUY": "#4ade80", "HOLD": "#fbbf24", "SELL": "#f87171"}.get(
                    stock["rec"], "#94a3b8"
                )
                st.markdown(f"""
                <div class="port-row">
                    <span class="port-ticker">{stock["ticker"]}</span>
                    <span class="port-qty">{stock.get("name", "")}</span>
                    <span style="color:{rec_col}; font-family:'IBM Plex Mono',monospace; font-size:13px;">
                        {stock["rec"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑  Clear portfolio"):
                st.session_state.portfolio = []
                st.rerun()

        st.markdown("---")
        st.markdown("**Quick Add**")
        manual_ticker = st.text_input("Ticker", placeholder="e.g. TSLA", key="manual_ticker")
        if st.button("Add manually") and manual_ticker.strip():
            t = manual_ticker.strip().upper()
            if not any(p["ticker"] == t for p in st.session_state.portfolio):
                st.session_state.portfolio.append({"ticker": t, "name": "", "rec": "N/A"})
                st.success(f"Added {t}")
                st.rerun()

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 0 60px;">
        <div style="font-size:48px; margin-bottom:16px;">📈</div>
        <div style="font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:10px;">
            Enter a stock ticker to begin
        </div>
        <div style="font-size:14px; color:#4a6080; font-family:'IBM Plex Mono',monospace;">
            Supports NSE (add .NS), BSE (.BO), NYSE, NASDAQ symbols
        </div>
    </div>

    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:16px; max-width:900px; margin:0 auto;">
        <div class="card" style="text-align:center;">
            <div style="font-size:24px;">🔍</div>
            <div style="font-weight:600; margin:8px 0 4px;">Fundamental Analysis</div>
            <div style="font-size:13px; color:#4a6080;">Revenue, margins, ROE, debt ratios</div>
        </div>
        <div class="card" style="text-align:center;">
            <div style="font-size:24px;">📈</div>
            <div style="font-weight:600; margin:8px 0 4px;">Price Chart</div>
            <div style="font-size:13px; color:#4a6080;">1Y history, 50 & 200-day MAs, volume</div>
        </div>
        <div class="card" style="text-align:center;">
            <div style="font-size:24px;">🤖</div>
            <div style="font-weight:600; margin:8px 0 4px;">AI Advisory</div>
            <div style="font-size:13px; color:#4a6080;">RAG + Gemini powered insight</div>
        </div>
        <div class="card" style="text-align:center;">
            <div style="font-size:24px;">💼</div>
            <div style="font-weight:600; margin:8px 0 4px;">Portfolio Tracker</div>
            <div style="font-size:13px; color:#4a6080;">Track your watchlist in session</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<div style="text-align:center; font-size:12px; color:#1e2d45; font-family:'IBM Plex Mono',monospace;">
    FinSight AI · For educational purposes only · Not financial advice
</div>
""", unsafe_allow_html=True)