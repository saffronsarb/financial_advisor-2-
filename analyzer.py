"""
analyzer.py — Financial analysis engine
Computes key ratios, scores, and investment recommendations.
"""

# Sector-aware margin benchmarks (profit_margin thresholds)
_SECTOR_MARGIN_THRESHOLDS = {
    "Technology":            {"good": 0.15, "poor": 0.05},
    "Financial Services":    {"good": 0.20, "poor": 0.08},
    "Healthcare":            {"good": 0.12, "poor": 0.04},
    "Consumer Cyclical":     {"good": 0.08, "poor": 0.02},
    "Consumer Defensive":    {"good": 0.07, "poor": 0.02},
    "Industrials":           {"good": 0.08, "poor": 0.03},
    "Energy":                {"good": 0.10, "poor": 0.02},
    "Basic Materials":       {"good": 0.08, "poor": 0.02},
    "Communication Services":{"good": 0.12, "poor": 0.04},
    "Utilities":             {"good": 0.10, "poor": 0.04},
    "Real Estate":           {"good": 0.15, "poor": 0.05},
}
_DEFAULT_MARGIN = {"good": 0.10, "poor": 0.03}

# Sector-aware PE benchmarks
_SECTOR_PE_THRESHOLDS = {
    "Technology":            {"high": 40, "fair": 25},
    "Financial Services":    {"high": 20, "fair": 12},
    "Healthcare":            {"high": 35, "fair": 20},
    "Consumer Cyclical":     {"high": 30, "fair": 18},
    "Consumer Defensive":    {"high": 25, "fair": 15},
    "Industrials":           {"high": 28, "fair": 18},
    "Energy":                {"high": 20, "fair": 12},
    "Basic Materials":       {"high": 22, "fair": 14},
    "Communication Services":{"high": 30, "fair": 18},
    "Utilities":             {"high": 22, "fair": 15},
    "Real Estate":           {"high": 35, "fair": 20},
}
_DEFAULT_PE = {"high": 30, "fair": 18}


def _safe_div(numerator, denominator):
    """Return numerator/denominator, or None if denominator is 0/None."""
    if not denominator:
        return None
    return numerator / denominator


def analyze_financials(data: dict) -> dict:
    info   = data.get("info", {})
    sector = info.get("sector", "")

    # --- Core figures (None if missing, NOT defaulting to 1) ---
    revenue        = info.get("totalRevenue")        or 0
    profit         = info.get("netIncomeToCommon")   or 0
    debt           = info.get("totalDebt")           or 0
    equity         = info.get("totalStockholderEquity")
    current_assets = info.get("totalCurrentAssets")  or 0
    current_liab   = info.get("totalCurrentLiabilities")
    operating_cf   = info.get("operatingCashflow")   or 0
    shares_out     = info.get("sharesOutstanding")
    market_cap     = info.get("marketCap")           or 0

    # --- Derived ratios (None when denominator is missing) ---
    profit_margin = _safe_div(profit, revenue)
    roe           = _safe_div(profit, equity)
    debt_to_equity= _safe_div(debt, equity)
    current_ratio = _safe_div(current_assets, current_liab)
    pe_ratio      = info.get("trailingPE")
    eps           = info.get("trailingEps") or _safe_div(profit, shares_out)
    price_to_book = info.get("priceToBook")
    ev_ebitda     = info.get("enterpriseToEbitda")

    # --- Sector-aware thresholds ---
    margin_thresh = _SECTOR_MARGIN_THRESHOLDS.get(sector, _DEFAULT_MARGIN)
    pe_thresh     = _SECTOR_PE_THRESHOLDS.get(sector, _DEFAULT_PE)

    # --- Composite risk score (0 = low risk, 1 = high risk) ---
    risk_factors = []
    if debt_to_equity is not None:
        risk_factors.append(min(debt_to_equity / 3, 1))
    if profit_margin is not None:
        risk_factors.append(1 - min(max(profit_margin, 0), 1))
    if current_ratio is not None:
        risk_factors.append(0 if current_ratio >= 1.5 else 0.5)
    risk_factors.append(0 if operating_cf > 0 else 0.5)
    risk_score = round(sum(risk_factors) / len(risk_factors), 2) if risk_factors else 0.5

    # --- Buy / sell signal logic (sector-aware) ---
    buy_signals = 0
    sell_signals = 0

    # Profitability
    if profit_margin is not None:
        if profit_margin > margin_thresh["good"]:
            buy_signals += 1
        elif profit_margin < 0:
            sell_signals += 1

    # Debt health
    if debt_to_equity is not None:
        if debt_to_equity < 1.0:
            buy_signals += 1
        elif debt_to_equity > 3.0:
            sell_signals += 1

    # Liquidity
    if current_ratio is not None:
        if current_ratio >= 1.5:
            buy_signals += 1
        elif current_ratio < 1.0:
            sell_signals += 1

    # Return on equity
    if roe is not None:
        if roe > 0.12:
            buy_signals += 1
        elif roe < 0:
            sell_signals += 1

    # Cash flow
    if operating_cf > 0:
        buy_signals += 1
    else:
        sell_signals += 1

    # Valuation (PE) — sector-aware
    if pe_ratio and pe_ratio > 0:
        if pe_ratio < pe_thresh["fair"]:
            buy_signals += 1
        elif pe_ratio > pe_thresh["high"]:
            sell_signals += 1

    # Price-to-book
    if price_to_book and price_to_book > 0:
        if price_to_book < 1.5:
            buy_signals += 1
        elif price_to_book > 5.0:
            sell_signals += 1

    # --- Recommendation ---
    if buy_signals >= 5 and sell_signals == 0:
        recommendation = "BUY"
        confidence = "High"
    elif buy_signals >= 4 and sell_signals <= 1:
        recommendation = "BUY"
        confidence = "Medium"
    elif sell_signals >= 3:
        recommendation = "SELL"
        confidence = "High"
    elif sell_signals >= 2:
        recommendation = "SELL"
        confidence = "Medium"
    elif buy_signals >= 2:
        recommendation = "HOLD"
        confidence = "Medium"
    else:
        recommendation = "HOLD"
        confidence = "Low"

    def _r(v, n=2):
        return round(v, n) if v is not None else None

    return {
        # Identifiers
        "ticker":         info.get("symbol", "N/A"),
        "company_name":   info.get("longName", "Unknown"),
        "sector":         sector or "N/A",
        "currency":       info.get("currency", "USD"),
        "market_cap":     market_cap,
        "revenue":        revenue,
        "profit":         profit,
        "eps":            _r(eps),

        # Ratios
        "profit_margin":   _r(profit_margin, 4),
        "roe":             _r(roe, 4),
        "debt_to_equity":  _r(debt_to_equity),
        "current_ratio":   _r(current_ratio),
        "pe_ratio":        _r(pe_ratio),
        "price_to_book":   _r(price_to_book),
        "ev_ebitda":       _r(ev_ebitda),
        "operating_cf":    operating_cf,

        # Thresholds used (for UI context)
        "margin_good_thresh": margin_thresh["good"],
        "pe_high_thresh":     pe_thresh["high"],

        # Composite scores
        "risk_score":      risk_score,
        "buy_signals":     buy_signals,
        "sell_signals":    sell_signals,

        # Decision
        "recommendation":  recommendation,
        "confidence":      confidence,
    }


def generate_advisory(analysis: dict) -> dict:
    rec        = analysis["recommendation"]
    confidence = analysis["confidence"]
    pm         = analysis["profit_margin"]
    r          = analysis["roe"]
    margin_pct = round(pm * 100, 1) if pm is not None else "N/A"
    roe_pct    = round(r  * 100, 1) if r  is not None else "N/A"
    name       = analysis.get("company_name", "This company")
    sector     = analysis.get("sector", "")
    sector_note = f" (sector: {sector})" if sector and sector != "N/A" else ""

    base = {
        "BUY": {
            "summary": (
                f"{name}{sector_note} shows strong fundamentals — {margin_pct}% profit margin, "
                f"{roe_pct}% ROE, and manageable debt. Financials support a positive outlook."
            ),
            "action":  "Consider a staggered entry (SIP or phased buying) to reduce timing risk.",
            "avoid":   "Don't deploy all capital at once. Wait for any short-term dips.",
            "horizon": "Medium to long term (1–3 years)",
        },
        "HOLD": {
            "summary": (
                f"{name}{sector_note} is financially stable but shows limited upside signals. "
                f"Profit margin at {margin_pct}% and ROE at {roe_pct}% are modest."
            ),
            "action":  "Hold existing positions. Re-evaluate on next earnings release.",
            "avoid":   "Avoid aggressive accumulation at current valuation levels.",
            "horizon": "Short to medium term (3–12 months)",
        },
        "SELL": {
            "summary": (
                f"{name}{sector_note} exhibits weak fundamentals — thin margins ({margin_pct}%), "
                f"elevated debt, or negative ROE ({roe_pct}%). Risk outweighs reward."
            ),
            "action":  "Consider reducing exposure gradually. Avoid catching a falling knife.",
            "avoid":   "Do not buy on hype, tips, or short-term price spikes.",
            "horizon": "Exit over 1–4 weeks to minimize market impact",
        },
    }

    advisory = base[rec].copy()
    advisory["confidence_note"] = (
        f"Signal confidence: {confidence}. "
        f"Based on {analysis['buy_signals']} bullish and {analysis['sell_signals']} bearish indicators."
    )
    return advisory