"""
rag_engine.py — Lightweight RAG engine using Gemini embeddings + generative model.
Falls back gracefully if the API key is missing or quota is exceeded.
"""

import os
import textwrap
from typing import List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=_GOOGLE_API_KEY)
    return _client


class RAGEngine:
    """
    Minimal vector store. Add text documents, then query by semantic similarity.
    Uses cosine similarity over Gemini embeddings.
    """

    def __init__(self):
        self.documents: List[str] = []
        self.vectors:   List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_document(self, text: str) -> None:
        """Embed and store a document."""
        if not text or not text.strip():
            return
        vec = self._embed(text)
        if vec is not None:
            self.documents.append(text)
            self.vectors.append(vec)
        else:
            # No embeddings available — store anyway for fallback retrieval
            self.documents.append(text)
            self.vectors.append(None)

    def query(self, query_text: str, k: int = 3) -> List[str]:
        """Return the k most relevant stored documents."""
        if not self.documents:
            return []

        # If no embeddings available, return all docs (there won't be many)
        valid_vectors = [v for v in self.vectors if v is not None]
        if not valid_vectors:
            return self.documents[:k]

        q_vec = self._embed(query_text)
        if q_vec is None:
            return self.documents[:k]

        sims = []
        for v in self.vectors:
            if v is not None:
                sim = float(
                    np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9)
                )
            else:
                sim = 0.0
            sims.append(sim)

        top_k = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _embed(self, text: str):
        if not _GOOGLE_API_KEY:
            return None
        try:
            client = _get_client()
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
            )
            return np.array(result.embeddings[0].values, dtype=float)
        except Exception as e:
            print(f"[RAGEngine] Embedding error: {e}")
            return None


# ----------------------------------------------------------------------- #
# Context builder — turns raw data into rich text chunks for RAG           #
# ----------------------------------------------------------------------- #

def build_rag_context(analysis: dict, data: dict) -> RAGEngine:
    """
    Populate a RAGEngine with multiple meaningful text documents derived
    from the fetched stock data and computed analysis.
    This gives the LLM genuinely varied context to retrieve from.
    """
    rag = RAGEngine()
    info = data.get("info", {}) if data else {}

    # 1. Fundamental ratios summary
    rag.add_document(
        f"FUNDAMENTAL ANALYSIS for {analysis.get('company_name')} ({analysis.get('ticker')}):\n"
        f"Sector: {analysis.get('sector')}\n"
        f"Market Cap: {analysis.get('market_cap')}\n"
        f"Revenue: {analysis.get('revenue')}\n"
        f"Net Profit: {analysis.get('profit')}\n"
        f"Profit Margin: {analysis.get('profit_margin')}\n"
        f"ROE: {analysis.get('roe')}\n"
        f"EPS: {analysis.get('eps')}\n"
        f"Operating Cash Flow: {analysis.get('operating_cf')}"
    )

    # 2. Valuation ratios
    rag.add_document(
        f"VALUATION RATIOS for {analysis.get('ticker')}:\n"
        f"P/E Ratio: {analysis.get('pe_ratio')} (sector high threshold: {analysis.get('pe_high_thresh')})\n"
        f"Price-to-Book: {analysis.get('price_to_book')}\n"
        f"EV/EBITDA: {analysis.get('ev_ebitda')}\n"
        f"Debt-to-Equity: {analysis.get('debt_to_equity')}\n"
        f"Current Ratio: {analysis.get('current_ratio')}"
    )

    # 3. Risk and signal summary
    rag.add_document(
        f"RISK & SIGNAL SUMMARY for {analysis.get('ticker')}:\n"
        f"Risk Score: {analysis.get('risk_score')} / 1.00  (0=low, 1=high)\n"
        f"Bullish signals: {analysis.get('buy_signals')}\n"
        f"Bearish signals: {analysis.get('sell_signals')}\n"
        f"Model Recommendation: {analysis.get('recommendation')} "
        f"(confidence: {analysis.get('confidence')})"
    )

    # 4. Company description from yfinance
    description = info.get("longBusinessSummary", "")
    if description:
        rag.add_document(f"COMPANY DESCRIPTION:\n{description[:800]}")

    # 5. Key officers / management (if available)
    officers = info.get("companyOfficers", [])
    if officers:
        names = ", ".join(
            f"{o.get('name')} ({o.get('title', 'Officer')})"
            for o in officers[:4]
        )
        rag.add_document(f"KEY MANAGEMENT of {analysis.get('ticker')}:\n{names}")

    # 6. Recent price context from history
    history = data.get("history") if data else None
    if history is not None:
        try:
            close = history["Close"].dropna()
            if len(close) >= 20:
                latest = close.iloc[-1]
                high_52w = close.max()
                low_52w  = close.min()
                ma_50    = close.iloc[-50:].mean() if len(close) >= 50 else close.mean()
                ma_200   = close.mean()
                rag.add_document(
                    f"PRICE HISTORY for {analysis.get('ticker')} (last 1 year):\n"
                    f"Current Price: {latest:.2f}\n"
                    f"52-week High: {high_52w:.2f}\n"
                    f"52-week Low:  {low_52w:.2f}\n"
                    f"50-day MA:    {ma_50:.2f}\n"
                    f"200-day MA:   {ma_200:.2f}\n"
                    f"Price vs 200d MA: {'above' if latest > ma_200 else 'below'}"
                )
        except Exception:
            pass

    return rag


# ----------------------------------------------------------------------- #
# Advice generation                                                        #
# ----------------------------------------------------------------------- #

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a professional financial analyst specialising in equity research.
    Answer in structured sections:
    1. **Decision** — BUY / HOLD / SELL
    2. **Key Reasons** — 2-3 bullet points
    3. **Action Plan** — specific and actionable
    4. **Risks to Watch** — 2-3 bullet points
    Keep the total response under 280 words. Be direct and data-driven.
""").strip()


def generate_advice(context: List[str], query: str) -> str:
    """
    Generate investment advice using the Gemini model.
    Falls back to a rule-based summary if the API key is absent.
    """
    if not _GOOGLE_API_KEY:
        return (
            "⚠️ **AI Insight unavailable** — add a `GOOGLE_API_KEY` to your `.env` file "
            "to enable Gemini-powered analysis."
        )

    context_text = "\n\n---\n\n".join(context) if context else "No context available."
    prompt = f"FINANCIAL DATA:\n{context_text}\n\nQUESTION:\n{query}"

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [{"text": _SYSTEM_PROMPT}]},
                {"role": "user", "parts": [{"text": prompt}]},
            ],
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not generate AI advice: {e}"