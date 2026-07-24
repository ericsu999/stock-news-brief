"""
STOCK NEWS BRIEF - web dashboard (v1.2)
---------------------------------------
Macro briefing panel (multi-source RSS) + per-ticker company briefs.

How to run it:
  streamlit run app.py
"""

import streamlit as st

from stock_brief import (
    get_news, summarize, log_to_csv, get_prices,
    get_macro_news, summarize_macro,
)

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Stock News Brief", page_icon="📈", layout="wide")

st.title("📈 Stock News Brief")
st.caption("Macro briefing + AI company news analysis. Not investment advice.")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Watchlist")
    raw_input_text = st.text_input(
        "Ticker(s), comma-separated:",
        placeholder="AAPL, TSLA, NVDA",
    )
    include_macro = st.checkbox("Include macro briefing", value=True)
    run = st.button("Generate briefs", type="primary", use_container_width=True)
    st.divider()
    st.caption("Tip: use the same watchlist daily to build a consistent sentiment dataset.")

# ---------- HELPER ----------
def sentiment_color(score):
    """Green for positive, red for negative, orange for the middle."""
    if score >= 7:
        return "green"
    elif score <= 4:
        return "red"
    else:
        return "orange"

# ---------- MAIN LOGIC ----------
if run:

    # ===== MACRO BRIEFING (market-wide picture, multi-source) =====
    if include_macro:
        with st.spinner("Reading macro headlines from CNBC, MarketWatch, Yahoo Finance..."):
            try:
                macro_stories = get_macro_news()
                if len(macro_stories) > 0:
                    macro = summarize_macro(macro_stories)

                    st.subheader("🌍 Macro briefing")
                    mood = macro["market_mood"]
                    mood_color = {"Risk-on": "green", "Risk-off": "red"}.get(mood, "orange")

                    col1, col2 = st.columns([1, 3])
                    col1.markdown(f"### :{mood_color}[{mood}]")
                    col1.caption(f"Based on {len(macro_stories)} headlines from {len(set(s['source'] for s in macro_stories))} sources")
                    col2.markdown(macro["big_picture"])

                    with st.expander("Key events & what to watch"):
                        for event in macro["key_events"]:
                            st.markdown(f"- {event}")
                        st.markdown(f"**Watch next:** {macro['watch_next']}")
                else:
                    st.warning("Couldn't reach the macro news feeds right now.")
            except Exception as e:
                st.error(f"Macro briefing failed: {e}")
        st.divider()

    # ===== COMPANY BRIEFS (per-ticker) =====
    if raw_input_text.strip():
        tickers = [t.strip().upper() for t in raw_input_text.split(",")]

        results = {}
        data = {}

        progress = st.progress(0, text="Starting...")
        for i, ticker in enumerate(tickers):
            progress.progress(i / len(tickers), text=f"Analyzing {ticker}...")
            try:
                news = get_news(ticker)
                if len(news) == 0:
                    st.warning(f"No news found for {ticker} - skipping.")
                    continue

                brief = summarize(ticker, news)
                closes, pct_change = get_prices(ticker)

                results[ticker] = brief
                data[ticker] = (news, brief, closes, pct_change)
                log_to_csv(ticker, brief)

            except Exception as e:
                st.error(f"Something went wrong with {ticker}: {e}")
                continue
        progress.empty()

        if len(data) > 0:
            st.subheader("🏢 Company briefs")
            tabs = st.tabs(list(data.keys()))

            for tab, ticker in zip(tabs, data.keys()):
                news, brief, closes, pct_change = data[ticker]
                with tab:
                    col1, col2, col3, col4 = st.columns(4)
                    color = sentiment_color(brief["sentiment_score"])
                    col1.metric("Sentiment", f"{brief['sentiment_score']}/10")
                    col2.metric("Tone", brief["sentiment_label"])
                    if pct_change is not None:
                        col3.metric("1-week price", f"{pct_change:+.1f}%")
                    col4.metric("Stories analyzed", len(news))

                    if closes is not None:
                        st.line_chart(closes, height=250)

                    st.markdown(f"### :{color}[{brief['sentiment_label']} sentiment]")
                    st.markdown(f"**Summary:** {brief['summary']}")
                    st.markdown(f"**Top risk:** {brief['top_risk']}")

                    if brief["rumor_flag"]:
                        st.warning("At least one story is single-source or unconfirmed.")

            # ---------- LEADERBOARD ----------
            if len(results) > 1:
                st.divider()
                st.subheader("🏆 Sentiment leaderboard")
                ranked = sorted(results.items(), key=lambda x: x[1]["sentiment_score"], reverse=True)
                cols = st.columns(len(ranked))
                for col, (ticker, brief) in zip(cols, ranked):
                    color = sentiment_color(brief["sentiment_score"])
                    col.markdown(f"### {ticker}")
                    col.markdown(f":{color}[**{brief['sentiment_score']}/10** — {brief['sentiment_label']}]")
    else:
        if not include_macro:
            st.info("👈 Enter tickers and/or enable the macro briefing.")
else:
    st.info("👈 Enter tickers in the sidebar and click Generate briefs.")