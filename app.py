"""
STOCK NEWS BRIEF - web dashboard (v1.3)
---------------------------------------
Macro briefing panel (multi-source RSS) + per-ticker company briefs.
v1.3: show how sentiment moved since last time, chart the history we've been
      logging all along, and call out when sentiment and price disagree.

How to run it:
  streamlit run app.py
"""

from datetime import date

import pandas as pd
import streamlit as st

from stock_brief import (
    get_news, summarize, log_to_csv, get_prices,
    get_macro_news, summarize_macro,
    get_history, get_sentiment_change, get_divergence,
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

                # Read the history and compare BEFORE logging today's run, so
                # we're looking at the previous score rather than the one we're
                # about to write.
                change = get_sentiment_change(ticker, brief["sentiment_score"])
                history = get_history(ticker)
                divergence = get_divergence(brief["sentiment_score"], pct_change)

                results[ticker] = brief
                data[ticker] = {
                    "news": news,
                    "brief": brief,
                    "closes": closes,
                    "pct_change": pct_change,
                    "change": change,
                    "history": history,
                    "divergence": divergence,
                }
                log_to_csv(ticker, brief)

            except Exception as e:
                st.error(f"Something went wrong with {ticker}: {e}")
                continue
        progress.empty()

        if len(data) > 0:
            st.subheader("🏢 Company briefs")
            tabs = st.tabs(list(data.keys()))

            for tab, ticker in zip(tabs, data.keys()):
                d = data[ticker]
                news, brief = d["news"], d["brief"]
                with tab:
                    col1, col2, col3, col4 = st.columns(4)
                    color = sentiment_color(brief["sentiment_score"])

                    # The metric's little arrow is exactly the move since our
                    # last run, so the change shows up right on the number.
                    if d["change"]:
                        delta_text = (f"{d['change']['delta']:+d} since "
                                      f"{d['change']['previous_date']}")
                        # A flat score shouldn't get a green "up" arrow.
                        delta_color = "off" if d["change"]["delta"] == 0 else "normal"
                    else:
                        delta_text, delta_color = None, "normal"

                    col1.metric("Sentiment", f"{brief['sentiment_score']}/10",
                                delta=delta_text, delta_color=delta_color)
                    col2.metric("Tone", brief["sentiment_label"])
                    if d["pct_change"] is not None:
                        col3.metric("1-week price", f"{d['pct_change']:+.1f}%")
                    col4.metric("Stories analyzed", len(news))

                    # When the news mood and the price disagree, say so loudly -
                    # that mismatch is the whole reason we track both.
                    if d["divergence"]:
                        st.warning(f"**⚡ {d['divergence']['headline']}** — "
                                   f"{d['divergence']['detail']}")

                    chart_cols = st.columns(2)

                    if d["closes"] is not None:
                        chart_cols[0].caption("Price — last month")
                        chart_cols[0].line_chart(d["closes"], height=250)

                    # The sentiment history we've been logging all along but
                    # never showed. Today's fresh score goes on the end.
                    if d["history"]:
                        series = pd.DataFrame(
                            [{"date": r["date"], "sentiment": r["sentiment_score"]}
                             for r in d["history"]]
                            + [{"date": date.today().strftime("%Y-%m-%d"),
                                "sentiment": brief["sentiment_score"]}]
                        ).set_index("date")
                        chart_cols[1].caption("Sentiment — every day we've logged")
                        chart_cols[1].line_chart(series, height=250)
                    else:
                        chart_cols[1].caption("Sentiment — every day we've logged")
                        chart_cols[1].info("First time scoring this ticker. Run it "
                                           "again tomorrow to start a trend.")

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