"""
STOCK NEWS BRIEF - engine (v0.8)
--------------------------------
News fetching (company + macro), AI summarization, price data, CSV logging.
v0.8:    read the sentiment log back, so each brief shows how sentiment moved
         since last time; flag when sentiment and price disagree; and log one
         row per ticker per day instead of one per run.
v0.7:    upgrade to Claude Opus 5 and use structured outputs, so the API
         guarantees the reply shape instead of us parsing it by hand.
v0.6.1: fetch RSS feeds with a browser user-agent (some sites block scripts).

How to run the terminal version:
  python3 stock_brief.py
"""

# ---------- IMPORTS ----------
import os                      # read environment variables (where we hide API keys)
import csv                     # read/write CSV files (our sentiment history log)
import requests                # make requests to APIs over the internet
import feedparser              # parse RSS feed content from news sites
import yfinance as yf          # historical price data from Yahoo Finance
from typing import Literal              # restrict a field to a fixed set of values
from datetime import date, timedelta   # work with dates
from dotenv import load_dotenv          # load our secret keys from the .env file
from pydantic import BaseModel, Field   # describe the exact shape we want back
from anthropic import Anthropic         # the official Claude library

# ---------- SETUP ----------
load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
claude = Anthropic()  # automatically finds ANTHROPIC_API_KEY in .env

# RSS feeds for macro/market-wide news.
MACRO_FEEDS = {
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
}

# Some news sites block requests that identify as scripts. This header makes
# our requests identify as a normal browser instead.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Safari/537.36"
}

MODEL = "claude-opus-5"

# Where the sentiment history lives. We anchor it to this file's own folder so
# the log always lands in the project, no matter which directory you happen to
# run the app from.
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_log.csv")
LOG_FIELDS = ["date", "ticker", "sentiment_score", "sentiment_label", "rumor_flag"]

# ---------- DIVERGENCE THRESHOLDS ----------
# The interesting case is when the news mood and the share price disagree:
# upbeat coverage while the stock falls, or gloomy coverage while it rises.
# That mismatch is easy to miss when you're only skimming headlines. These
# numbers decide what counts as "disagreeing".
BULLISH_SCORE = 7      # at or above this, coverage reads positive
BEARISH_SCORE = 4      # at or below this, coverage reads negative
PRICE_MOVE_PCT = 3.0   # a weekly move smaller than this is just noise


# ---------- RESPONSE SHAPES ----------
# These classes describe the exact structure we want back from Claude. We hand
# them to the API, which then *guarantees* the reply matches - no more asking
# nicely in the prompt and hoping, and no more stripping stray text by hand.
class CompanyBrief(BaseModel):
    """Claude's analysis of one company's recent news."""
    sentiment_score: int = Field(ge=1, le=10,
                                 description="1 = very negative, 10 = very positive")
    sentiment_label: Literal["Positive", "Negative", "Mixed"]
    summary: str = Field(description="3-4 sentence summary of the most important developments")
    top_risk: str = Field(description="The single biggest risk or concern in this news")
    rumor_flag: bool = Field(description="True if any major story is single-source or unconfirmed")


class MacroBrief(BaseModel):
    """Claude's read on the market-wide picture."""
    market_mood: Literal["Risk-on", "Risk-off", "Mixed"]
    big_picture: str = Field(description="3-4 sentences on the most important macro themes "
                                         "right now (central banks, policy, geopolitics, data)")
    key_events: list[str] = Field(description="Three short bullets on what is driving markets")
    watch_next: str = Field(description="One sentence: the upcoming event markets care about most")


# ---------- COMPANY NEWS (Finnhub) ----------
def get_news(ticker):
    """
    Asks Finnhub for company news from the last 7 days.
    Returns a list of news items (each is a dictionary with title, summary, etc.)
    """
    today = date.today()
    week_ago = today - timedelta(days=7)

    url = "https://finnhub.io/api/v1/company-news"

    params = {
        "symbol": ticker,
        "from": week_ago.strftime("%Y-%m-%d"),
        "to": today.strftime("%Y-%m-%d"),
        "token": FINNHUB_KEY,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    news_items = response.json()
    return news_items[:10]


# ---------- MACRO NEWS (multi-source RSS) ----------
def get_macro_news():
    """
    Pulls recent headlines from several major outlets' RSS feeds.
    Fetches with a browser user-agent (some sites block scripts), then
    parses the feed content with feedparser.
    Returns a list of dictionaries: {"source": ..., "title": ..., "summary": ...}
    """
    stories = []
    for source_name, feed_url in MACRO_FEEDS.items():
        try:
            # Step 1: download the feed ourselves, disguised as a browser.
            response = requests.get(feed_url, headers=BROWSER_HEADERS, timeout=10)
            response.raise_for_status()

            # Step 2: hand the downloaded content to feedparser to interpret.
            feed = feedparser.parse(response.content)

            for entry in feed.entries[:5]:
                stories.append({
                    "source": source_name,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:300],
                })
        except Exception:
            # If one feed is down or blocked, skip it - don't crash everything.
            continue
    return stories


def summarize_macro(stories):
    """
    Sends macro headlines from multiple sources to Claude and gets back
    a structured big-picture market briefing.
    """
    headlines_text = ""
    for s in stories:
        headlines_text += f"- [{s['source']}] {s['title']}\n"
        if s["summary"]:
            headlines_text += f"  {s['summary']}\n"
        headlines_text += "\n"

    prompt = f"""You are a macro market analyst. Below are current headlines from
several major financial news outlets.

{headlines_text}

Focus on market-wide forces (rates, policy, conflicts, economic data), NOT
individual company stories. Be factual. Do not give buy/sell advice."""

    message = claude.messages.parse(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        output_format=MacroBrief,   # the API now guarantees this exact shape
    )

    # .parsed_output is a validated MacroBrief. Hand back a plain dictionary so
    # the rest of the app keeps working with macro["market_mood"] etc.
    return message.parsed_output.model_dump()


# ---------- PRICE HISTORY ----------
def get_prices(ticker):
    """
    Gets the last month of daily closing prices from Yahoo Finance.
    Returns (closing prices table, week % change), or (None, None) if unavailable.
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period="1mo")

    if history.empty:
        return None, None

    closes = history["Close"]
    week_ago_price = closes.iloc[-6] if len(closes) >= 6 else closes.iloc[0]
    latest_price = closes.iloc[-1]
    pct_change = (latest_price - week_ago_price) / week_ago_price * 100

    return closes, pct_change


# ---------- COMPANY SUMMARIZATION (structured JSON) ----------
def summarize(ticker, news_items):
    """
    Sends company headlines to Claude and gets back structured data:
    a dictionary with sentiment_score, summary, top_risk, etc.
    """
    headlines_text = ""
    for item in news_items:
        headlines_text += f"- HEADLINE: {item['headline']}\n"
        headlines_text += f"  SOURCE: {item['source']}\n"
        headlines_text += f"  SUMMARY: {item['summary']}\n\n"

    prompt = f"""You are a financial news analyst. Below are recent news items
about the stock {ticker}.

{headlines_text}

Focus on COMPANY-SPECIFIC news (products, earnings, legal, management), not
general market conditions. Be factual. Do not give buy/sell advice."""

    message = claude.messages.parse(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        output_format=CompanyBrief,   # the API now guarantees this exact shape
    )

    # .parsed_output is a validated CompanyBrief. Hand back a plain dictionary so
    # the rest of the app keeps working with brief["sentiment_score"] etc.
    return message.parsed_output.model_dump()


# ---------- PRINT ONE TICKER'S BRIEF (terminal version) ----------
def print_brief(ticker, brief, change=None, divergence=None):
    """
    Nicely formats a single ticker's brief in the terminal.

    `change` and `divergence` are optional - when we have them, we show how
    sentiment moved since last time and whether it disagrees with the price.
    """
    print("-" * 50)
    print(f"  {ticker}")
    print("-" * 50)

    # Sentiment, plus how it moved since the last time we scored this ticker.
    line = f"SENTIMENT: {brief['sentiment_label']} ({brief['sentiment_score']}/10)"
    if change:
        arrow = {"up": "▲", "down": "▼", "flat": "="}[change["direction"]]
        line += (f"  {arrow} from {change['previous_score']}/10"
                 f" on {change['previous_date']}")
    print(line)
    print()

    if divergence:
        print(f"⚡ DIVERGENCE - {divergence['headline']}")
        print(f"   {divergence['detail']}")
        print()

    print(f"SUMMARY: {brief['summary']}")
    print()
    print(f"TOP RISK: {brief['top_risk']}")
    if brief["rumor_flag"]:
        print()
        print("⚠️  WARNING: at least one story is single-source or unconfirmed")
    print()


# ---------- READ THE LOG BACK ----------
def read_log():
    """
    Reads the whole sentiment history back as a list of dictionaries.
    Returns an empty list if we've never logged anything yet.

    Everything in a CSV file is text, so we convert the score back to a number
    and rumor_flag back to a real True/False on the way out.
    """
    if not os.path.exists(LOG_PATH):
        return []

    rows = []
    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["sentiment_score"] = int(row["sentiment_score"])
            except (TypeError, ValueError, KeyError):
                continue    # skip a malformed line rather than crashing
            row["rumor_flag"] = str(row.get("rumor_flag", "")).strip().lower() == "true"
            rows.append(row)
    return rows


def get_history(ticker):
    """
    Every past score for one ticker, oldest first, ignoring today's run.
    This is what lets the app say "up from 6 last week" instead of just "8".
    """
    today = date.today().strftime("%Y-%m-%d")
    past = [r for r in read_log() if r["ticker"] == ticker and r["date"] < today]
    return sorted(past, key=lambda r: r["date"])


def get_sentiment_change(ticker, current_score):
    """
    Compares today's score against the most recent earlier one.
    Returns a dictionary describing the move, or None if this is the first
    time we've ever scored this ticker.
    """
    history = get_history(ticker)
    if not history:
        return None

    previous = history[-1]
    delta = current_score - previous["sentiment_score"]

    return {
        "previous_score": previous["sentiment_score"],
        "previous_date": previous["date"],
        "delta": delta,
        "direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
    }


# ---------- SENTIMENT vs PRICE DIVERGENCE ----------
def get_divergence(sentiment_score, pct_change):
    """
    Checks whether the news mood and the 1-week price move point opposite ways.
    Returns a dictionary describing the mismatch, or None when the two agree
    (or when the price move is too small to read anything into).
    """
    if pct_change is None:
        return None

    if sentiment_score >= BULLISH_SCORE and pct_change <= -PRICE_MOVE_PCT:
        return {
            "type": "sentiment_ahead",
            "headline": "Positive coverage, falling price",
            "detail": (f"The news reads positive ({sentiment_score}/10) but the stock is "
                       f"{pct_change:+.1f}% this week. The market isn't buying the story, "
                       f"or it's reacting to something the headlines haven't caught yet."),
        }

    if sentiment_score <= BEARISH_SCORE and pct_change >= PRICE_MOVE_PCT:
        return {
            "type": "price_ahead",
            "headline": "Negative coverage, rising price",
            "detail": (f"The news reads negative ({sentiment_score}/10) but the stock is "
                       f"{pct_change:+.1f}% this week. The bad news may already be priced "
                       f"in, or buyers are looking past it."),
        }

    return None


# ---------- LOG RESULTS TO CSV ----------
def log_to_csv(ticker, brief):
    """
    Records today's score for this ticker - one row per ticker per day.

    Running the app twice in one day now updates that day's row instead of
    adding a duplicate, so the history stays a clean one-point-per-day series
    that we can actually chart.
    """
    today = date.today().strftime("%Y-%m-%d")

    # Keep every existing row except an earlier run of this ticker today.
    kept = [r for r in read_log()
            if not (r["date"] == today and r["ticker"] == ticker)]

    kept.append({
        "date": today,
        "ticker": ticker,
        "sentiment_score": brief["sentiment_score"],
        "sentiment_label": brief["sentiment_label"],
        "rumor_flag": brief["rumor_flag"],
    })

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


# ---------- TERMINAL VERSION ----------
def main():
    print("=" * 50)
    print("  STOCK NEWS BRIEF - mini terminal v0.8")
    print("=" * 50)

    raw_input_text = input("\nEnter ticker(s), comma-separated (e.g. AAPL, TSLA): ")
    tickers = [t.strip().upper() for t in raw_input_text.split(",")]

    results = {}

    for ticker in tickers:
        print(f"\nFetching news for {ticker}...")
        news = get_news(ticker)

        if len(news) == 0:
            print(f"No news found for {ticker} - skipping.")
            continue

        print(f"Found {len(news)} stories. Asking Claude to summarize...\n")
        brief = summarize(ticker, news)
        results[ticker] = brief

        # Compare against history and against the price BEFORE logging today's
        # run, so we're reading yesterday's number rather than the one we just
        # wrote.
        change = get_sentiment_change(ticker, brief["sentiment_score"])
        _, pct_change = get_prices(ticker)
        divergence = get_divergence(brief["sentiment_score"], pct_change)

        print_brief(ticker, brief, change=change, divergence=divergence)
        log_to_csv(ticker, brief)

    if len(results) > 1:
        print("=" * 50)
        print("  SENTIMENT LEADERBOARD")
        print("=" * 50)
        ranked = sorted(results.items(), key=lambda x: x[1]["sentiment_score"], reverse=True)
        for ticker, brief in ranked:
            print(f"  {ticker}: {brief['sentiment_score']}/10  ({brief['sentiment_label']})")


if __name__ == "__main__":
    main()