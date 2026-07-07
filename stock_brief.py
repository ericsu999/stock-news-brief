"""
STOCK NEWS BRIEF - your mini Bloomberg terminal (v0.3)
------------------------------------------------------
What this does:
  1. Asks you for one or more stock tickers (like AAPL, TSLA)
  2. Fetches recent news headlines for each from Finnhub
  3. Sends them to Claude, which returns structured JSON (sentiment score etc.)
  4. Prints a formatted brief per ticker + a sentiment leaderboard

How to run it:
  python3 stock_brief.py
"""

# ---------- IMPORTS ----------
import os                      # read environment variables (where we hide API keys)
import json                    # convert JSON text <-> Python dictionaries
import requests                # make requests to APIs over the internet
from datetime import date, timedelta   # work with dates (news from the last week)
from dotenv import load_dotenv          # load our secret keys from the .env file
from anthropic import Anthropic         # the official Claude library

# ---------- SETUP ----------
load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
claude = Anthropic()  # automatically finds ANTHROPIC_API_KEY in .env


# ---------- STEP 1: GET NEWS HEADLINES ----------
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
    response.raise_for_status()   # stop with an error if something went wrong

    news_items = response.json()
    return news_items[:10]        # keep only the first 10 stories


# ---------- STEP 2: SUMMARIZE WITH CLAUDE (structured JSON output) ----------
def summarize(ticker, news_items):
    """
    Sends the headlines to Claude and gets back structured data:
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

Respond with ONLY a JSON object, no other text before or after, in exactly
this format:
{{
  "sentiment_score": <integer 1-10, where 1=very negative, 10=very positive>,
  "sentiment_label": "<Positive, Negative, or Mixed>",
  "summary": "<3-4 sentence summary of the most important developments>",
  "top_risk": "<the single biggest risk or concern in this news>",
  "rumor_flag": <true if any major story is single-source or unconfirmed, else false>
}}

Be factual. Do not give buy/sell advice."""

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text

    # Sometimes the model wraps JSON in ```json fences - strip them if present.
    raw = raw.replace("```json", "").replace("```", "").strip()

    return json.loads(raw)


# ---------- STEP 3: PRINT ONE TICKER'S BRIEF ----------
def print_brief(ticker, brief):
    """Nicely formats a single ticker's brief in the terminal."""
    print("-" * 50)
    print(f"  {ticker}")
    print("-" * 50)
    print(f"SENTIMENT: {brief['sentiment_label']} ({brief['sentiment_score']}/10)")
    print()
    print(f"SUMMARY: {brief['summary']}")
    print()
    print(f"TOP RISK: {brief['top_risk']}")
    if brief["rumor_flag"]:
        print()
        print("⚠️  WARNING: at least one story is single-source or unconfirmed")
    print()


# ---------- STEP 4: PUT IT ALL TOGETHER ----------
def main():
    print("=" * 50)
    print("  STOCK NEWS BRIEF - mini terminal v0.3")
    print("=" * 50)

    raw_input_text = input("\nEnter ticker(s), comma-separated (e.g. AAPL, TSLA): ")
    # Split on commas, clean up spaces, uppercase everything -> a list of tickers
    tickers = [t.strip().upper() for t in raw_input_text.split(",")]

    results = {}   # will map ticker -> its brief

    for ticker in tickers:
        print(f"\nFetching news for {ticker}...")
        news = get_news(ticker)

        if len(news) == 0:
            print(f"No news found for {ticker} - skipping.")
            continue   # move on to the next ticker

        print(f"Found {len(news)} stories. Asking Claude to summarize...\n")
        brief = summarize(ticker, news)
        results[ticker] = brief

        print_brief(ticker, brief)

    # ----- comparison leaderboard (only if we did more than one ticker) -----
    if len(results) > 1:
        print("=" * 50)
        print("  SENTIMENT LEADERBOARD")
        print("=" * 50)
        ranked = sorted(results.items(), key=lambda x: x[1]["sentiment_score"], reverse=True)
        for ticker, brief in ranked:
            print(f"  {ticker}: {brief['sentiment_score']}/10  ({brief['sentiment_label']})")


if __name__ == "__main__":
    main()