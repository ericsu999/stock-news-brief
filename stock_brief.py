"""
STOCK NEWS BRIEF - your mini Bloomberg terminal (v0.1)
------------------------------------------------------
What this does:
  1. Asks you for a stock ticker (like AAPL or TSLA)
  2. Fetches recent news headlines about that company from Finnhub
  3. Sends those headlines to Claude, which writes a short investor brief
  4. Prints the brief in your terminal

How to run it (after setup):
  python stock_brief.py
"""

# ---------- IMPORTS ----------
# "import" pulls in code other people wrote so we don't reinvent the wheel.

import os                      # lets us read environment variables (where we hide API keys)
import requests                # lets us make requests to websites/APIs over the internet
from datetime import date, timedelta   # for working with dates (news from the last week)
from dotenv import load_dotenv          # loads our secret keys from the .env file
from anthropic import Anthropic         # the official Claude library

# ---------- SETUP ----------
# This reads the file called ".env" in the same folder and loads the keys inside it.
load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")      # your Finnhub key from the .env file
claude = Anthropic()  # the Anthropic library automatically finds ANTHROPIC_API_KEY in .env


# ---------- STEP 1: GET NEWS HEADLINES ----------
def get_news(ticker):
    """
    Asks Finnhub for company news from the last 7 days.
    Returns a list of news items (each one is a dictionary with title, summary, etc.)
    """
    today = date.today()
    week_ago = today - timedelta(days=7)

    # This is the web address (API endpoint) we're requesting data from.
    url = "https://finnhub.io/api/v1/company-news"

    # These are the "parameters" - extra info the API needs from us.
    params = {
        "symbol": ticker,                        # which stock
        "from": week_ago.strftime("%Y-%m-%d"),   # start date, formatted like 2026-06-29
        "to": today.strftime("%Y-%m-%d"),        # end date
        "token": FINNHUB_KEY,                    # proves we're allowed to use the API
    }

    response = requests.get(url, params=params)  # actually make the request

    # If something went wrong (bad key, no internet), stop and show the error.
    response.raise_for_status()

    news_items = response.json()   # convert the response into Python data (a list)

    # Keep only the first 10 stories so we don't overwhelm Claude (or your wallet).
    return news_items[:10]


# ---------- STEP 2: SUMMARIZE WITH CLAUDE ----------
def summarize(ticker, news_items):
    """
    Sends the headlines to Claude and asks for a short investor brief.
    """
    # Build one big text block out of all the headlines + summaries.
    headlines_text = ""
    for item in news_items:
        headlines_text += f"- HEADLINE: {item['headline']}\n"
        headlines_text += f"  SOURCE: {item['source']}\n"
        headlines_text += f"  SUMMARY: {item['summary']}\n\n"

    # The "prompt" - our instructions to Claude. Prompt design is a real skill;
    # notice we tell it exactly what format and length we want.
    prompt = f"""You are a financial news analyst. Below are recent news items
about the stock {ticker}.

{headlines_text}

Write a brief for an investor with exactly these three sections:
1. SUMMARY: 3-4 sentences covering the most important developments.
2. TONE: One word (Positive / Negative / Mixed) plus one sentence explaining why.
3. WATCH OUT: One sentence flagging anything that looks like rumor or is only
   reported by a single source, or say "Nothing notable" if all looks solid.

Be concise and factual. Do not give buy/sell advice."""

    # This is the actual call to Claude.
    message = claude.messages.create(
        model="claude-sonnet-4-6",   # which Claude model to use
        max_tokens=500,               # cap on response length (keeps costs tiny)
        messages=[{"role": "user", "content": prompt}],
    )

    # Claude's reply comes back as a list of blocks; we want the text of the first one.
    return message.content[0].text


# ---------- STEP 3: PUT IT ALL TOGETHER ----------
def main():
    print("=" * 50)
    print("  STOCK NEWS BRIEF - mini terminal v0.1")
    print("=" * 50)

    ticker = input("\nEnter a ticker symbol (e.g. AAPL): ").strip().upper()

    print(f"\nFetching news for {ticker}...")
    news = get_news(ticker)

    if len(news) == 0:
        print("No news found. Check the ticker symbol and try again.")
        return   # stop here

    print(f"Found {len(news)} stories. Asking Claude to summarize...\n")
    brief = summarize(ticker, news)

    print("-" * 50)
    print(brief)
    print("-" * 50)


# This line means: "only run main() if this file is run directly"
# (a standard Python convention - you'll see it everywhere).
if __name__ == "__main__":
    main()