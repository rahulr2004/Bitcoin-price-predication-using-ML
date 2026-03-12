"""
sentiment.py
------------
Fetches recent Bitcoin-related news and scores market sentiment
using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Two modes:
  1. NewsAPI  — requires a free API key from https://newsapi.org
  2. Fallback — scrapes RSS feeds from CoinDesk & CryptoPanic
                (no key required, limited to 10 headlines)

Usage:
    from src.sentiment import get_sentiment_score, get_news_headlines

    score, headlines = get_sentiment_score()
    # score: float in [-1.0, 1.0]
    #   > +0.2  → positive / bullish
    #   < -0.2  → negative / bearish
    #   else    → neutral
"""

import os
import re
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# ── Data classes ──────────────────────────────────────────────

@dataclass
class NewsItem:
    title:     str
    source:    str
    url:       str
    sentiment: float   # VADER compound score [-1, 1]

    @property
    def label(self) -> str:
        if self.sentiment >= 0.05:
            return "🟢 Positive"
        elif self.sentiment <= -0.05:
            return "🔴 Negative"
        return "🟡 Neutral"


# ── NewsAPI (primary, requires free key) ─────────────────────

def _fetch_newsapi(api_key: str, query: str = "bitcoin", n: int = 20) -> List[NewsItem]:
    """Fetch headlines via NewsAPI (https://newsapi.org — free tier)."""
    try:
        from newsapi import NewsApiClient
        client   = NewsApiClient(api_key=api_key)
        articles = client.get_everything(
            q=query, language="en",
            sort_by="publishedAt", page_size=n,
        )["articles"]

        items = []
        for a in articles:
            title = a.get("title") or ""
            score = analyzer.polarity_scores(title)["compound"]
            items.append(NewsItem(
                title=title,
                source=a.get("source", {}).get("name", "NewsAPI"),
                url=a.get("url", ""),
                sentiment=score,
            ))
        return items

    except Exception as e:
        print(f"[sentiment] NewsAPI error: {e}")
        return []


# ── RSS fallback (no key required) ───────────────────────────

_RSS_FEEDS = [
    ("CoinDesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CryptoPanic",   "https://cryptopanic.com/news/bitcoin/rss/"),
    ("Decrypt",       "https://decrypt.co/feed"),
]

def _fetch_rss(n: int = 15) -> List[NewsItem]:
    """Parse Bitcoin headlines from public RSS feeds (no API key needed)."""
    import xml.etree.ElementTree as ET

    items: List[NewsItem] = []
    for source, url in _RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for entry in root.iter("item"):
                title_el = entry.find("title")
                link_el  = entry.find("link")
                if title_el is None:
                    continue
                title = re.sub(r"<[^>]+>", "", title_el.text or "").strip()
                if not title:
                    continue
                # Only keep bitcoin-related headlines
                if not re.search(r"bitcoin|btc|crypto|blockchain", title, re.I):
                    continue

                score = analyzer.polarity_scores(title)["compound"]
                items.append(NewsItem(
                    title=title,
                    source=source,
                    url=link_el.text if link_el is not None else "",
                    sentiment=score,
                ))

            if len(items) >= n:
                break

        except Exception as e:
            print(f"[sentiment] RSS {source} error: {e}")
            continue

    return items[:n]


# ── Public API ────────────────────────────────────────────────

def get_news_headlines(
    newsapi_key: Optional[str] = None,
    n: int = 20,
) -> List[NewsItem]:
    """
    Retrieve recent Bitcoin news headlines.

    Tries NewsAPI first (if key provided), then falls back to RSS.

    Args:
        newsapi_key: Optional NewsAPI key (get free at newsapi.org)
        n:           Max headlines to retrieve

    Returns:
        List of NewsItem objects with title + sentiment score
    """
    key = newsapi_key or os.getenv("NEWSAPI_KEY")

    if key:
        items = _fetch_newsapi(key, n=n)
        if items:
            print(f"[sentiment] Fetched {len(items)} headlines via NewsAPI")
            return items

    print("[sentiment] Using RSS fallback …")
    items = _fetch_rss(n=n)
    print(f"[sentiment] Fetched {len(items)} headlines via RSS")
    return items


def get_sentiment_score(
    newsapi_key: Optional[str] = None,
    n: int = 20,
) -> Tuple[float, List[NewsItem]]:
    """
    Compute an aggregate Bitcoin market sentiment score.

    Returns:
        (score, headlines)
        score: float [-1.0, 1.0]
            > +0.05  bullish
            < -0.05  bearish
            else     neutral
    """
    headlines = get_news_headlines(newsapi_key=newsapi_key, n=n)

    if not headlines:
        print("[sentiment] No headlines found — returning neutral score 0.0")
        return 0.0, []

    avg_score = sum(h.sentiment for h in headlines) / len(headlines)
    avg_score = round(avg_score, 4)

    label = "BULLISH 🟢" if avg_score > 0.05 else ("BEARISH 🔴" if avg_score < -0.05 else "NEUTRAL 🟡")
    print(f"[sentiment] Score: {avg_score:+.4f}  →  {label}  ({len(headlines)} headlines)")
    return avg_score, headlines


def sentiment_label(score: float) -> str:
    """Human-readable label for a sentiment score."""
    if score > 0.20:   return "Strongly Bullish 🚀"
    if score > 0.05:   return "Bullish 🟢"
    if score < -0.20:  return "Strongly Bearish 💥"
    if score < -0.05:  return "Bearish 🔴"
    return "Neutral 🟡"
