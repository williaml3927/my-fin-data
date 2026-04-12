#!/usr/bin/env python3
"""
market_news.py — Market-Moving Headlines Pipeline
==================================================
Fetches the top market-moving headlines from NewsAPI, classifies each story
by impact type and affected market segment, and writes a single JSON file to
the GitHub data repo so the app can display a live-feel news feed on the
landing page without any runtime API calls.

Run schedule: daily via GitHub Actions (recommended: 7am UTC / 8am BST)

Output:  data_news/market_headlines.json
Schema:
  {
    "generated_at": "2025-04-12 07:00",
    "headlines": [
      {
        "id":           "1",
        "title":        "Fed holds rates steady, signals two cuts in 2025",
        "summary":      "One plain-English sentence for a beginner investor.",
        "source":       "Reuters",
        "url":          "https://...",
        "published_at": "2025-04-12T06:30:00Z",
        "category":     "Macro",          // Macro | Sector | Single Stock | Crypto | Earnings
        "impact":       "High",           // High | Medium | Low
        "impact_colour": "#ef4444",       // red=High, amber=Medium, grey=Low
        "affected":     "Bonds, Equities", // what asset classes / sectors are affected
        "sentiment":    "Bearish",        // Bullish | Bearish | Neutral
        "sentiment_colour": "#ef4444"
      },
      ...
    ]
  }
"""

import os
import json
import time
import base64
import requests
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

NEWSAPI_KEY       = os.environ.get("NEWSAPI_KEY", "")
BENZINGA_API_KEY  = os.environ.get("BENZINGA_API_KEY", "")
FINNHUB_API_KEY   = os.environ.get("FINNHUB_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# GitHub target repo — same pattern as your existing pipelines
GITHUB_TOKEN     = os.environ.get("GH_PAT", "")
GITHUB_REPO      = "williaml3927/my-fin-data"
OUTPUT_PATH      = "data_news/market_headlines.json"
GITHUB_API_BASE  = "https://api.github.com"

# How many headlines to keep in the final output
MAX_HEADLINES    = 10

# ── API endpoints ────────────────────────────────────────────────────────────
BENZINGA_URL  = "https://api.benzinga.com/api/v2/news"
FINNHUB_URL   = "https://finnhub.io/api/v1/news"
NEWSAPI_URL   = "https://newsapi.org/v2/top-headlines"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

IMPACT_COLOURS = {
    "High":   "#ef4444",   # red
    "Medium": "#f59e0b",   # amber
    "Low":    "#6b7280",   # grey
}

SENTIMENT_COLOURS = {
    "Bullish":  "#10b981",  # green
    "Bearish":  "#ef4444",  # red
    "Neutral":  "#6b7280",  # grey
}


def _safe_str(v, default=""):
    return str(v).strip() if v else default


def fetch_from_benzinga() -> list[dict]:
    """
    Primary source — Benzinga News API.
    Returns normalised article dicts with pre-tagged tickers and importance.
    Benzinga stories already carry sentiment and ticker tags so Claude's
    classification job is much lighter.
    """
    if not BENZINGA_API_KEY:
        return []
    try:
        resp = requests.get(
            BENZINGA_URL,
            params={
                "token":       BENZINGA_API_KEY,
                "pageSize":    30,
                "displayOutput": "full",
                "sort":        "created:desc",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  [BENZINGA] {resp.status_code} — {resp.text[:100]}")
            return []
        items = resp.json() if isinstance(resp.json(), list) else resp.json().get("stories", [])
        normalised = []
        for a in items:
            title = _safe_str(a.get("title"))
            if not title or "[Removed]" in title:
                continue
            # Benzinga provides ticker tags — join them for the "affected" field
            tickers = [t.get("name", "") for t in (a.get("stocks") or []) if t.get("name")]
            normalised.append({
                "title":        title,
                "source":       _safe_str(a.get("source") or "Benzinga"),
                "description":  _safe_str(a.get("teaser") or a.get("body", "")[:200]),
                "url":          _safe_str(a.get("url")),
                "published_at": _safe_str(a.get("created")),
                "tickers":      tickers[:5],     # pre-tagged tickers
                "importance":   a.get("importance", 0),  # 0=low 1=med 2=high
                "_source_api":  "benzinga",
            })
        print(f"  [BENZINGA] {len(normalised)} articles fetched")
        return normalised
    except Exception as e:
        print(f"  [BENZINGA] Failed: {e}")
        return []


def fetch_from_finnhub() -> list[dict]:
    """
    Fallback source — Finnhub General Market News.
    Free, fast, and covers major macro and company stories.
    """
    if not FINNHUB_API_KEY:
        return []
    try:
        resp = requests.get(
            FINNHUB_URL,
            params={"category": "general", "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  [FINNHUB] {resp.status_code} — {resp.text[:100]}")
            return []
        items = resp.json() if isinstance(resp.json(), list) else []
        normalised = []
        for a in items[:30]:
            title = _safe_str(a.get("headline"))
            if not title:
                continue
            # Convert unix timestamp
            ts = a.get("datetime", 0)
            published = (
                datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
                if ts else ""
            )
            normalised.append({
                "title":        title,
                "source":       _safe_str(a.get("source") or "Finnhub"),
                "description":  _safe_str(a.get("summary", "")[:200]),
                "url":          _safe_str(a.get("url")),
                "published_at": published,
                "tickers":      [],
                "importance":   1,
                "_source_api":  "finnhub",
            })
        print(f"  [FINNHUB] {len(normalised)} articles fetched")
        return normalised
    except Exception as e:
        print(f"  [FINNHUB] Failed: {e}")
        return []


def fetch_from_newsapi() -> list[dict]:
    """
    Last-resort source — NewsAPI business headlines.
    Used when both Benzinga and Finnhub fail or are unavailable.
    """
    if not NEWSAPI_KEY:
        return []
    try:
        resp = requests.get(
            NEWSAPI_URL,
            params={
                "language": "en",
                "category": "business",
                "pageSize": 30,
                "apiKey":   NEWSAPI_KEY,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  [NEWSAPI] {resp.status_code} — {resp.text[:100]}")
            return []
        items = resp.json().get("articles", [])
        normalised = []
        for a in items:
            title = _safe_str(a.get("title"))
            if not title or "[Removed]" in title:
                continue
            normalised.append({
                "title":        title,
                "source":       _safe_str(a.get("source", {}).get("name")),
                "description":  _safe_str(a.get("description", "")[:200]),
                "url":          _safe_str(a.get("url")),
                "published_at": _safe_str(a.get("publishedAt")),
                "tickers":      [],
                "importance":   1,
                "_source_api":  "newsapi",
            })
        print(f"  [NEWSAPI] {len(normalised)} articles fetched")
        return normalised
    except Exception as e:
        print(f"  [NEWSAPI] Failed: {e}")
        return []


def fetch_headlines() -> list[dict]:
    """
    Tiered fetch: Benzinga → Finnhub → NewsAPI.
    Uses the best available source, falling back automatically.
    Deduplicates by title similarity across sources.
    """
    articles = fetch_from_benzinga()

    # If Benzinga gave us fewer than 10 stories, supplement with Finnhub
    if len(articles) < 10:
        print("  [FETCH] Benzinga thin — supplementing with Finnhub")
        finnhub = fetch_from_finnhub()
        # Simple dedup: skip if title already seen (first 40 chars)
        seen_titles = {a["title"][:40].lower() for a in articles}
        for a in finnhub:
            if a["title"][:40].lower() not in seen_titles:
                articles.append(a)
                seen_titles.add(a["title"][:40].lower())

    # If still thin, fall back to NewsAPI
    if len(articles) < 10:
        print("  [FETCH] Still thin — supplementing with NewsAPI")
        newsapi = fetch_from_newsapi()
        seen_titles = {a["title"][:40].lower() for a in articles}
        for a in newsapi:
            if a["title"][:40].lower() not in seen_titles:
                articles.append(a)
                seen_titles.add(a["title"][:40].lower())

    print(f"  [FETCH] Total after merge: {len(articles)} articles")
    return articles


def classify_headlines(articles: list[dict]) -> list[dict]:
    """
    Use Claude Haiku to classify, score, and summarise each headline.
    Returns a list of structured headline dicts ready for the JSON output.
    Falls back to rule-based classification if API key is missing.
    """
    if not articles:
        return []

    # Build a compact list for the prompt — just title + source + description
    article_list = []
    for i, a in enumerate(articles[:25]):
        title = _safe_str(a.get("title"))
        if not title or "[Removed]" in title:
            continue
        entry = {
            "index":        i,
            "title":        title,
            "source":       _safe_str(a.get("source")),
            "description":  _safe_str(a.get("description"))[:200],
            "url":          _safe_str(a.get("url")),
            "published_at": _safe_str(a.get("published_at")),
            "source_api":   a.get("_source_api", "unknown"),
        }
        # Pass through Benzinga's pre-tagged tickers and importance
        # so Claude can use them directly rather than inferring
        if a.get("tickers"):
            entry["tickers"] = a["tickers"]
        if a.get("importance") is not None:
            entry["benzinga_importance"] = a["importance"]
        article_list.append(entry)

    if not article_list:
        return []

    if not (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip()):
        print("  [NEWS] No ANTHROPIC_API_KEY — using rule-based classification")
        return _rule_based_classify(article_list, articles)

    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")

    prompt = f"""You are a senior financial editor curating a market-moving headlines feed 
for a beginner-investor app. Today is {today}.

Below are up to 25 news headlines. Your job is to:
1. Select the {MAX_HEADLINES} most market-moving stories (ignore duplicates, PR pieces, 
   and listicles — focus on stories that actually move asset prices).
2. For each selected story, produce a structured classification.

Articles:
{json.dumps(article_list, indent=2)}

Return ONLY a JSON array of exactly up to {MAX_HEADLINES} objects, ordered by market 
importance (most important first). Each object must have:

- "index":      (int) the original index from the input list
- "summary":    (str) ONE plain-English sentence (max 20 words) explaining what happened 
                and why it matters to an investor. No jargon. Written for a beginner.
- "category":   (str) one of: "Macro" | "Sector" | "Single Stock" | "Crypto" | "Earnings"
- "impact":     (str) one of: "High" | "Medium" | "Low"
- "affected":   (str) brief description of what is affected, e.g. "Tech stocks, Growth equities"
                or "US Dollar, Bonds" or "Bitcoin, Crypto market"
- "sentiment":  (str) one of: "Bullish" | "Bearish" | "Neutral"

Rules:
- "High" impact = moves broad market indices or a major asset class
- "Medium" impact = affects a sector or well-known company meaningfully
- "Low" impact = relevant context but unlikely to move prices today
- Earnings beats/misses for major companies = always at least "Medium"
- Fed/ECB/central bank decisions = always "High" and "Macro"
- Single company CEO changes, minor product launches = "Low"

Respond ONLY with a valid JSON array, no other text."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=40,
        )
        if response.status_code != 200:
            print(f"  [NEWS] Claude API {response.status_code} — falling back to rules")
            return _rule_based_classify(article_list, articles)

        raw = response.json()["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        classified = json.loads(raw)

        # Merge classifications back with original article data
        result = []
        for i, item in enumerate(classified[:MAX_HEADLINES]):
            idx = item.get("index", 0)
            if idx >= len(article_list):
                continue
            base = article_list[idx]
            original = articles[idx] if idx < len(articles) else {}

            impact    = item.get("impact", "Medium")
            sentiment = item.get("sentiment", "Neutral")

            result.append({
                "id":             str(i + 1),
                "title":          base["title"],
                "summary":        item.get("summary", ""),
                "source":         base["source"],
                "url":            base["url"],
                "published_at":   base["published_at"],
                "category":       item.get("category", "Macro"),
                "impact":         impact,
                "impact_colour":  IMPACT_COLOURS.get(impact, "#6b7280"),
                "affected":       item.get("affected", ""),
                "sentiment":      sentiment,
                "sentiment_colour": SENTIMENT_COLOURS.get(sentiment, "#6b7280"),
            })

        print(f"  [NEWS] Classified {len(result)} market-moving headlines via Claude")
        return result

    except Exception as e:
        print(f"  [NEWS] Claude classification failed ({e}) — falling back to rules")
        return _rule_based_classify(article_list, articles)


def _rule_based_classify(article_list: list[dict], articles: list[dict]) -> list[dict]:
    """
    Fallback classifier when no API key is set.
    Uses keyword matching to assign rough categories and impact.
    Not as good as Claude but produces usable output.
    """
    MACRO_KEYWORDS    = ["fed", "federal reserve", "ecb", "interest rate", "inflation",
                         "gdp", "recession", "central bank", "treasury", "yield curve",
                         "jobs report", "cpi", "ppi", "fomc", "powell", "lagarde"]
    EARNINGS_KEYWORDS = ["earnings", "revenue", "profit", "quarterly", "eps", "beat",
                         "miss", "guidance", "forecast", "q1", "q2", "q3", "q4"]
    CRYPTO_KEYWORDS   = ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
                         "defi", "nft", "solana", "binance", "coinbase"]
    HIGH_KEYWORDS     = ["fed", "rate", "inflation", "recession", "crash", "surge",
                         "plunge", "soars", "tumbles", "federal reserve", "tariff"]

    result = []
    for i, item in enumerate(article_list[:MAX_HEADLINES]):
        title_lower = item["title"].lower()
        desc_lower  = item["description"].lower()
        combined    = title_lower + " " + desc_lower

        # Category
        if any(k in combined for k in CRYPTO_KEYWORDS):
            category = "Crypto"
        elif any(k in combined for k in EARNINGS_KEYWORDS):
            category = "Earnings"
        elif any(k in combined for k in MACRO_KEYWORDS):
            category = "Macro"
        else:
            category = "Sector"

        # Impact
        impact = "High" if any(k in combined for k in HIGH_KEYWORDS) else "Medium"

        # Sentiment — simple heuristic
        bearish_words = ["fall", "drop", "plunge", "tumble", "crash", "decline",
                         "loss", "miss", "warn", "concern", "risk", "fear"]
        bullish_words = ["rise", "surge", "soar", "beat", "record", "gain",
                         "growth", "strong", "rally", "boost", "climb"]
        bear_count = sum(1 for w in bearish_words if w in combined)
        bull_count = sum(1 for w in bullish_words if w in combined)

        if bull_count > bear_count:
            sentiment = "Bullish"
        elif bear_count > bull_count:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        # Summary — truncate description
        summary = item["description"][:120] + "..." if len(item["description"]) > 120 else item["description"]
        if not summary:
            summary = item["title"]

        result.append({
            "id":              str(i + 1),
            "title":           item["title"],
            "summary":         summary,
            "source":          item["source"],
            "url":             item["url"],
            "published_at":    item["published_at"],
            "category":        category,
            "impact":          impact,
            "impact_colour":   IMPACT_COLOURS.get(impact, "#6b7280"),
            "affected":        "",
            "sentiment":       sentiment,
            "sentiment_colour": SENTIMENT_COLOURS.get(sentiment, "#6b7280"),
        })

    print(f"  [NEWS] Rule-based classification: {len(result)} headlines")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_github(content: dict) -> bool:
    """
    Write the headlines JSON to GitHub via the Contents API.
    Creates the file if it doesn't exist, updates it if it does.
    Same pattern as calculate_stocks.py uses for partition files.
    """
    if not GITHUB_TOKEN:
        print("  [GITHUB] No GH_PAT set — writing locally only")
        with open("market_headlines.json", "w") as f:
            json.dump(content, f, indent=2)
        print("  [GITHUB] Written to market_headlines.json")
        return True

    url     = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/contents/{OUTPUT_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
    }

    # Check if file exists (need SHA to update)
    sha = None
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass

    payload_str  = json.dumps(content, indent=2, ensure_ascii=False)
    payload_b64  = base64.b64encode(payload_str.encode()).decode()

    body = {
        "message": f"[bot] Update market headlines — {content['generated_at']}",
        "content": payload_b64,
    }
    if sha:
        body["sha"] = sha

    try:
        resp = requests.put(url, headers=headers, json=body, timeout=20)
        if resp.status_code in (200, 201):
            print(f"  [GITHUB] Uploaded to {OUTPUT_PATH} ✓")
            return True
        else:
            print(f"  [GITHUB] Upload failed: {resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  [GITHUB] Upload error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Market Headlines Pipeline")
    print(f"Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Fetch raw headlines from NewsAPI
    print("\n[1] Fetching headlines from NewsAPI...")
    articles = fetch_headlines()

    if not articles:
        print("  No articles fetched — exiting without writing file")
        return

    # 2. Classify via Claude Haiku (or rule-based fallback)
    print(f"\n[2] Classifying {len(articles)} articles...")
    headlines = classify_headlines(articles)

    if not headlines:
        print("  No headlines classified — exiting")
        return

    # 3. Build output payload
    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "headline_count": len(headlines),
        "headlines": headlines,
    }

    print(f"\n[3] Built {len(headlines)} market-moving headlines")
    for h in headlines:
        print(f"  [{h['impact']:<6}] [{h['category']:<14}] {h['title'][:70]}")

    # 4. Upload to GitHub
    print(f"\n[4] Uploading to GitHub ({OUTPUT_PATH})...")
    upload_to_github(output)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
