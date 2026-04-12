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

NEWSAPI_KEY      = os.environ.get("NEWSAPI_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# GitHub target repo — same pattern as your existing pipelines
GITHUB_TOKEN     = os.environ.get("GH_PAT", "")
GITHUB_REPO      = "williaml3927/my-fin-data"
OUTPUT_PATH      = "data_news/market_headlines.json"
GITHUB_API_BASE  = "https://api.github.com"

# How many headlines to fetch and store
MAX_HEADLINES    = 10

# NewsAPI endpoint — business + financial news, English only, sorted by relevance
NEWSAPI_URL      = "https://newsapi.org/v2/top-headlines"

# Market-moving search queries — cast wide to capture macro, earnings, Fed, etc.
NEWSAPI_PARAMS   = {
    "language": "en",
    "category": "business",
    "pageSize": 30,         # Fetch 30, trim to best 10 after classification
    "apiKey":   NEWSAPI_KEY,
}

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


def fetch_headlines() -> list[dict]:
    """
    Pull top business headlines from NewsAPI.
    Returns a list of raw article dicts.
    """
    if not NEWSAPI_KEY:
        print("  [NEWS] No NEWSAPI_KEY set — skipping fetch")
        return []

    try:
        resp = requests.get(NEWSAPI_URL, params=NEWSAPI_PARAMS, timeout=15)
        if resp.status_code != 200:
            print(f"  [NEWS] NewsAPI returned {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        articles = data.get("articles", [])
        print(f"  [NEWS] Fetched {len(articles)} raw articles from NewsAPI")
        return articles
    except Exception as e:
        print(f"  [NEWS] Fetch failed: {e}")
        return []


def classify_headlines(articles: list[dict]) -> list[dict]:
    """
    Use Claude Haiku to classify, score, and summarise each headline.
    Returns a list of structured headline dicts ready for the JSON output.
    Falls back to rule-based classification if API key is missing.
    """
    if not articles:
        return []

    # Build a compact list for the prompt — just title + source + description
    article_list = [
        {
            "index":       i,
            "title":       _safe_str(a.get("title")),
            "source":      _safe_str(a.get("source", {}).get("name")),
            "description": _safe_str(a.get("description"))[:200],
            "url":         _safe_str(a.get("url")),
            "published_at": _safe_str(a.get("publishedAt")),
        }
        for i, a in enumerate(articles[:25])   # Cap at 25 for prompt size
        if a.get("title") and "[Removed]" not in _safe_str(a.get("title"))
    ]

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
