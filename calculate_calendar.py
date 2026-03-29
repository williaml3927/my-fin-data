#!/usr/bin/env python3
"""
VI Pro — US Economic Calendar Pipeline
=======================================
Generates data_calendar/economic_calendar.json

Data sources (all free, no scraping):
  1. FRED API  — scheduled release dates for CPI, PPI, NFP, GDP, retail sales etc.
     Sign up free at https://fred.stlouisfed.org/docs/api/api_key.html
     Takes 2 minutes — just enter an email address.

  2. Hardcoded FOMC schedule — Fed publishes the full year's meeting dates in January.
     Updated manually each January when the Fed releases the schedule.
     Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

  3. US Treasury auction calendar — fetched from TreasuryDirect (no key needed).

Output: data_calendar/economic_calendar.json
  {
    "last_updated": "2026-03-29 10:00",
    "events": [
      {
        "date": "2026-04-02",
        "time": "08:30",
        "event": "Non-Farm Payrolls",
        "category": "Employment",
        "impact": "HIGH",
        "description": "...",
        "previous": "151K",
        "forecast": "140K",
        "actual": null,
        "source": "BLS"
      },
      ...
    ]
  }

Run:
  python calculate_calendar.py

Schedule: run weekly (e.g. every Sunday) via GitHub Actions cron.
"""

import json
import os
import requests
import time
from datetime import datetime, timedelta, date

# =============================================================================
# CONFIG
# =============================================================================
FRED_API_KEY = "8f93cb5bc6cf363bf8c830ae138ca7cf"

OUTPUT_DIR  = "data_calendar"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "economic_calendar.json")

# How many days ahead to fetch events
DAYS_AHEAD = 90

# =============================================================================
# FOMC MEETING SCHEDULE
# Hard-coded from the Fed's official calendar published each January.
# Update this list every January with the new year's dates.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# =============================================================================
FOMC_SCHEDULE = [
    # 2025 FOMC meetings (decision day = last day of 2-day meeting)
    {"date": "2025-01-29", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-03-19", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-05-07", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-06-18", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-07-30", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-09-17", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-11-05", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2025-12-17", "event": "FOMC Rate Decision",    "time": "14:00"},
    # FOMC Minutes (published ~3 weeks after each meeting)
    {"date": "2025-02-19", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2025-04-09", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2025-05-28", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2025-07-09", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2025-08-20", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2025-10-08", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2025-11-26", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-01-07", "event": "FOMC Minutes",          "time": "14:00"},
    # 2026 FOMC meetings
    {"date": "2026-01-28", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-03-18", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-05-06", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-06-17", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-07-29", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-09-16", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-11-04", "event": "FOMC Rate Decision",    "time": "14:00"},
    {"date": "2026-12-16", "event": "FOMC Rate Decision",    "time": "14:00"},
    # 2026 FOMC Minutes
    {"date": "2026-02-18", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-04-08", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-05-27", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-07-08", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-08-19", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-10-07", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2026-11-25", "event": "FOMC Minutes",          "time": "14:00"},
    {"date": "2027-01-06", "event": "FOMC Minutes",          "time": "14:00"},
]

# =============================================================================
# FRED SERIES — maps to human-readable event metadata
# Each entry: (series_id, event_name, category, impact, description, source)
# =============================================================================
FRED_SERIES = [
    # ── Employment ────────────────────────────────────────────────────────────
    ("PAYEMS",     "Non-Farm Payrolls",              "Employment",  "HIGH",
     "Total number of paid US workers excluding farm, government, private households "
     "and non-profit employees. The single most market-moving monthly release — "
     "a strong print tends to strengthen the dollar and pressure equities and crypto.",
     "BLS"),

    ("UNRATE",     "Unemployment Rate",              "Employment",  "HIGH",
     "Percentage of the labour force that is jobless and actively seeking work. "
     "Released with NFP on the first Friday of each month.",
     "BLS"),

    ("JTSJOL",     "JOLTS Job Openings",             "Employment",  "MEDIUM",
     "Number of unfilled job positions across the US economy. "
     "A high reading signals labour market tightness which can influence Fed policy.",
     "BLS"),

    ("ICSA",       "Initial Jobless Claims",         "Employment",  "MEDIUM",
     "Weekly count of people filing for unemployment benefits for the first time. "
     "Released every Thursday — a leading indicator of labour market health.",
     "DOL"),

    # ── Inflation ─────────────────────────────────────────────────────────────
    ("CPIAUCSL",   "CPI (Consumer Price Index)",     "Inflation",   "HIGH",
     "Measures the average change in prices paid by consumers for goods and services. "
     "The Fed's primary inflation gauge — a hot CPI print typically pressures risk assets "
     "as it raises expectations for higher interest rates.",
     "BLS"),

    ("CPILFESL",   "Core CPI (ex-Food & Energy)",    "Inflation",   "HIGH",
     "CPI excluding volatile food and energy prices. Considered a cleaner measure "
     "of underlying inflation trends and closely watched by the Federal Reserve.",
     "BLS"),

    ("PPIACO",     "PPI (Producer Price Index)",     "Inflation",   "MEDIUM",
     "Measures price changes from the perspective of sellers rather than consumers. "
     "Often seen as a leading indicator of future consumer inflation.",
     "BLS"),

    ("PCEPILFE",   "Core PCE Price Index",           "Inflation",   "HIGH",
     "The Fed's preferred inflation measure — Personal Consumption Expenditures "
     "excluding food and energy. Drives Federal Reserve rate decisions more directly "
     "than CPI.",
     "BEA"),

    # ── Growth ────────────────────────────────────────────────────────────────
    ("GDP",        "GDP (Advance Estimate)",         "Growth",      "HIGH",
     "First estimate of Gross Domestic Product growth for the prior quarter. "
     "Released approximately one month after the quarter ends. "
     "Two consecutive negative quarters technically defines a recession.",
     "BEA"),

    ("RSAFS",      "Retail Sales",                   "Growth",      "HIGH",
     "Monthly measure of total receipts at stores selling merchandise and related services. "
     "Consumer spending drives ~70% of US GDP — a key growth indicator.",
     "Census"),

    ("INDPRO",     "Industrial Production",          "Growth",      "MEDIUM",
     "Measures real output of manufacturing, mining, and electric and gas utilities. "
     "A gauge of factory activity and broader economic momentum.",
     "Fed"),

    ("TCU",        "Capacity Utilization",           "Growth",      "LOW",
     "Percentage of productive capacity being used in manufacturing, mining, and utilities. "
     "High utilization can signal inflationary pressure.",
     "Fed"),

    # ── Housing ───────────────────────────────────────────────────────────────
    ("HOUST",      "Housing Starts",                 "Housing",     "MEDIUM",
     "Number of new residential construction projects begun in a given month. "
     "A leading indicator of economic activity given the ripple effects on materials, "
     "appliances, and employment.",
     "Census"),

    ("EXHOSLUSM495S", "Existing Home Sales",         "Housing",     "MEDIUM",
     "Number of previously-owned homes sold during the month. "
     "Reflects consumer confidence and the health of the housing market.",
     "NAR"),

    # ── Consumer ─────────────────────────────────────────────────────────────
    ("UMCSENT",    "Michigan Consumer Sentiment",    "Consumer",    "MEDIUM",
     "Survey-based measure of consumer confidence in economic conditions. "
     "A leading indicator of consumer spending, which drives ~70% of US GDP.",
     "U-Michigan"),

    ("CSCICP03USM665S", "Conference Board Consumer Confidence", "Consumer", "MEDIUM",
     "Monthly survey of consumer attitudes about current and future economic conditions. "
     "High confidence typically supports consumer spending and risk asset prices.",
     "Conference Board"),

    # ── Trade & Business ─────────────────────────────────────────────────────
    ("BOPGSTB",    "Trade Balance",                  "Trade",       "MEDIUM",
     "Difference between US exports and imports of goods and services. "
     "A widening deficit can weigh on GDP and the dollar.",
     "BEA/Census"),

    ("DGORDER",    "Durable Goods Orders",           "Business",    "MEDIUM",
     "New orders placed with manufacturers for goods expected to last 3+ years. "
     "A leading indicator of manufacturing sector health and business investment.",
     "Census"),

    ("ISRATIO",    "ISM Manufacturing PMI",          "Business",    "HIGH",
     "Purchasing Managers Index — a reading above 50 signals expansion, below 50 "
     "signals contraction in the manufacturing sector. "
     "Released on the first business day of each month.",
     "ISM"),
]

# =============================================================================
# IMPACT COLOURS for AI Studio display
# =============================================================================
IMPACT_COLOURS = {
    "HIGH":   "#ef4444",   # red
    "MEDIUM": "#f59e0b",   # amber
    "LOW":    "#6b7280",   # grey
}

CATEGORY_ICONS = {
    "Employment":  "👥",
    "Inflation":   "📈",
    "Growth":      "📊",
    "Housing":     "🏠",
    "Consumer":    "🛒",
    "Trade":       "⚖️",
    "Business":    "🏭",
    "Fed Policy":  "🏦",
}


# =============================================================================
# FRED API HELPERS
# =============================================================================
def fred_get(endpoint, params):
    """Make a FRED API call. Returns parsed JSON or None on failure."""
    if not FRED_API_KEY or not FRED_API_KEY.strip():
        return None
    try:
        params["api_key"]   = FRED_API_KEY
        params["file_type"] = "json"
        resp = requests.get(
            f"https://api.stlouisfed.org/fred/{endpoint}",
            params=params,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"  [FRED] {endpoint} → HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [FRED] {endpoint} failed: {str(e)[:60]}")
    return None


def get_fred_release_dates(series_id, days_ahead=90):
    """
    Get upcoming scheduled release dates for a FRED series.
    Returns list of date strings ["2026-04-03", ...] sorted ascending.
    """
    today     = date.today()
    end_date  = today + timedelta(days=days_ahead)

    # Step 1: get the release_id for this series
    series_info = fred_get("series", {"series_id": series_id})
    if not series_info:
        return []
    release_id = series_info.get("seriess", [{}])[0].get("release_id")
    if not release_id:
        return []

    # Step 2: get upcoming release dates for that release
    data = fred_get("release/dates", {
        "release_id":         release_id,
        "realtime_start":     today.strftime("%Y-%m-%d"),
        "realtime_end":       end_date.strftime("%Y-%m-%d"),
        "sort_order":         "asc",
        "include_release_dates_with_no_data": "true",
    })
    if not data:
        return []

    dates = [d["date"] for d in data.get("release_dates", [])
             if today.strftime("%Y-%m-%d") <= d["date"] <= end_date.strftime("%Y-%m-%d")]
    return dates


def get_fred_latest_value(series_id):
    """Get the most recent actual value and its date for a series."""
    data = fred_get("series/observations", {
        "series_id":      series_id,
        "sort_order":     "desc",
        "limit":          2,
        "observation_end": date.today().strftime("%Y-%m-%d"),
    })
    if not data:
        return None, None
    obs = [o for o in data.get("observations", []) if o.get("value") != "."]
    if not obs:
        return None, None
    return obs[0].get("value"), obs[0].get("date")


# =============================================================================
# FOMC EVENT BUILDER
# =============================================================================
def build_fomc_events(days_ahead=90):
    """Build FOMC events from hardcoded schedule within the date window."""
    today    = date.today()
    end_date = today + timedelta(days=days_ahead)
    events   = []

    for item in FOMC_SCHEDULE:
        event_date = datetime.strptime(item["date"], "%Y-%m-%d").date()
        if today <= event_date <= end_date:
            is_decision = "Rate Decision" in item["event"]
            events.append({
                "date":        item["date"],
                "time":        item["time"],
                "event":       item["event"],
                "category":    "Fed Policy",
                "impact":      "HIGH" if is_decision else "MEDIUM",
                "impact_colour": IMPACT_COLOURS["HIGH"] if is_decision else IMPACT_COLOURS["MEDIUM"],
                "category_icon": CATEGORY_ICONS["Fed Policy"],
                "description": (
                    "The Federal Open Market Committee announces its federal funds rate decision. "
                    "This is the single most impactful event for all financial markets — "
                    "rate hikes strengthen the dollar and pressure equities and crypto, "
                    "while rate cuts or dovish signals tend to boost risk assets."
                    if is_decision else
                    "The FOMC releases minutes from its most recent policy meeting, "
                    "providing detailed insight into how committee members discussed "
                    "inflation, employment, and the rate outlook. "
                    "Can move markets if the tone is more hawkish or dovish than expected."
                ),
                "previous":    None,
                "forecast":    None,
                "actual":      None,
                "source":      "Federal Reserve",
                "days_until":  (event_date - today).days,
            })

    return events


# =============================================================================
# FRED EVENT BUILDER
# =============================================================================
def build_fred_events(days_ahead=90):
    """Fetch upcoming release dates from FRED and build event dicts."""
    if not FRED_API_KEY or not FRED_API_KEY.strip():
        print("  [CALENDAR] No FRED API key — skipping FRED economic releases.")
        print("             Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return []

    today    = date.today()
    events   = []

    for (series_id, event_name, category, impact, description, source) in FRED_SERIES:
        print(f"  [FRED] Fetching {event_name} ({series_id})...")
        release_dates = get_fred_release_dates(series_id, days_ahead)

        # Get most recent actual value for "previous" field
        prev_value, prev_date = get_fred_latest_value(series_id)

        for d in release_dates:
            event_date = datetime.strptime(d, "%Y-%m-%d").date()
            events.append({
                "date":          d,
                "time":          "08:30",   # most BLS/BEA releases at 8:30 ET
                "event":         event_name,
                "category":      category,
                "impact":        impact,
                "impact_colour": IMPACT_COLOURS.get(impact, "#6b7280"),
                "category_icon": CATEGORY_ICONS.get(category, "📅"),
                "description":   description,
                "previous":      prev_value,
                "forecast":      None,   # forecasts require a paid API
                "actual":        None,   # populated after release
                "source":        source,
                "days_until":    (event_date - today).days,
            })

        time.sleep(0.3)   # respect FRED rate limits (120 req/min free tier)

    return events


# =============================================================================
# FALLBACK EVENTS — fires when FRED key is absent
# A minimal hardcoded set of next-quarter events so the calendar
# is never completely empty.
# =============================================================================
def build_fallback_events(days_ahead=90):
    """
    Returns a small set of well-known recurring events with approximate dates.
    Used when the FRED API key is not configured.
    These are approximate — the first Friday of each month for NFP etc.
    """
    today    = date.today()
    end_date = today + timedelta(days=days_ahead)
    events   = []

    # Generate approximate first-Friday NFP dates for the next 3 months
    def first_friday(y, m):
        d = date(y, m, 1)
        while d.weekday() != 4:   # 4 = Friday
            d += timedelta(days=1)
        return d

    # NFP releases on the first Friday of each month
    for month_offset in range(4):
        ref = today.replace(day=1) + timedelta(days=32 * month_offset)
        nfp_date = first_friday(ref.year, ref.month)
        if today <= nfp_date <= end_date:
            events.append({
                "date":          nfp_date.strftime("%Y-%m-%d"),
                "time":          "08:30",
                "event":         "Non-Farm Payrolls",
                "category":      "Employment",
                "impact":        "HIGH",
                "impact_colour": IMPACT_COLOURS["HIGH"],
                "category_icon": CATEGORY_ICONS["Employment"],
                "description":   (
                    "Total number of paid US workers excluding farm, government, private "
                    "households and non-profit employees. The single most market-moving "
                    "monthly release — a strong print tends to strengthen the dollar and "
                    "pressure equities and crypto."
                ),
                "previous":      None,
                "forecast":      None,
                "actual":        None,
                "source":        "BLS",
                "days_until":    (nfp_date - today).days,
            })

    # CPI releases approximately the 10th-12th of each month
    for month_offset in range(4):
        ref = today.replace(day=1) + timedelta(days=32 * month_offset)
        cpi_date = date(ref.year, ref.month, 10)
        # Shift to next business day if weekend
        while cpi_date.weekday() >= 5:
            cpi_date += timedelta(days=1)
        if today <= cpi_date <= end_date:
            events.append({
                "date":          cpi_date.strftime("%Y-%m-%d"),
                "time":          "08:30",
                "event":         "CPI (Consumer Price Index)",
                "category":      "Inflation",
                "impact":        "HIGH",
                "impact_colour": IMPACT_COLOURS["HIGH"],
                "category_icon": CATEGORY_ICONS["Inflation"],
                "description":   (
                    "Measures the average change in prices paid by consumers for goods "
                    "and services. The Fed's primary inflation gauge — a hot CPI print "
                    "typically pressures risk assets as it raises expectations for higher "
                    "interest rates."
                ),
                "previous":      None,
                "forecast":      None,
                "actual":        None,
                "source":        "BLS",
                "days_until":    (cpi_date - today).days,
            })

    return events


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("VI Pro — Economic Calendar Pipeline")
    print("=" * 60)

    today     = date.today()
    end_date  = today + timedelta(days=DAYS_AHEAD)
    print(f"Fetching events: {today} → {end_date}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_events = []

    # ── FOMC events (always available — hardcoded) ────────────────────────────
    print("[1/3] Building FOMC schedule...")
    fomc_events = build_fomc_events(DAYS_AHEAD)
    all_events.extend(fomc_events)
    print(f"      {len(fomc_events)} FOMC events in window")

    # ── FRED economic releases ────────────────────────────────────────────────
    print("\n[2/3] Fetching FRED economic release dates...")
    fred_events = build_fred_events(DAYS_AHEAD)
    all_events.extend(fred_events)
    print(f"      {len(fred_events)} FRED events fetched")

    # ── Fallback if FRED key not set ──────────────────────────────────────────
    if not fred_events:
        print("\n[2/3] Using fallback approximate dates (set FRED_API_KEY for exact dates)...")
        fallback_events = build_fallback_events(DAYS_AHEAD)
        all_events.extend(fallback_events)
        print(f"      {len(fallback_events)} fallback events generated")

    # ── Sort by date ──────────────────────────────────────────────────────────
    print("\n[3/3] Sorting and saving...")
    all_events.sort(key=lambda x: (x["date"], x["time"]))

    # ── Deduplicate (same date + event name) ──────────────────────────────────
    seen = set()
    deduped = []
    for e in all_events:
        key = (e["date"], e["event"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # ── Group by date for easy rendering ─────────────────────────────────────
    by_date = {}
    for e in deduped:
        d = e["date"]
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(e)

    # ── Count by impact ───────────────────────────────────────────────────────
    high_count   = sum(1 for e in deduped if e["impact"] == "HIGH")
    medium_count = sum(1 for e in deduped if e["impact"] == "MEDIUM")
    low_count    = sum(1 for e in deduped if e["impact"] == "LOW")

    output = {
        "last_updated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window_start":  today.strftime("%Y-%m-%d"),
        "window_end":    end_date.strftime("%Y-%m-%d"),
        "total_events":  len(deduped),
        "high_impact":   high_count,
        "medium_impact": medium_count,
        "low_impact":    low_count,
        "fred_key_set":  bool(FRED_API_KEY and FRED_API_KEY.strip()),
        "note": (
            "Exact release dates from FRED API."
            if FRED_API_KEY and FRED_API_KEY.strip() else
            "FOMC dates are exact. Economic release dates are approximate. "
            "Set FRED_API_KEY for precise scheduled dates from the Federal Reserve."
        ),
        "events":       deduped,
        "by_date":      by_date,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved {len(deduped)} events to {OUTPUT_FILE}")
    print(f"  HIGH impact:   {high_count}")
    print(f"  MEDIUM impact: {medium_count}")
    print(f"  LOW impact:    {low_count}")

    # ── Preview next 7 days ───────────────────────────────────────────────────
    upcoming = [e for e in deduped if 0 <= e["days_until"] <= 7]
    if upcoming:
        print(f"\nNext 7 days ({len(upcoming)} events):")
        for e in upcoming:
            icon   = e.get("category_icon", "📅")
            impact = e["impact"]
            flag   = "🔴" if impact == "HIGH" else "🟡" if impact == "MEDIUM" else "⚪"
            print(f"  {flag} {e['date']} {e['time']} ET  {icon} {e['event']}")
    else:
        print("\nNo events in the next 7 days.")

    print()


if __name__ == "__main__":
    main()
