import json
import requests
import concurrent.futures
from datetime import datetime, timedelta
import os
import time
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

# CoinGecko Demo API key — get free at https://www.coingecko.com/en/api
# Leave empty to use the public endpoint (slower, more rate limited)
COINGECKO_API_KEY = ""

# DeFiLlama — completely free, no key needed
DEFILLAMA_BASE    = "https://api.llama.fi"
DEFILLAMA_FEES    = "https://api.llama.fi/overview/fees"

# Gold market cap for Monetary Premium calculation (Method 6)
# Updated periodically — gold is ~$21T as of early 2026
GOLD_MARKET_CAP_USD = 21_000_000_000_000

# Bitcoin energy cost estimate for Method 8 (Cost of Production)
# Average all-in mining cost per BTC in USD — rough industry estimate
BTC_MINING_COST_USD = 35_000

# Number of top coins to process
TOP_N_COINS = 1000   # 4 pages × 250 per page

# =============================================================================
# BUCKET CLASSIFICATION
#
# Bucket A — Cash-Flow Tokens: valued on protocol fees, revenue, network utility
# Bucket B — Store of Value:   valued on monetary premium, security, scarcity
#
# All others default to Bucket A with limited data where DeFiLlama has no fees
# =============================================================================
BUCKET_B_SYMBOLS = {
    # Proof-of-Work / Store of Value assets — valued on scarcity,
    # not fee revenue. ETC added alongside BTC/LTC/DOGE family.
    "BTC", "LTC", "DOGE", "BCH", "XMR", "ZEC", "DASH", "BSV",
    "ETC", "KAS", "RVN", "ERG", "XMG",
}

# DeFiLlama protocol slug mapping — maps CoinGecko symbol → DeFiLlama slug
# Only needed for Bucket A tokens that have fee data on DeFiLlama
DEFILLAMA_SLUGS = {
    # Layer 1 blockchains
    "ETH":   "ethereum",
    "SOL":   "solana",
    "BNB":   "binance-smart-chain",
    "AVAX":  "avalanche",
    "TRX":   "tron",
    "TON":   "the-open-network",
    "ADA":   "cardano",
    "DOT":   "polkadot",
    "ATOM":  "cosmos",
    "NEAR":  "near",
    "FTM":   "fantom",
    "SUI":   "sui",
    "APT":   "aptos",
    "INJ":   "injective",
    "SEI":   "sei",
    "HBAR":  "hedera",
    "ALGO":  "algorand",
    "ONE":   "harmony",
    "ZIL":   "zilliqa",
    # Layer 2 / scaling
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "MATIC": "polygon",
    "IMX":   "immutable-x",
    "METIS": "metis",
    "MANTA": "manta",
    "BLAST": "blast",
    "SCROLL":"scroll",
    "ZKSYNC":"zksync-era",
    # DeFi protocols
    "UNI":   "uniswap",
    "AAVE":  "aave",
    "COMP":  "compound",
    "MKR":   "makerdao",
    "SNX":   "synthetix",
    "CRV":   "curve",
    "BAL":   "balancer",
    "SUSHI": "sushiswap",
    "1INCH": "1inch",
    "CAKE":  "pancakeswap",
    "JUP":   "jupiter",
    "RAY":   "raydium",
    "PENDLE":"pendle",
    "GMX":   "gmx",
    "DYDX":  "dydx",
    "PERP":  "perpetual-protocol",
    "GNS":   "gains-network",
    # Liquid staking
    "LDO":   "lido",
    "RPL":   "rocket-pool",
    "ANKR":  "ankr",
    "SFRXETH":"frax-ether",
    # Lending
    "MORPHO":"morpho",
    "EULER": "euler",
    # Infrastructure
    "LINK":  "chainlink",
    "BAND":  "band-protocol",
    "API3":  "api3",
    "GRT":   "the-graph",
    "FIL":   "filecoin",
    "STORJ": "storj",
    "AR":    "arweave",
    "RNDR":  "render-token",
    "AKT":   "akash-network",
    # Oracles / data
    "PYTH":  "pyth-network",
    # Stablecoins / RWA
    "FRAX":  "frax",
    "CRVUSD":"crvusd",
    "ONDO":  "ondo-finance",
}

# =============================================================================
# HELPERS
# =============================================================================
def _safe(val, fallback=None):
    if val is None: return fallback
    try:
        v = float(val)
        return v if v == v else fallback   # NaN check
    except (TypeError, ValueError):
        return fallback

def _headers():
    h = {"accept": "application/json"}
    if COINGECKO_API_KEY:
        h["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return h

# =============================================================================
# DATA FETCHERS
# =============================================================================
def fetch_coingecko_page(page_num):
    """Fetch one page of 250 coins from CoinGecko /coins/markets."""
    url = (
        f"https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&order=market_cap_desc&per_page=250&page={page_num}"
        f"&sparkline=false&price_change_percentage=7d,30d"
    )
    try:
        resp = requests.get(url, headers=_headers(), timeout=15)
        if resp.status_code == 429:
            time.sleep(60)
            resp = requests.get(url, headers=_headers(), timeout=15)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def fetch_coingecko_detail(coin_id):
    """
    Fetch extended data for a single coin — developer activity, community,
    active addresses (where available). Used for Method 7 (Metcalfe).
    Only called for top 100 coins to stay within rate limits.
    """
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        f"?localization=false&tickers=false&market_data=true"
        f"&community_data=true&developer_data=true&sparkline=false"
    )
    try:
        time.sleep(1.5)   # respect free tier rate limits
        resp = requests.get(url, headers=_headers(), timeout=15)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def fetch_defillama_fees():
    """
    Fetch annual protocol fee data from DeFiLlama free API.
    Returns dict: { protocol_slug: { annual_fees, annual_revenue,
                                     annual_holders_revenue } }
    """
    result = {}
    try:
        # Total fees paid by users
        resp = requests.get(
            f"{DEFILLAMA_FEES}?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true",
            timeout=20
        )
        if resp.status_code != 200:
            return result
        data = resp.json()

        for p in data.get("protocols", []):
            slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
            result[slug] = {
                "annual_fees":            _safe(p.get("total24h", 0)) * 365,
                "annual_revenue":         _safe(p.get("totalAllTime")) or
                                          (_safe(p.get("total24h", 0)) * 365 * 0.15),
                "annual_holders_revenue": None,
                "tvl":                    _safe(p.get("tvl")),
            }
    except Exception as e:
        print(f"  [DeFiLlama] fees fetch failed: {e}")

    # Fetch revenue to holders separately
    try:
        resp2 = requests.get(
            f"{DEFILLAMA_FEES}?excludeTotalDataChart=true"
            f"&excludeTotalDataChartBreakdown=true&dataType=dailyHoldersRevenue",
            timeout=20
        )
        if resp2.status_code == 200:
            data2 = resp2.json()
            for p in data2.get("protocols", []):
                slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
                if slug in result:
                    result[slug]["annual_holders_revenue"] = (
                        _safe(p.get("total24h", 0)) * 365
                    )
    except Exception:
        pass

    return result


def fetch_defillama_chain_fees():
    """
    Fetch chain-level fee data from DeFiLlama.
    L1 blockchains like Solana, Ethereum, Avalanche report fees
    under the chains endpoint, not the protocols endpoint.
    Returns dict: { chain_name_lower: { annual_fees, annual_revenue } }
    """
    result = {}
    try:
        resp = requests.get(
            "https://api.llama.fi/overview/fees?excludeTotalDataChart=true"
            "&excludeTotalDataChartBreakdown=true&dataType=dailyFees",
            timeout=20
        )
        if resp.status_code == 200:
            for chain in resp.json().get("allChains", []):
                name  = str(chain).lower().replace(" ", "-")
                result[name] = {"annual_fees": None, "annual_revenue": None}

        # Get actual fee values per chain
        resp2 = requests.get(
            "https://api.llama.fi/overview/fees/chains"
            "?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true",
            timeout=20
        )
        if resp2.status_code == 200:
            for p in resp2.json().get("protocols", []):
                name = (p.get("name") or "").lower().replace(" ", "-")
                result[name] = {
                    "annual_fees":    (_safe(p.get("total24h")) or 0) * 365,
                    "annual_revenue": (_safe(p.get("totalAllTime"))) or
                                      ((_safe(p.get("total24h")) or 0) * 365 * 0.3),
                    "tvl": _safe(p.get("tvl")),
                }
    except Exception as e:
        print(f"  [DeFiLlama] chain fees fetch failed: {e}")
    return result


# Mapping from token symbol to DeFiLlama chain name for L1 fee lookup.
# These tokens report fees at the chain level, not the protocol level.
CHAIN_FEE_MAP = {
    "SOL":   "solana",
    "ETH":   "ethereum",
    "BNB":   "bsc",
    "AVAX":  "avalanche",
    "MATIC": "polygon",
    "TRX":   "tron",
    "ADA":   "cardano",
    "DOT":   "polkadot",
    "NEAR":  "near",
    "FTM":   "fantom",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "ATOM":  "cosmos",
    "SUI":   "sui",
    "APT":   "aptos",
    "INJ":   "injective",
    "TON":   "ton",
    "HBAR":  "hedera",
    "ALGO":  "algorand",
    "SEI":   "sei",
    "XRP":   "ripple",
    "XLM":   "stellar",
    "VET":   "vechain",
    "ONE":   "harmony",
    "ZIL":   "zilliqa",
    "IMX":   "immutable",
}


def fetch_bitcoin_hashrate():
    """
    Fetch current Bitcoin hashrate from CoinGecko.
    Returns hashrate in TH/s or None.
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin"
            "?localization=false&tickers=false&market_data=true"
            "&community_data=false&developer_data=false",
            headers=_headers(), timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            return _safe(data.get("market_data", {}).get("max_supply"))
    except Exception:
        pass
    return None


def fetch_gold_price():
    """
    Fetch current gold price per troy oz from a free public API.
    Falls back to hardcoded estimate if unavailable.
    """
    try:
        # Use exchangerate-api metals endpoint (free)
        resp = requests.get(
            "https://api.metals.live/v1/spot/gold",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            price = _safe(data[0].get("gold") if isinstance(data, list) else data.get("gold"))
            if price:
                # Gold market cap = price × total ounces above ground (~208,874 tonnes)
                troy_oz = 208_874 * 1000 * 32.1507
                return price * troy_oz
    except Exception:
        pass
    return GOLD_MARKET_CAP_USD


# =============================================================================
# VALUATION METHODS
# =============================================================================

# ── Bucket A Methods ──────────────────────────────────────────────────────────

def method_1_dcf_fees(annual_fees, growth_rate=0.15, discount_rate=0.30,
                       years=5, fdv=None):
    """
    Method 1 — DCF on Protocol Fees
    Projects annual fees forward, discounts to present value.
    Uses a high discount rate (30%) appropriate for crypto risk.
    Returns fair value of the entire protocol (not per token).
    """
    if not annual_fees or annual_fees <= 0:
        return None
    try:
        pv = 0
        cf = annual_fees
        for t in range(1, years + 1):
            cf *= (1 + growth_rate)
            pv += cf / (1 + discount_rate) ** t
        # Terminal value using Gordon Growth Model (5% perpetual growth)
        terminal = (cf * 1.05) / (discount_rate - 0.05)
        pv += terminal / (1 + discount_rate) ** years
        return round(pv, 2)
    except Exception:
        return None


def method_2_ps_analog(market_cap, annual_fees):
    """
    Method 2 — Market Cap / Fees (P/S Analog)
    Lower is cheaper. Returns the ratio.
    """
    if not annual_fees or annual_fees <= 0 or not market_cap or market_cap <= 0:
        return None
    return round(market_cap / annual_fees, 2)


def method_3_pe_analog(fdv, annual_holders_revenue):
    """
    Method 3 — FDV / Net Fee Capture (P/E Analog)
    Fee revenue actually accruing to token holders or burned.
    Returns the ratio — lower is cheaper.
    """
    if not annual_holders_revenue or annual_holders_revenue <= 0:
        return None
    if not fdv or fdv <= 0:
        return None
    return round(fdv / annual_holders_revenue, 2)


def method_7_metcalfe(market_cap, active_addresses=None, volume=None,
                       prev_market_cap=None):
    """
    Method 7 — Metcalfe's Law Network Valuation
    V ∝ N² where N = network participants.

    If active_addresses are available use them directly.
    Otherwise proxy with volume/price as a transaction count estimate.

    Returns a Metcalfe ratio: actual_mc / metcalfe_value
    < 1 = undervalued relative to network size
    > 1 = overvalued relative to network size
    """
    if not market_cap or market_cap <= 0:
        return None
    try:
        if active_addresses and active_addresses > 0:
            metcalfe_value = (active_addresses ** 2) * 1e-6   # scale constant
            if metcalfe_value > 0:
                return round(market_cap / metcalfe_value, 4)

        # Proxy: use volume as network activity signal
        if volume and volume > 0:
            # Ratio of market cap to volume — lower = more genuine activity
            return round(market_cap / volume, 2)
    except Exception:
        pass
    return None


def method_10_dilution(circulating_supply, total_supply, max_supply,
                        price, market_cap):
    """
    Method 10 — Dilution-Adjusted Valuation
    Measures how much supply overhang could dilute current holders.

    dilution_risk: 0-100%, lower is better (less future dilution)
    fully_diluted_premium: FDV / MC ratio — how much premium for locked supply
    """
    if not circulating_supply or circulating_supply <= 0:
        return None
    try:
        effective_total = max_supply or total_supply or circulating_supply
        dilution_pct    = round((circulating_supply / effective_total) * 100, 2)
        fdv             = price * effective_total if price else None
        fd_premium      = round(fdv / market_cap, 2) if fdv and market_cap else None
        return {
            "circulating_pct":   dilution_pct,       # % of max supply circulating
            "supply_overhang":   round(100 - dilution_pct, 2),  # % still to be issued
            "fully_diluted_valuation": round(fdv, 2) if fdv else None,
            "fdv_to_mc_ratio":   fd_premium,          # 1.0 = fully circulating
        }
    except Exception:
        return None


# ── Bucket B Methods ──────────────────────────────────────────────────────────

def method_6_monetary_premium(market_cap, gold_market_cap):
    """
    Method 6 — Monetary Premium / TAM vs Gold
    What percentage of gold's market cap has this asset captured?
    What price would BTC be at various gold capture scenarios?
    """
    if not market_cap or market_cap <= 0 or not gold_market_cap:
        return None
    try:
        gold_capture_pct = round((market_cap / gold_market_cap) * 100, 4)
        return {
            "gold_capture_pct":     gold_capture_pct,
            "price_at_25pct_gold":  None,   # filled in by caller who has price+supply
            "price_at_50pct_gold":  None,
            "price_at_100pct_gold": None,
        }
    except Exception:
        return None


def method_8_cost_of_production(symbol, market_cap, circulating_supply,
                                  current_price):
    """
    Method 8 — Cost of Production / Security Floor
    For PoW assets: estimated mining cost provides a lower-bound floor.

    Bitcoin: uses industry estimate of all-in mining cost per coin.
    Other PoW: estimated proportionally from hashrate security spend.

    Returns:
      production_cost_usd: estimated cost to produce one coin
      premium_to_cost: current price / production cost
    """
    if symbol == "BTC":
        cost = BTC_MINING_COST_USD
    elif symbol in {"LTC", "BCH", "BSV"}:
        cost = BTC_MINING_COST_USD * 0.05   # rough proxy — far less security spend
    elif symbol == "DOGE":
        cost = BTC_MINING_COST_USD * 0.02
    elif symbol == "XMR":
        cost = BTC_MINING_COST_USD * 0.10
    else:
        return None   # no reliable estimate for unknown PoW coins

    if not current_price or current_price <= 0:
        return None

    return {
        "estimated_production_cost_usd": cost,
        "current_price_usd":             round(current_price, 4),
        "premium_to_production_cost":    round(current_price / cost, 3),
        "note": "Production cost is an industry estimate, not an exact figure",
    }


# =============================================================================
# QUALITY SCORING FOR CRYPTO
#
# Six signals per bucket — all computable from free APIs.
# Returns a dict with individual signal scores AND a final_score_pct
# so AI Studio can display the breakdown rather than just a number.
#
# ── Bucket A — Cash-Flow Protocols ───────────────────────────────────────────
#
#  1. Fee Growth (0-2 pts)
#     YoY change in annual fees — is the protocol growing its revenue?
#     Proxy: fee/TVL ratio trend using current snapshot vs market context
#     >20% growth → 2  |  >0% → 1  |  declining → 0
#
#  2. Capital Efficiency (0-2 pts)
#     Fee/TVL ratio — how much fee revenue per dollar of capital locked
#     >5% → 2  |  >1% → 1  |  <1% → 0
#
#  3. Holder Value Accrual (0-2 pts)
#     Holders revenue as % of total fees — does value reach token holders?
#     >30% → 2  |  >10% → 1  |  <10% or unknown → 0
#
#  4. Network Demand (0-2 pts)
#     Volume/MC ratio — genuine economic activity vs passive holding
#     >5% → 2  |  >1% → 1  |  <1% → 0
#
#  5. Dilution Control (0-1 pt)
#     Circulating % of max supply — less future dilution = stronger
#     >80% → 1  |  <80% → 0
#
#  6. Protocol Maturity (0-1 pt)
#     MC rank proxy — top-50 tokens have survived multiple cycles
#     rank ≤ 50 → 1  |  else → 0
#
# ── Bucket B — Store of Value ─────────────────────────────────────────────────
#
#  1. Scarcity (0-2 pts)
#     Supply overhang — less remaining issuance = stronger scarcity
#     overhang <5% → 2  |  <20% → 1  |  >20% → 0
#
#  2. Adoption Momentum (0-2 pts)
#     Gold capture % — higher = more monetary adoption
#     >10% → 2  |  >1% → 1  |  <1% → 0
#
#  3. Security Premium (0-2 pts)
#     Price vs production cost — above cost = sustainable, extreme premium = risk
#     1x-5x above cost → 2  |  5x-20x → 1  |  below cost or >20x → 0
#
#  4. Monetary Premium Quality (0-2 pts)
#     FDV/MC ratio — close to 1.0 means supply is already in circulation
#     <1.05 → 2  |  <1.20 → 1  |  >1.20 → 0
#
#  5. Dilution Control (0-1 pt)
#     Same as Bucket A
#
#  6. Market Resilience (0-1 pt)
#     30d price change vs broader market — holding value in downturns
#     30d change > -10% → 1  |  worse → 0
# =============================================================================
def score_crypto_quality(bucket, mc_ps_ratio, mc_pe_ratio, dilution,
                          metcalfe_ratio, gold_capture_pct,
                          premium_to_cost, price_change_7d,
                          price_change_30d, annual_fees=None,
                          annual_holders_rev=None, tvl=None,
                          volume=None, mc=None, rank=None,
                          fdv_mc_ratio=None, network_econ=None,
                          symbol=None):
    """
    network_econ: the pre-computed network_economics dict from
                  fetch_network_economics(). When provided, quality signals
                  read directly from chart data so the Quality tab and
                  Network Economics charts always show identical values.

    Returns:
      {
        "scores": {
            signal_name: { "score": int, "max": int, "value": raw_value,
                           "chart_source": str }
        },
        "final_score":     int (0-10),
        "final_score_pct": float,
        "classification":  str,
      }
    """
    signals = {}
    current_year = str(datetime.now().year)

    # ── Extract values from Network Economics charts where available ──────────
    # This ensures the quality score and the charts always agree.
    # Falls back to raw inputs if network_econ is not available.

    ne = network_econ or {}

    # Chart 4 — Capital Efficiency → fee/TVL ratio
    cap_eff_chart  = ne.get("capital_efficiency", {})
    curr_cap_eff   = cap_eff_chart.get(current_year, {})
    fee_tvl_chart  = curr_cap_eff.get("fee_tvl_ratio_pct")
    real_yield_chart = curr_cap_eff.get("real_yield_pct")

    # Chart 1 — Protocol Revenue & Fees → fees and revenue current year
    rev_fees_chart = ne.get("protocol_revenue_and_fees", {})
    curr_rev_fees  = rev_fees_chart.get(current_year, {})
    fees_chart_m   = curr_rev_fees.get("fees_usd")      # $M
    rev_chart_m    = curr_rev_fees.get("revenue_usd")   # $M
    holders_pct_chart = (
        round(rev_chart_m / fees_chart_m * 100, 1)
        if rev_chart_m and fees_chart_m and fees_chart_m > 0 else None
    )

    # Chart 3 — Issuance vs Burns → supply overhang and FDV/MC
    issuance_chart = ne.get("issuance_vs_burns", {})
    curr_issuance  = issuance_chart.get(current_year, {})
    overhang_chart = curr_issuance.get("supply_overhang_pct")
    fdv_mc_chart   = curr_issuance.get("fdv_to_mc")

    # Chart 2 — Treasury vs Emissions (used for context, not direct scoring)
    treasury_chart = ne.get("treasury_vs_emissions", {})
    curr_treasury  = treasury_chart.get(current_year, {})
    treasury_m     = curr_treasury.get("treasury_usd_m")

    # ── Determine whether to use chart values or raw fallbacks ───────────────
    fee_tvl     = fee_tvl_chart  if fee_tvl_chart  is not None else (
                  (annual_fees / tvl * 100) if annual_fees and tvl and tvl > 0 else None)
    holders_pct = holders_pct_chart if holders_pct_chart is not None else (
                  (annual_holders_rev / annual_fees * 100)
                  if annual_holders_rev and annual_fees and annual_fees > 0 else None)
    overhang    = overhang_chart if overhang_chart is not None else (
                  (dilution or {}).get("supply_overhang"))
    fdv_mc_use  = fdv_mc_chart   if fdv_mc_chart   is not None else fdv_mc_ratio

    # ── Remaining helpers from raw data (no chart equivalent) ────────────────
    circ_pct   = (dilution or {}).get("circulating_pct")
    vol_mc_pct = (volume / mc * 100) if volume and mc and mc > 0 else None

    # ── Chart source label for display ───────────────────────────────────────
    def _src(used_chart):
        return "Network Economics chart" if used_chart else "raw calculation"

    if bucket == "A":
        # All signals scored 0-10 for gradation.
        # KEY RULE: missing data scores NEUTRAL (5), never 0.
        # Absence of DeFiLlama coverage ≠ absence of protocol activity.
        # L1 chains (SOL, ETH) and infrastructure tokens (LINK, GRT)
        # generate real fees but through channels DeFiLlama doesn't
        # always expose under a single protocol slug.

        fee_data_available  = fee_tvl is not None
        mc_fee_available    = mc_ps_ratio is not None
        holders_available   = holders_pct is not None

        # Tokens in CHAIN_FEE_MAP or known infrastructure get staking proxy
        _sym = symbol or ""
        is_l1_or_infra = _sym in CHAIN_FEE_MAP or _sym in {
            # Infrastructure / oracle / storage — distribute via node rewards
            "LINK", "GRT", "RNDR", "FIL", "AR", "AKT", "PYTH",
            "BAND", "API3", "STORJ",
            # Metaverse / gaming — revenue from platform fees not DeFi
            "SAND", "MANA", "BLUR", "IMX",
            # Identity / social
            "WLD",
        }

        # 1. Fee Growth — fee/TVL ratio when available, vol/MC proxy otherwise
        if fee_data_available:
            if fee_tvl >= 20:   s1 = 10
            elif fee_tvl >= 10: s1 = 8
            elif fee_tvl >= 5:  s1 = 6
            elif fee_tvl >= 2:  s1 = 4
            elif fee_tvl >= 1:  s1 = 2
            else:               s1 = 1
        else:
            # Volume proxy — high vol/MC shows genuine network activity
            if vol_mc_pct and vol_mc_pct >= 10:   s1 = 6
            elif vol_mc_pct and vol_mc_pct >= 5:  s1 = 5
            elif vol_mc_pct and vol_mc_pct >= 2:  s1 = 4
            else:                                  s1 = 5   # neutral
        signals["fee_growth"] = {
            "label": "Fee Growth", "score": s1, "max": 10,
            "value": round(fee_tvl, 2) if fee_tvl else None,
            "unit": "fee/tvl %" if fee_data_available else "vol/MC proxy",
            "chart_source": _src(fee_tvl_chart is not None),
            "chart_ref": "Network Economics → Capital Efficiency",
            "data_available": fee_data_available,
        }

        # 2. Capital Efficiency — MC/Fees when available, rank proxy otherwise
        if mc_fee_available:
            if mc_ps_ratio < 5:    s2 = 10
            elif mc_ps_ratio < 10: s2 = 8
            elif mc_ps_ratio < 20: s2 = 6
            elif mc_ps_ratio < 50: s2 = 4
            elif mc_ps_ratio < 100:s2 = 2
            else:                  s2 = 1
        else:
            # MC rank proxy — top protocols earn their valuation through usage
            if rank and rank <= 10:   s2 = 7
            elif rank and rank <= 25: s2 = 6
            elif rank and rank <= 50: s2 = 5
            else:                     s2 = 5   # neutral
        signals["capital_efficiency"] = {
            "label": "Capital Efficiency", "score": s2, "max": 10,
            "value": mc_ps_ratio,
            "unit": "MC/fees ratio" if mc_fee_available else "rank proxy",
            "chart_source": "Protocol Metrics (MC/Fees)" if mc_fee_available
                            else "rank proxy (fee data unavailable)",
            "chart_ref": "Network Economics → Capital Efficiency",
            "data_available": mc_fee_available,
        }

        # 3. Holder Value Accrual — direct fee split when available
        # L1 chains distribute value via staking/validator rewards — this
        # is real holder value accrual but DeFiLlama labels it differently.
        if holders_available:
            if holders_pct >= 50:   s3 = 10
            elif holders_pct >= 30: s3 = 8
            elif holders_pct >= 20: s3 = 6
            elif holders_pct >= 10: s3 = 4
            elif holders_pct >= 5:  s3 = 2
            else:                   s3 = 1
        elif is_l1_or_infra:
            # L1/infra tokens distribute value through staking rewards
            # Award partial credit based on network maturity (rank proxy)
            if rank and rank <= 10:   s3 = 6
            elif rank and rank <= 30: s3 = 5
            else:                     s3 = 4
        else:
            s3 = 5   # truly unknown → neutral
        signals["holder_value_accrual"] = {
            "label": "Holder Value Accrual", "score": s3, "max": 10,
            "value": round(holders_pct, 1) if holders_pct else None,
            "unit": "% of fees to holders" if holders_available
                    else "staking/validator proxy",
            "chart_source": _src(holders_pct_chart is not None),
            "chart_ref": "Network Economics → Protocol Revenue & Fees",
            "data_available": holders_available,
        }

        # 4. Network Demand — volume/MC % (0-10), neutral if unavailable
        if vol_mc_pct is not None:
            if vol_mc_pct >= 20:   s4 = 10
            elif vol_mc_pct >= 10: s4 = 8
            elif vol_mc_pct >= 5:  s4 = 6
            elif vol_mc_pct >= 2:  s4 = 4
            elif vol_mc_pct >= 1:  s4 = 2
            else:                  s4 = 1
        else:
            s4 = 5   # neutral
        signals["network_demand"] = {
            "label": "Network Demand", "score": s4, "max": 10,
            "value": round(vol_mc_pct, 2) if vol_mc_pct else None,
            "unit": "volume/MC %",
            "chart_source": "raw calculation (CoinGecko volume)",
            "chart_ref": None,
            "data_available": vol_mc_pct is not None,
        }

        # 5. Dilution Control — circulating % of max supply (0-10)
        s5 = 0
        if circ_pct is not None:
            if circ_pct >= 95:   s5 = 10
            elif circ_pct >= 85: s5 = 8
            elif circ_pct >= 70: s5 = 6
            elif circ_pct >= 50: s5 = 4
            elif circ_pct >= 30: s5 = 2
        signals["dilution_control"] = {
            "label": "Dilution Control", "score": s5, "max": 10,
            "value": circ_pct, "unit": "circulating %",
            "chart_source": _src(overhang_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # 6. Protocol Maturity — MC rank (0-10)
        s6 = 0
        if rank is not None:
            if rank <= 10:    s6 = 10
            elif rank <= 25:  s6 = 8
            elif rank <= 50:  s6 = 6
            elif rank <= 100: s6 = 4
            elif rank <= 200: s6 = 2
        signals["protocol_maturity"] = {
            "label": "Protocol Maturity", "score": s6, "max": 10,
            "value": rank, "unit": "MC rank",
            "chart_source": "raw calculation (CoinGecko rank)",
            "chart_ref": None,
        }

    elif bucket == "B":
        # 1. Scarcity — supply overhang % (0-10, lower overhang = better)
        # Neutral (5) if data unavailable
        s1 = 5   # default neutral
        if overhang is not None:
            ov = overhang
            if ov <= 2:    s1 = 10
            elif ov <= 5:  s1 = 8
            elif ov <= 10: s1 = 6
            elif ov <= 20: s1 = 4
            elif ov <= 40: s1 = 2
            else:          s1 = 1
        signals["scarcity"] = {
            "label": "Scarcity", "score": s1, "max": 10,
            "value": overhang, "unit": "supply overhang %",
            "chart_source": _src(overhang_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # 2. Adoption Momentum — gold capture % (0-10)
        # For small PoW coins (DOGE, RVN, ERG etc.) gold_capture_pct will
        # be tiny (<0.1%) which scored 0 — unfair. Use MC rank as a
        # relative adoption proxy alongside gold capture.
        s2 = 0
        if gold_capture_pct is not None and gold_capture_pct >= 0.5:
            if gold_capture_pct >= 50:    s2 = 10
            elif gold_capture_pct >= 20:  s2 = 8
            elif gold_capture_pct >= 10:  s2 = 6
            elif gold_capture_pct >= 3:   s2 = 4
            elif gold_capture_pct >= 0.5: s2 = 2
        else:
            # Rank-based adoption proxy for small-cap PoW coins
            if rank and rank <= 20:    s2 = 5
            elif rank and rank <= 50:  s2 = 4
            elif rank and rank <= 100: s2 = 3
            elif rank and rank <= 200: s2 = 2
            else:                      s2 = 1
        signals["adoption_momentum"] = {
            "label": "Adoption Momentum", "score": s2, "max": 10,
            "value": round(gold_capture_pct, 4) if gold_capture_pct else None,
            "unit": "gold capture %" if (gold_capture_pct and gold_capture_pct >= 0.5)
                    else "rank proxy",
            "chart_source": "SOV Metrics (monetary premium)",
            "chart_ref": None,
        }

        # 3. Security Premium — price / production cost (0-10)
        # Non-PoW coins in Bucket B (XMG) don't have production cost
        # Award neutral (5) when unavailable — not penalised
        s3 = 5   # default neutral
        if premium_to_cost is not None:
            if 1.0 < premium_to_cost <= 2:    s3 = 10
            elif 1.0 < premium_to_cost <= 3:  s3 = 8
            elif 1.0 < premium_to_cost <= 5:  s3 = 6
            elif 1.0 < premium_to_cost <= 10: s3 = 4
            elif premium_to_cost > 10:        s3 = 2
            elif premium_to_cost <= 1:        s3 = 1
        signals["security_premium"] = {
            "label": "Security Premium", "score": s3, "max": 10,
            "value": premium_to_cost, "unit": "price / production cost",
            "chart_source": "SOV Metrics (cost of production)",
            "chart_ref": None,
        }

        # 4. Monetary Premium Quality — FDV/MC ratio (0-10)
        s4 = 0
        if fdv_mc_use is not None:
            if fdv_mc_use <= 1.02:  s4 = 10
            elif fdv_mc_use <= 1.05:s4 = 8
            elif fdv_mc_use <= 1.10:s4 = 6
            elif fdv_mc_use <= 1.20:s4 = 4
            elif fdv_mc_use <= 1.50:s4 = 2
        signals["monetary_premium_quality"] = {
            "label": "Monetary Premium Quality", "score": s4, "max": 10,
            "value": fdv_mc_use, "unit": "FDV/MC ratio",
            "chart_source": _src(fdv_mc_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # 5. Dilution Control (0-10)
        s5 = 0
        if circ_pct is not None:
            if circ_pct >= 95:   s5 = 10
            elif circ_pct >= 85: s5 = 8
            elif circ_pct >= 70: s5 = 6
            elif circ_pct >= 50: s5 = 4
            elif circ_pct >= 30: s5 = 2
        signals["dilution_control"] = {
            "label": "Dilution Control", "score": s5, "max": 10,
            "value": circ_pct, "unit": "circulating %",
            "chart_source": _src(overhang_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # 6. Market Resilience — 30d price vs broader market (0-10)
        s6 = 0
        chg = price_change_30d or 0
        if chg >= 20:    s6 = 10
        elif chg >= 10:  s6 = 8
        elif chg >= 0:   s6 = 6
        elif chg >= -10: s6 = 4
        elif chg >= -25: s6 = 2
        signals["market_resilience"] = {
            "label": "Market Resilience", "score": s6, "max": 10,
            "value": round(price_change_30d, 1) if price_change_30d else None,
            "unit": "30d price change %",
            "chart_source": "raw calculation (CoinGecko price)",
            "chart_ref": None,
        }

    # ── Aggregate ─────────────────────────────────────────────────────────────
    # All 6 signals now score 0-10, total max = 60
    total_score = sum(s["score"] for s in signals.values())
    total_max   = sum(s["max"]   for s in signals.values())   # should be 60
    final_pct   = round(total_score / total_max * 100, 1) if total_max > 0 else 0
    final_10    = round(total_score / total_max * 10, 1)  if total_max > 0 else 0

    if final_pct >= 70:    classification = "Strong"
    elif final_pct >= 50:  classification = "Moderate"
    elif final_pct >= 30:  classification = "Weak"
    else:                  classification = "Poor"

    return {
        "scores":          signals,
        "final_score":     final_10,
        "final_score_pct": final_pct,
        "classification":  classification,
    }


# =============================================================================
# FEAR & GREED INDEX — PER TOKEN
#
# Calculates a token-specific Fear & Greed score (0-100) using signals
# already available from CoinGecko markets data. No extra API calls needed.
#
# Components and weights:
#
#   Price Momentum (30%)
#     30-day price change — rising prices = greed, falling = fear
#
#   Volume Momentum (25%)
#     Volume / Market Cap ratio vs typical baseline — high volume on up
#     moves = greed, high volume on down moves = fear
#
#   Volatility (20%)
#     Magnitude of recent price swings — extreme moves in either direction
#     indicate fear (panic selling) or greed (FOMO buying)
#     Combined with direction to determine which
#
#   Market Cap Rank Trend (15%)
#     Short-term rank change — rising rank = greed, falling = fear
#     (rank not available directly, proxied from 7d vs 30d momentum)
#
#   Supply Dilution Pressure (10%)
#     High supply overhang = fear (selling pressure expected)
#     Low supply overhang = greed (scarcity)
#
# Score interpretation:
#   0-20   Extreme Fear
#   21-40  Fear
#   41-60  Neutral
#   61-80  Greed
#   81-100 Extreme Greed
#
# Additionally fetches the overall market Fear & Greed from alternative.me
# once per run and attaches it to every coin as context.
# =============================================================================

MARKET_FEAR_GREED_CACHE = {}   # cached once per run

def fetch_market_fear_greed():
    """
    Fetch the overall crypto market Fear & Greed Index from alternative.me.
    Free, no API key needed. Cached for the whole run.
    Returns dict with value (0-100), classification and timestamp.
    """
    global MARKET_FEAR_GREED_CACHE
    if MARKET_FEAR_GREED_CACHE:
        return MARKET_FEAR_GREED_CACHE
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            entry = data.get("data", [{}])[0]
            MARKET_FEAR_GREED_CACHE = {
                "value":          int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
                "timestamp":      entry.get("timestamp"),
            }
            return MARKET_FEAR_GREED_CACHE
    except Exception:
        pass
    MARKET_FEAR_GREED_CACHE = {
        "value": 50, "classification": "Neutral", "timestamp": None
    }
    return MARKET_FEAR_GREED_CACHE


def _fg_label(score):
    """Convert 0-100 score to label."""
    if score >= 81:  return "Extreme Greed"
    if score >= 61:  return "Greed"
    if score >= 41:  return "Neutral"
    if score >= 21:  return "Fear"
    return "Extreme Fear"


def calc_fear_greed(price_change_7d, price_change_30d, volume, market_cap,
                    dilution_data, bucket):
    """
    Calculate token-specific Fear & Greed score (0-100).

    Parameters are all from CoinGecko /coins/markets — no extra API needed.
    Returns:
      {
        "score":           int (0-100),
        "label":           str,
        "components": {
            "price_momentum":    int,
            "volume_momentum":   int,
            "volatility":        int,
            "momentum_trend":    int,
            "dilution_pressure": int,
        }
      }
    """
    try:
        components = {}
        chg30 = price_change_30d or 0
        chg7  = price_change_7d  or 0

        # ── Component 1: Price Momentum 30d (30 pts max) ─────────────────────
        # Neutral baseline is 10 (not 15) — flat market should lean Fear,
        # not Neutral, since crypto markets trend up in good times.
        if chg30 >= 100:   pm = 30
        elif chg30 >= 50:  pm = 26
        elif chg30 >= 20:  pm = 22
        elif chg30 >= 10:  pm = 18
        elif chg30 >= 5:   pm = 14
        elif chg30 >= 0:   pm = 10   # flat = mild fear baseline
        elif chg30 >= -5:  pm = 7
        elif chg30 >= -15: pm = 5
        elif chg30 >= -30: pm = 3
        elif chg30 >= -50: pm = 1
        else:              pm = 0
        components["price_momentum"] = pm

        # ── Component 2: Volume Momentum (25 pts max) ────────────────────────
        vol_ratio = (volume / market_cap * 100) if market_cap and market_cap > 0 else 0
        if vol_ratio >= 20:    vm = 25
        elif vol_ratio >= 10:  vm = 20
        elif vol_ratio >= 5:   vm = 16
        elif vol_ratio >= 2:   vm = 12
        elif vol_ratio >= 1:   vm = 8
        elif vol_ratio >= 0.5: vm = 5
        else:                  vm = 3
        # Stronger penalty: high volume on falling price = selling panic
        if chg30 < -5 and vol_ratio >= 5:
            vm = max(0, vm - 10)
        elif chg30 < -15 and vol_ratio >= 2:
            vm = max(0, vm - 6)
        components["volume_momentum"] = vm

        # ── Component 3: Volatility (20 pts max) ─────────────────────────────
        vol_mag = abs(chg7)
        if vol_mag >= 50:      vlt = 19 if chg7 > 0 else 1
        elif vol_mag >= 30:    vlt = 16 if chg7 > 0 else 3
        elif vol_mag >= 15:    vlt = 13 if chg7 > 0 else 6
        elif vol_mag >= 5:     vlt = 10 if chg7 > 0 else 7
        else:                  vlt = 8    # low volatility = slight fear (stagnation)
        components["volatility"] = vlt

        # ── Component 4: Momentum Trend 7d vs 30d (15 pts max) ───────────────
        if chg7 is not None and chg30 is not None:
            if chg7 > 0 and chg7 > abs(chg30) * 0.5:  mt = 15  # strong recovery
            elif chg7 > 0 and chg30 > 0:               mt = 12  # both positive
            elif chg7 > 0 and chg30 <= 0:              mt = 9   # short bounce in downtrend
            elif chg7 <= 0 and chg30 > 0:              mt = 6   # recent weakness
            elif chg7 <= 0 and chg30 <= 0:             mt = 3   # both negative
            else:                                       mt = 5
        else:
            mt = 5
        components["momentum_trend"] = mt

        # ── Component 5: Dilution Pressure (10 pts max) ───────────────────────
        if dilution_data:
            overhang = dilution_data.get("supply_overhang", 50)
            if overhang <= 5:     dp = 10
            elif overhang <= 15:  dp = 8
            elif overhang <= 30:  dp = 6
            elif overhang <= 50:  dp = 3
            elif overhang <= 70:  dp = 1
            else:                 dp = 0
        else:
            dp = 4   # unknown = slight fear
        components["dilution_pressure"] = dp

        total = pm + vm + vlt + mt + dp
        score = min(100, max(0, total))

        return {
            "score":      score,
            "label":      _fg_label(score),
            "components": components,
        }

    except Exception:
        return {"score": 50, "label": "Neutral", "components": {}}


# =============================================================================
# MAIN COIN ANALYSER
# =============================================================================
def analyze_coin(coin, defillama_data, gold_mc, defillama_chain_data=None):
    """Full valuation analysis for a single coin."""
    try:
        symbol   = coin["symbol"].upper()
        coin_id  = coin.get("id", "")
        name     = coin.get("name", symbol)
        price    = _safe(coin.get("current_price"))
        mc       = _safe(coin.get("market_cap"), 0)
        volume   = _safe(coin.get("total_volume"), 0)
        circ     = _safe(coin.get("circulating_supply"), 0)
        total    = _safe(coin.get("total_supply")) or circ
        max_sup  = _safe(coin.get("max_supply"))
        rank     = coin.get("market_cap_rank")
        chg_7d   = _safe(coin.get("price_change_percentage_7d_in_currency"))
        chg_30d  = _safe(coin.get("price_change_percentage_30d_in_currency"))

        bucket = "B" if symbol in BUCKET_B_SYMBOLS else "A"

        # ── Common: Method 10 — Dilution ────────────────────────────────────
        dilution = method_10_dilution(circ, total, max_sup, price, mc)

        # ── Common: Method 7 — Metcalfe ─────────────────────────────────────
        metcalfe_ratio = method_7_metcalfe(mc, volume=volume)

        # ── Bucket A: Fee-based methods ──────────────────────────────────────
        bucket_a_data = {}
        fdv = (price * (max_sup or total or circ)) if price else None

        if bucket == "A":
            slug        = DEFILLAMA_SLUGS.get(symbol)
            fee_data    = defillama_data.get(slug, {}) if slug else {}

            # For L1 blockchains, DeFiLlama reports fees at the chain level
            # not the protocol level — check chain endpoint as fallback
            chain_slug  = CHAIN_FEE_MAP.get(symbol)
            chain_data  = (defillama_chain_data or {}).get(chain_slug, {})                           if chain_slug else {}

            # Prefer protocol-level data; fall back to chain-level
            annual_fees = fee_data.get("annual_fees") or chain_data.get("annual_fees")
            annual_rev  = fee_data.get("annual_revenue") or chain_data.get("annual_revenue")
            holders_rev = fee_data.get("annual_holders_revenue")
            tvl         = fee_data.get("tvl") or chain_data.get("tvl")

            # Method 1 — DCF on fees
            dcf_protocol_value = method_1_dcf_fees(annual_fees)

            # Method 2 — P/S analog
            ps_ratio = method_2_ps_analog(mc, annual_fees)

            # Method 3 — P/E analog
            pe_ratio = method_3_pe_analog(fdv, holders_rev)

            # Per-token fair value from DCF (protocol value / circulating supply)
            dcf_per_token = None
            if dcf_protocol_value and circ and circ > 0:
                dcf_per_token = round(dcf_protocol_value / circ, 4)

            bucket_a_data = {
                "annual_protocol_fees_usd":    round(annual_fees, 2) if annual_fees else None,
                "annual_revenue_usd":          round(annual_rev, 2)  if annual_rev  else None,
                "annual_holders_revenue_usd":  round(holders_rev, 2) if holders_rev else None,
                "tvl_usd":                     round(tvl, 2)         if tvl         else None,
                "dcf_protocol_value_usd":      dcf_protocol_value,
                "dcf_fair_value_per_token":    dcf_per_token,
                "ps_ratio":                    ps_ratio,
                "pe_ratio_fdv_to_holders_rev": pe_ratio,
                "data_source": "defillama" if slug and fee_data else "unavailable",
            }

        # ── Bucket B: SoV methods ─────────────────────────────────────────────
        bucket_b_data = {}
        if bucket == "B":
            # Method 6 — Monetary premium vs gold
            mon_prem = method_6_monetary_premium(mc, gold_mc)
            if mon_prem and circ and circ > 0 and price:
                mon_prem["price_at_25pct_gold"]  = round((gold_mc * 0.25) / circ, 2)
                mon_prem["price_at_50pct_gold"]  = round((gold_mc * 0.50) / circ, 2)
                mon_prem["price_at_100pct_gold"] = round((gold_mc * 1.00) / circ, 2)

            # Method 8 — Cost of production
            prod_cost = method_8_cost_of_production(symbol, mc, circ, price)

            bucket_b_data = {
                "monetary_premium": mon_prem,
                "cost_of_production": prod_cost,
            }

        # ── Network Economics charts ─────────────────────────────────────────
        # Computed BEFORE quality scoring so chart values feed into scores
        # ensuring the Quality tab and Network Economics charts always agree.
        slug_for_charts  = DEFILLAMA_SLUGS.get(symbol)
        chain_slug_charts = CHAIN_FEE_MAP.get(symbol)
        # Merge protocol-level and chain-level fee summaries so that
        # L1 chains (SOL, AVAX etc.) get their fee data into the charts
        _proto_summary = defillama_data.get(slug_for_charts, {}) if slug_for_charts else {}
        _chain_summary = (defillama_chain_data or {}).get(chain_slug_charts, {}) if chain_slug_charts else {}
        _fee_summary   = {**_chain_summary, **_proto_summary}  # protocol wins on overlap

        network_econ    = fetch_network_economics(
            slug              = slug_for_charts or chain_slug_charts,
            symbol            = symbol,
            chain_slug        = chain_slug_charts,
            coin_data         = {
                "current_price": price,
                "market_cap":    mc,
                "dilution":      method_10_dilution(circ, total, max_sup, price, mc),
            },
            defillama_summary = _fee_summary,
        )

        # ── Fear & Greed — token specific ────────────────────────────────────
        fear_greed = calc_fear_greed(
            price_change_7d  = chg_7d,
            price_change_30d = chg_30d,
            volume           = volume,
            market_cap       = mc,
            dilution_data    = dilution,
            bucket           = bucket,
        )

        # ── Quality score ─────────────────────────────────────────────────────
        # Reads from network_econ charts where possible — same values
        # displayed in the Network Economics tab, ensuring full consistency.
        _slug_data   = defillama_data.get(DEFILLAMA_SLUGS.get(symbol), {})
        _ann_fees    = _slug_data.get("annual_fees")
        _holders_rev = _slug_data.get("annual_holders_revenue")
        _tvl         = _slug_data.get("tvl")
        _gold_cap    = (bucket_b_data.get("monetary_premium") or {}).get("gold_capture_pct") if bucket == "B" else None
        _prem_cost   = (bucket_b_data.get("cost_of_production") or {}).get("premium_to_production_cost") if bucket == "B" else None
        _fdv_mc      = (dilution or {}).get("fdv_to_mc_ratio")

        quality = score_crypto_quality(
            bucket             = bucket,
            mc_ps_ratio        = bucket_a_data.get("ps_ratio"),
            mc_pe_ratio        = bucket_a_data.get("pe_ratio_fdv_to_holders_rev"),
            dilution           = dilution,
            metcalfe_ratio     = metcalfe_ratio,
            gold_capture_pct   = _gold_cap,
            premium_to_cost    = _prem_cost,
            price_change_7d    = chg_7d,
            price_change_30d   = chg_30d,
            annual_fees        = _ann_fees,
            annual_holders_rev = _holders_rev,
            tvl                = _tvl,
            volume             = volume,
            mc                 = mc,
            rank               = rank,
            fdv_mc_ratio       = _fdv_mc,
            network_econ       = network_econ,
            symbol             = symbol,
        )

        # ── Valuation Chart — Forecast Adjustment Framework ──────────────────
        #
        # Philosophy — no compounding growth lines for crypto:
        #   Unlike stocks, crypto has no EPS or analyst consensus growth rate.
        #   Projecting a compounding line gives false precision. Instead the
        #   chart uses FLAT VALUATION BANDS derived from the spread of
        #   valuation methods already computed — honest and grounded.
        #
        # Structure:
        #   Historical section  — actual market prices from CoinGecko
        #   Current year        — market price + base IV from valuation methods
        #   Forecast section    — flat bands held constant at current IV
        #                         widened/narrowed by quality and bull/bear signals
        #
        # Three bands:
        #   base  = median of all valid valuation method outputs
        #   bull  = base × bull_multiplier (quality + bull signal adjusted)
        #   bear  = base × bear_multiplier (survivability + bear signal adjusted)
        #
        # Survivability constraint (maps to quality signals):
        #   Bucket A: avg of dilution_control + protocol_maturity + holder_value_accrual
        #   Bucket B: avg of scarcity + dilution_control + monetary_premium_quality
        #   Strong survivability → bands stay wide (token can handle volatility)
        #   Weak survivability  → bear band compressed downward (existential risk)
        #
        # Bull/Bear signal tilt:
        #   Pre-computed bull_bear.source and number of strong bull vs bear
        #   signals adjusts which band gets emphasis — this is the qualitative
        #   layer on top of the quantitative bands.
        # ------------------------------------------------------------------
        valuation_chart = {}
        forecast_meta   = {}
        try:
            def _sr(v):
                """Safe round — returns None for NaN/Inf."""
                try:
                    f = float(v)
                    return round(f, 4) if f == f and abs(f) != float('inf') else None
                except Exception:
                    return None

            # ── Step 1: Collect all valid valuation outputs ───────────────
            valid_ivs = []
            if bucket == "A":
                proto = bucket_a_data or {}
                dcf_v = proto.get("dcf_fair_value_per_token")
                if dcf_v and dcf_v > 0: valid_ivs.append(dcf_v)
                # PS analog: if fees exist, price * (median_ps / current_ps)
                ps = proto.get("ps_ratio")
                if ps and ps > 0 and price:
                    median_ps = 20   # reasonable median for cash-flow protocols
                    valid_ivs.append(price * median_ps / ps)
            elif bucket == "B":
                sov = bucket_b_data or {}
                prod = (sov.get("cost_of_production") or {})
                cost = prod.get("estimated_production_cost_usd")
                prem = prod.get("premium_to_production_cost")
                if cost and prem and cost > 0:
                    valid_ivs.append(cost * prem)
                mon  = (sov.get("monetary_premium") or {})
                gcp  = mon.get("gold_capture_pct")
                if gcp and gcp > 0 and price:
                    # Fair value implied by current gold capture + median growth
                    valid_ivs.append(price * (1 + gcp / 100))

            # Base IV = median of valid outputs, fallback to current price
            if valid_ivs:
                valid_ivs_sorted = sorted(valid_ivs)
                mid = len(valid_ivs_sorted) // 2
                base_iv = valid_ivs_sorted[mid]
            else:
                base_iv = price or 0

            if not base_iv or base_iv <= 0:
                base_iv = price or 0

            if base_iv and base_iv > 0:

                # ── Step 2: Survivability score ───────────────────────────
                # Derived from quality signals most relevant to long-term
                # protocol sustainability — not a single score, a composite.
                q_scores = quality.get("scores", {})
                if bucket == "A":
                    surv_signals = [
                        (q_scores.get("dilution_control")    or {}).get("score", 0),
                        (q_scores.get("protocol_maturity")   or {}).get("score", 0),
                        (q_scores.get("holder_value_accrual") or {}).get("score", 0),
                    ]
                    surv_maxes = [
                        (q_scores.get("dilution_control")    or {}).get("max", 1),
                        (q_scores.get("protocol_maturity")   or {}).get("max", 1),
                        (q_scores.get("holder_value_accrual") or {}).get("max", 2),
                    ]
                else:
                    surv_signals = [
                        (q_scores.get("scarcity")                  or {}).get("score", 0),
                        (q_scores.get("dilution_control")          or {}).get("score", 0),
                        (q_scores.get("monetary_premium_quality")  or {}).get("score", 0),
                    ]
                    surv_maxes = [
                        (q_scores.get("scarcity")                  or {}).get("max", 2),
                        (q_scores.get("dilution_control")          or {}).get("max", 1),
                        (q_scores.get("monetary_premium_quality")  or {}).get("max", 2),
                    ]
                total_surv = sum(surv_signals)
                max_surv   = sum(surv_maxes) or 1
                surv_pct   = total_surv / max_surv   # 0.0 – 1.0

                # Survivability → band multiplier
                # Strong: 0.75–1.0   bands stay wide — token is resilient
                # Moderate: 0.5–0.75 moderate compression
                # Weak: 0.25–0.5     bear band pulled in — higher failure risk
                # Poor: 0–0.25       severe compression — existential risk
                if surv_pct >= 0.75:   surv_mult = 1.00
                elif surv_pct >= 0.50: surv_mult = 0.80
                elif surv_pct >= 0.25: surv_mult = 0.60
                else:                  surv_mult = 0.35

                # Overall quality multiplier for bull band
                q_score = quality.get("final_score", 5) or 5
                if q_score >= 8:   q_mult = 1.00
                elif q_score >= 6: q_mult = 0.90
                elif q_score >= 4: q_mult = 0.75
                elif q_score >= 2: q_mult = 0.55
                else:              q_mult = 0.35

                # ── Step 3: Flat band multipliers ────────────────────────
                # Bull band: base × (1 + bull_spread * quality_mult)
                # Bear band: base × (1 - bear_spread * survivability_mult)
                # Crypto bands are wider than stocks due to inherent volatility
                bull_spread = 0.60   # 60% upside for bull band max
                bear_spread = 0.50   # 50% downside for bear band max

                bull_mult = 1.0 + (bull_spread * q_mult)
                bear_mult = 1.0 - (bear_spread * (1 - surv_mult))

                bull_iv = _sr(base_iv * bull_mult)
                bear_iv = _sr(base_iv * bear_mult)

                was_constrained = surv_mult < 1.0 or q_mult < 1.0

                # ── Step 4: Fetch 10Y price history from CoinGecko ───────
                # Uses daily interval with 3650 days — CoinGecko free tier
                # returns daily data for ranges > 90 days automatically.
                # Then we sample one price per year (year-end closing).
                price_history = {}
                try:
                    time.sleep(1.2)   # respect rate limit
                    # Daily interval gives full history on free tier
                    hist_resp = requests.get(
                        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                        f"/market_chart?vs_currency=usd&days=3650",
                        headers=_headers(), timeout=20
                    )
                    if hist_resp.status_code == 200:
                        prices_raw = hist_resp.json().get("prices", [])
                        # Sample: keep last price per calendar year
                        year_prices = {}
                        for ts_ms, p_val in prices_raw:
                            yr = str(datetime.fromtimestamp(ts_ms / 1000).year)
                            year_prices[yr] = round(p_val, 4)  # last entry per yr wins
                        price_history = year_prices
                    elif hist_resp.status_code == 429:
                        time.sleep(30)   # rate limit — wait and skip
                except Exception as e:
                    print(f"  [CRYPTO HIST] {coin_id}: {str(e)[:50]}")

                # ── Step 5: Build historical section ─────────────────────
                # Bands are back-projected as flat multiples — no compounding.
                # The historical section exists purely to show market price
                # context, not to project what IV "should have been."
                for yr, hist_price in sorted(price_history.items()):
                    valuation_chart[yr] = {
                        "market_price":    hist_price,
                        "intrinsic_value": round(base_iv, 4),
                        "bull_case":       bull_iv,
                        "bear_case":       bear_iv,
                        "is_historical":   True,
                    }

                # ── Step 6: Current year ──────────────────────────────────
                valuation_chart[str(current_year)] = {
                    "market_price":    round(price, 4) if price else None,
                    "intrinsic_value": round(base_iv, 4),
                    "bull_case":       bull_iv,
                    "bear_case":       bear_iv,
                    "is_historical":   False,
                }

                # ── Step 7: Forecast years — flat bands, 5 years ─────────
                # No compounding — bands stay at current levels.
                # AI Studio's qualitative bull/bear analysis can annotate
                # which band is more likely based on the thesis.
                for i in range(1, 6):
                    valuation_chart[str(current_year + i)] = {
                        "market_price":    None,
                        "intrinsic_value": round(base_iv, 4),
                        "bull_case":       bull_iv,
                        "bear_case":       bear_iv,
                        "is_historical":   False,
                    }

                # ── Step 8: Forecast metadata ─────────────────────────────
                forecast_meta = {
                    "base_iv":              round(base_iv, 4),
                    "bull_iv":              bull_iv,
                    "bear_iv":              bear_iv,
                    "bull_upside_pct":      round((bull_mult - 1) * 100, 1),
                    "bear_downside_pct":    round((1 - bear_mult) * 100, 1),
                    "quality_score":        q_score,
                    "quality_multiplier":   q_mult,
                    "survivability_pct":    round(surv_pct * 100, 1),
                    "survivability_mult":   surv_mult,
                    "growth_constrained":   was_constrained,
                    "bucket":               bucket,
                    "iv_sources":           len(valid_ivs),
                    "constraint_note": (
                        f"Bands constrained — quality {q_score}/10, "
                        f"survivability {round(surv_pct*100,1)}%. "
                        f"Bull upside capped at {round((bull_mult-1)*100,1)}%, "
                        f"bear downside at {round((1-bear_mult)*100,1)}%."
                    ) if was_constrained else None,
                }

        except Exception:
            valuation_chart = {}
            forecast_meta   = {}

        return symbol, {
            "name":              name,
            "coin_id":           coin_id,
            "bucket":            bucket,
            "bucket_label":      "Store of Value" if bucket == "B" else "Cash-Flow Protocol",
            "current_price":     price,
            "market_cap":        mc,
            "fully_diluted_valuation": round(fdv, 2) if fdv else None,
            "market_cap_rank":   rank,
            "price_change_7d_pct":  chg_7d,
            "price_change_30d_pct": chg_30d,
            "dilution":          dilution,
            "metcalfe_ratio":    metcalfe_ratio,
            "quality":           quality,
            "fear_greed":        fear_greed,
            "network_economics": network_econ,
            "valuation_chart":   valuation_chart,
            "forecast_meta":     forecast_meta,
            # Top-level intrinsic_value for AI Studio hero card display
            # Same as forecast_meta["base_iv"] but accessible without
            # nested lookup — prevents "N/A - Insufficient Data" display.
            "intrinsic_value":   forecast_meta.get("base_iv") if forecast_meta else None,
            "iv_method":         (
                "DCF on protocol fees" if bucket == "A" and
                (forecast_meta or {}).get("iv_sources", 0) > 0
                else "Monetary premium / production cost" if bucket == "B"
                else "Market price (no fee data available)"
            ),
            "bull_bear":         generate_crypto_bull_bear(
                                     symbol       = symbol,
                                     name         = name,
                                     bucket       = bucket,
                                     price        = price,
                                     mc           = mc,
                                     rank         = rank,
                                     quality      = quality,
                                     dilution     = dilution,
                                     fear_greed   = fear_greed,
                                     network_econ = network_econ,
                                     bucket_a_data= bucket_a_data,
                                     bucket_b_data= bucket_b_data,
                                     chg_7d       = chg_7d,
                                     chg_30d      = chg_30d,
                                 ),
            **({"protocol_metrics": bucket_a_data} if bucket == "A" else {}),
            **({"sov_metrics":      bucket_b_data} if bucket == "B" else {}),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    except Exception as e:
        print(f"  [ERROR] {coin.get('symbol','?')}: {str(e)[:80]}")
        return None


# =============================================================================
# NETWORK ECONOMICS CHARTS
# (Tab rename: "Financials" → "Network Economics")
#
# Four charts that mirror the stock Financials tab, adapted for crypto:
#
#   Chart 1 — Protocol Revenue & Fees
#     Equivalent to: 10Y Revenue & Net Income
#     Annual fees paid by users (= revenue) and protocol revenue retained
#     (= net income analog). Shows whether the protocol is growing its
#     economic activity over time.
#     Source: DeFiLlama /summary/fees/{slug}?dataType=dailyFees (historical)
#
#   Chart 2 — Treasury vs Token Emissions
#     Equivalent to: 10Y Cash & Total Debt
#     Protocol treasury value (= cash) vs annual token issuance in USD
#     (= debt analog). A healthy protocol's treasury grows or holds while
#     emissions shrink — distressed ones issue more than they hold.
#     Source: DeFiLlama treasury + CoinGecko supply data
#
#   Chart 3 — Token Supply: Issuance vs Burns
#     Equivalent to: 10Y Shares Outstanding & Buyback Spend
#     New tokens issued per year (= share issuance) vs tokens burned/
#     destroyed (= buyback equivalent). Net issuance = dilution pressure.
#     Source: CoinGecko historical supply (approximated from market data)
#
#   Chart 4 — Capital Efficiency
#     Equivalent to: 10Y Returns Trajectory
#     Fee/TVL ratio (= ROIC analog) and Real Yield % (= ROE analog) over
#     time. Shows how efficiently the protocol converts locked capital
#     into fees.
#     Source: DeFiLlama fees + TVL historical data
#
# DATA AVAILABILITY NOTE:
#   DeFiLlama free API provides summary-level current data for most protocols
#   but historical time-series data (per-year breakdown) requires the Pro API.
#   For tokens WITHOUT historical data, charts will contain only the current
#   year estimate with a clear note to users.
#
# DATA COVERAGE LABELS stored in network_economics_meta:
#   "full"    — 3+ years of historical data available
#   "limited" — 1-2 years only
#   "current_only" — only current year estimate, no history
#   "unavailable"  — no DeFiLlama data for this token
# =============================================================================
def fetch_network_economics(slug, symbol, coin_data, defillama_summary,
                             chain_slug=None):
    """
    Build the four Network Economics chart datasets for a single token.

    Data sources (in priority order):
      1. DeFiLlama /protocol/{slug} — protocol-level TVL + fee history
      2. DeFiLlama /v2/historicalChainTvl/{chain} — L1 chain TVL history
      3. defillama_summary — current-year estimates from batch fetch
      4. CoinGecko coin_data — supply, price, market cap (always available)

    Charts 3 (Issuance vs Burns) and partial Chart 4 are always populated
    from CoinGecko data even when DeFiLlama coverage is unavailable.
    """
    current_year = datetime.now().year
    result = {
        "protocol_revenue_and_fees":   {},
        "treasury_vs_emissions":       {},
        "issuance_vs_burns":           {},
        "capital_efficiency":          {},
        "meta": {
            "data_coverage":    "unavailable",
            "years_available":  0,
            "data_note":        None,
            "last_fetched":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    }

    # ── Helpers ───────────────────────────────────────────────────────────────
    price        = coin_data.get("current_price") or 0
    mc           = coin_data.get("market_cap") or 0
    circ         = (coin_data.get("dilution") or {}).get("circulating_pct", 100)
    fee_summary  = defillama_summary or {}
    annual_fees  = fee_summary.get("annual_fees") or 0
    annual_rev   = fee_summary.get("annual_revenue") or 0
    holders_rev  = fee_summary.get("annual_holders_revenue") or 0
    tvl          = fee_summary.get("tvl") or 0

    has_fee_data = annual_fees > 0

    # ── Attempt DeFiLlama historical protocol data ─────────────────────────
    historical_tvl   = {}   # { year_str: tvl_usd }
    historical_fees  = {}   # { year_str: fees_usd }
    years_of_history = 0

    # ── Try protocol-level endpoint first ────────────────────────────────────
    if slug:
        try:
            resp = requests.get(
                f"https://api.llama.fi/protocol/{slug}",
                timeout=15
            )
            if resp.status_code == 200:
                pdata = resp.json()
                # TVL history
                for entry in pdata.get("tvl", []):
                    ts  = entry.get("date")
                    val = entry.get("totalLiquidityUSD")
                    if ts and val:
                        yr = str(datetime.fromtimestamp(ts).year)
                        historical_tvl[yr] = round(float(val), 2)
                # Fee history (daily accumulate → yearly)
                for entry in pdata.get("feesHistory", []):
                    ts  = entry.get("date")
                    val = entry.get("dailyFees")
                    if ts and val:
                        yr = str(datetime.fromtimestamp(ts).year)
                        historical_fees[yr] = historical_fees.get(yr, 0) + float(val)
        except Exception:
            pass

    # ── Try chain-level TVL endpoint for L1 blockchains ──────────────────────
    # DeFiLlama stores chain TVL separately from protocol TVL.
    # e.g. Solana chain TVL is at /v2/historicalChainTvl/Solana
    # This gives real historical TVL for chains like SOL, AVAX, MATIC.
    if chain_slug and not historical_tvl:
        try:
            chain_name = chain_slug.capitalize()
            resp2 = requests.get(
                f"https://api.llama.fi/v2/historicalChainTvl/{chain_name}",
                timeout=15
            )
            if resp2.status_code == 200:
                for entry in resp2.json():
                    ts  = entry.get("date")
                    val = entry.get("tvl")
                    if ts and val:
                        yr = str(datetime.fromtimestamp(ts).year)
                        historical_tvl[yr] = round(float(val), 2)
        except Exception:
            pass

    years_of_history = len(set(historical_tvl.keys()) | set(historical_fees.keys()))

    # ── Determine data coverage label and user note ───────────────────────
    if years_of_history >= 3:
        coverage = "full"
        note     = None
    elif years_of_history >= 1:
        coverage = "limited"
        note     = (f"Only {years_of_history} year(s) of historical data available "
                    f"for this protocol on DeFiLlama. Charts show available history "
                    f"plus current year estimates.")
    elif has_fee_data:
        coverage = "current_only"
        note     = ("Historical time-series data is not available for this token "
                    "on the free DeFiLlama API. Charts show current year estimates only. "
                    "Older or smaller protocols may have limited on-chain data coverage.")
    else:
        coverage = "unavailable"
        note     = ("No protocol fee or TVL data found for this token on DeFiLlama. "
                    "Charts 1 and 2 are unavailable. Charts 3 and 4 show "
                    "current-year supply and dilution data from CoinGecko.")
        # Do NOT return early — Charts 3 (issuance/burns) and 4 (capital
        # efficiency) can still be partially populated from CoinGecko data.

    # ── Chart 1: Protocol Revenue & Fees ─────────────────────────────────
    # Historical years from DeFiLlama, current year from summary estimate
    rev_fees = {}
    for yr, fee_val in historical_fees.items():
        rev_fees[yr] = {
            "fees_usd":     round(fee_val / 1e6, 2),    # in $M
            "revenue_usd":  round(fee_val * 0.15 / 1e6, 2),  # ~15% retained estimate
            "is_estimated": False,
        }
    # Fill TVL-years that have no fee data with N/A
    for yr in historical_tvl:
        if yr not in rev_fees and has_fee_data:
            rev_fees[yr] = {"fees_usd": None, "revenue_usd": None, "is_estimated": True}
    # Current year estimate from summary
    if has_fee_data:
        rev_fees[str(current_year)] = {
            "fees_usd":     round(annual_fees / 1e6, 2),
            "revenue_usd":  round(annual_rev  / 1e6, 2),
            "is_estimated": True,
        }
    result["protocol_revenue_and_fees"] = rev_fees

    # ── Chart 2: Treasury vs Token Emissions ─────────────────────────────
    # Treasury approximated from TVL history (DeFiLlama protocol TVL ≈ treasury
    # for many protocols). Emissions estimated from supply change × price.
    treas_emit = {}
    sorted_tvl_years = sorted(historical_tvl.keys())
    for i, yr in enumerate(sorted_tvl_years):
        tvl_val = historical_tvl[yr]
        # Emission estimate: if we have consecutive years, infer from supply change
        emission_usd = None
        treas_emit[yr] = {
            "treasury_usd_m":  round(tvl_val / 1e6, 2) if tvl_val else None,
            "emissions_usd_m": emission_usd,
            "is_estimated":    True,
            "note":            "Treasury proxied from protocol TVL",
        }
    # Current year
    if tvl:
        treas_emit[str(current_year)] = {
            "treasury_usd_m":  round(tvl / 1e6, 2),
            "emissions_usd_m": None,
            "is_estimated":    True,
            "note":            "Treasury proxied from protocol TVL",
        }
    result["treasury_vs_emissions"] = treas_emit

    # ── Chart 3: Issuance vs Burns ────────────────────────────────────────
    # Derived from CoinGecko dilution data (circulating supply changes)
    # CoinGecko markets endpoint doesn't give historical supply, so this
    # is limited to current snapshot with direction indicators.
    dilution     = coin_data.get("dilution") or {}
    supply_ovhng = dilution.get("supply_overhang", 0) or 0
    fdv_mc_ratio = dilution.get("fdv_to_mc_ratio") or 1
    circ_supply  = (coin_data.get("market_cap") / price) if price and mc else None

    issuance_burns = {
        str(current_year): {
            "circulating_supply_m": round(circ_supply / 1e6, 2) if circ_supply else None,
            "supply_overhang_pct":  supply_ovhng,
            "fdv_to_mc":            fdv_mc_ratio,
            "burn_rate":            None,   # requires on-chain data not in free APIs
            "net_issuance_note":    (
                "Current snapshot only. Historical supply data requires "
                "on-chain indexing not available via free APIs."
            ),
            "is_estimated": True,
        }
    }
    result["issuance_vs_burns"] = issuance_burns

    # ── Chart 4: Capital Efficiency ───────────────────────────────────────
    # Fee/TVL ratio per year = ROIC analog
    # Real Yield = holders_revenue / TVL = ROE analog
    cap_eff = {}
    for yr in sorted(set(historical_tvl) | set(historical_fees)):
        tvl_yr  = historical_tvl.get(yr)
        fees_yr = historical_fees.get(yr)
        fee_tvl_ratio  = round(fees_yr / tvl_yr * 100, 4) if fees_yr and tvl_yr and tvl_yr > 0 else None
        cap_eff[yr] = {
            "fee_tvl_ratio_pct": fee_tvl_ratio,   # ROIC analog
            "real_yield_pct":    None,             # holders_rev/TVL — needs per-year holders rev
            "tvl_usd_m":         round(tvl_yr / 1e6, 2) if tvl_yr else None,
            "fees_usd_m":        round(fees_yr / 1e6, 2) if fees_yr else None,
        }
    # Current year
    if tvl and annual_fees:
        fee_tvl_curr = round(annual_fees / tvl * 100, 4) if tvl > 0 else None
        real_yield   = round(holders_rev / tvl * 100, 4) if holders_rev and tvl > 0 else None
        cap_eff[str(current_year)] = {
            "fee_tvl_ratio_pct": fee_tvl_curr,
            "real_yield_pct":    real_yield,
            "tvl_usd_m":         round(tvl / 1e6, 2),
            "fees_usd_m":        round(annual_fees / 1e6, 2),
            "is_estimated":      True,
        }
    result["capital_efficiency"] = cap_eff

    # ── Update meta ───────────────────────────────────────────────────────
    all_years = (set(rev_fees) | set(treas_emit) | set(cap_eff)) - {str(current_year)}
    result["meta"].update({
        "data_coverage":   coverage,
        "years_available": len(all_years),
        "data_note":       note,
    })

    return result


# =============================================================================
# BULL / BEAR THESIS GENERATOR — CRYPTO
#
# Same two-layer approach as the stocks pipeline:
#   Layer 1 — Rule-based signals extracted from already-computed data
#              (quality scores, network economics, valuation methods,
#               fear/greed, dilution). Zero extra API calls.
#   Layer 2 — Anthropic API (Claude Haiku) narrativises the signals into
#              coherent bull/bear arguments. Falls back to rules if no key.
#
# Crypto-specific signals differ by bucket:
#   Bucket A — fee growth, capital efficiency, TVL trend, holder accrual
#   Bucket B — gold capture, production cost, scarcity, monetary premium
# =============================================================================
ANTHROPIC_API_KEY = ""   # ← paste your Anthropic API key when ready

def generate_crypto_bull_bear(symbol, name, bucket, price, mc, rank,
                               quality, dilution, fear_greed,
                               network_econ, bucket_a_data, bucket_b_data,
                               chg_7d, chg_30d):
    """
    Generate 3 bull and 3 bear points for a crypto token.
    Returns:
      {
        "bull_points":  [...],
        "bear_points":  [...],
        "source":       "api" | "rules",
        "generated_at": "YYYY-MM-DD HH:MM"
      }
    """
    bull_rules = []
    bear_rules = []

    quality_scores  = quality.get("scores", {})
    final_score_pct = quality.get("final_score_pct", 0)
    classification  = quality.get("classification", "")
    dilution_data   = dilution or {}
    circ_pct        = dilution_data.get("circulating_pct")
    overhang        = dilution_data.get("supply_overhang")
    fdv_mc          = dilution_data.get("fdv_to_mc_ratio")
    fg_score        = (fear_greed or {}).get("score", 50)
    fg_label        = (fear_greed or {}).get("label", "Neutral")
    ne_meta         = (network_econ or {}).get("meta", {})
    ne_coverage     = ne_meta.get("data_coverage", "unavailable")

    # ── Bucket A — Cash-Flow Protocol signals ────────────────────────────────
    if bucket == "A":
        proto = bucket_a_data or {}
        annual_fees = proto.get("annual_protocol_fees_usd")
        ps_ratio    = proto.get("ps_ratio")
        pe_ratio    = proto.get("pe_ratio_fdv_to_holders_rev")
        dcf_token   = proto.get("dcf_fair_value_per_token")
        tvl         = proto.get("tvl_usd")

        # Valuation signals
        if dcf_token and price:
            mos = round((dcf_token - price) / dcf_token * 100, 1)
            if mos > 20:
                bull_rules.append(
                    f"DCF model suggests {name} is trading at a "
                    f"{mos}% discount to fair value (est. ${dcf_token:,.2f})"
                )
            elif mos < -20:
                bear_rules.append(
                    f"DCF model suggests {name} is trading at a "
                    f"{abs(mos)}% premium to fair value (est. ${dcf_token:,.2f})"
                )

        if ps_ratio is not None:
            if ps_ratio < 15:
                bull_rules.append(
                    f"Low MC/Fees ratio of {ps_ratio}x suggests the protocol "
                    f"is generating strong fee revenue relative to its valuation"
                )
            elif ps_ratio > 100:
                bear_rules.append(
                    f"High MC/Fees ratio of {ps_ratio}x means investors are paying "
                    f"a large premium relative to actual protocol fee generation"
                )

        if pe_ratio is not None:
            if pe_ratio < 30:
                bull_rules.append(
                    f"FDV/Holders Revenue of {pe_ratio}x indicates strong value "
                    f"accrual to token holders relative to fully diluted valuation"
                )
            elif pe_ratio > 200:
                bear_rules.append(
                    f"FDV/Holders Revenue of {pe_ratio}x — very little fee revenue "
                    f"currently reaches token holders, limiting investment returns"
                )

        # Fee and TVL signals
        if annual_fees and annual_fees > 1e8:
            bull_rules.append(
                f"Protocol generates ${annual_fees/1e9:.2f}B in annual fees — "
                f"demonstrating real, sustained user demand"
            )
        elif annual_fees and annual_fees < 1e6:
            bear_rules.append(
                f"Annual protocol fees of ${annual_fees/1e6:.1f}M are low, "
                f"suggesting limited organic user demand"
            )

        # Capital efficiency score
        cap_eff = quality_scores.get("capital_efficiency", {})
        if cap_eff.get("score", 0) == 2:
            bull_rules.append(
                f"Strong capital efficiency — the protocol converts locked capital "
                f"into fee revenue at an above-average rate"
            )
        elif cap_eff.get("score", 0) == 0 and cap_eff.get("value") is not None:
            bear_rules.append(
                f"Poor capital efficiency — large TVL generates relatively "
                f"little fee revenue, suggesting capital is not being productively deployed"
            )

        # Network economics data quality
        if ne_coverage == "full":
            bull_rules.append(
                f"Multi-year on-chain data available — protocol has an established "
                f"track record of fee generation across market cycles"
            )
        elif ne_coverage == "unavailable":
            bear_rules.append(
                f"No historical fee data available on DeFiLlama — protocol may be "
                f"too new or too small to have an established economic track record"
            )

    # ── Bucket B — Store of Value signals ────────────────────────────────────
    elif bucket == "B":
        sov = bucket_b_data or {}
        mon_prem  = sov.get("monetary_premium") or {}
        prod_cost = sov.get("cost_of_production") or {}

        gold_cap  = mon_prem.get("gold_capture_pct")
        p_100gold = mon_prem.get("price_at_100pct_gold")
        cost_usd  = prod_cost.get("estimated_production_cost_usd")
        prem_cost = prod_cost.get("premium_to_production_cost")

        if gold_cap:
            if gold_cap > 5:
                bull_rules.append(
                    f"Has captured {gold_cap:.1f}% of gold's market cap — "
                    f"significant monetary adoption as a store of value"
                )
            else:
                bull_rules.append(
                    f"At {gold_cap:.2f}% of gold's market cap, substantial "
                    f"upside remains if monetary adoption continues to grow"
                )

        if p_100gold and price:
            upside = round((p_100gold / price - 1) * 100, 0)
            if upside > 0:
                bull_rules.append(
                    f"Full gold parity would imply a price of "
                    f"${p_100gold:,.0f} — {upside:.0f}% above current price"
                )

        if prem_cost:
            if 1 < prem_cost <= 3:
                bull_rules.append(
                    f"Trading at {prem_cost:.1f}x estimated production cost — "
                    f"a healthy premium that incentivises miners without being excessive"
                )
            elif prem_cost < 1:
                bear_rules.append(
                    f"Trading below estimated production cost of "
                    f"${cost_usd:,} — miners may reduce security spend, "
                    f"weakening the network"
                )
            elif prem_cost > 10:
                bear_rules.append(
                    f"Trading at {prem_cost:.0f}x production cost — "
                    f"significant premium above mining economics may not be sustainable"
                )

        scarcity = quality_scores.get("scarcity", {})
        if scarcity.get("score", 0) == 2:
            bull_rules.append(
                f"Over {100 - (overhang or 0):.0f}% of maximum supply already "
                f"in circulation — minimal future dilution risk"
            )
        elif overhang and overhang > 50:
            bear_rules.append(
                f"Only {circ_pct:.0f}% of maximum supply currently circulating — "
                f"significant future issuance could dilute existing holders"
            )

    # ── Common signals (both buckets) ────────────────────────────────────────
    # Fear & Greed context
    if fg_score >= 75:
        bear_rules.append(
            f"Token-specific Fear & Greed score of {fg_score} ({fg_label}) "
            f"suggests short-term euphoria — historically a caution signal"
        )
    elif fg_score <= 25:
        bull_rules.append(
            f"Fear & Greed score of {fg_score} ({fg_label}) indicates "
            f"significant pessimism — often a contrarian buying opportunity"
        )

    # Price momentum
    if chg_30d is not None:
        if chg_30d > 30:
            bear_rules.append(
                f"Up {chg_30d:.0f}% over 30 days — strong momentum but "
                f"increases the risk of a short-term correction"
            )
        elif chg_30d < -30:
            bull_rules.append(
                f"Down {abs(chg_30d):.0f}% over 30 days — significant "
                f"drawdown may represent a long-term entry opportunity"
            )

    # Overall quality
    if final_score_pct >= 70:
        bull_rules.append(
            f"Strong quality score of {quality.get('final_score', 0):.1f}/10 "
            f"({classification}) across all key metrics"
        )
    elif final_score_pct <= 30:
        bear_rules.append(
            f"Weak quality score of {quality.get('final_score', 0):.1f}/10 "
            f"({classification}) — most key metrics are below expectations"
        )

    # FDV/MC ratio
    if fdv_mc and fdv_mc > 2:
        bear_rules.append(
            f"FDV is {fdv_mc:.1f}x market cap — large token unlocks ahead "
            f"could create sustained sell pressure"
        )

    # Market rank
    if rank and rank <= 10:
        bull_rules.append(
            f"Top-{rank} by market cap — established network effect and "
            f"deep liquidity across major exchanges"
        )

    # ── Pad to 3 points minimum ───────────────────────────────────────────────
    generic_bull = [
        f"{name} maintains a top-{rank or 'N/A'} market cap position with established liquidity",
        f"Long-term adoption thesis remains intact for {bucket_label_str(bucket)} assets",
        f"Broader crypto market development continues to benefit established protocols",
    ]
    generic_bear = [
        f"Crypto market volatility remains high — macro conditions could impact all digital assets",
        f"Regulatory uncertainty across major markets presents an ongoing risk",
        f"Competition from newer protocols could erode {name}'s market position over time",
    ]
    i = 0
    while len(bull_rules) < 3:
        bull_rules.append(generic_bull[i % len(generic_bull)])
        i += 1
    i = 0
    while len(bear_rules) < 3:
        bear_rules.append(generic_bear[i % len(generic_bear)])
        i += 1

    # ── Anthropic API call ────────────────────────────────────────────────────
    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip():
        context = {
            "symbol": symbol, "name": name, "bucket": bucket,
            "current_price": price, "market_cap_usd": mc,
            "rank": rank, "price_change_30d_pct": chg_30d,
            "quality_score": quality.get("final_score"),
            "quality_classification": classification,
            "fear_greed_score": fg_score,
            "fear_greed_label": fg_label,
            "dilution_circulating_pct": circ_pct,
            "fdv_to_mc_ratio": fdv_mc,
            "rule_based_bull": bull_rules[:3],
            "rule_based_bear": bear_rules[:3],
        }
        try:
            prompt = (
                f"You are a senior crypto analyst. Based on the data below, "
                f"generate exactly 3 bull and 3 bear arguments for {name} ({symbol}), "
                f"classified as a {bucket_label_str(bucket)}.\n\n"
                f"Data:\n{json.dumps(context, indent=2)}\n\n"
                f"Requirements:\n"
                f"- Each point must be specific and grounded in the data\n"
                f"- Do not repeat the rule-based signals verbatim — expand or add new insights\n"
                f"- Keep each point to 1-2 sentences\n"
                f"- Respond ONLY with valid JSON, no other text:\n"
                f'{{ "bull_points": ["...", "...", "..."], '
                f'"bear_points": ["...", "...", "..."] }}'
            )
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      "claude-haiku-4-5-20251001",
                    "max_tokens": 500,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                raw = resp.json()["content"][0]["text"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                return {
                    "bull_points":  parsed.get("bull_points", bull_rules[:3]),
                    "bear_points":  parsed.get("bear_points", bear_rules[:3]),
                    "source":       "api",
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
        except Exception as e:
            print(f"  [BULL/BEAR] {symbol}: API failed ({str(e)[:50]}), using rules")

    return {
        "bull_points":  bull_rules[:3],
        "bear_points":  bear_rules[:3],
        "source":       "rules",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def bucket_label_str(bucket):
    return "Cash-Flow Protocol" if bucket == "A" else "Store of Value"


# =============================================================================
# JSON SAVER
# =============================================================================
def _sanitize(obj):
    """Replace NaN/Inf with None recursively."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
        return None
    return obj


def save_partitions(results):
    """Save A-Z partitioned JSON files to data_crypto/."""
    os.makedirs("data_crypto", exist_ok=True)
    buckets = {}
    for sym, data in results.items():
        letter = sym[0].upper()
        if not letter.isalpha():
            letter = "0-9"
        buckets.setdefault(letter, {})[sym] = data

    for letter, content in buckets.items():
        path = f"data_crypto/crypto_{letter}.json"
        with open(path, "w") as f:
            json.dump(_sanitize(content), f, indent=4)
    print(f"✅ Saved {len(results)} coins to data_crypto/ ({len(buckets)} files)")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("=" * 60)
    print("  CRYPTO ANALYSIS PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Step 1 — Fetch gold market cap
    print("\n[1] Fetching gold market cap...")
    gold_mc = fetch_gold_price()
    print(f"    Gold market cap: ${gold_mc/1e12:.1f}T")

    # Step 2 — Fetch DeFiLlama protocol fees (Bucket A)
    print("\n[2] Fetching DeFiLlama protocol fee data...")
    defillama_data       = fetch_defillama_fees()
    defillama_chain_data = fetch_defillama_chain_fees()
    print(f"    Found fee data for {len(defillama_data)} protocols")

    # Step 3 — Fetch top 1000 coins from CoinGecko
    print(f"\n[3] Fetching top {TOP_N_COINS} coins from CoinGecko...")
    all_coins = []
    pages = range(1, (TOP_N_COINS // 250) + 1)
    for page in pages:
        print(f"    Fetching page {page}...", end="", flush=True)
        coins = fetch_coingecko_page(page)
        all_coins.extend(coins)
        print(f" {len(coins)} coins")
        time.sleep(1.5)   # rate limit
    print(f"    Total fetched: {len(all_coins)} coins")

    # Step 4 — Fetch overall market Fear & Greed (once, free API)
    print("\n[4] Fetching overall market Fear & Greed index...")
    market_fg = fetch_market_fear_greed()
    print(f"    Market Fear & Greed: {market_fg['value']} — {market_fg['classification']}")

    # Step 5 — Analyse each coin
    print(f"\n[5] Analysing {len(all_coins)} coins...")
    master_results = {}
    bucket_a_count = bucket_b_count = 0

    for coin in all_coins:
        res = analyze_coin(coin, defillama_data, gold_mc, defillama_chain_data)
        if res:
            sym, data = res
            master_results[sym] = data
            if data["bucket"] == "A": bucket_a_count += 1
            else:                     bucket_b_count += 1

    print(f"    Bucket A (Cash-Flow): {bucket_a_count}")
    print(f"    Bucket B (Store of Value): {bucket_b_count}")

    # Attach market Fear & Greed to every coin
    for sym in master_results:
        master_results[sym]["market_fear_greed"] = market_fg

    # Step 6 — Save
    print("\n[6] Saving partitioned JSON files...")
    save_partitions(master_results)
    print("\n✅ Crypto pipeline complete.")


if __name__ == "__main__":
    main()
