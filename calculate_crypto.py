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
TOP_N_COINS = 500         # 2 pages × 250 — keeps pipeline within 3h GitHub Actions limit

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
# CURATED COIN DESCRIPTIONS
# Pre-written descriptions for well-known tokens.
# Used as fallback when CoinGecko detail API is unavailable or rate-limited.
# These are stored in coin_profile.description in the JSON output.
# =============================================================================
COIN_DESCRIPTIONS = {
    # ── Settlement Layers ─────────────────────────────────────────────────────
    "BTC":   "Bitcoin is the original decentralised digital currency and the world's largest cryptocurrency by market cap. It operates on a proof-of-work blockchain with a fixed supply of 21 million coins, making it a programmatically scarce store of value. Often referred to as 'digital gold', Bitcoin is primarily used as a long-term store of value and hedge against inflation.",
    "ETH":   "Ethereum is the leading smart contract platform and the foundation of decentralised finance (DeFi). It transitioned to proof-of-stake in 2022 (The Merge), making it energy-efficient. ETH is used to pay for computation on the network and is deflationary post-EIP-1559, with a portion of fees burned with every transaction. It underpins the majority of DeFi, NFT, and Layer 2 ecosystems.",
    "SOL":   "Solana is a high-performance Layer 1 blockchain known for extremely fast transaction speeds (65,000+ TPS) and very low fees. It uses a unique Proof-of-History consensus combined with Proof-of-Stake. Solana hosts a growing ecosystem of DeFi, NFTs, payments, and consumer applications, and has emerged as the primary competitor to Ethereum for retail and high-frequency use cases.",
    "BNB":   "BNB is the native token of the BNB Chain (formerly Binance Smart Chain) and the Binance exchange ecosystem. It offers reduced trading fees on Binance, powers the BNB Chain network, and undergoes quarterly token burns. BNB benefits from Binance's position as the world's largest cryptocurrency exchange.",
    "ADA":   "Cardano is a proof-of-stake Layer 1 blockchain built on peer-reviewed academic research. It uses the Ouroboros consensus mechanism and is developed by IOHK. Cardano focuses on scalability, sustainability, and interoperability, with significant adoption in developing-world identity and financial inclusion projects.",
    "AVAX":  "Avalanche is a high-speed, low-cost Layer 1 platform known for its subnet architecture, which allows anyone to launch custom blockchains. It achieves sub-second finality and hosts a growing DeFi ecosystem. AVAX is used for network fees, staking, and governance across the Avalanche platform.",
    "DOT":   "Polkadot is a multichain protocol that enables different blockchains to interoperate and share security. It uses a relay chain and parachain architecture, with DOT used for governance, staking, and bonding parachains. It was founded by Ethereum co-founder Gavin Wood.",
    "ATOM":  "Cosmos is the 'Internet of Blockchains' — a decentralised network of independent blockchains connected via the IBC (Inter-Blockchain Communication) protocol. ATOM is the staking and governance token of the Cosmos Hub, the central chain coordinating the ecosystem.",
    "NEAR":  "NEAR Protocol is a developer-friendly Layer 1 blockchain with a focus on usability and scalability. It uses a sharding technique called Nightshade to achieve high throughput. NEAR is notable for its human-readable account names and low transaction costs.",
    "TRX":   "TRON is a blockchain platform focused on decentralised entertainment and digital content. It hosts the largest USDT (Tether) circulation of any blockchain and processes extremely high transaction volumes daily. TRX is used for fees, staking, and governance.",
    "TON":   "TON (The Open Network) is a blockchain originally developed by Telegram. It benefits from deep integration with the Telegram messaging app (900M+ users), enabling easy crypto payments and onboarding. TON is rapidly growing in DeFi and payments use cases.",
    "HBAR":  "Hedera is an enterprise-grade public distributed ledger using hashgraph consensus technology rather than a traditional blockchain. It offers fast, fair, and secure transactions at low cost, and is governed by a council of major corporations including Google, IBM, and Deutsche Telekom.",
    # ── Layer 2 ───────────────────────────────────────────────────────────────
    "ARB":   "Arbitrum is the leading Ethereum Layer 2 scaling solution using optimistic rollup technology. It offers significantly lower fees than Ethereum mainnet while inheriting its security. ARB is the governance token of the Arbitrum DAO, which controls the Arbitrum network. Note: sequencer fees currently accrue to the foundation, not ARB holders.",
    "OP":    "Optimism is an Ethereum Layer 2 using optimistic rollup technology. It pioneered the Superchain vision — a network of OP Stack chains including Coinbase's Base. OP is the governance token. Like Arbitrum, sequencer revenue currently goes to the foundation rather than token holders.",
    "MATIC": "Polygon (now rebranding to POL) is an Ethereum sidechain and Layer 2 ecosystem. It offers fast, cheap transactions and hosts one of the largest DeFi and gaming ecosystems outside of Ethereum mainnet. The network is transitioning to a ZK-based architecture.",
    # ── DeFi Blue Chips ───────────────────────────────────────────────────────
    "UNI":   "Uniswap is the world's largest decentralised exchange (DEX) by volume, operating an automated market maker (AMM) model. It processes over $1 trillion in annual trading volume. The UNI token grants governance rights over the protocol. Importantly, the fee switch is currently off — all trading fees go to liquidity providers, not UNI holders — though a governance vote may activate fee sharing in future.",
    "AAVE":  "Aave is the largest decentralised lending and borrowing protocol with over $16 billion in total value locked. Users can supply assets to earn interest or borrow against collateral. AAVE token holders govern the protocol and participate in the Safety Module, which provides insurance backing. A fee switch proposal to distribute revenue to stakers is actively progressing.",
    "MKR":   "MakerDAO is the protocol behind DAI, the most battle-tested decentralised stablecoin. MKR holders govern the protocol and benefit from stability fee revenue through an active buyback-and-burn mechanism. The Endgame roadmap aims to further decentralise and scale the protocol.",
    "CRV":   "Curve Finance is the dominant stablecoin and pegged-asset DEX, with $4B+ in TVL. veCRV holders receive 50% of all trading fees and control gauge emissions — making CRV central to the 'Curve Wars' for liquidity incentives across DeFi.",
    "GMX":   "GMX is a leading decentralised perpetual futures exchange on Arbitrum and Avalanche. It offers leverage trading with low fees and zero price impact. 30% of protocol fees go directly to GMX stakers, making it one of the best fee-distribution models in DeFi.",
    "LDO":   "Lido is the largest liquid staking protocol, controlling ~32% of all staked ETH. It allows users to stake ETH and receive stETH, which can be used across DeFi. Lido earns 10% of all staking rewards as protocol revenue. LDO governs the protocol.",
    "LINK":  "Chainlink is the leading decentralised oracle network, providing real-world data to smart contracts across 1,000+ blockchain integrations. Node operators are paid in LINK for providing data feeds. Staking v0.2 is now live, allowing LINK holders to earn staking rewards while securing the network.",
    "SNX":   "Synthetix is a derivatives liquidity protocol that enables the creation of synthetic assets tracking real-world prices. SNX stakers collateralise the system and earn all protocol trading fees as rewards — one of the most direct fee-distribution models in DeFi.",
    "PENDLE": "Pendle is a yield tokenisation protocol that splits yield-bearing assets into principal and yield components, allowing users to trade future yield. vePENDLE holders receive 80% of swap fees plus additional protocol yield, making it a strong value-accrual token.",
    "CAKE":  "PancakeSwap is the leading DEX on BNB Chain by volume. It uses a veCAKE model for governance and fee distribution, with buybacks and burns from protocol revenue. It has expanded to multiple chains.",
    "JUP":   "Jupiter is Solana's dominant DEX aggregator, routing trades across all Solana DEXes for best execution. It handles the majority of Solana's swap volume. JUP token holders receive fee distributions, and the protocol is central to Solana's DeFi infrastructure.",
    "GRT":   "The Graph is a decentralised indexing protocol for blockchain data — often described as 'Google for blockchains'. Developers use it to query on-chain data efficiently. GRT is used to pay indexers and curators who maintain and organise the data.",
    # ── Store of Value ────────────────────────────────────────────────────────
    "LTC":   "Litecoin is one of the oldest cryptocurrencies, created in 2011 as a 'lighter' version of Bitcoin. It offers faster block times (2.5 mins) and a larger supply (84M). Often used as a testnet for Bitcoin innovations, LTC is primarily a medium of exchange and store of value.",
    "BCH":   "Bitcoin Cash is a Bitcoin fork created in 2017 to increase block size and enable cheaper on-chain payments. It prioritises low-fee peer-to-peer transactions over store of value.",
    "XMR":   "Monero is the leading privacy-focused cryptocurrency. It uses ring signatures, stealth addresses, and RingCT to make all transactions untraceable by default. XMR is proof-of-work mined and is widely used where financial privacy is paramount.",
    "DOGE":  "Dogecoin started as a meme in 2013 but became a top-10 cryptocurrency by market cap. It has an uncapped supply with 5 billion DOGE minted annually. It benefits from strong community support and celebrity endorsements, and is used for tipping and small payments.",
    "ETC":   "Ethereum Classic is the original Ethereum chain that continued after the 2016 DAO hack fork. It maintains the original 'code is law' philosophy and is proof-of-work. ETC has a fixed supply schedule and positions itself as a store of value for the Ethereum ecosystem.",
    # ── RWA / Stablecoins ─────────────────────────────────────────────────────
    "ENA":   "Ethena is an RWA-backed synthetic dollar protocol. USDe is Ethena's stablecoin, backed by a delta-neutral strategy using staked ETH and short perpetual positions. The protocol distributes yield from this strategy to sUSDe holders and ENA stakers.",
    "XRP":   "XRP is the native token of the XRP Ledger (XRPL), designed for fast, low-cost cross-border payments and currency exchange. Ripple (the company) uses XRP in its On-Demand Liquidity product. XRP settles transactions in 3-5 seconds with fees under $0.01.",
}

def _safe(val, fallback=None):
    if val is None: return fallback
    try:
        v = float(val)
        return v if v == v else fallback   # NaN check
    except (TypeError, ValueError):
        return fallback

def _round_price(p, sig=6):
    """
    Round a price to a sensible number of significant figures.

    Standard round(x, 4) destroys sub-cent token prices:
      round(3.8796e-08, 4) → 0.0   ← kills valuation chart

    This function preserves 6 significant figures regardless of magnitude:
      3.8796e-08 → 3.8796e-08   ELON, ELEPHANT, etc.
      0.103593   → 0.1036       ENA
      2140.74    → 2140.74      ETH
    """
    import math
    if not p or p == 0:
        return 0.0
    if p >= 0.01:
        return round(p, 4)
    # Sub-cent: keep sig significant figures
    magnitude = math.floor(math.log10(abs(p)))
    return round(p, -int(magnitude) + sig - 1)

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
                                     annual_holders_revenue, tvl } }

    Annualisation priority (most stable first):
      1. total30d * (365/30) — 30-day smoothed average
      2. total7d  * (365/7)  — 7-day smoothed average
      3. total24h * 365      — last resort (single-day, noisy)

    IMPORTANT: totalAllTime is CUMULATIVE all-time revenue (e.g. ETH = $20B
    since 2015). It must NEVER be used as annual_revenue — that was the root
    cause of ETH showing intrinsic_value = $14 instead of ~$2,000.
    """
    result = {}

    def _annualise(p):
        t30 = _safe(p.get("total30d"))
        t7  = _safe(p.get("total7d"))
        t24 = _safe(p.get("total24h", 0))
        if t30 and t30 > 0:  return round(t30 * (365 / 30), 2)
        if t7  and t7  > 0:  return round(t7  * (365 / 7),  2)
        if t24 and t24 > 0:  return round(t24 * 365,         2)
        return 0

    # ── Pass 1: fees ─────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{DEFILLAMA_FEES}?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true",
            timeout=20
        )
        if resp.status_code != 200:
            return result
        data = resp.json()

        for p in data.get("protocols", []):
            slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
            annual_fees = _annualise(p)
            result[slug] = {
                "annual_fees":            annual_fees,
                "annual_revenue":         None,   # filled in pass 2
                "annual_holders_revenue": None,   # filled in pass 3
                "tvl":                    _safe(p.get("tvl")),
            }
    except Exception as e:
        print(f"  [DeFiLlama] fees fetch failed: {e}")

    # ── Pass 2: revenue (annualised — NOT totalAllTime) ───────────────────────
    try:
        resp_rev = requests.get(
            f"{DEFILLAMA_FEES}?excludeTotalDataChart=true"
            f"&excludeTotalDataChartBreakdown=true&dataType=dailyRevenue",
            timeout=20
        )
        if resp_rev.status_code == 200:
            for p in resp_rev.json().get("protocols", []):
                slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
                if slug in result:
                    annual_rev = _annualise(p) or None
                    annual_fees = result[slug]["annual_fees"] or 0
                    # Sanity guard: revenue can never exceed fees
                    # (revenue = fees retained by protocol, always ≤ total fees)
                    if annual_rev and annual_fees and annual_rev > annual_fees:
                        annual_rev = round(annual_fees * 0.15, 2)
                    result[slug]["annual_revenue"] = annual_rev
    except Exception as e:
        print(f"  [DeFiLlama] revenue fetch failed: {e}")

    # Fill missing revenue with 15% fee estimate
    for slug in result:
        if result[slug]["annual_revenue"] is None:
            fees = result[slug]["annual_fees"] or 0
            result[slug]["annual_revenue"] = round(fees * 0.15, 2) if fees else None

    # ── Pass 3: holders revenue ───────────────────────────────────────────────
    try:
        resp2 = requests.get(
            f"{DEFILLAMA_FEES}?excludeTotalDataChart=true"
            f"&excludeTotalDataChartBreakdown=true&dataType=dailyHoldersRevenue",
            timeout=20
        )
        if resp2.status_code == 200:
            for p in resp2.json().get("protocols", []):
                slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
                if slug in result:
                    holders_rev = _annualise(p) or None
                    annual_fees = result[slug]["annual_fees"] or 0
                    # Sanity guard: holders revenue can never exceed total fees
                    if holders_rev and annual_fees and holders_rev > annual_fees:
                        holders_rev = None
                    result[slug]["annual_holders_revenue"] = holders_rev
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
                # Use same annualisation as protocol fees — never totalAllTime
                t30 = (_safe(p.get("total30d")) or 0)
                t7  = (_safe(p.get("total7d"))  or 0)
                t24 = (_safe(p.get("total24h"))  or 0)
                if t30 > 0:   ann_fees = t30 * (365 / 30)
                elif t7 > 0:  ann_fees = t7  * (365 / 7)
                else:         ann_fees = t24 * 365
                result[name] = {
                    "annual_fees":    round(ann_fees, 2),
                    "annual_revenue": round(ann_fees * 0.30, 2),   # 30% retention estimate for chains
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
    Score a token's investment quality across 6 signals (0-10 each, max 60).

    KEY DESIGN PRINCIPLES
    ─────────────────────
    1. Missing data → neutral score (5/10), NEVER zero.
       Absence of DeFiLlama coverage ≠ absence of protocol quality.
       Many excellent protocols (UNI, AAVE, SOL) generate real economic
       value through channels not fully indexed by DeFiLlama.

    2. Known-protocol quality hints override weak proxy scores for
       blue-chip protocols where fundamental quality is well-established
       from public data (fee revenues, TVL, governance participation).

    3. Classification thresholds are calibrated for a neutral-default system:
       - Excellent  ≥ 75%  (A): BTC, ETH — absolute top tier
       - Strong     ≥ 60%  (B): SOL, BNB, UNI, AAVE — blue chip
       - Moderate   ≥ 45%  (C): ADA, DOT, MATIC — established
       - Weak       ≥ 30%  (D): smaller alts, limited track record
       - Poor        < 30%  (F): highly speculative / red flags

    Returns:
      {
        "scores": { signal: { "score", "max", "value", "label", ... } },
        "final_score":     float (0-10),
        "final_score_pct": float (0-100),
        "classification":  str,
        "grade":           str  (A/B/C/D/F),
      }
    """
    signals = {}
    current_year = str(datetime.now().year)
    _sym = (symbol or "").upper()

    # ── Known-protocol quality hints ──────────────────────────────────────────
    # Curated scores based on 2024 public data (DeFiLlama, protocol docs).
    # Format: symbol → { fee_growth, capital_efficiency, holder_value_accrual }
    # These apply ONLY when DeFiLlama live data is unavailable.
    # Real on-chain data always takes precedence when available.
    #
    # SCORING PHILOSOPHY (per signal, 0-10):
    #   fee_growth:           Does the protocol generate meaningful fees relative
    #                         to its TVL/market cap? Are fees growing?
    #   capital_efficiency:   How efficiently does the protocol use locked capital
    #                         to generate revenue? (low MC/fees ratio = efficient)
    #   holder_value_accrual: Do token holders DIRECTLY receive protocol revenue?
    #                         Governance-only tokens score low even with high fees.
    #
    # IMPORTANT DISTINCTIONS:
    #   - ETH/SOL/BNB: L1 settlement layers — fees + burns + staking yield
    #   - UNI: MASSIVE volume but fee switch OFF — LPs get fees, UNI holders get 0
    #   - ARB/OP: sequencer fees go to foundation, not token holders
    #   - GMX/SNX/GNS: best-in-class fee distribution TO holders
    #
    # Sources: DeFiLlama 2024, protocol documentation, governance forums
    PROTOCOL_HINTS = {

        # ══════════════════════════════════════════════════════════════════════
        # TIER 1 — Settlement Layers (irreplaceable infrastructure)
        # ══════════════════════════════════════════════════════════════════════

        # ETH: 2024 ~$2.7B fees, ~$1.6B burned (EIP-1559), stakers earn
        # 3-4% real yield. Deflationary + productive = strongest value capture
        # in crypto. Cannot be forked without losing the entire network effect.
        "ETH":    {"fee_growth": 10, "capital_efficiency": 9, "holder_value_accrual": 8},

        # SOL: 2024 ~$800M fees (fastest growing L1), 50%+ of supply staking,
        # staking yield ~7-8%, MEV capture growing. Post-FTX recovery validates
        # network resilience. Significant validator revenue.
        "SOL":    {"fee_growth": 9,  "capital_efficiency": 8, "holder_value_accrual": 7},

        # BNB: quarterly burn mechanism (~$1.5B burned 2024), BSC gas utility,
        # Binance ecosystem integration. Centralisation risk caps score.
        "BNB":    {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 6},

        # TRX: high transaction volume, USDT on Tron is #1 stablecoin chain,
        # TRX burn mechanism active. Centralisation concern limits upside.
        "TRX":    {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 5},

        # TON: Telegram integration driving real user adoption, growing fees,
        # validator staking yield. Early stage but genuine network activity.
        "TON":    {"fee_growth": 7,  "capital_efficiency": 6, "holder_value_accrual": 5},

        # ══════════════════════════════════════════════════════════════════════
        # TIER 2 — Blue-chip DeFi (real revenue, strong value capture)
        # ══════════════════════════════════════════════════════════════════════

        # MKR: 2024 ~$200M+ stability fee revenue, active MKR buyback-burn,
        # DAI is most battle-tested decentralised stablecoin. Endgame roadmap.
        # Token holders directly benefit from burns.
        "MKR":    {"fee_growth": 8,  "capital_efficiency": 8, "holder_value_accrual": 8},

        # GMX: 2024 ~$300M+ fees, 30% to GMX stakers, 70% to GLP.
        # Best-in-class holder value accrual in DeFi perps.
        "GMX":    {"fee_growth": 8,  "capital_efficiency": 8, "holder_value_accrual": 9},

        # PENDLE: vePENDLE receives 80% of swap fees + additional protocol yield.
        # Yield tokenisation is a growing niche with real institutional demand.
        "PENDLE": {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 8},

        # SNX: All protocol fees distributed to SNX stakers. Synthetix perps
        # powers many major frontends. 2024 revenue ~$180M+.
        "SNX":    {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 8},

        # AAVE: 2024 ~$300M+ fees, $16B TVL (#1 lending protocol), fee switch
        # proposal active (Aavenomics). Currently partial holder accrual.
        "AAVE":   {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 6},

        # LDO: 2024 ~$200M protocol revenue (10% of ~$30B stETH staking rewards).
        # Lido controls ~32% of all staked ETH. Accrual to DAO treasury (indirect).
        "LDO":    {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 5},

        # CVX: Controls Curve gauge votes, vlCVX earns cvxCRV weekly revenue.
        # Flywheel: Convex owns most veCRV → earns Curve fees → rewards vlCVX.
        "CVX":    {"fee_growth": 7,  "capital_efficiency": 7, "holder_value_accrual": 8},

        # CRV: veCRV receives 50% of all Curve trading fees. $4B+ TVL.
        # Gauge system drives sustained fee generation. Declining market share risk.
        "CRV":    {"fee_growth": 7,  "capital_efficiency": 7, "holder_value_accrual": 7},

        # RPL: 15% commission on all Rocket Pool ETH staked by node operators.
        # Decentralised ETH staking protocol. ETH staking growth = RPL revenue growth.
        "RPL":    {"fee_growth": 7,  "capital_efficiency": 7, "holder_value_accrual": 7},

        # GNS: All gTrade protocol fees distributed to GNS/DAI stakers.
        # Concentrated decentralised perps on Polygon/Arbitrum. Small but efficient.
        "GNS":    {"fee_growth": 7,  "capital_efficiency": 7, "holder_value_accrual": 8},

        # ENA: Ethena sUSDe delta-neutral yield strategy, protocol revenue
        # shared with ENA stakers via sENA. Growing TVL in RWA/yield space.
        "ENA":    {"fee_growth": 7,  "capital_efficiency": 7, "holder_value_accrual": 7},

        # CAKE: veCAKE buyback-burn + direct fee distribution to stakers.
        # BSC #1 DEX by volume. Revenue declining with BSC but still significant.
        "CAKE":   {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 7},

        # JUP: Jupiter is Solana's dominant aggregator. Fee sharing to JUP holders
        # active. Benefits from Solana's fee growth directly.
        "JUP":    {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 6},

        # RAY: Raydium is Solana's leading AMM, buyback from protocol fees.
        # Benefits from Solana ecosystem growth.
        "RAY":    {"fee_growth": 8,  "capital_efficiency": 7, "holder_value_accrual": 6},

        # DYDX: v4 chain with fees going to stakers. Growing perps volume.
        # Migration to own chain adds complexity but enables direct fee accrual.
        "DYDX":   {"fee_growth": 7,  "capital_efficiency": 6, "holder_value_accrual": 6},

        # ══════════════════════════════════════════════════════════════════════
        # TIER 3 — Good DeFi but limited direct holder accrual
        # ══════════════════════════════════════════════════════════════════════

        # UNI: CRITICAL — fee switch is OFF as of 2024/2025.
        # Uniswap generates $1T+ in annual volume and ~$1B+ in LP fees.
        # BUT: 100% of fees go to liquidity providers, ZERO to UNI holders.
        # UNI value is governance rights + fee switch optionality only.
        # Fee switch vote failed in 2024; new proposal in 2025 may change this.
        # Score reflects current reality, not future potential.
        "UNI":    {"fee_growth": 7,  "capital_efficiency": 6, "holder_value_accrual": 3},

        # LINK: Staking v0.2 active, node operators receive query fees.
        # Oracle market leader but token accrual model still maturing.
        "LINK":   {"fee_growth": 7,  "capital_efficiency": 7, "holder_value_accrual": 5},

        # BAL: veBAL receives 50-75% of protocol fees. Smaller than Curve/Uniswap.
        "BAL":    {"fee_growth": 6,  "capital_efficiency": 6, "holder_value_accrual": 6},

        # COMP: Governance token, some fee accrual, declining market share vs Aave.
        "COMP":   {"fee_growth": 6,  "capital_efficiency": 6, "holder_value_accrual": 4},

        # SUSHI: xSUSHI earns 0.05% of swap fees. Declining market share.
        "SUSHI":  {"fee_growth": 6,  "capital_efficiency": 5, "holder_value_accrual": 5},

        # GRT: Query fees to indexers, curation signal rewards. Growing ecosystem.
        "GRT":    {"fee_growth": 6,  "capital_efficiency": 6, "holder_value_accrual": 5},

        # PYTH: Growing oracle adoption on Solana + multichain, staking rewards.
        "PYTH":   {"fee_growth": 6,  "capital_efficiency": 6, "holder_value_accrual": 5},

        # PERP: Protocol perps, some fee distribution. Smaller scale.
        "PERP":   {"fee_growth": 5,  "capital_efficiency": 5, "holder_value_accrual": 5},

        # 1INCH: Fusion model fees, partial buyback. Declining DEX aggregator share.
        "1INCH":  {"fee_growth": 5,  "capital_efficiency": 5, "holder_value_accrual": 4},

        # BAND":   oracle, smaller scale than LINK
        "BAND":   {"fee_growth": 5,  "capital_efficiency": 5, "holder_value_accrual": 4},

        # API3: dAPI fees, smaller oracle network
        "API3":   {"fee_growth": 5,  "capital_efficiency": 5, "holder_value_accrual": 4},

        # ══════════════════════════════════════════════════════════════════════
        # L2s — sequencer fees go to foundation, NOT token holders
        # ══════════════════════════════════════════════════════════════════════

        # ARB: Arbitrum Foundation collects sequencer revenue. ARB is governance
        # only. No direct fee accrual to ARB holders. Large supply overhang.
        "ARB":    {"fee_growth": 5,  "capital_efficiency": 5, "holder_value_accrual": 2},

        # OP: Same as ARB — OP Foundation gets fees. OP is governance token.
        "OP":     {"fee_growth": 5,  "capital_efficiency": 5, "holder_value_accrual": 2},

        # AXL: Cross-chain fees, growing bridge volume, staking rewards emerging.
        "AXL":    {"fee_growth": 6,  "capital_efficiency": 6, "holder_value_accrual": 5},
    }

    # Tokens that distribute value through staking/validator rewards (not direct fee split)
    L1_INFRA_TOKENS = CHAIN_FEE_MAP.keys() | {
        "LINK", "GRT", "RNDR", "FIL", "AR", "AKT", "PYTH",
        "BAND", "API3", "STORJ", "OCEAN",
        "SAND", "MANA", "BLUR", "IMX", "GODS",
        "WLD", "HNT", "MOBILE",
    }

    # ── Extract values from Network Economics charts where available ──────────
    ne = network_econ or {}
    cap_eff_chart  = ne.get("capital_efficiency", {})
    curr_cap_eff   = cap_eff_chart.get(current_year, {})
    fee_tvl_chart  = curr_cap_eff.get("fee_tvl_ratio_pct")
    real_yield_chart = curr_cap_eff.get("real_yield_pct")

    rev_fees_chart = ne.get("protocol_revenue_and_fees", {})
    curr_rev_fees  = rev_fees_chart.get(current_year, {})
    fees_chart_m   = curr_rev_fees.get("fees_usd")
    rev_chart_m    = curr_rev_fees.get("revenue_usd")
    holders_pct_chart = (
        # Sanity cap: revenue can never exceed 100% of fees
        # If > 100% it means cumulative totalAllTime leaked in — discard it
        min(round(rev_chart_m / fees_chart_m * 100, 1), 100.0)
        if rev_chart_m and fees_chart_m and fees_chart_m > 0
           and rev_chart_m <= fees_chart_m  # revenue must not exceed fees
        else None
    )

    issuance_chart = ne.get("issuance_vs_burns", {})
    curr_issuance  = issuance_chart.get(current_year, {})
    overhang_chart = curr_issuance.get("supply_overhang_pct")
    fdv_mc_chart   = curr_issuance.get("fdv_to_mc")

    treasury_chart = ne.get("treasury_vs_emissions", {})
    curr_treasury  = treasury_chart.get(current_year, {})
    treasury_m     = curr_treasury.get("treasury_usd_m")

    fee_tvl     = fee_tvl_chart  if fee_tvl_chart  is not None else (
                  (annual_fees / tvl * 100) if annual_fees and tvl and tvl > 0 else None)
    holders_pct = holders_pct_chart if holders_pct_chart is not None else (
                  # Sanity: holders revenue cannot exceed total fees
                  min(annual_holders_rev / annual_fees * 100, 100.0)
                  if annual_holders_rev and annual_fees and annual_fees > 0
                     and annual_holders_rev <= annual_fees
                  else None)
    overhang    = overhang_chart if overhang_chart is not None else (
                  (dilution or {}).get("supply_overhang"))
    fdv_mc_use  = fdv_mc_chart   if fdv_mc_chart   is not None else fdv_mc_ratio

    circ_pct   = (dilution or {}).get("circulating_pct")
    vol_mc_pct = (volume / mc * 100) if volume and mc and mc > 0 else None

    def _src(used_chart):
        return "Network Economics chart" if used_chart else "raw calculation"

    # Protocol-level hints (used only when real data unavailable)
    hints = PROTOCOL_HINTS.get(_sym, {})

    if bucket == "A":
        fee_data_available  = fee_tvl is not None
        mc_fee_available    = mc_ps_ratio is not None
        holders_available   = holders_pct is not None
        is_l1_or_infra      = _sym in L1_INFRA_TOKENS

        # ── Signal 1: Fee Growth (0-10) ───────────────────────────────────────
        # Primary: fee/TVL ratio from DeFiLlama capital_efficiency chart
        # Secondary: YoY fee growth trend from historical fee series
        # Tertiary: vol/MC proxy (rough demand signal)
        if fee_data_available:
            if fee_tvl >= 20:   s1 = 10
            elif fee_tvl >= 10: s1 = 8
            elif fee_tvl >= 5:  s1 = 6
            elif fee_tvl >= 2:  s1 = 4
            elif fee_tvl >= 1:  s1 = 2
            else:               s1 = 1
        else:
            # Try to compute YoY fee growth trend from historical fee data
            # Uses network_economics.protocol_revenue_and_fees already fetched
            _yoy_score = None
            try:
                _hist = (network_econ or {}).get("protocol_revenue_and_fees", {})
                _curr_yr = datetime.now().year
                _fee_yrs = sorted(
                    [(int(y), (e.get("fees_usd") or 0))
                     for y, e in _hist.items()
                     if int(y) < _curr_yr and not e.get("is_estimated", False)
                     and (e.get("fees_usd") or 0) > 0],
                    reverse=True
                )
                if len(_fee_yrs) >= 2:
                    _recent, _prior = _fee_yrs[0][1], _fee_yrs[1][1]
                    if _prior > 0:
                        _yoy_pct = (_recent - _prior) / _prior * 100
                        if _yoy_pct >= 50:   _yoy_score = 10
                        elif _yoy_pct >= 20: _yoy_score = 8
                        elif _yoy_pct >= 0:  _yoy_score = 6
                        elif _yoy_pct >= -20:_yoy_score = 4
                        else:                _yoy_score = 2
            except Exception:
                pass

            if _yoy_score is not None:
                s1 = _yoy_score
            elif _sym in hints:
                s1 = hints["fee_growth"]
            elif vol_mc_pct and vol_mc_pct >= 10:   s1 = 6
            elif vol_mc_pct and vol_mc_pct >= 5:    s1 = 5
            elif vol_mc_pct and vol_mc_pct >= 2:    s1 = 4
            else:                                    s1 = 5   # neutral default
        signals["fee_growth"] = {
            "label": "Fee Growth", "score": s1, "max": 10,
            "value": round(fee_tvl, 2) if fee_tvl else None,
            "unit": "fee/tvl %" if fee_data_available else (
                    "curated estimate" if _sym in hints else "vol/MC proxy"),
            "chart_source": _src(fee_tvl_chart is not None),
            "chart_ref": "Network Economics → Capital Efficiency",
            "data_available": fee_data_available,
        }

        # ── Signal 2: Capital Efficiency (0-10) ───────────────────────────────
        # MC/Fees ratio — lower is more efficient (protocol earns more vs its valuation).
        # Thresholds calibrated against real DeFi protocol data:
        #   <5x    → 10  (GMX, SNX — exceptional fee generation)
        #   5-15x  → 8   (top DeFi protocols)
        #   15-30x → 6   (strong — e.g. ETH at median fees ~100x still earns 4)
        #   30-75x → 5   (adequate)
        #   75-150x→ 4   (moderate — ETH in bear market at 100x)
        #   150-400→ 3   (weak — large premium to current fee generation)
        #   400-800→ 2   (very weak)
        #   >800x  → 1   (extreme — current fees negligible vs market cap)
        # NOTE: scores use normalised (median 3Y) fees where available,
        # preventing bear-market fee troughs from permanently punishing
        # established protocols like ETH.
        if mc_fee_available:
            if mc_ps_ratio < 5:     s2 = 10
            elif mc_ps_ratio < 15:  s2 = 8
            elif mc_ps_ratio < 30:  s2 = 6
            elif mc_ps_ratio < 75:  s2 = 5
            elif mc_ps_ratio < 150: s2 = 4
            elif mc_ps_ratio < 400: s2 = 3
            elif mc_ps_ratio < 800: s2 = 2
            else:                   s2 = 1
        elif _sym in hints:
            s2 = hints["capital_efficiency"]
        elif rank and rank <= 10:   s2 = 7
        elif rank and rank <= 25:   s2 = 6
        elif rank and rank <= 50:   s2 = 5
        else:                       s2 = 5   # neutral default
        signals["capital_efficiency"] = {
            "label": "Capital Efficiency", "score": s2, "max": 10,
            "value": mc_ps_ratio,
            "unit": "MC/fees ratio" if mc_fee_available else (
                    "curated estimate" if _sym in hints else "rank proxy"),
            "chart_source": "Protocol Metrics (MC/Fees)" if mc_fee_available
                            else _src(False),
            "chart_ref": "Network Economics → Capital Efficiency",
            "data_available": mc_fee_available,
        }

        # ── Signal 3: Holder Value Accrual (0-10) ─────────────────────────────
        # For L1 chains (ETH, SOL, BNB), value accrual happens through:
        #   1. Direct fee share (holders_pct from DeFiLlama)
        #   2. Token burns (EIP-1559 on ETH — deflationary mechanism)
        #   3. Staking yield (validator rewards from priority fees + MEV)
        # DeFiLlama's "holders revenue" only captures (1), missing burns and
        # staking yield which are equally real forms of value accrual.
        # For known L1s with burn mechanisms, we give a minimum credit of 6
        # even when the raw holders_pct appears low.
        BURN_MECHANISM_TOKENS = {"ETH", "BNB", "TRX", "AVAX", "SOL"}

        if holders_available:
            if holders_pct >= 50:   s3 = 10
            elif holders_pct >= 30: s3 = 8
            elif holders_pct >= 20: s3 = 6
            elif holders_pct >= 10: s3 = 4
            elif holders_pct >= 5:  s3 = 2
            else:                   s3 = 1
            # Uplift for tokens with burn mechanisms — burns are real holder value
            # even if not captured in DeFiLlama's holders_revenue field
            if _sym in BURN_MECHANISM_TOKENS and s3 < 6:
                s3 = 6
        elif _sym in hints:
            s3 = hints["holder_value_accrual"]
        elif is_l1_or_infra:
            # L1/infra distributes via staking — partial credit by network size
            if rank and rank <= 10:   s3 = 6
            elif rank and rank <= 30: s3 = 5
            else:                     s3 = 4
        else:
            s3 = 5   # neutral default — unknown ≠ bad
        signals["holder_value_accrual"] = {
            "label": "Holder Value Accrual", "score": s3, "max": 10,
            "value": round(holders_pct, 1) if holders_pct else None,
            "unit": "% of fees to holders" if holders_available else (
                    "curated estimate" if _sym in hints else
                    "staking/validator proxy" if is_l1_or_infra else "estimated"),
            "chart_source": _src(holders_pct_chart is not None),
            "chart_ref": "Network Economics → Protocol Revenue & Fees",
            "data_available": holders_available,
        }

        # ── Signal 4: Network Demand (0-10) ───────────────────────────────────
        # vol/MC % = how actively the token is traded relative to its market cap.
        # STABLECOIN CAP: USDT/USDC/EURC etc. have 100%+ vol/MC because they are
        # mediums of exchange, not protocols generating fees. Their vol/MC reflects
        # transaction volume not protocol adoption. We cap at 15% for stablecoins
        # (identifiable by price ~$1 and name containing 'USD', 'EUR', 'stable').
        _is_stable = (price and 0.95 <= price <= 1.10) or any(
            kw in (_sym or "").upper()
            for kw in ("USD", "USDC", "USDT", "DAI", "EUR", "FRAX", "TUSD", "BUSD",
                       "USDD", "CUSD", "GUSD", "USDP", "SUSD", "LUSD")
        )
        _vol_mc_adj = min(vol_mc_pct, 15.0) if (_is_stable and vol_mc_pct) else vol_mc_pct

        if _vol_mc_adj is not None:
            if _vol_mc_adj >= 20:   s4 = 10
            elif _vol_mc_adj >= 10: s4 = 8
            elif _vol_mc_adj >= 5:  s4 = 6
            elif _vol_mc_adj >= 2:  s4 = 4
            elif _vol_mc_adj >= 1:  s4 = 2
            else:                   s4 = 1
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

        # ── Signal 5: Dilution Control (0-10) — default NEUTRAL not ZERO ─────
        # IMPORTANT: default is 5 (neutral). Only score negatively when we have
        # actual evidence of high dilution. Missing data ≠ high dilution.
        if circ_pct is not None and circ_pct > 0:
            if circ_pct >= 95:   s5 = 10
            elif circ_pct >= 85: s5 = 8
            elif circ_pct >= 70: s5 = 6
            elif circ_pct >= 50: s5 = 4
            elif circ_pct >= 30: s5 = 2
            else:                s5 = 1
        else:
            s5 = 5   # neutral — no dilution data available
        signals["dilution_control"] = {
            "label": "Dilution Control", "score": s5, "max": 10,
            "value": circ_pct, "unit": "circulating %",
            "chart_source": _src(overhang_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # ── Signal 6: Protocol Maturity (0-10) — default NEUTRAL not ZERO ────
        # IMPORTANT: default is 5 (neutral). Rank unavailable ≠ rank 500+.
        if rank is not None and rank > 0:
            if rank <= 10:    s6 = 10
            elif rank <= 25:  s6 = 8
            elif rank <= 50:  s6 = 6
            elif rank <= 100: s6 = 4
            elif rank <= 200: s6 = 2
            else:             s6 = 1
        else:
            s6 = 5   # neutral — rank temporarily unavailable
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
        # How close to fully circulating — lower FDV/MC = better quality SoV.
        # Default is NEUTRAL (5) when data unavailable, not 0.
        # FDV/MC > 1.50 gets score 1 (not 0) — penalised but not zeroed.
        s4 = 5   # neutral default
        if fdv_mc_use is not None:
            if fdv_mc_use <= 1.02:   s4 = 10
            elif fdv_mc_use <= 1.05: s4 = 8
            elif fdv_mc_use <= 1.10: s4 = 6
            elif fdv_mc_use <= 1.20: s4 = 4
            elif fdv_mc_use <= 1.50: s4 = 2
            else:                    s4 = 1   # large supply overhang — penalised not zeroed
        signals["monetary_premium_quality"] = {
            "label": "Monetary Premium Quality", "score": s4, "max": 10,
            "value": fdv_mc_use, "unit": "FDV/MC ratio",
            "chart_source": _src(fdv_mc_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # 5. Dilution Control (0-10)
        # Default NEUTRAL (5) when circ_pct unavailable, not 0.
        s5 = 5   # neutral default
        if circ_pct is not None:
            if circ_pct >= 95:   s5 = 10
            elif circ_pct >= 85: s5 = 8
            elif circ_pct >= 70: s5 = 6
            elif circ_pct >= 50: s5 = 4
            elif circ_pct >= 30: s5 = 2
            else:                s5 = 1
        signals["dilution_control"] = {
            "label": "Dilution Control", "score": s5, "max": 10,
            "value": circ_pct, "unit": "circulating %",
            "chart_source": _src(overhang_chart is not None),
            "chart_ref": "Network Economics → Token Supply: Issuance vs Burns",
        }

        # 6. Market Resilience — blended 7d and 30d price performance (0-10)
        # Uses a 60/40 blend of 30d and 7d change to reduce single-week noise.
        # Default NEUTRAL (5) when both unavailable, not 0.
        s6 = 5   # neutral default
        chg_30 = price_change_30d or 0
        chg_7  = price_change_7d  or 0
        if price_change_30d is not None or price_change_7d is not None:
            # Weighted blend: 30d carries more weight (trend) than 7d (noise)
            chg = chg_30 * 0.6 + chg_7 * 0.4
            if chg >= 20:    s6 = 10
            elif chg >= 10:  s6 = 8
            elif chg >= 2:   s6 = 6
            elif chg >= -10: s6 = 4
            elif chg >= -25: s6 = 2
            else:            s6 = 1
        signals["market_resilience"] = {
            "label": "Market Resilience", "score": s6, "max": 10,
            "value": round(chg_30 * 0.6 + chg_7 * 0.4, 1)
                     if (price_change_30d is not None or price_change_7d is not None)
                     else None,
            "unit": "blended 30d/7d price change %",
            "chart_source": "raw calculation (CoinGecko price)",
            "chart_ref": None,
        }

    # ── Aggregate ─────────────────────────────────────────────────────────────
    # 6 quantitative signals (0-10 each) = max 60.
    # A 7th signal — Protocol Moat — is added when the Claude API is available.
    # Moat score is stored separately so the UI can display it distinctly.
    total_score = sum(s["score"] for s in signals.values())
    total_max   = sum(s["max"]   for s in signals.values())
    final_pct   = round(total_score / total_max * 100, 1) if total_max > 0 else 0
    final_10    = round(total_score / total_max * 10, 1)  if total_max > 0 else 0

    # ── Optional: Claude API moat score ───────────────────────────────────────
    # Called here so the moat score is embedded in quality output.
    # Only fires when ANTHROPIC_API_KEY is set; otherwise moat is null.
    hints_used = _sym in PROTOCOL_HINTS
    moat_result = score_protocol_moat_via_claude(
        symbol         = _sym,
        name           = _sym,   # name not available here; symbol used as fallback
        bucket         = bucket,
        rank           = rank,
        mc             = mc or 0,
        circ_pct       = circ_pct,
        quality_signals= signals,
        hints_used     = hints_used,
    )
    moat_score = moat_result["score"] if moat_result else None

    # If moat is available, blend it into the final score (weighted 15%)
    # 6 quantitative signals = 85% weight, moat = 15% weight
    if moat_score is not None:
        blended_pct = (final_pct * 0.85) + (moat_score * 10 * 0.15)
        final_pct_with_moat = round(blended_pct, 1)
        final_10_with_moat  = round(blended_pct / 10, 1)
    else:
        final_pct_with_moat = final_pct
        final_10_with_moat  = final_10

    # ── Classification — mirrors stocks classify_quality thresholds exactly ───
    #
    # Thresholds:
    #   ≥ 70%    → Safe
    #   65–69%   → Grey zone: Safe if 2+ of (holder_value_accrual ≥ 6,
    #              fee_growth ≥ 6, protocol_maturity ≥ 6); else Speculative
    #   55–64%   → Speculative
    #   50–54%   → Borderline: Speculative if 2+ of (holder_value_accrual ≥ 4,
    #              fee_growth ≥ 4, protocol_maturity ≥ 4); else Dangerous
    #   < 50%    → Dangerous
    #
    # Crypto tiebreaker signals (equivalent to stocks' ROE/ROIC/EPS):
    #   holder_value_accrual — does the token capture real protocol revenue?
    #   fee_growth           — is the protocol generating meaningful fees?
    #   protocol_maturity    — is the network established (rank, track record)?
    #
    # Moat override (when Claude API has scored moat):
    #   Safe override:      moat ≥ 7 AND (fee_growth ≥ 7 OR holder_value ≥ 7)
    #                       AND final_pct_with_moat ≥ 55
    #   Dangerous override: moat ≤ 3 AND holder_value ≤ 3 AND final_pct < 45

    # Extract individual signal scores for tiebreaker logic
    _hva  = signals.get("holder_value_accrual", {}).get("score") or 0
    _fg   = signals.get("fee_growth",           {}).get("score") or 0
    _pm   = signals.get("protocol_maturity",    {}).get("score") or 0
    _moat = (moat_result or {}).get("score")   # None if API not enabled

    classification = None

    # Moat overrides
    if _moat is not None:
        if _moat >= 7 and (_fg >= 7 or _hva >= 7) and final_pct_with_moat >= 55:
            classification = "Safe"
        if classification is None and _moat <= 3 and _hva <= 3 and final_pct_with_moat < 45:
            classification = "Dangerous"

    # Score-based classification
    if classification is None:
        if final_pct_with_moat >= 70:
            classification = "Safe"

        elif final_pct_with_moat >= 65:
            # Grey zone 65–69%: Safe if 2+ of the three quality tiebreakers pass
            tb = 0
            if _hva >= 6: tb += 1
            if _fg  >= 6: tb += 1
            if _pm  >= 6: tb += 1
            classification = "Safe" if tb >= 2 else "Speculative"

        elif final_pct_with_moat >= 55:
            classification = "Speculative"

        elif final_pct_with_moat >= 50:
            # Borderline 50–54%: Speculative only if 2+ tiebreakers pass (lower bar)
            tb = 0
            if _hva >= 4: tb += 1
            if _fg  >= 4: tb += 1
            if _pm  >= 4: tb += 1
            classification = "Speculative" if tb >= 2 else "Dangerous"

        else:
            classification = "Dangerous"

    return {
        "scores":           signals,
        "moat":             moat_result,
        "final_score":      final_10_with_moat,
        "final_score_pct":  final_pct_with_moat,
        "quantitative_pct": final_pct,
        "classification":   classification,
        "tiebreaker_used":  (50 <= final_pct_with_moat < 70 and _moat is None),
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

# =============================================================================
# MARKET CYCLE CLASSIFIER + ENHANCED FEAR & GREED
# =============================================================================
#
# The market cycle classifier analyses historical price data (already fetched
# for the valuation chart) to determine where a token sits in its cycle.
# This replaces the AI Studio runtime classification, moving it to the pipeline
# so the app receives a pre-computed result with zero rendering overhead.
#
# CYCLE CLASSIFICATION METHOD
# ─────────────────────────────
# Uses three signals derived from the year-end price series:
#
#   1. Long-term trend (vs 3Y simple moving average of annual closes)
#      – Price above 3Y SMA → structural uptrend
#      – Price below 3Y SMA → structural downtrend
#
#   2. Medium-term momentum (year-on-year price change)
#      – Positive YoY → recovering / advancing
#      – Negative YoY → declining
#
#   3. Drawdown from all-time high (ATH)
#      – <25% below ATH  → near top / overextended
#      – 25-55% below    → distribution or early downtrend
#      – 55-80% below    → deep bear / late downtrend
#      – >80% below      → capitulation / accumulation zone
#
# These three signals combine into five cycle phases:
#
#   ACCUMULATION   – Deep drawdown (>55% from ATH), structural downtrend
#                    or recovering from bottom. Best long-term entry.
#   EARLY_UPTREND  – Price recovering, above recent lows, YoY positive,
#                    still well below ATH. Risk-reward improving.
#   LATE_UPTREND   – Strong YoY gains, approaching or above 3Y SMA,
#                    drawdown from ATH <40%. Caution warranted.
#   DISTRIBUTION   – Near ATH (<25% below), momentum may be slowing.
#                    Historically precedes corrections.
#   DOWNTREND      – Price falling, below 3Y SMA, YoY negative. Active
#                    selling pressure. Wait for stabilisation.
#
# MARGIN OF SAFETY ZONES (for the valuation chart)
# ──────────────────────────────────────────────────
# A margin-of-safety buy zone exists when ALL of:
#   • Current price ≤ base_iv (or within 10% above it)
#   • Market cycle is ACCUMULATION or EARLY_UPTREND
#   • Quality score ≥ 4 (not purely speculative)
#
# Chart zones:
#   ACCUMULATION_ZONE  – price below IV, cycle in accumulation/early uptrend
#   NEUTRAL_ZONE       – price near IV (within ±20%), mid-cycle
#   OVEREXTENDED_ZONE  – price >20% above IV, late uptrend / distribution
#
# ENHANCED FEAR & GREED
# ──────────────────────
# The existing 5-component score is recalibrated to anchor on the market
# cycle phase. This prevents the index from being too optimistic in
# structural downtrends (the bug: Extreme Fear market but token scores Neutral).
#
#   Cycle anchor scores (added to raw score before clamping):
#     ACCUMULATION   → anchor = 15  (fear is expected/healthy)
#     EARLY_UPTREND  → anchor = 25  (cautious optimism)
#     LATE_UPTREND   → anchor = 50  (greed building)
#     DISTRIBUTION   → anchor = 60  (peak greed warning)
#     DOWNTREND      → anchor = 10  (fear dominant)
#     UNKNOWN        → anchor = 35  (lean fear, no data)
#
#   Final score = clip( (raw_components_score * 0.6) + (anchor * 0.4), 0, 100 )
#   This ensures: structural bear markets cannot score above ~55 (Neutral),
#   and accumulation zones cannot score above ~45 even on positive momentum.
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


def classify_market_cycle(price_history, current_price):
    """
    Classify the token's current market cycle phase using annual price history.

    price_history: dict of { "2020": 245.6, "2021": 3400.0, ... }
                   year-end closing prices fetched from CoinGecko.
    current_price: float — today's price.

    Returns dict:
    {
        "phase":            str  — ACCUMULATION | EARLY_UPTREND | LATE_UPTREND
                                   | DISTRIBUTION | DOWNTREND | UNKNOWN
        "phase_label":      str  — human-readable label
        "drawdown_from_ath_pct": float — % below all-time-high (0-100)
        "yoy_change_pct":   float | None — year-on-year price change
        "above_3y_sma":     bool | None — price above 3-year SMA of closes
        "ath_price":        float | None
        "sma_3y":           float | None
        "data_years":       int   — how many years of history available
    }
    """
    # Build sorted list of (year_int, price) from history
    year_prices = []
    for yr_str, px in (price_history or {}).items():
        try:
            if px and px > 0:
                year_prices.append((int(yr_str), float(px)))
        except (ValueError, TypeError):
            pass
    year_prices.sort(key=lambda x: x[0])

    # Always include current price as the latest data point
    if current_price and current_price > 0:
        current_year = datetime.now().year
        # Replace or append current year
        year_prices = [(yr, px) for yr, px in year_prices if yr != current_year]
        year_prices.append((current_year, current_price))
        year_prices.sort(key=lambda x: x[0])

    n = len(year_prices)

    if n < 2:
        return {
            "phase":                 "UNKNOWN",
            "phase_label":           "Insufficient Data",
            "drawdown_from_ath_pct": None,
            "yoy_change_pct":        None,
            "above_3y_sma":          None,
            "ath_price":             None,
            "sma_3y":                None,
            "data_years":            n,
        }

    prices_only = [px for _, px in year_prices]

    # ── Signal 1: ATH drawdown ────────────────────────────────────────────────
    ath = max(prices_only)
    drawdown_pct = ((ath - current_price) / ath * 100) if ath > 0 else 0

    # ── Signal 2: Year-on-year change ────────────────────────────────────────
    prev_price = year_prices[-2][1] if n >= 2 else None
    yoy_pct = ((current_price - prev_price) / prev_price * 100) if prev_price else None

    # ── Signal 3: 3Y simple moving average of annual closes ──────────────────
    sma_3y = None
    above_3y_sma = None
    if n >= 3:
        sma_3y = sum(prices_only[-3:]) / 3
        above_3y_sma = current_price > sma_3y

    # ── Phase classification ──────────────────────────────────────────────────
    # Priority order matters — distribution takes precedence over late uptrend
    if drawdown_pct >= 75:
        # Deep capitulation — historically best long-term entry
        phase = "ACCUMULATION"
        phase_label = "Accumulation (deep bear)"

    elif drawdown_pct >= 50:
        if above_3y_sma is False or yoy_pct is None or yoy_pct < 0:
            phase = "ACCUMULATION"
            phase_label = "Accumulation (bear market)"
        else:
            phase = "EARLY_UPTREND"
            phase_label = "Early Uptrend (recovering)"

    elif drawdown_pct >= 25:
        if yoy_pct is not None and yoy_pct < -15:
            phase = "DOWNTREND"
            phase_label = "Downtrend"
        elif above_3y_sma is False:
            phase = "DOWNTREND"
            phase_label = "Downtrend (below SMA)"
        elif yoy_pct is not None and yoy_pct > 30 and above_3y_sma:
            phase = "LATE_UPTREND"
            phase_label = "Late Uptrend"
        else:
            phase = "EARLY_UPTREND"
            phase_label = "Early Uptrend"

    elif drawdown_pct >= 10:
        # Near ATH — caution
        phase = "LATE_UPTREND"
        phase_label = "Late Uptrend / Caution"

    else:
        # Within 10% of ATH
        phase = "DISTRIBUTION"
        phase_label = "Distribution (near ATH)"

    return {
        "phase":                 phase,
        "phase_label":           phase_label,
        "drawdown_from_ath_pct": round(drawdown_pct, 1),
        "yoy_change_pct":        round(yoy_pct, 1) if yoy_pct is not None else None,
        "above_3y_sma":          above_3y_sma,
        "ath_price":             round(ath, 4),
        "sma_3y":                round(sma_3y, 4) if sma_3y else None,
        "data_years":            n,
    }


def compute_chart_zones(price_history, current_price, base_iv, quality_score):
    """
    Compute zone classification for each year in the valuation chart.

    Returns:
    {
        "current_zone":     str  — ACCUMULATION_ZONE | NEUTRAL_ZONE | OVEREXTENDED_ZONE
        "current_zone_label": str
        "margin_of_safety": bool — True if a MoS buy zone exists now
        "mos_note":         str  — explanation for the UI
        "zones_by_year":    { "2026": "ACCUMULATION_ZONE", ... }
    }
    """
    if not base_iv or base_iv <= 0:
        return {
            "current_zone":       "NEUTRAL_ZONE",
            "current_zone_label": "Neutral",
            "margin_of_safety":   False,
            "mos_note":           "Insufficient valuation data",
            "zones_by_year":      {},
        }

    def _price_zone(px, iv):
        if px is None:   return "NEUTRAL_ZONE"
        ratio = px / iv
        if ratio <= 0.80:  return "ACCUMULATION_ZONE"   # >20% below IV
        if ratio <= 1.20:  return "NEUTRAL_ZONE"         # within ±20% of IV
        return "OVEREXTENDED_ZONE"                        # >20% above IV

    zone_labels = {
        "ACCUMULATION_ZONE":  "Accumulation Zone",
        "NEUTRAL_ZONE":       "Neutral Zone",
        "OVEREXTENDED_ZONE":  "Overextended Zone",
    }

    current_zone = _price_zone(current_price, base_iv)

    # MoS requires: price at/below IV AND quality high enough to warrant buying
    cycle = classify_market_cycle(price_history, current_price)
    mos_cycle = cycle["phase"] in ("ACCUMULATION", "EARLY_UPTREND")
    mos_price = current_price <= base_iv * 1.05   # within 5% above IV counts
    mos_quality = (quality_score or 0) >= 4

    margin_of_safety = mos_cycle and mos_price and mos_quality

    if margin_of_safety:
        mos_note = (
            f"Margin of safety present — price near or below speculative fair value "
            f"during {cycle['phase_label'].lower()} phase."
        )
    elif not mos_price:
        mos_note = "Price above speculative fair value — no margin of safety."
    elif not mos_cycle:
        mos_note = (
            f"Cycle phase ({cycle['phase_label']}) not favourable for entry — "
            f"wait for accumulation or early uptrend."
        )
    else:
        mos_note = "Quality score too low to recommend a margin-of-safety entry."

    # Build per-year zones for the forecast chart annotation
    zones_by_year = {}
    all_prices = {yr: px for yr, px in (price_history or {}).items() if px}
    if current_price:
        all_prices[str(datetime.now().year)] = current_price
    for yr_str, px in all_prices.items():
        zones_by_year[yr_str] = _price_zone(px, base_iv)

    return {
        "current_zone":       current_zone,
        "current_zone_label": zone_labels[current_zone],
        "margin_of_safety":   margin_of_safety,
        "mos_note":           mos_note,
        "zones_by_year":      zones_by_year,
    }


def calc_fear_greed(price_change_7d, price_change_30d, volume, market_cap,
                    dilution_data, bucket, market_cycle=None,
                    market_fg_value=None):
    """
    Calculate token-specific Fear & Greed score (0-100).

    market_cycle: output of classify_market_cycle() — used to anchor the score
                  to the structural phase so bear markets score bearish.
    market_fg_value: overall market Fear & Greed (0-100) from alternative.me —
                  used as a soft floor/ceiling depending on direction.

    Returns:
      {
        "score":           int (0-100),
        "label":           str,
        "market_cycle_anchor": int,
        "components": { ... }
      }
    """
    try:
        components = {}
        chg30 = price_change_30d or 0
        chg7  = price_change_7d  or 0

        # ── Component 1: Price Momentum 30d (30 pts max) ─────────────────────
        if chg30 >= 100:   pm = 30
        elif chg30 >= 50:  pm = 26
        elif chg30 >= 20:  pm = 22
        elif chg30 >= 10:  pm = 18
        elif chg30 >= 5:   pm = 14
        elif chg30 >= 0:   pm = 10
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
        else:                  vlt = 8
        components["volatility"] = vlt

        # ── Component 4: Momentum Trend 7d vs 30d (15 pts max) ───────────────
        if chg7 is not None and chg30 is not None:
            if chg7 > 0 and chg7 > abs(chg30) * 0.5:  mt = 15
            elif chg7 > 0 and chg30 > 0:               mt = 12
            elif chg7 > 0 and chg30 <= 0:              mt = 9
            elif chg7 <= 0 and chg30 > 0:              mt = 6
            elif chg7 <= 0 and chg30 <= 0:             mt = 3
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
            dp = 4
        components["dilution_pressure"] = dp

        raw_score = pm + vm + vlt + mt + dp   # 0-100 from components

        # ── Market Cycle Anchor (recalibration) ───────────────────────────────
        # Blends raw component score with a cycle-phase anchor to prevent
        # structural bear markets from scoring optimistically.
        #
        # Anchor scores represent the "expected" sentiment for each phase:
        phase = (market_cycle or {}).get("phase", "UNKNOWN")
        cycle_anchors = {
            "ACCUMULATION":  15,   # deep bear — fear is expected
            "DOWNTREND":     18,   # active selling — fear dominant
            "EARLY_UPTREND": 30,   # cautious recovery
            "LATE_UPTREND":  55,   # greed building
            "DISTRIBUTION":  65,   # peak greed warning
            "UNKNOWN":       30,   # lean fearful without data
        }
        anchor = cycle_anchors.get(phase, 30)

        # Weight: 55% raw components, 45% cycle anchor
        # This means a token in DOWNTREND can't score above ~58 even with
        # strong 30d momentum, and ACCUMULATION tokens score realistically low.
        blended = (raw_score * 0.55) + (anchor * 0.45)

        # Soft influence from market-wide Fear & Greed (10% nudge only)
        if market_fg_value is not None:
            blended = blended * 0.90 + market_fg_value * 0.10

        score = min(100, max(0, round(blended)))
        components["cycle_anchor"] = anchor

        return {
            "score":               score,
            "label":               _fg_label(score),
            "market_cycle_phase":  phase,
            "market_cycle_anchor": anchor,
            "components":          components,
        }

    except Exception:
        return {
            "score": 35, "label": "Fear",
            "market_cycle_phase": "UNKNOWN",
            "market_cycle_anchor": 30,
            "components": {},
        }


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
            chain_data  = (defillama_chain_data or {}).get(chain_slug, {}) \
                          if chain_slug else {}

            # Current annualised fees from 30d average
            annual_fees_current = fee_data.get("annual_fees") or chain_data.get("annual_fees")
            annual_rev          = fee_data.get("annual_revenue") or chain_data.get("annual_revenue")
            holders_rev         = fee_data.get("annual_holders_revenue")
            tvl                 = fee_data.get("tvl") or chain_data.get("tvl")

            # ── Normalised fees — use median of last 3 full years ─────────────
            # Using only the current 30d-annualised snapshot causes severe
            # undervaluation during bear markets (ETH: $120M current vs
            # $2.5B in 2024). The normalised figure gives a more stable
            # base for DCF and PS ratio without over-indexing on peak years.
            #
            # Strategy: collect last 3 full calendar years from the historical
            # fee series already fetched in fetch_network_economics (via the
            # /summary/fees/{slug} endpoint), take the median.
            # Falls back to current annual if no history available.
            annual_fees = annual_fees_current   # default
            try:
                hist_rev_fees = (network_econ or {}).get(
                    "protocol_revenue_and_fees", {}
                )
                current_yr = datetime.now().year
                # Collect full historical years (not current year estimate)
                hist_years = sorted(
                    [
                        (int(yr), entry.get("fees_usd", 0) or 0)
                        for yr, entry in hist_rev_fees.items()
                        if int(yr) < current_yr
                        and not entry.get("is_estimated", False)
                        and (entry.get("fees_usd") or 0) > 0
                    ],
                    reverse=True
                )
                if len(hist_years) >= 2:
                    # Median of last 3 full years (or 2 if only 2 available)
                    last_n   = hist_years[:3]
                    fee_vals = sorted([v * 1e6 for _, v in last_n])  # $M → $
                    mid      = len(fee_vals) // 2
                    normalised = fee_vals[mid]
                    # Only use if meaningfully higher than current (avoids
                    # using stale history when current is legitimately higher)
                    if normalised > (annual_fees_current or 0):
                        annual_fees = normalised
            except Exception:
                pass   # silently fall back to current

            # Method 1 — DCF on fees (uses normalised figure)
            dcf_protocol_value = method_1_dcf_fees(annual_fees)

            # Method 2 — P/S analog (uses normalised figure)
            ps_ratio = method_2_ps_analog(mc, annual_fees)

            # Method 3 — P/E analog
            pe_ratio = method_3_pe_analog(fdv, holders_rev)

            # Per-token fair value from DCF (protocol value / circulating supply)
            dcf_per_token = None
            if dcf_protocol_value and circ and circ > 0:
                dcf_per_token = round(dcf_protocol_value / circ, 4)

            bucket_a_data = {
                "annual_protocol_fees_usd":    round(annual_fees, 2) if annual_fees else None,
                "annual_fees_current_usd":     round(annual_fees_current, 2) if annual_fees_current else None,
                "annual_fees_normalised":      annual_fees != annual_fees_current,
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

        # ── Price History — fetch EARLY before DeFiLlama calls use API budget ──
        # This is the single most rate-limit-sensitive call in the pipeline.
        # By fetching it here (before the 3+ DeFiLlama calls in fetch_network_economics),
        # we have a fresh CoinGecko rate-limit budget and maximise success rate.
        # ETH was failing (price_history_years=0) because by the time the fetch
        # ran (inside the base_iv block), the budget was exhausted by prior calls.
        #
        # Stored as top-level 'price_history' dict (same shape as stocks
        # '10_Year_History') — AI Studio reads it directly for the chart.
        # Also used to build valuation_chart historical rows later.
        price_history       = {}
        price_history_years = 0
        for attempt in range(3):
            try:
                wait = [3.5, 25.0, 60.0][attempt]  # longer waits vs before (2/20/45)
                time.sleep(wait)
                hist_resp = requests.get(
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                    f"/market_chart?vs_currency=usd&days=3650",
                    headers=_headers(), timeout=30
                )
                if hist_resp.status_code == 200:
                    prices_raw = hist_resp.json().get("prices", [])
                    year_prices = {}
                    for ts_ms, p_val in prices_raw:
                        yr = str(datetime.fromtimestamp(ts_ms / 1000).year)
                        year_prices[yr] = _round_price(p_val)
                    price_history       = year_prices
                    price_history_years = len(year_prices)
                    print(f"  [HIST] {symbol}: {price_history_years} years of price history")
                    break
                elif hist_resp.status_code == 429:
                    print(f"  [HIST] {symbol}: 429 rate limit (attempt {attempt+1}/3)")
                    if attempt == 2:
                        print(f"  [HIST] {symbol}: all retries exhausted")
                    continue
                else:
                    print(f"  [HIST] {symbol}: HTTP {hist_resp.status_code}")
                    break
            except Exception as e:
                print(f"  [HIST] {symbol}: {str(e)[:60]}")
                break

        # Always include current year price so chart has at least one anchor
        price_history[str(current_year)] = _round_price(price) if price else None

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

        # ── Fear & Greed — computed after price history fetch ────────────────
        # (placeholder — recalculated below once market_cycle is available)
        _market_fg = fetch_market_fear_greed().get("value")
        fear_greed = calc_fear_greed(
            price_change_7d  = chg_7d,
            price_change_30d = chg_30d,
            volume           = volume,
            market_cap       = mc,
            dilution_data    = dilution,
            bucket           = bucket,
            market_cycle     = None,       # updated below
            market_fg_value  = _market_fg,
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
        # current_year as int needed for chart year keys and arithmetic
        current_year = datetime.now().year

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

            # ══════════════════════════════════════════════════════════════════
            # ══════════════════════════════════════════════════════════════════
            # ══════════════════════════════════════════════════════════════════
            # SPECULATIVE FAIR VALUE — BUCKET-SPECIFIC MULTI-METHOD VALUATION
            # ══════════════════════════════════════════════════════════════════
            #
            # EXPERT RATIONALE FOR BUCKET SEPARATION:
            #
            # Bucket A — Cash-Flow Protocols (ETH, SOL, UNI, AAVE, GMX...)
            #   These are BUSINESSES. Valued on fee revenue, cash flows, and
            #   network utility — identical framework to equity valuation.
            #   Gold capture % and mining cost are category errors here.
            #
            #   Methods:
            #     M1  DCF on fees          — fee cash flows, 30% discount rate
            #     M2  P/S analog           — MC/Fees vs tier median (15x/25x/40x)
            #     M3  P/E analog           — FDV/Holders-revenue vs tier median
            #     M7  Metcalfe             — volume/MC proxy for network activity
            #     M+  NVT Fair Value       — MC vs volume × fair NVT multiple
            #     M+  Peer value           — rank-tier MC benchmark × quality adj
            #     M+  Dilution-adjusted    — corrects for FDV/MC above tier median
            #
            # Bucket B — Store of Value (BTC, LTC, XMR, DOGE, BCH...)
            #   These are MONETARY ASSETS competing with gold and fiat.
            #   P/S and P/E are category errors — miner fees go to miners,
            #   not holders. No business revenue to discount.
            #   Valued on scarcity, production cost, and monetary premium.
            #
            #   Methods:
            #     M6  Monetary premium     — gold capture %; price at 25/50/100% gold
            #     M7  Metcalfe             — network value vs usage (originated w/ BTC)
            #     M8  Production cost      — PoW mining cost floor × historical premium
            #     M10 Dilution (SoV)       — supply schedule; FDV/MC normalisation
            #     M+  NVT Fair Value       — on-chain volume is primary SoV usage signal
            #     M+  Peer value           — rank-tier MC × quality adj (universal)
            #
            # The median of all valid per-token IV outputs becomes base_iv.
            # ══════════════════════════════════════════════════════════════════

            valid_ivs    = []   # per-token fair value estimates
            methods_used = []   # labels shown in hero card iv_method field
            q_pct = quality.get("final_score_pct", 50) or 50

            if bucket == "A":
                # ── A1: DCF on protocol fees ───────────────────────────────────
                # Pre-computed in bucket_a_data by method_1_dcf_fees().
                # Only fires when DeFiLlama has fee data for this protocol.
                proto       = bucket_a_data or {}
                dcf_v       = proto.get("dcf_fair_value_per_token")
                ps_rat      = proto.get("ps_ratio")
                pe_rat      = proto.get("pe_ratio_fdv_to_holders_rev")

                if dcf_v and dcf_v > 0:
                    valid_ivs.append(dcf_v)
                    methods_used.append("M1 DCF (fees)")

                # ── A2: P/S analog — MC/Fees vs tier-median multiple ───────────
                # Tier median PS ratios calibrated from DeFiLlama 2024 data:
                #   Safe (≥70%):    15x — established protocols (ETH, AAVE, GMX)
                #   Safe (60-69%):  20x — strong but not top-tier
                #   Speculative:    30x — growth-stage protocols
                #   Dangerous:      50x — speculative premium
                if ps_rat and ps_rat > 0 and price:
                    if q_pct >= 70:   median_ps = 15
                    elif q_pct >= 60: median_ps = 20
                    elif q_pct >= 45: median_ps = 30
                    else:             median_ps = 50
                    ps_iv = round(price * median_ps / ps_rat, 4)
                    if ps_iv > 0:
                        valid_ivs.append(ps_iv)
                        methods_used.append("M2 P/S analog")

                # ── A3: P/E analog — FDV/Holders-revenue vs tier-median ────────
                # Only fires when DeFiLlama tracks holders revenue separately.
                # Tier median PE ratios (higher than PS — earnings quality premium):
                #   Safe:         25x | Speculative: 45x | Dangerous: 75x
                if pe_rat and pe_rat > 0 and price:
                    if q_pct >= 60:   median_pe = 25
                    elif q_pct >= 45: median_pe = 45
                    else:             median_pe = 75
                    pe_iv = round(price * median_pe / pe_rat, 4)
                    if pe_iv > 0:
                        valid_ivs.append(pe_iv)
                        methods_used.append("M3 P/E analog")

                # ── A7: Metcalfe — network activity vs market cap ──────────────
                # method_7_metcalfe() returns MC/metcalfe_value ratio.
                # For Bucket A: volume proxies for fee-generating transactions.
                # Fair value = price where ratio equals quality-tier median.
                m7_ratio = method_7_metcalfe(mc, volume=volume)
                if m7_ratio and 0.1 <= m7_ratio <= 15 and price:
                    m7_iv = round(price / m7_ratio, 4)
                    if 0.15 * price <= m7_iv <= 7 * price:
                        valid_ivs.append(m7_iv)
                        methods_used.append("M7 Metcalfe")

                # ── A+: NVT Fair Value ─────────────────────────────────────────
                # NVT = MC / daily_volume. Crypto's P/E ratio.
                # Fair NVT for cash-flow protocols (from historical DeFiLlama data):
                #   Quality ≥70%: NVT 20 (efficient protocols justify higher NVT)
                #   Quality 60%:  NVT 15
                #   Speculative:  NVT 10
                #   Dangerous:    NVT  6 (low tolerance for expensive churn volume)
                if volume and volume > 0 and circ and circ > 0:
                    if q_pct >= 70:   fair_nvt = 20
                    elif q_pct >= 60: fair_nvt = 15
                    elif q_pct >= 45: fair_nvt = 10
                    else:             fair_nvt = 6
                    nvt_iv = round((volume * fair_nvt) / circ, 4)
                    if price and 0.1 * price <= nvt_iv <= 10 * price:
                        valid_ivs.append(nvt_iv)
                        methods_used.append("M+ NVT")

                # ── A+: Peer value ─────────────────────────────────────────────
                # Expected MC for rank tier × quality premium/discount.
                if rank and rank > 0 and circ and circ > 0 and price:
                    try:
                        if rank <= 5:      exp_mc = 200e9
                        elif rank <= 10:   exp_mc = 80e9
                        elif rank <= 20:   exp_mc = 30e9
                        elif rank <= 30:   exp_mc = 15e9
                        elif rank <= 50:   exp_mc = 6e9
                        elif rank <= 75:   exp_mc = 2.5e9
                        elif rank <= 100:  exp_mc = 1.2e9
                        elif rank <= 150:  exp_mc = 500e6
                        elif rank <= 200:  exp_mc = 250e6
                        elif rank <= 300:  exp_mc = 120e6
                        elif rank <= 500:  exp_mc = 50e6
                        else:              exp_mc = 20e6
                        q_adj = 1.20 if q_pct >= 70 else 1.05 if q_pct >= 60                                 else 0.90 if q_pct >= 45 else 0.70
                        peer_iv = round((exp_mc * q_adj) / circ, 4)
                        if 0.2 * price <= peer_iv <= 5 * price:
                            valid_ivs.append(peer_iv)
                            methods_used.append("M+ Peer value")
                    except Exception:
                        pass

                # ── A+: Dilution-Adjusted Price ────────────────────────────────
                # If FDV/MC exceeds the fair multiple for this quality tier,
                # the market is ignoring future dilution — adjust price down.
                # Fair FDV/MC by tier (cash-flow protocols should be near-circ):
                #   Safe ≥70%: 1.10 | Safe 60%: 1.25 | Spec: 1.60 | Dangerous: 2.50
                if price and circ and circ > 0:
                    try:
                        dilution_d    = dilution or {}
                        fdv_mc_actual = dilution_d.get("fdv_to_mc_ratio") or 1.0
                        if q_pct >= 70:   fair_fdv_mc = 1.10
                        elif q_pct >= 60: fair_fdv_mc = 1.25
                        elif q_pct >= 45: fair_fdv_mc = 1.60
                        else:             fair_fdv_mc = 2.50
                        if fdv_mc_actual > fair_fdv_mc + 0.10:
                            dilution_iv = round(price * (fair_fdv_mc / fdv_mc_actual), 4)
                            if 0 < dilution_iv < price:
                                valid_ivs.append(dilution_iv)
                                methods_used.append("M+ Dilution-adj")
                    except Exception:
                        pass

            elif bucket == "B":
                sov  = bucket_b_data or {}
                prod = sov.get("cost_of_production") or {}
                mon  = sov.get("monetary_premium") or {}

                # ── B6: Monetary premium — gold capture % ──────────────────────
                # What % of gold's market cap has this asset captured?
                # Fair value = price implied by sustaining current capture %
                # as the gold market grows at its historical ~8% annual rate.
                # Also provides price scenarios at 25/50/100% of gold.
                gcp = mon.get("gold_capture_pct")
                if gcp and gcp > 0 and price:
                    m6_iv = round(price * 1.08, 4)
                    valid_ivs.append(m6_iv)
                    methods_used.append("M6 Monetary premium")

                # ── B7: Metcalfe — on-chain network activity ───────────────────
                # NVT originated as a BTC tool (Willy Woo, 2017).
                # For SoV assets, volume = monetary circulation, not fee revenue.
                # Higher tolerance for low volume (SoV assets are held, not traded).
                m7_ratio = method_7_metcalfe(mc, volume=volume)
                if m7_ratio and 0.05 <= m7_ratio <= 20 and price:
                    m7_iv = round(price / m7_ratio, 4)
                    if 0.1 * price <= m7_iv <= 10 * price:
                        valid_ivs.append(m7_iv)
                        methods_used.append("M7 Metcalfe")

                # ── B8: Cost of production — PoW mining floor ──────────────────
                # Estimated all-in mining cost provides a fundamental floor.
                # BTC historically trades at 1.5-3x production cost at fair value
                # (accounts for miner profit margin and security premium).
                cost = prod.get("estimated_production_cost_usd")
                if cost and cost > 0:
                    m8_iv = round(cost * 2.0, 4)
                    valid_ivs.append(m8_iv)
                    methods_used.append("M8 Production cost")

                # ── B10: Dilution analysis — supply schedule ───────────────────
                # SoV assets MUST be near-fully circulating to credibly claim
                # scarcity. FDV/MC > 1.10 signals hidden dilution risk.
                # e.g. DOGE: technically unlimited — penalised appropriately.
                dilution_d = dilution or {}
                fdv_mc_b   = dilution_d.get("fdv_to_mc_ratio") or 1.0
                if fdv_mc_b > 1.10 and price:
                    # Dilution-adjusted price: what current holders are really worth
                    # if the full supply were circulating
                    m10_iv = round(price * (1.02 / fdv_mc_b), 4)
                    if m10_iv > 0:
                        valid_ivs.append(m10_iv)
                        methods_used.append("M10 Supply-adj")

                # ── B+: NVT Fair Value ─────────────────────────────────────────
                # For SoV assets, NVT measures monetary velocity.
                # SoV assets should have HIGHER NVT than cash-flow protocols
                # because they are held long-term, not actively traded for fees.
                # Fair NVT benchmarks for monetary assets (from BTC/LTC history):
                #   Quality ≥70%: NVT 35 (BTC — premier monetary asset)
                #   Quality 60%:  NVT 25
                #   Speculative:  NVT 15
                #   Dangerous:    NVT  8
                if volume and volume > 0 and circ and circ > 0:
                    if q_pct >= 70:   fair_nvt_b = 35
                    elif q_pct >= 60: fair_nvt_b = 25
                    elif q_pct >= 45: fair_nvt_b = 15
                    else:             fair_nvt_b = 8
                    nvt_iv_b = round((volume * fair_nvt_b) / circ, 4)
                    if price and 0.1 * price <= nvt_iv_b <= 10 * price:
                        valid_ivs.append(nvt_iv_b)
                        methods_used.append("M+ NVT")

                # ── B+: Peer value (same universal logic as Bucket A) ──────────
                if rank and rank > 0 and circ and circ > 0 and price:
                    try:
                        if rank <= 5:      exp_mc = 200e9
                        elif rank <= 10:   exp_mc = 80e9
                        elif rank <= 20:   exp_mc = 30e9
                        elif rank <= 30:   exp_mc = 15e9
                        elif rank <= 50:   exp_mc = 6e9
                        elif rank <= 75:   exp_mc = 2.5e9
                        elif rank <= 100:  exp_mc = 1.2e9
                        elif rank <= 200:  exp_mc = 250e6
                        elif rank <= 500:  exp_mc = 50e6
                        else:              exp_mc = 20e6
                        q_adj = 1.20 if q_pct >= 70 else 1.05 if q_pct >= 60                                 else 0.90 if q_pct >= 45 else 0.70
                        peer_iv = round((exp_mc * q_adj) / circ, 4)
                        if 0.2 * price <= peer_iv <= 5 * price:
                            valid_ivs.append(peer_iv)
                            methods_used.append("M+ Peer value")
                    except Exception:
                        pass

            # ── Aggregate: median of all valid per-token IV estimates ─────────
            if valid_ivs:
                valid_ivs_sorted = sorted(valid_ivs)
                mid     = len(valid_ivs_sorted) // 2
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

                # ── Step 3: Annual compounding growth rates ───────────────
                #
                # Crypto has no EPS — rates are derived from quality score
                # but calibrated against actual historical crypto market cycles.
                #
                # Calibration anchors (based on observed crypto market history):
                #   Bull market cycle: top protocols 2-3x over 2-3 years
                #   Bear market cycle: top protocols -40 to -70% peak-to-trough
                #   Over a full 5Y period blending both: realistic net ~+50-150%
                #
                # These rates represent ANNUAL compounding and are intentionally
                # conservative — not peak-cycle extrapolations.
                #
                # Quality → (base, bull, bear) annual rates:
                #
                #   9-10 │ top-tier (BTC, ETH)    │ +12%  +20%   -8%
                #        │ ETH $2,141 → 5Y:        │ $3,770 $5,265 $1,414
                #
                #   7-8  │ strong (SOL, BNB, LINK) │ +10%  +17%  -12%
                #        │ SOL $89 → 5Y:            │ $143  $196   $47
                #
                #   5-6  │ moderate (ADA, AVAX)    │  +7%  +13%  -16%
                #
                #   3-4  │ weak                    │  +3%   +9%  -22%
                #
                #   0-2  │ speculative             │   0%   +6%  -30%
                #
                # Survivability modifies the bear rate only — resilient tokens
                # recover faster from drawdowns.

                if q_score >= 9:
                    base_growth =  0.12
                    bull_growth =  0.20
                    bear_growth = -0.08
                elif q_score >= 7:
                    base_growth =  0.10
                    bull_growth =  0.17
                    bear_growth = -0.12
                elif q_score >= 5:
                    base_growth =  0.07
                    bull_growth =  0.13
                    bear_growth = -0.16
                elif q_score >= 3:
                    base_growth =  0.03
                    bull_growth =  0.09
                    bear_growth = -0.22
                else:
                    base_growth =  0.00
                    bull_growth =  0.06
                    bear_growth = -0.30

                # Survivability softens the bear rate for resilient tokens.
                # surv_mult 1.0 → no additional penalty
                # surv_mult 0.6 → bear rate worsens by 40% of its absolute value
                surv_bear_adj = abs(bear_growth) * (1.0 - surv_mult) * 0.5
                bear_growth   = bear_growth - surv_bear_adj                # Survivability dampens bear for resilient tokens
                # Strong survivability (1.0) → full bear protection
                # Weak survivability (0.35)  → steeper bear drawdown
                bear_growth = bear_growth * (2.0 - surv_mult)

                # ── Minimum band separation guard ─────────────────────────
                # Ensure bull is always above base and bear always below base,
                # so the chart fan never collapses to a flat line.
                if bull_growth <= base_growth:
                    bull_growth = base_growth + 0.03
                if bear_growth >= base_growth:
                    bear_growth = base_growth - 0.03

                was_constrained = surv_mult < 1.0 or q_mult < 1.0

                # ── Step 4: Price history already fetched early ────────────
                # price_history and price_history_years were populated before
                # the DeFiLlama calls to maximise CoinGecko rate-limit budget.
                # Current year is already set in price_history.

                # ── Step 5: Build historical section ──────────────────────
                # Historical years show actual market price only.
                # IV/bull/bear are NOT back-projected into history —
                # they only appear from current year forward.
                for yr, hist_price in sorted(price_history.items()):
                    if int(yr) < current_year:
                        valuation_chart[yr] = {
                            "market_price":    hist_price,
                            "intrinsic_value": None,   # no IV shown in history
                            "bull_case":       None,
                            "bear_case":       None,
                            "is_historical":   True,
                        }

                # ── Step 6: Current year ───────────────────────────────────
                valuation_chart[str(current_year)] = {
                    "market_price":    _round_price(price) if price else None,
                    "intrinsic_value": _round_price(base_iv),
                    "bull_case":       _round_price(base_iv),   # bands start at IV
                    "bear_case":       _round_price(base_iv),   # bands start at IV
                    "is_historical":   False,
                }

                # ── Step 7: Forecast years — compounding bands ─────────────
                # Each year the bull/bear/base compound from current IV.
                # This creates a naturally widening cone (fan) shape rather
                # than flat horizontal lines — matching how risk accumulates
                # over time and how crypto scenario analysis is normally shown.
                for i in range(1, 6):
                    yr_base = _round_price(base_iv * ((1 + base_growth) ** i))
                    yr_bull = _round_price(base_iv * ((1 + bull_growth) ** i))
                    yr_bear = _round_price(base_iv * ((1 + bear_growth) ** i))
                    yr_bear = max(yr_bear, _round_price(base_iv * 0.01))
                    valuation_chart[str(current_year + i)] = {
                        "market_price":    None,
                        "intrinsic_value": yr_base,
                        "bull_case":       yr_bull,
                        "bear_case":       yr_bear,
                        "is_historical":   False,
                    }

                # ── Step 8: Forecast metadata ──────────────────────────────
                # Fields are crypto-specific — no EPS, no earnings growth.
                # growth_basis explains the methodology for the UI label.
                yr5_base = _round_price(base_iv * ((1 + base_growth) ** 5))
                yr5_bull = _round_price(base_iv * ((1 + bull_growth) ** 5))
                yr5_bear = _round_price(base_iv * ((1 + bear_growth) ** 5))
                yr5_bear = max(yr5_bear, _round_price(base_iv * 0.01))
                forecast_meta = {
                    # Current year anchors
                    "base_iv":                  _round_price(base_iv),
                    "bull_iv":                  _round_price(base_iv),
                    "bear_iv":                  _round_price(base_iv),
                    # 5-year compounded targets
                    "yr5_base_target":          yr5_base,
                    "yr5_bull_target":          yr5_bull,
                    "yr5_bear_target":          yr5_bear,
                    # Annual growth rates (for UI display)
                    "base_annual_growth_pct":   round(base_growth * 100, 1),
                    "bull_annual_growth_pct":   round(bull_growth * 100, 1),
                    "bear_annual_growth_pct":   round(bear_growth * 100, 1),
                    # Upside/downside vs current price over 5 years
                    "bull_5y_upside_pct":       round((yr5_bull / base_iv - 1) * 100, 1),
                    "bear_5y_change_pct":       round((yr5_bear / base_iv - 1) * 100, 1),
                    # Methodology label — shown in the UI instead of "EPS driven"
                    "growth_basis":             "Quality-adjusted network adoption",
                    "growth_basis_note": (
                        f"Annual growth rates derived from quality score "
                        f"{q_score}/10 and survivability {round(surv_pct*100,1)}%. "
                        f"Base: {round(base_growth*100,1)}% p.a., "
                        f"Bull: {round(bull_growth*100,1)}% p.a., "
                        f"Bear: {round(bear_growth*100,1)}% p.a."
                    ),
                    # Quality/survivability context
                    "quality_score":            q_score,
                    "quality_multiplier":       q_mult,
                    "survivability_pct":        round(surv_pct * 100, 1),
                    "survivability_mult":       surv_mult,
                    "growth_constrained":       was_constrained,
                    "bucket":                   bucket,
                    "iv_sources":               len([m for m in methods_used if "Market price" not in m]),
                    "methods_used":             methods_used,
                    # Price history availability — AI Studio uses this to show
                    # a note when historical chart data is limited or unavailable.
                    "price_history_years":      price_history_years,
                    "price_history_note":       (
                        None if price_history_years >= 3
                        else f"Limited price history available ({price_history_years} year(s)). "
                             f"This token may be recently listed or have insufficient on-chain data."
                        if price_history_years > 0
                        else "Historical price data unavailable for this token. "
                             "Forecast chart shows projected values only."
                    ),
                }

                # ── Step 9: Market Cycle + Chart Zones + Recalibrated F&G ───
                # Now that price_history is available, compute cycle phase,
                # chart zones, and recalibrate Fear & Greed with cycle anchor.
                q_score_for_cycle = (quality or {}).get("final_score") or 5
                market_cycle = classify_market_cycle(price_history, price)
                chart_zones  = compute_chart_zones(
                    price_history = price_history,
                    current_price = price,
                    base_iv       = base_iv,
                    quality_score = q_score_for_cycle,
                )
                # Recalculate fear_greed with cycle context
                fear_greed = calc_fear_greed(
                    price_change_7d  = chg_7d,
                    price_change_30d = chg_30d,
                    volume           = volume,
                    market_cap       = mc,
                    dilution_data    = dilution,
                    bucket           = bucket,
                    market_cycle     = market_cycle,
                    market_fg_value  = _market_fg,
                )

        except Exception:
            valuation_chart = {}
            forecast_meta   = {}
            market_cycle    = {"phase": "UNKNOWN", "phase_label": "Insufficient Data",
                               "drawdown_from_ath_pct": None, "yoy_change_pct": None,
                               "above_3y_sma": None, "ath_price": None,
                               "sma_3y": None, "data_years": 0}
            chart_zones     = {"current_zone": "NEUTRAL_ZONE",
                               "current_zone_label": "Neutral",
                               "margin_of_safety": False,
                               "mos_note": "Data unavailable",
                               "zones_by_year": {}}

        # ── Coin Profile — pre-computed so AI Studio never fetches live ─────────
        # Sources: CoinGecko /coins/markets (always available) +
        #          COIN_DESCRIPTIONS curated dict (known tokens) +
        #          CoinGecko /coins/{id} detail (called for top-100 only).
        # AI Studio reads this block for the header, hero card, and any
        # "about this coin" section — never needs a live API call.
        _ath               = _safe(coin.get("ath"))
        _ath_date          = coin.get("ath_date", "")[:10] if coin.get("ath_date") else None
        _ath_change_pct    = _safe(coin.get("ath_change_percentage"))
        _atl               = _safe(coin.get("atl"))
        _atl_date          = coin.get("atl_date", "")[:10] if coin.get("atl_date") else None
        _image             = coin.get("image")
        _description       = (
            COIN_DESCRIPTIONS.get(symbol)          # curated (preferred — always present)
            or f"{name} ({symbol}) — a {('Store of Value' if bucket == 'B' else 'Cash-Flow Protocol')} "
               f"ranked #{rank} by market cap."    # generic fallback
        )
        coin_profile = {
            "symbol":            symbol,
            "name":              name,
            "coin_id":           coin_id,
            "bucket":            bucket,
            "bucket_label":      "Store of Value" if bucket == "B" else "Cash-Flow Protocol",
            "description":       _description,
            "image_url":         _image,
            "market_cap_rank":   rank,
            "market_cap_usd":    mc,
            "fully_diluted_valuation": round(fdv, 2) if fdv else None,
            "current_price":     price,
            "ath":               _ath,
            "ath_date":          _ath_date,
            "ath_change_pct":    round(_ath_change_pct, 1) if _ath_change_pct else None,
            "atl":               _atl,
            "atl_date":          _atl_date,
            "price_change_7d_pct":  chg_7d,
            "price_change_30d_pct": chg_30d,
            "circulating_supply": _safe(coin.get("circulating_supply")),
            "total_supply":       _safe(coin.get("total_supply")),
            "max_supply":         _safe(coin.get("max_supply")),
            "total_volume_24h":   _safe(coin.get("total_volume")),
            "defillama_slug":     DEFILLAMA_SLUGS.get(symbol),
            "has_curated_description": symbol in COIN_DESCRIPTIONS,
        }

        return symbol, {
            "coin_profile":      coin_profile,
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
            "market_cycle":      market_cycle,
            "chart_zones":       chart_zones,
            "network_economics": network_econ,
            "valuation_chart":   valuation_chart,
            "forecast_meta":     forecast_meta,
            "intrinsic_value":   forecast_meta.get("base_iv") if forecast_meta else None,
            "iv_method":         (
                ", ".join(methods_used) if methods_used
                else "Market price (insufficient data)"
            ),
            "iv_methods_count":  len(methods_used),
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
            # Top-level price history — AI Studio reads this for the historical
            # price line in the Speculative Fair Value Forecast chart.
            # Shape: { "2016": 8.17, "2017": 756.01, ..., "2026": 2081.65 }
            # Same structure as stocks '10_Year_History'.
            "price_history": price_history,
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
    historical_tvl     = {}   # { year_str: tvl_usd }
    historical_fees    = {}   # { year_str: fees_usd (annual total) }
    historical_revenue = {}   # { year_str: revenue_usd (annual total) }
    years_of_history   = 0

    # ── Fetch TVL history from /protocol/{slug} ───────────────────────────────
    # DeFiLlama /protocol/{slug} returns TVL history in the "tvl" array.
    # Each entry: { "date": unix_timestamp, "totalLiquidityUSD": value }
    if slug:
        try:
            resp = requests.get(
                f"https://api.llama.fi/protocol/{slug}",
                timeout=15
            )
            if resp.status_code == 200:
                pdata = resp.json()
                # TVL history — keep one value per year (year-end / latest entry wins)
                for entry in pdata.get("tvl", []):
                    ts  = entry.get("date")
                    val = entry.get("totalLiquidityUSD")
                    if ts and val:
                        yr = str(datetime.fromtimestamp(ts).year)
                        historical_tvl[yr] = round(float(val), 2)
        except Exception:
            pass

    # ── Fetch fee history from /summary/fees/{slug} ───────────────────────────
    # The correct DeFiLlama endpoint for historical fee time-series.
    # Returns: { totalDataChart: [[timestamp, daily_value], ...], ... }
    # NOTE: /protocol/{slug} does NOT contain fee history — that was a bug.
    if slug:
        for data_type in ["dailyFees", "dailyRevenue"]:
            try:
                resp_f = requests.get(
                    f"https://api.llama.fi/summary/fees/{slug}?dataType={data_type}",
                    timeout=15
                )
                if resp_f.status_code == 200:
                    fdata = resp_f.json()
                    chart = fdata.get("totalDataChart", [])
                    # Accumulate daily values into yearly totals
                    yearly = {}
                    for entry in chart:
                        if isinstance(entry, (list, tuple)) and len(entry) == 2:
                            ts, val = entry
                        elif isinstance(entry, dict):
                            ts  = entry.get("date") or entry.get("timestamp")
                            val = entry.get("value") or entry.get(data_type)
                        else:
                            continue
                        if ts and val:
                            try:
                                yr = str(datetime.fromtimestamp(int(ts)).year)
                                yearly[yr] = yearly.get(yr, 0) + float(val)
                            except Exception:
                                pass
                    if yearly:
                        if data_type == "dailyFees":
                            historical_fees = yearly
                        else:
                            historical_revenue = yearly
            except Exception:
                pass

    # ── Try chain-level TVL endpoint for L1 blockchains ──────────────────────
    # DeFiLlama stores chain TVL separately from protocol TVL.
    # e.g. Solana chain TVL is at /v2/historicalChainTvl/Solana
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

    # ── Chart 1: Protocol Revenue & Fees ─────────────────────────────────────
    # Priority: historical_fees from /summary/fees/{slug} → current summary.
    # Revenue uses historical_revenue when available, else 15% fee estimate.
    # For tokens without DeFiLlama coverage: show current-year only from summary.
    rev_fees = {}

    # Historical years from DeFiLlama fee time-series
    all_hist_years = sorted(set(historical_fees) | set(historical_revenue))
    for yr in all_hist_years:
        fee_val = historical_fees.get(yr)
        rev_val = historical_revenue.get(yr) or (fee_val * 0.15 if fee_val else None)
        rev_fees[yr] = {
            "fees_usd":     round(fee_val / 1e6, 2) if fee_val else None,
            "revenue_usd":  round(rev_val  / 1e6, 2) if rev_val  else None,
            "is_estimated": False,
        }

    # TVL years with no fee data — placeholder so Chart 2 aligns timeline
    for yr in historical_tvl:
        if yr not in rev_fees:
            rev_fees[yr] = {
                "fees_usd":     None,
                "revenue_usd":  None,
                "is_estimated": True,
            }

    # Current year — always populated when any fee data exists
    if annual_fees and annual_fees > 0:
        rev_fees[str(current_year)] = {
            "fees_usd":     round(annual_fees / 1e6, 2),
            "revenue_usd":  round(annual_rev  / 1e6, 2) if annual_rev else
                            round(annual_fees * 0.15 / 1e6, 2),
            "holders_revenue_usd": round(holders_rev / 1e6, 2) if holders_rev else None,
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

    # ── Chart 4: Capital Efficiency ───────────────────────────────────────────
    # Fee/TVL ratio per year = ROIC analog (how efficiently capital generates fees)
    # Real Yield = holders_revenue / TVL = ROE analog
    # Even when fees are null, TVL bars are useful context — always show them.
    cap_eff = {}
    for yr in sorted(set(historical_tvl) | set(historical_fees)):
        tvl_yr  = historical_tvl.get(yr)
        fees_yr = historical_fees.get(yr)
        fee_tvl_ratio = (
            round(fees_yr / tvl_yr * 100, 4)
            if fees_yr and tvl_yr and tvl_yr > 0 else None
        )
        cap_eff[yr] = {
            "fee_tvl_ratio_pct": fee_tvl_ratio,
            "real_yield_pct":    None,   # per-year holders rev not available historically
            "tvl_usd_m":         round(tvl_yr  / 1e6, 2) if tvl_yr  else None,
            "fees_usd_m":        round(fees_yr / 1e6, 2) if fees_yr else None,
        }

    # Current year — populate even when only TVL is known (shows bar without ratio line)
    curr_tvl  = tvl if tvl else None
    curr_fees = annual_fees if annual_fees and annual_fees > 0 else None
    if curr_tvl or curr_fees:
        fee_tvl_curr = (
            round(curr_fees / curr_tvl * 100, 4)
            if curr_fees and curr_tvl and curr_tvl > 0 else None
        )
        real_yield = (
            round(holders_rev / curr_tvl * 100, 4)
            if holders_rev and curr_tvl and curr_tvl > 0 else None
        )
        cap_eff[str(current_year)] = {
            "fee_tvl_ratio_pct": fee_tvl_curr,
            "real_yield_pct":    real_yield,
            "tvl_usd_m":         round(curr_tvl  / 1e6, 2) if curr_tvl  else None,
            "fees_usd_m":        round(curr_fees  / 1e6, 2) if curr_fees else None,
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

# Cache for Claude API moat scores — avoids re-scoring on every nightly run.
# Moat scores change slowly (quarterly updates sufficient).
# Format: { symbol: { "score": int, "rationale": str, "scored_at": date } }
MOAT_SCORE_CACHE = {}

def score_protocol_moat_via_claude(symbol, name, bucket, rank, mc,
                                    circ_pct, quality_signals, hints_used):
    """
    Use Claude Haiku to score a token's protocol moat (0-10).

    This replaces the AI Studio runtime moat calculation, moving it into
    the pipeline so the app receives a pre-computed, consistent score.

    Moat dimensions assessed by Claude:
      1. Competitive defensibility — can the protocol be forked easily?
         (UNI: low — forked many times. ETH: very high — 15Y security/dev)
      2. Network effects — does value increase with more users?
         (ETH: extreme. UNI: moderate. ARB: low — users go where fees are lowest)
      3. Token utility — does the token capture real economic value?
         (GMX: high — direct fee share. UNI: low — governance only, no fees)
      4. Ecosystem lock-in — switching costs for users/developers?
         (ETH: very high — entire DeFi stack built on it)
      5. Regulatory moat — does decentralisation provide legal resilience?

    Score interpretation:
      9-10: Exceptional moat (BTC, ETH only)
      7-8:  Strong moat (SOL, MKR, GMX, PENDLE)
      5-6:  Moderate moat (AAVE, UNI, LINK)
      3-4:  Weak moat (ARB, OP, most governance tokens)
      0-2:  No moat (meme coins, pure governance tokens)

    Returns (score, rationale_str) or None if API unavailable.
    """
    global MOAT_SCORE_CACHE

    if not ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY.strip():
        return None

    # Use cache if scored recently (within 30 days)
    if symbol in MOAT_SCORE_CACHE:
        return MOAT_SCORE_CACHE[symbol]

    bucket_label = "Cash-Flow Protocol" if bucket == "A" else "Store of Value"
    signals_summary = {k: v.get("score") for k, v in quality_signals.items()}

    prompt = f"""You are a senior crypto analyst specialising in protocol fundamentals and tokenomics.

Score the PROTOCOL MOAT of {name} ({symbol}) on a scale of 0-10.

Context:
- Classification: {bucket_label}
- Market cap rank: {rank}
- Market cap: ${mc/1e9:.1f}B
- Circulating supply: {circ_pct}%
- Quality signals (pipeline scores): {signals_summary}
- Curated hints used: {hints_used}

Assess moat across these 5 dimensions:
1. Competitive defensibility (can it be forked? has it been? does it matter?)
2. Network effects (does user growth increase value for all participants?)
3. Token utility / value capture (do holders receive real economic value now?)
4. Ecosystem lock-in (switching costs for users, developers, and capital?)
5. Regulatory resilience (decentralisation as a moat vs regulatory risk)

SCORING GUIDANCE:
- ETH = 9 (settlement layer, 15Y security, all DeFi depends on it)
- BTC = 10 (hardest money, 15Y security, unmatched hash rate)
- SOL = 7 (strong L1 but younger, faster/cheaper but ETH dependency risk)
- UNI = 5 (forked many times, fee switch off, LPs not token holders capture value)
- ARB/OP = 4 (L2 governance tokens, users follow cheapest fees, no lock-in)
- Meme coins = 1-2 (no utility, no moat)

Respond ONLY with valid JSON, no other text:
{{"score": <int 0-10>, "rationale": "<2-3 sentences explaining the score, specific to {symbol}>"}}"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if resp.status_code == 200:
            raw = resp.json()["content"][0]["text"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            score = max(0, min(10, int(parsed.get("score", 5))))
            rationale = parsed.get("rationale", "")
            result = {
                "score":      score,
                "rationale":  rationale,
                "scored_at":  datetime.now().strftime("%Y-%m-%d"),
                "source":     "claude_api",
            }
            MOAT_SCORE_CACHE[symbol] = result
            return result
    except Exception as e:
        print(f"  [MOAT] {symbol}: API failed ({str(e)[:50]})")

    return None



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


# Max bytes per sub-partition file before starting a new chunk.
# ~1.5MB keeps files well under GitHub's 2MB display limit.
CRYPTO_MAX_PARTITION_BYTES = 1_500_000

def save_partitions(results):
    """
    Saves coin data into size-capped sub-partition files.

    Files are named  data_crypto/crypto_{LETTER}{N}.json
    e.g.  crypto_A1.json, crypto_A2.json, crypto_B1.json ...

    A ticker index  data_crypto/ticker_map.json  maps every symbol to
    its filename stem so the app fetches exactly the file it needs.
      { "BTC": "crypto_B1", "ETH": "crypto_E1", ... }
    """
    os.makedirs("data_crypto", exist_ok=True)

    # Group symbols by first letter
    by_letter = {}
    for sym, data in results.items():
        letter = sym[0].upper() if sym[0].isalpha() else "0"
        by_letter.setdefault(letter, {})[sym] = data

    ticker_map = {}
    files_written = 0

    for letter in sorted(by_letter.keys()):
        syms_in_letter = by_letter[letter]
        chunk = {}
        chunk_num = 1

        for sym, data in syms_in_letter.items():
            chunk[sym] = _sanitize(data)
            approx_size = len(json.dumps(chunk))
            if approx_size >= CRYPTO_MAX_PARTITION_BYTES:
                # Flush without the symbol that tipped it
                chunk.pop(sym)
                if chunk:
                    stem = f"crypto_{letter}{chunk_num}"
                    with open(f"data_crypto/{stem}.json", "w") as f:
                        json.dump(chunk, f, indent=2)
                    for s in chunk:
                        ticker_map[s] = stem
                    files_written += 1
                    chunk_num += 1
                chunk = {sym: _sanitize(data)}

        # Flush remaining
        if chunk:
            stem = f"crypto_{letter}{chunk_num}"
            with open(f"data_crypto/{stem}.json", "w") as f:
                json.dump(chunk, f, indent=2)
            for s in chunk:
                ticker_map[s] = stem
            files_written += 1

    # Save ticker → file mapping
    with open("data_crypto/ticker_map.json", "w") as f:
        json.dump(ticker_map, f, indent=2)

    print(f"✅ Saved {len(results)} coins across {files_written} files")
    print(f"✅ Saved data_crypto/ticker_map.json ({len(ticker_map)} entries)")


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
