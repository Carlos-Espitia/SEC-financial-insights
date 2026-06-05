import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMPANIES_FILE = Path(__file__).parent.parent / "data" / "companies.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "metrics"
OUTPUT_FILE = OUTPUT_DIR / "financials.csv"

HEADERS = {"User-Agent": "CustomerInsightsProject carlos3212345@gmail.com"}

# Revenue GAAP concepts tried in priority order — covers most US public companies
REVENUE_CANDIDATES = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerNetOfAllowances",
]

# Standard GAAP concepts — same tag across companies
CONCEPTS = {
    "gross_profit":           "GrossProfit",
    "operating_income":       "OperatingIncomeLoss",
    "net_income":             "NetIncomeLoss",
    "total_assets":           "Assets",
    "stockholders_equity":    "StockholdersEquity",
    "long_term_debt":         "LongTermDebt",
    "rd_expense":             "ResearchAndDevelopmentExpense",
    "capex":                  "PaymentsToAcquirePropertyPlantAndEquipment",
    "operating_cash_flow":    "NetCashProvidedByUsedInOperatingActivities",
    "finance_lease_payments": "FinanceLeasePrincipalPayments",
}


def load_companies() -> dict:
    """Load company registry from data/companies.json."""
    if not COMPANIES_FILE.exists():
        return {}
    return json.loads(COMPANIES_FILE.read_text(encoding="utf-8"))


def save_companies(companies: dict) -> None:
    COMPANIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMPANIES_FILE.write_text(json.dumps(companies, indent=2), encoding="utf-8")


def lookup_cik(ticker: str) -> tuple[str, str]:
    """
    Looks up ticker in SEC's full company list.
    Returns (cik_formatted, company_name) or raises ValueError.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            cik = f"CIK{str(entry['cik_str']).zfill(10)}"
            return cik, entry["title"]
    raise ValueError(f"Ticker '{ticker}' not found in SEC company database")


def register_company(ticker: str) -> dict:
    """
    Registers a new company: looks up CIK, detects revenue GAAP concept,
    and saves to companies.json. Returns the company entry dict.
    Raises ValueError on failure.
    """
    ticker = ticker.upper().strip()
    companies = load_companies()
    if ticker in companies:
        return companies[ticker]

    logger.info(f"Looking up CIK for {ticker}...")
    cik, name = lookup_cik(ticker)

    logger.info(f"Fetching XBRL facts to detect revenue concept for {ticker}...")
    facts = _fetch_facts(cik)
    if not facts:
        raise ValueError(f"Could not fetch XBRL data for {ticker}")
    gaap = facts.get("facts", {}).get("us-gaap", {})

    revenue_concept = _detect_revenue_concept(gaap)
    if not revenue_concept:
        raise ValueError(
            f"No annual revenue data found for {ticker}. "
            f"The company may not file standard US GAAP financials."
        )

    entry = {"name": name, "cik": cik, "revenue_concept": revenue_concept}
    companies[ticker] = entry
    save_companies(companies)
    logger.info(f"Registered {ticker} ({name}): revenue_concept={revenue_concept}")
    return entry


def _detect_revenue_concept(gaap: dict) -> str | None:
    """Try common revenue GAAP concepts and return the first one with annual data."""
    for concept in REVENUE_CANDIDATES:
        if concept in gaap and _get_annual_map(gaap, concept):
            return concept
    return None


def _get_all_annual_concepts(gaap: dict, exclude: set[str]) -> dict[str, dict]:
    """
    Auto-discovers every USD-denominated us-gaap concept that has annual 10-K data.
    Skips concepts already covered by the curated CONCEPTS dict and the revenue concept
    to avoid redundant columns. Returns {concept_name: {period: value}}.
    """
    result = {}
    for concept, data in gaap.items():
        if concept in exclude:
            continue
        if "USD" not in data.get("units", {}):
            continue
        annual_map = _get_annual_map(gaap, concept)
        if annual_map:
            result[concept] = annual_map
    return result


def build_financials() -> pd.DataFrame:
    companies = load_companies()
    if not companies:
        logger.warning("No companies in registry. Add companies to data/companies.json first.")
        return pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for ticker, info in companies.items():
        logger.info(f"Fetching XBRL facts for {ticker}...")
        facts = _fetch_facts(info["cik"])
        if not facts:
            continue
        gaap = facts.get("facts", {}).get("us-gaap", {})
        rows = _extract_annual_rows(ticker, info["name"], gaap, info["revenue_concept"])
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = _compute_derived_metrics(df)
    df = df.sort_values(["ticker", "period"]).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    return df


def _fetch_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch {cik}: {e}")
        return {}


def _extract_annual_rows(ticker: str, company: str, gaap: dict, revenue_concept: str) -> list[dict]:
    revenue_map = _get_annual_map(gaap, revenue_concept)
    if not revenue_map:
        logger.warning(f"  {ticker}: no annual revenue data for concept '{revenue_concept}'")
        return []

    # Precompute curated maps once per company (not once per period)
    curated_maps = {col: _get_annual_map(gaap, concept) for col, concept in CONCEPTS.items()}

    # Auto-discover all other USD annual concepts, skipping already-curated ones
    exclude = set(CONCEPTS.values()) | {revenue_concept}
    discovered = _get_all_annual_concepts(gaap, exclude)
    logger.info(f"  {ticker}: {len(discovered)} additional XBRL concepts discovered")

    rows = []
    for period, revenue in revenue_map.items():
        row = {
            "ticker": ticker,
            "company": company,
            "period": period,
            "fiscal_year": period[:4],
            "revenue": revenue,
        }
        for col, concept_map in curated_maps.items():
            row[col] = concept_map.get(period)
        for concept, concept_map in discovered.items():
            if period in concept_map:
                row[f"xbrl__{concept}"] = concept_map[period]
        if row["revenue"] is None and row.get("net_income") is None:
            continue
        rows.append(row)

    return rows


def _get_annual_map(gaap: dict, concept: str) -> dict:
    """
    Returns {period_end_date: value} for all FY annual entries of a concept.
    When a period is filed multiple times (amendments), keeps the latest filing.
    """
    if concept not in gaap:
        return {}

    entries = gaap[concept].get("units", {}).get("USD", [])

    def _is_annual(e: dict) -> bool:
        if e.get("form") != "10-K":
            return False
        start, end = e.get("start"), e.get("end")
        if not start or not end:
            return e.get("fp") == "FY"
        days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
        return days >= 340

    annual = [e for e in entries if _is_annual(e)]

    latest: dict[str, dict] = {}
    for e in annual:
        end = e["end"]
        if end not in latest or e["filed"] > latest[end]["filed"]:
            latest[end] = e

    return {end: entry["val"] for end, entry in latest.items()}


def _compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["gross_margin"]     = df["gross_profit"]     / df["revenue"]
    df["operating_margin"] = df["operating_income"] / df["revenue"]
    df["net_margin"]       = df["net_income"]        / df["revenue"]
    df["rd_pct_revenue"]   = df["rd_expense"]        / df["revenue"]

    # FCF = operating cash flow - capex - finance lease principal payments
    # fillna(0) treats companies with no disclosed finance leases as $0
    df["free_cash_flow"] = (
        df["operating_cash_flow"]
        - df["capex"]
        - df["finance_lease_payments"].fillna(0)
    )

    df["debt_to_equity"] = df["long_term_debt"] / df["stockholders_equity"]

    df = df.sort_values(["ticker", "period"])
    df["revenue_growth_yoy"] = df.groupby("ticker")["revenue"].pct_change()

    return df


def load_financials() -> pd.DataFrame:
    """Load the pre-built financials CSV. Builds it if it doesn't exist."""
    if not OUTPUT_FILE.exists():
        logger.info("financials.csv not found — building from XBRL API...")
        return build_financials()
    return pd.read_csv(OUTPUT_FILE)


if __name__ == "__main__":
    df = build_financials()
    print(df[["ticker", "period", "revenue", "net_income", "operating_margin", "revenue_growth_yoy"]].to_string())
