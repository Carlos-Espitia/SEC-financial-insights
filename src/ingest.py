import json
import logging
from pathlib import Path
from edgar import Company, set_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FILING_CONFIG = {
    "10-K": 4,
    "10-Q": 12,
}

COMPANIES_FILE = Path(__file__).parent.parent / "data" / "companies.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sec-filings"
EMAIL = "CustomerInsightsProject carlos3212345@gmail.com"


def load_companies() -> dict:
    if not COMPANIES_FILE.exists():
        return {}
    return json.loads(COMPANIES_FILE.read_text(encoding="utf-8"))


def filing_path(ticker: str, form_type: str, period: str) -> Path:
    """data/sec-filings/MSFT/10-K/2022-06-30.json"""
    return OUTPUT_DIR / ticker / form_type / f"{period}.json"


def fetch_company(ticker: str, start_year: int | None = None, filing_config: dict | None = None) -> None:
    """
    Download and save filings for a single ticker.
    start_year: if set, fetch all filings from that year onward (overrides count limits).
    """
    if filing_config is None:
        filing_config = FILING_CONFIG
    set_identity(EMAIL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for form_type, limit in filing_config.items():
        logger.info(f"Fetching {form_type} for {ticker} (start_year={start_year or 'recent'})...")
        try:
            _fetch_filings(ticker, form_type, limit, start_year=start_year)
        except Exception as e:
            logger.error(f"  Failed {ticker} {form_type}: {e}")


def fetch_all(tickers: list[str] | None = None) -> None:
    """Fetch filings for all registered companies, or a specific list of tickers."""
    if tickers is None:
        companies = load_companies()
        tickers = list(companies.keys())

    if not tickers:
        logger.warning("No companies registered. Add companies to data/companies.json first.")
        return

    set_identity(EMAIL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        for form_type, limit in FILING_CONFIG.items():
            logger.info(f"Fetching {limit}x {form_type} for {ticker}...")
            try:
                _fetch_filings(ticker, form_type, limit)
            except Exception as e:
                logger.error(f"  Failed {ticker} {form_type}: {e}")

    logger.info(f"\nDone. Files saved to: {OUTPUT_DIR}")
    _print_summary(tickers)


def _fetch_filings(ticker: str, form_type: str, limit: int, start_year: int | None = None) -> None:
    company = Company(ticker)
    # When fetching by year range, use a generous cap so we don't miss older filings.
    # edgartools returns filings newest-first, so we break early once past start_year.
    if start_year is None:
        filings = company.get_filings(form=form_type).head(limit)
    else:
        max_historical = 50 if form_type == "10-K" else 200
        filings = company.get_filings(form=form_type).head(max_historical)

    for filing in filings:
        try:
            obj = filing.obj()
            period = str(obj.period_of_report)

            if start_year is not None and int(period[:4]) < start_year:
                break  # Filings are newest-first; everything past here is older

            out_path = filing_path(ticker, form_type, period)

            if out_path.exists():
                logger.info(f"  Skipping {ticker} {form_type} {period} (already fetched)")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            sections = _extract_sections(obj, form_type)
            record = {
                "ticker": ticker,
                "company": str(obj.company),
                "form_type": form_type,
                "period": period,
                "filed_date": str(obj.filing_date),
                "accession": filing.accession_no,
                "sections": sections,
            }
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"  Saved {ticker} {form_type} {period}")
        except Exception as e:
            logger.warning(f"  Could not parse {filing.accession_no}: {e}")


def _extract_sections(obj, form_type: str) -> dict:
    sections = {}

    if form_type == "10-K":
        for key, attr in [
            ("business", "business"),
            ("risk_factors", "risk_factors"),
            ("mda", "management_discussion"),
        ]:
            try:
                sections[key] = str(getattr(obj, attr) or "")
            except Exception:
                sections[key] = ""

    elif form_type == "10-Q":
        try:
            sections["mda"] = str(obj["Part I, Item 2"] or "")
        except Exception:
            sections["mda"] = ""

    for key, attr in [
        ("income_statement", "income_statement"),
        ("balance_sheet", "balance_sheet"),
        ("cash_flow_statement", "cash_flow_statement"),
    ]:
        try:
            sections[key] = str(getattr(obj, attr) or "")
        except Exception:
            sections[key] = ""

    return sections


def _print_summary(tickers: list[str]) -> None:
    files = list(OUTPUT_DIR.glob("**/*.json"))
    logger.info(f"\n--- Summary: {len(files)} total filings ---")
    for ticker in tickers:
        for form_type in FILING_CONFIG:
            folder = OUTPUT_DIR / ticker / form_type
            count = len(list(folder.glob("*.json"))) if folder.exists() else 0
            logger.info(f"  {ticker} {form_type}: {count} filings")


if __name__ == "__main__":
    fetch_all()
