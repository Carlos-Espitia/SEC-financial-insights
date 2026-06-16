import json
import logging
from pathlib import Path
from edgar import Company, set_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FORM_TYPES = ["10-K", "10-Q"]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sec-filings"
EMAIL = "CustomerInsightsProject carlos3212345@gmail.com"


def filing_path(ticker: str, form_type: str, period: str) -> Path:
    """data/sec-filings/MSFT/10-K/2022-06-30.json"""
    return OUTPUT_DIR / ticker / form_type / f"{period}.json"


def fetch_company(ticker: str, start_year: int) -> None:
    """
    Download and save all 10-K and 10-Q filings for a ticker from start_year onward.
    """
    set_identity(EMAIL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for form_type in FORM_TYPES:
        logger.info(f"Fetching {form_type} for {ticker} (from {start_year})...")
        try:
            _fetch_filings(ticker, form_type, start_year)
        except Exception as e:
            logger.error(f"  Failed {ticker} {form_type}: {e}")


def _fetch_filings(ticker: str, form_type: str, start_year: int) -> None:
    company = Company(ticker)
    # edgartools returns filings newest-first; cap generously and break once past start_year.
    max_historical = 50 if form_type == "10-K" else 200
    filings = company.get_filings(form=form_type).head(max_historical)

    for filing in filings:
        try:
            obj = filing.obj()
            period = str(obj.period_of_report)

            if int(period[:4]) < start_year:
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
