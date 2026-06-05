import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from src.rag import RAGPipeline
from src.metrics import load_companies, register_company, build_financials, load_financials
from src.ingest import fetch_company
from src.indexer import build_index

SEC_FILINGS_DIR = Path(__file__).parent / "data" / "sec-filings"

COLOR_PALETTE = [
    "#00A4EF", "#4285F4", "#0082FB", "#EA4335",
    "#34A853", "#FBBC04", "#FF6D00", "#9C27B0",
    "#00BCD4", "#795548",
]


def get_indexed_periods(ticker: str, form_type: str = "10-K") -> set[str]:
    folder = SEC_FILINGS_DIR / ticker / form_type
    if not folder.exists():
        return set()
    return {p.stem for p in folder.glob("*.json")}


def get_colors(tickers: list[str]) -> dict[str, str]:
    return {t: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, t in enumerate(sorted(tickers))}


def get_coverage() -> dict[str, dict[str, list[str]]]:
    """Returns {ticker: {"10-K": [sorted periods], "10-Q": [sorted periods]}} from disk."""
    coverage: dict = {}
    if not SEC_FILINGS_DIR.exists():
        return coverage
    for ticker_dir in sorted(SEC_FILINGS_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        coverage[ticker] = {}
        for form_type in ("10-K", "10-Q"):
            form_dir = ticker_dir / form_type
            periods = sorted(f.stem for f in form_dir.glob("*.json")) if form_dir.exists() else []
            coverage[ticker][form_type] = periods
    return coverage


def get_filing_year_range() -> str:
    """Compute min/max fiscal year from actual filing filenames on disk."""
    if not SEC_FILINGS_DIR.exists():
        return ""
    years = []
    for f in SEC_FILINGS_DIR.glob("**/*.json"):
        try:
            year = int(f.stem[:4])
            if 2000 <= year <= 2030:
                years.append(year)
        except (ValueError, IndexError):
            pass
    if not years:
        return ""
    return f"{min(years)} – {max(years)}"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEC Financial Insights",
    page_icon="📊",
    layout="wide",
)

METRIC_OPTIONS = {
    "Revenue":             "revenue",
    "Gross Profit":        "gross_profit",
    "Operating Income":    "operating_income",
    "Net Income":          "net_income",
    "Gross Margin":        "gross_margin",
    "Operating Margin":    "operating_margin",
    "Net Margin":          "net_margin",
    "Free Cash Flow":      "free_cash_flow",
    "R&D Expense":         "rd_expense",
    "R&D % of Revenue":    "rd_pct_revenue",
    "Revenue Growth YoY":  "revenue_growth_yoy",
}

DOLLAR_METRICS  = {"revenue", "gross_profit", "operating_income", "net_income",
                   "free_cash_flow", "rd_expense", "long_term_debt", "capex"}
PERCENT_METRICS = {"gross_margin", "operating_margin", "net_margin",
                   "rd_pct_revenue", "revenue_growth_yoy"}


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_data(show_spinner="Loading financial data...")
def get_financials() -> pd.DataFrame:
    return load_financials()


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_value(val, col: str) -> str:
    if pd.isna(val):
        return "N/A"
    if col in DOLLAR_METRICS:
        return f"${val / 1e9:.1f}B"
    if col in PERCENT_METRICS:
        return f"{val * 100:.1f}%"
    return f"{val:.2f}"


def make_line_chart(
    df: pd.DataFrame, col: str, title: str,
    companies: dict[str, str], colors: dict[str, str],
) -> go.Figure:
    fig = go.Figure()
    for ticker, name in companies.items():
        sub = df[df["ticker"] == ticker].sort_values("period")
        if sub.empty:
            continue
        y = sub[col] * 100 if col in PERCENT_METRICS else sub[col] / 1e9
        fig.add_trace(go.Scatter(
            x=sub["period"], y=y, mode="lines+markers",
            name=name, line=dict(color=colors[ticker], width=2),
        ))
    y_label = "%" if col in PERCENT_METRICS else "USD (billions)"
    fig.update_layout(
        title=title, xaxis_title="Fiscal Year End",
        yaxis_title=y_label, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=420,
    )
    return fig


# ── Load companies (dynamic) ──────────────────────────────────────────────────
COMPANIES = load_companies()  # {ticker: {name, cik, revenue_concept}}
COMPANY_NAMES = {t: info["name"] for t, info in COMPANIES.items()}
COLORS = get_colors(list(COMPANIES.keys()))

ticker_list_str = " · ".join(sorted(COMPANIES.keys())) if COMPANIES else "No companies yet"
year_range = get_filing_year_range()
filing_count = len(list(SEC_FILINGS_DIR.glob("**/*.json"))) if SEC_FILINGS_DIR.exists() else 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SEC Filing Analyzer")
    st.caption(f"**{ticker_list_str}**")
    if year_range:
        st.caption(f"{filing_count} filings indexed  ·  {year_range}")
    st.divider()

    company_filter = st.selectbox(
        "Filter Q&A by company",
        ["All companies"] + [f"{t} – {n}" for t, n in COMPANY_NAMES.items()],
    )
    selected_ticker = None if company_filter == "All companies" \
        else company_filter.split(" – ")[0]

    st.divider()

    # ── Coverage panel ────────────────────────────────────────────────────
    with st.expander("📁 Indexed coverage"):
        coverage = get_coverage()
        if not coverage:
            st.caption("No filings indexed yet.")
        else:
            for tkr, forms in coverage.items():
                tenk = forms.get("10-K", [])
                tenq = forms.get("10-Q", [])
                tenk_str = f"{tenk[0][:4]}–{tenk[-1][:4]} ({len(tenk)})" if tenk else "—"
                tenq_str = f"{tenq[0][:4]}–{tenq[-1][:4]} ({len(tenq)})" if tenq else "—"
                st.markdown(
                    f"**{tkr}** &nbsp; 10-K: `{tenk_str}` &nbsp; 10-Q: `{tenq_str}`"
                )

    # ── Add / fetch more filings ──────────────────────────────────────────
    with st.expander("➕ Add / fetch filings"):
        new_ticker = st.text_input(
            "Ticker symbol", placeholder="e.g. NVDA", key="new_ticker_input"
        ).upper().strip()

        from datetime import datetime as _dt
        current_year = _dt.now().year
        start_year = st.number_input(
            "Fetch filings from year",
            min_value=1993,
            max_value=current_year,
            value=current_year - 4,
            step=1,
            help="SEC EDGAR electronic filings begin around 1993.",
        )

        is_existing = new_ticker in COMPANIES
        btn_label = "Fetch More History" if is_existing else "Add Company"
        if new_ticker and is_existing:
            coverage_now = get_coverage().get(new_ticker, {})
            tenk_now = coverage_now.get("10-K", [])
            if tenk_now:
                st.caption(f"{new_ticker} already indexed: 10-K {tenk_now[0][:4]}–{tenk_now[-1][:4]}")

        if st.button(btn_label, disabled=not new_ticker):
            with st.status(f"Processing {new_ticker}...", expanded=True) as status:
                try:
                    if not is_existing:
                        st.write(f"Looking up {new_ticker} in SEC database...")
                        entry = register_company(new_ticker)
                        st.write(f"Found: **{entry['name']}**")
                    else:
                        entry = COMPANIES[new_ticker]
                        st.write(f"Fetching more history for **{entry['name']}**...")

                    st.write(f"Downloading filings from {start_year} onward...")
                    fetch_company(new_ticker, start_year=start_year)

                    st.write("Indexing new filings into ChromaDB...")
                    build_index()

                    st.write("Rebuilding financial metrics from XBRL...")
                    build_financials()

                    status.update(
                        label=f"✅ Done — {entry['name']} updated!",
                        state="complete",
                    )
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                except ValueError as e:
                    status.update(label="❌ Failed", state="error")
                    st.error(str(e))
                except Exception as e:
                    status.update(label="❌ Failed", state="error")
                    st.error(f"Unexpected error: {e}")

    st.divider()
    st.caption("Data sources: SEC EDGAR (edgartools) · XBRL API")
    st.caption("LLM: Claude Haiku · Embeddings: all-MiniLM-L6-v2")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_charts, tab_analysis = st.tabs([
    "💬 Q&A Chat", "📈 Financial Charts", "🔍 Deep Analysis"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A Chat
# ══════════════════════════════════════════════════════════════════════════════
STARTER_QUESTIONS = [
    "What drove Meta's operating margin improvement from 2022 to 2023?",
    "How did Microsoft's cloud revenue grow in fiscal year 2024?",
    "Which company spent the highest percentage of revenue on R&D in 2023?",
    "How did Google Cloud's profitability change from 2022 to 2023?",
    "What was Meta's free cash flow in 2022 and how did it change in 2023?",
    "What were the main risks Alphabet highlighted in their 2023 annual report?",
]

with tab_chat:
    st.subheader("Ask anything about the filings")
    indexed_tickers = sorted(
        d.name for d in SEC_FILINGS_DIR.iterdir() if d.is_dir()
    ) if SEC_FILINGS_DIR.exists() else []
    range_str = f" ({year_range})" if year_range else ""
    st.caption(
        f"Answers are grounded in 10-K and 10-Q filings for "
        f"{', '.join(indexed_tickers) or 'no companies yet'}{range_str}. "
        f"Every claim is cited to its source document."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources & metadata"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **{s['company']}** {s['form_type']} "
                            f"`{s['period']}` · {s['section'].upper()}"
                        )
                    st.caption(f"Rewritten query: *{msg.get('rewritten_query', '–')}*")
                    label = "✅ Grounded" if msg.get("is_grounded", True) else "⚠️ Possibly not fully grounded"
                    st.caption(label)

    # Starter questions — shown only on empty chat
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        q_cols = st.columns(2)
        for i, q in enumerate(STARTER_QUESTIONS):
            with q_cols[i % 2]:
                if st.button(q, key=f"starter_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()
        st.divider()

    # Chat input — always rendered
    typed_input = st.chat_input("Ask about any SEC filing...")

    # Pending question (from starter button) takes priority over typed input
    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None
    elif typed_input:
        prompt = typed_input
    else:
        prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching filings…"):
                pipeline = get_pipeline()
                result = pipeline.query(prompt, ticker=selected_ticker)

            st.markdown(result.answer)
            with st.expander("Sources & metadata"):
                for s in result.sources:
                    st.markdown(
                        f"- **{s['company']}** {s['form_type']} "
                        f"`{s['period']}` · {s['section'].upper()}"
                    )
                st.caption(f"Rewritten query: *{result.rewritten_query}*")
                label = "✅ Grounded" if result.is_grounded else "⚠️ Possibly not fully grounded"
                st.caption(label)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "rewritten_query": result.rewritten_query,
            "is_grounded": result.is_grounded,
        })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Financial Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    df = get_financials()

    if df.empty or not COMPANIES:
        st.info("No financial data yet. Add a company using the sidebar.")
        st.stop()

    companies_in_data = sorted(df["ticker"].unique())
    ticker_names_str = " · ".join(
        COMPANY_NAMES.get(t, t) for t in companies_in_data
    )
    st.subheader(f"Financial comparison: {ticker_names_str}")
    st.caption("All figures from SEC EDGAR XBRL API. Computed metrics show formula inputs.")

    # ── Summary metric cards (latest year per company) ──
    st.markdown("**Latest annual figures**")
    cols = st.columns(max(len(companies_in_data), 1))
    for i, ticker in enumerate(companies_in_data):
        name = COMPANY_NAMES.get(ticker, ticker)
        ticker_df = df[df["ticker"] == ticker].sort_values("period")
        if ticker_df.empty:
            continue
        latest = ticker_df.iloc[-1]
        with cols[i]:
            st.metric(
                name,
                fmt_value(latest["revenue"], "revenue"),
                delta=fmt_value(latest["revenue_growth_yoy"], "revenue_growth_yoy"),
            )
            st.caption(f"Op. margin: {fmt_value(latest['operating_margin'], 'operating_margin')}")
            st.caption(f"Net margin: {fmt_value(latest['net_margin'], 'net_margin')}")
            st.caption(f"Period: {latest['period']}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        metric_label = st.selectbox("Select metric", list(METRIC_OPTIONS.keys()))
    metric_col = METRIC_OPTIONS[metric_label]

    chart_companies = {t: COMPANY_NAMES.get(t, t) for t in companies_in_data}
    chart_colors = {t: COLORS.get(t, COLOR_PALETTE[i % len(COLOR_PALETTE)])
                    for i, t in enumerate(companies_in_data)}
    fig = make_line_chart(df, metric_col, f"{metric_label} over time", chart_companies, chart_colors)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw data"):
        rd_col1, rd_col2 = st.columns(2)
        with rd_col1:
            rd_ticker = st.selectbox(
                "Company", ["All"] + companies_in_data, key="rd_ticker",
                format_func=lambda t: COMPANY_NAMES.get(t, t) if t != "All" else "All companies",
            )
        with rd_col2:
            show_xbrl = st.checkbox("Show all XBRL concepts", value=False, key="rd_xbrl")

        rd_df = df if rd_ticker == "All" else df[df["ticker"] == rd_ticker]

        if show_xbrl:
            xbrl_cols = [c for c in rd_df.columns if c.startswith("xbrl__")]
            # Only keep columns that have at least one non-null value for this selection
            xbrl_cols = [c for c in xbrl_cols if rd_df[c].notna().any()]
            display_cols = ["ticker", "company", "period"] + xbrl_cols
            st.caption(f"{len(xbrl_cols)} XBRL concepts with data. Column names are raw us-gaap tags.")
        else:
            display_cols = ["ticker", "company", "period", "revenue", "operating_margin",
                            "net_margin", "revenue_growth_yoy", "free_cash_flow"]

        st.dataframe(
            rd_df[display_cols].sort_values(["ticker", "period"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Deep Analysis (Quant → Narrative)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.subheader("Quant → Narrative linkage")
    st.caption(
        "Pick a company and metric. The dashboard identifies the biggest year-over-year "
        "change and retrieves what management said about it from the actual filing."
    )

    df = get_financials()

    if df.empty or not COMPANIES:
        st.info("No financial data yet. Add a company using the sidebar.")
        st.stop()

    available_tickers = sorted(df["ticker"].unique())

    col1, col2 = st.columns(2)
    with col1:
        an_ticker = st.selectbox(
            "Company", available_tickers,
            format_func=lambda t: COMPANY_NAMES.get(t, t),
        )
    with col2:
        an_metric_label = st.selectbox("Metric", [
            "Operating Margin", "Net Margin", "Revenue", "Gross Margin",
            "Free Cash Flow", "R&D % of Revenue",
        ], key="an_metric")
    an_metric_col = METRIC_OPTIONS[an_metric_label]

    indexed_periods = get_indexed_periods(an_ticker, "10-K")
    company_df = (
        df[(df["ticker"] == an_ticker) & (df["period"].isin(indexed_periods))]
        .sort_values("period")
        .copy()
    )
    if company_df.empty:
        st.warning("No indexed 10-K text found for this company.")
        st.stop()
    st.caption(f"Analysis limited to years with indexed 10-K text: {', '.join(sorted(indexed_periods))}")

    an_color = COLORS.get(an_ticker, COLOR_PALETTE[0])
    an_name = COMPANY_NAMES.get(an_ticker, an_ticker)

    y_vals = (company_df[an_metric_col] * 100
              if an_metric_col in PERCENT_METRICS
              else company_df[an_metric_col] / 1e9)
    y_label = "%" if an_metric_col in PERCENT_METRICS else "USD (billions)"

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=company_df["period"], y=y_vals, mode="lines+markers",
        line=dict(color=an_color, width=2),
        name=an_name,
    ))

    company_df["_change"] = company_df[an_metric_col].diff().abs()
    if not company_df["_change"].isna().all():
        peak_idx = company_df["_change"].idxmax()
        peak_row = company_df.loc[peak_idx]
        peak_y = (peak_row[an_metric_col] * 100
                  if an_metric_col in PERCENT_METRICS
                  else peak_row[an_metric_col] / 1e9)
        fig2.add_trace(go.Scatter(
            x=[peak_row["period"]], y=[peak_y],
            mode="markers", marker=dict(color="red", size=12),
            name="Biggest change",
        ))

    fig2.update_layout(
        title=f"{an_name} — {an_metric_label}",
        xaxis_title="Fiscal Year End", yaxis_title=y_label,
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if not company_df["_change"].isna().all():
        peak_period = peak_row["period"]
        prev_rows = company_df.loc[company_df["period"] < peak_period, an_metric_col]
        prev_val = fmt_value(prev_rows.iloc[-1] if not prev_rows.empty else None, an_metric_col)
        curr_val = fmt_value(peak_row[an_metric_col], an_metric_col)

        st.info(
            f"Biggest change detected: **{an_metric_label}** moved from "
            f"**{prev_val}** to **{curr_val}** in **{peak_period}**"
        )

        if st.button("Explain this change using the filing"):
            query = (
                f"In the Management Discussion and Analysis for the fiscal year ending "
                f"{peak_period}, what did {an_name} explain about "
                f"{an_metric_label} performance? What drove the results and how did "
                f"revenue, costs, and operating expenses contribute?"
            )
            with st.spinner("Searching filing narrative…"):
                pipeline = get_pipeline()
                result = pipeline.query(
                    query,
                    ticker=an_ticker,
                    sections=["mda"],
                    period=peak_period,
                    k=10,
                )

            st.markdown("**Management explanation (from filing):**")
            st.markdown(result.answer)
            with st.expander("Sources"):
                for s in result.sources:
                    st.markdown(
                        f"- {s['form_type']} `{s['period']}` · {s['section'].upper()}"
                    )
                label = "✅ Grounded" if result.is_grounded else "⚠️ Possibly not fully grounded"
                st.caption(label)
