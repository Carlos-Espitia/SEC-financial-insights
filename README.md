# SEC Financial Insights

A RAG-powered financial analysis dashboard that answers questions about SEC filings, visualises XBRL financial metrics, and links quantitative changes to management's own explanations from the filing text.

## What it does

- **Q&A Chat** — Ask anything about indexed 10-K and 10-Q filings. The pipeline auto-detects which company, filing period, and section to retrieve from, cites every claim to a source document, and verifies answers are grounded before returning them.
- **Financial Charts** — Interactive multi-company charts for any XBRL metric (revenue, margins, FCF, R&D spend, etc.) across the full filing history.
- **Deep Analysis** — Picks the biggest year-over-year change in a selected metric and retrieves what management actually said about it in the corresponding MD&A.
- **Any company** — Add any SEC-listed company from the sidebar. The system looks up the CIK, downloads filings, indexes them, and rebuilds metrics automatically.

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

The filing JSONs are included in the repo (`data/sec-filings/`). You only need to build the local vector index before launching.

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set up environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Build the vector index**

Embeds all included filings into a local ChromaDB store (~2 minutes):

```bash
python src/indexer.py
```

**4. Launch the app**

```bash
python -m streamlit run app.py
```

The financial metrics CSV is fetched from the SEC XBRL API automatically on first load (~30 seconds).

---

## Testing guide

The repo includes filings for **Microsoft, Alphabet, Meta, Apple, and NVIDIA** across multiple years. Below are concrete things to try in each tab.

### Q&A Chat

The system auto-detects which company and filing period to search — no filtering needed. Try these questions:

**Factual lookups**
- `What was Meta's total revenue in Q3 2023?`
- `How many employees did Alphabet have as of December 31, 2023?`
- `What was Microsoft's Azure growth rate in Q2 fiscal year 2024?`
- `What was Meta's free cash flow in 2022?`

**Analytical questions**
- `What drove Meta's operating margin improvement from 2022 to 2023?`
- `How did Google Cloud's profitability change from 2022 to 2023?`
- `What was Meta's Reality Labs operating loss in 2022?`

**Cross-company comparisons**
- `Which company — Microsoft, Alphabet, or Meta — spent the highest percentage of revenue on R&D in 2023?`
- `How did Alphabet's employee count change from end of 2022 to end of 2023?`

**Gap detection (should refuse with a helpful message)**
- `What was Meta's revenue in 2015?` — outside indexed range
- `How does Amazon Web Services revenue compare to Microsoft Azure?` — Amazon not in corpus
- `What is the monthly price of Microsoft Copilot?` — not in SEC filings

**Sidebar filter** — Use "Filter Q&A by company" to pre-narrow retrieval when you know the target company.

---

### Financial Charts

1. Open the **📈 Financial Charts** tab
2. Use the metric selector to switch between Revenue, Operating Margin, Free Cash Flow, R&D %, etc.
3. Hover over any data point to see the exact value and filing period
4. Open **View raw data** → select a company → check **Show all XBRL concepts** to see every financial metric that company has ever filed via XBRL

---

### Deep Analysis

1. Open the **🔍 Deep Analysis** tab
2. Select **NVIDIA** and **R&D % of Revenue** — the system finds FY2020 (ended Jan 26, 2020) as the biggest shift: revenue fell 7% while R&D kept growing, spiking the ratio from 20.3% to 25.9%
3. Click **Explain this change using the filing** to retrieve NVIDIA's own explanation from the MD&A
4. Try **Meta** + **Operating Margin** to see the "Year of Efficiency" story: 25% → 35% driven by headcount cuts and ad revenue recovery
5. Try **Microsoft** + **Free Cash Flow** for the cloud-driven FCF expansion

---

### Adding a new company

1. Open the sidebar → **➕ Add / fetch filings**
2. Enter any ticker (e.g. `TSLA`, `AMZN`, `NFLX`)
3. Set the start year (default: last 4 years)
4. Click **Add Company**

The app handles CIK lookup → filing download → indexing → metrics rebuild automatically. The new company appears in all dropdowns immediately after.

---

## Running the evaluation

```bash
python evaluation/run_eval.py
```

Runs 17 labeled questions through the full pipeline and saves results to `evaluation/results.json`. No explicit filters — the pipeline extracts all retrieval metadata from the question itself.

To print the score summary:

```bash
python evaluation/run_eval.py --score
```

Expected results: ~86% correct on answerable questions, 0% hallucination rate, 100% correct refusal on unanswerable questions.

---

## Project structure

```
src/
  ingest.py      — SEC EDGAR filing downloader (edgartools)
  indexer.py     — Chunking, embedding, ChromaDB indexing
  rag.py         — 5-step RAG pipeline (intent → gap check → rewrite → retrieve → generate + verify)
  metrics.py     — XBRL financial metrics + auto-discovery of all USD concepts

app.py           — Streamlit UI (Q&A, Charts, Deep Analysis tabs)

data/
  companies.json         — Registry of indexed companies (ticker, CIK, revenue concept)
  sec-filings/           — Raw filing JSON by ticker/form/period (included in repo)
  index/                 — ChromaDB vector store (gitignored — build with indexer.py)
  metrics/financials.csv — XBRL metrics (auto-built on first load)

evaluation/
  test_set_v2.json  — 17 labeled questions with verified expected answers
  run_eval.py       — Evaluation runner and scorer
  results.json      — Last run results with grades

WRITEUP.md  — Architecture, evaluation results, design decisions
```

## Data sources

- **Filing text** — SEC EDGAR via [edgartools](https://github.com/dgunning/edgartools) (no API key required)
- **Financial metrics** — SEC EDGAR XBRL API (`data.sec.gov/api/xbrl/`) (no API key required)
- **LLM** — Anthropic Claude Haiku (intent detection, query rewriting, answer generation, grounding verification)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key required)
