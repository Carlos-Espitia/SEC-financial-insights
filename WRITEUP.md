# WRITEUP

## Architecture

The system has five components, each in a dedicated source file:

**Ingestion (`src/ingest.py`)** — Fetches 10-K and 10-Q filings for any SEC-listed company via `edgartools`. Each filing is parsed into clean text sections (Business, Risk Factors, MD&A, and financial statements) and saved as structured JSON under `data/sec-filings/{TICKER}/{FORM_TYPE}/{PERIOD}.json`. Companies are registered in `data/companies.json` (a lightweight registry storing ticker, CIK, and GAAP revenue concept). `fetch_company(ticker, start_year)` accepts an optional start year so users can pull filings going back to 1993 (the start of SEC EDGAR electronic filing). I initially used `sec-edgar-downloader` but switched to `edgartools` after discovering the raw SGML bundles it produced required complex parsing; edgartools extracts clean, section-level text directly.

**Indexing (`src/indexer.py`)** — Each JSON file is chunked using LangChain's `RecursiveCharacterTextSplitter` (~1,200 chars, 150-char overlap), embedded with `sentence-transformers/all-MiniLM-L6-v2` (local, free), and stored in a ChromaDB vector database at `data/index/`. Every chunk carries metadata: ticker, company, form type, period, filed date, and section name. This metadata drives both filtering and citations. The indexer is fully dynamic — it globs all JSON files in the filings directory and skips already-indexed documents, so adding a new company requires no code changes.

**RAG Pipeline (`src/rag.py`)** — A five-step agentic loop:

1. **Intent detection** — Claude Haiku extracts retrieval metadata (ticker, period, form\_type, sections) from the natural-language question using a dynamically-built prompt that reflects exactly what's indexed. The prompt is generated at startup by scanning `data/sec-filings/` — no hardcoded company list. Adding a new company re-teaches the intent detector at the next app restart.
2. **Gap detection** — Before retrieval, checks whether the detected ticker exists in the corpus and whether any year referenced in the question predates the earliest indexed filing. Returns an explicit, actionable refusal ("I don't have NVDA filings from 2010; the earliest indexed is 2002") rather than silently retrieving the wrong data.
3. **Query rewriting** — Rewrites the user question into a keyword-rich search query optimised for semantic retrieval over SEC filing text.
4. **Retrieval** — ChromaDB vector search with metadata filters for ticker, section, period, and form type. Multi-ticker queries run a separate retrieval per company and merge results by cosine distance, preventing one company's filings from dominating cross-company comparisons.
5. **Generation + verification** — Claude generates a cited answer from the retrieved chunks, then a separate grounding-verification call checks whether every factual claim is supported by the context. Ungrounded answers are flagged; unanswerable questions return a standardised refusal.

**Metrics (`src/metrics.py`)** — Pulls structured financial facts from the SEC EDGAR XBRL API (`/api/xbrl/companyfacts/{CIK}.json`). New companies are registered automatically: the system looks up the CIK from SEC's public company tickers list, fetches the XBRL fact blob, and auto-detects the correct revenue GAAP concept by trying six common tags in priority order. Annual figures are filtered by period duration (≥340 days) to exclude the quarterly breakdowns that Microsoft incorrectly tags as `fp=FY` in their 10-K XBRL submissions. Beyond the curated set of ~10 standard metrics, the system auto-discovers every USD-denominated us-gaap concept with annual 10-K data for each company, adding them as `xbrl__ConceptName` columns. This surfaces company-specific line items (segment revenue, restructuring charges, etc.) that the hardcoded list would miss. Free cash flow subtracts finance lease principal payments in addition to capex, matching Meta's disclosed FCF definition.

**UI (`app.py`)** — Streamlit app with three tabs:
- **Q&A Chat** — RAG chat interface with source citations and grounding status. Starter questions guide new users. The sidebar "Filter by company" pass-through lets users narrow retrieval when they know the target company.
- **Financial Charts** — Interactive line charts for any XBRL metric across all indexed companies, with metric cards showing the latest annual figures. A "View raw data" expander with per-company filtering and a toggle to show all auto-discovered XBRL concepts.
- **Deep Analysis (Quant → Narrative)** — Identifies the biggest year-over-year change in a selected metric and retrieves what management said about it in the corresponding MD&A. Works across the full indexed history (e.g., 25 years of NVDA filings).

The sidebar also shows a live coverage panel (which periods are indexed per company), a filing fetch panel with start-year control, and a one-click "Add company" flow that handles CIK lookup → filing download → indexing → metrics rebuild in sequence.

---

## Hallucination Rate and Evaluation

A labeled test set of 17 questions (`evaluation/test_set_v2.json`) was run through the RAG pipeline in three stages. All expected answers were verified directly against the raw JSON filing files before use.

### Stage 1 — Baseline (no retrieval filters)

| Category | Score |
|---|---|
| Answerable — correct | 3/14 (21%) |
| Answerable — partial | 1/14 (7%) |
| Answerable — wrong | 10/14 (71%) |
| Grounding rate (answerable) | 14/14 (100%) |
| Unanswerable — correctly refused | 3/3 (100%) |
| Hallucination rate | 0% |

### Stage 2 — Hardcoded per-question filters (eval gaming)

Explicit `ticker`, `period`, and `form_type` filters were manually specified per question in the eval runner — the kind of hints that are only available if you already know the answer.

| Category | Score |
|---|---|
| Answerable — correct | 7/14 (50%) |
| Answerable — partial | 3/14 (21%) |
| Answerable — wrong | 4/14 (29%) |
| Grounding rate (answerable) | 11/14 (79%) |
| Unanswerable — correctly refused | 3/3 (100%) |
| Hallucination rate | 0% |

This approach was discarded: it required knowing the answer before asking the question, which is not how the system works in production.

### Stage 3 — Intent detection (production-equivalent)

The hardcoded hints were removed. The same eval runner calls `pipeline.query(question, k=8)` with no explicit filters — the system extracts all retrieval metadata from the question itself.

| Category | Score |
|---|---|
| Answerable — correct | 12/14 (86%) |
| Answerable — partial | 0/14 (0%) |
| Answerable — wrong | 2/14 (14%) |
| Grounding rate (answerable) | 13/14 (93%) |
| Unanswerable — correctly refused | 3/3 (100%) |
| Hallucination rate | 0% |

**The consistent finding across all three stages:** every wrong answer is a retrieval failure, not a hallucination. The system never invented a number — it either found the right answer or refused. Hallucination rate held at 0% throughout.

### Root causes of remaining failures

**F01 — Meta FCF 2022:** Intent detection correctly identifies the filing and section. The problem is chunking: Meta's FCF reconciliation table spans multiple lines in the cash flow statement, and the row containing the final `$18,439M` figure was not in the top-8 retrieved chunks. The finance lease adjustment (~$850M) that separates operating CF minus capex from Meta's disclosed FCF definition falls below the chunk boundary. Increasing `k` or targeting the liquidity section more precisely would fix this.

**A01 — Meta margin attribution 2022→2023:** This is a cross-year question with no specific period to pin. Without a period filter, chunks from 2024 and 2025 MDA sections — which discuss the same margin recovery retrospectively — outrank the 2023 filing's original attribution paragraph on cosine similarity. A smarter approach would detect year-span questions and retrieve from both endpoints.

---

## Most Interesting Insights

**Meta's "Year of Efficiency" in the XBRL data:** Operating margin collapsed from 39.6% in 2021 to 24.8% in 2022 — a 14.8-point drop — then recovered sharply to 34.7% in 2023 and 42.2% in 2024. The 2022 10-K MD&A attributes this directly to the metaverse bet: increased infrastructure investment and a deteriorating advertising market hitting revenue simultaneously. By 2023, headcount fell from ~87,000 to ~67,000 while ad revenue recovered, reversing the cost trajectory. Confidence is high: the margin numbers come from XBRL (exact, SEC-filed figures) and the narrative attribution is verified against the actual filing text.

**NVIDIA's FY2020 as the trough before the AI explosion:** The Quant → Narrative feature on NVDA's R&D % of revenue immediately surfaces FY2020 (ended January 26, 2020) as the biggest year-over-year change: R&D jumped from 20.3% to 25.9% of revenue. The driver was a 7% revenue decline (gaming crypto hangover + data center softness) combined with continued R&D investment — management kept hiring even as revenue fell. This period is the clearest inflection in NVDA's 25-year filing history: the company that looked overinvested in FY2020 became the defining infrastructure company of the AI era four years later.

---

## Failure Found and Diagnosed

**Problem:** The quant-to-narrative feature for Microsoft's operating margin retrieved `RISK_FACTORS` chunks instead of `MDA` chunks, causing Claude to answer about hypothetical future risks rather than explaining what actually drove the margin change.

**Root cause:** The rewritten query contained terms like "operating margin performance drivers" which semantically matched risk factor text ("operational risks to margins") more than MDA text (which uses language like "revenue increased", "costs declined"). ChromaDB's cosine similarity does not distinguish between a risk description of a metric and a performance explanation of the same metric.

**Diagnosis:** Confirmed by inspecting returned chunk metadata — all six retrieved chunks were from `RISK_FACTORS` sections despite the filing containing 53,000 chars of MDA with directly relevant content.

**Fix:** Added `sections=["mda"]` and `period=peak_period` filters to the retrieval call for the narrative query. Since we already know which filing and which section to look in, semantic search should be used for *what within the MDA*, not *which filing*.

**Lesson:** Semantic search works well for "find relevant content across many documents" but poorly for "find a specific section of a specific document." When the target is known, use metadata filters; save semantic search for intra-document relevance ranking.

---

## How I Used AI Tools

I used Claude Code throughout the entire build — it wrote first drafts of all source files, debugged the `sec-edgar-downloader` API version mismatch, and identified the XBRL duration-filter fix.

**One specific override:** Claude initially generated the XBRL annual filter as `e.get("form") == "10-K" and e.get("fp") == "FY"`. When I ran it, Microsoft's data included quarterly entries (Q1–Q3 values like $21.9B) tagged with `fp="FY"` and `form="10-K"` — a real quirk of how MSFT submits XBRL. Claude had not anticipated this edge case. I inspected the raw API response, identified that Microsoft tags quarterly breakdowns within its 10-K as FY periods, and overrode the filter with a duration-based check (`end_date - start_date >= 340 days`). This was the difference between clean annual data and garbage output.

**A design disagreement I won:** Claude's first instinct for the evaluation runner was to add hardcoded `ticker` and `period` hints per question to improve scores. I pushed back — if the hints encode knowledge you only have after reading the answer, the eval is measuring the hints, not the system. The right fix was to build intent detection that extracts those filters from the question itself, so it generalises to any question a real user would ask.

---

## Framework Reflections

**Where `edgartools` helped:** Removing 200+ lines of SGML parsing code. Section extraction (`tenk.management_discussion`, `tenk.risk_factors`) works reliably for 10-K filings and eliminated the need to locate and parse HTML within a multi-document bundle.

**Where I fought it:** The `TenQ` object does not have a `management_discussion` attribute — 10-Q MD&A requires `tenq["Part I, Item 2"]` instead. This inconsistency between `TenK` and `TenQ` APIs is undocumented and required trial and error.

**Where I dropped to raw API:** The XBRL financial metrics use the SEC's REST API directly rather than through edgartools. edgartools provides parsed financial statement objects, but they use a custom table format that is harder to aggregate across years and companies than the raw JSON from the API. The raw API gives a complete history in one call and lets me apply my own filtering logic (duration-based annual detection, deduplication by filed date, auto-discovery of all USD concepts).

**The XBRL comparability problem:** Even within the standardised us-gaap taxonomy, the same economic concept can have multiple valid tag names — a product of accounting standard changes over time (ASC 606 in 2018 changed revenue recognition tags) and industry-specific reporting conventions. The system handles this for revenue by trying six candidate tags in priority order. For other metrics, auto-discovered `xbrl__*` columns capture whatever a company actually reports, but cross-company comparison of those columns requires verifying that the underlying concepts are economically equivalent — the tag names alone are not sufficient.
