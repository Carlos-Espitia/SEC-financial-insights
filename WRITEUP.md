# WRITEUP

## Architecture

The system has five components, each in a dedicated source file:

**Ingestion (`src/ingest.py`)** — Fetches 10-K and 10-Q filings for any SEC-listed company via `edgartools`. Each filing is parsed into clean text sections (Business, Risk Factors, MD&A, and financial statements) and saved as structured JSON under `data/sec-filings/{TICKER}/{FORM_TYPE}/{PERIOD}.json`. Companies are registered in `data/companies.json` (a lightweight registry storing ticker, CIK, and GAAP revenue concept). `fetch_company(ticker, start_year)` accepts an optional start year so users can pull filings going back to 1993 (the start of SEC EDGAR electronic filing). I initially used `sec-edgar-downloader` but switched to `edgartools` after discovering the raw SGML bundles it produced required complex parsing; edgartools extracts clean, section-level text directly.

**Indexing (`src/indexer.py`)** — Each JSON file is chunked using LangChain's `RecursiveCharacterTextSplitter` (~1,200 chars, 150-char overlap), embedded with `sentence-transformers/all-MiniLM-L6-v2` (local, free), and stored in a ChromaDB vector database at `data/index/`. Every chunk carries metadata: ticker, company, form type, period, filed date, and section name. This metadata drives both filtering and citations. The indexer is fully dynamic — it globs all JSON files in the filings directory and skips already-indexed documents, so adding a new company requires no code changes. A complementary in-memory BM25 sparse index is built from the same chunks at pipeline startup, giving retrieval a keyword channel alongside the dense vectors.

**RAG Pipeline (`src/rag.py`)** — A five-step agentic loop:

1. **Intent detection** — Claude Haiku extracts retrieval metadata (ticker(s), period, form\_type, sections) from the natural-language question using a dynamically-built prompt that reflects exactly what's indexed. The prompt is generated at startup by scanning `data/sec-filings/` — no hardcoded company list. Adding a new company re-teaches the intent detector at the next app restart. For multi-year comparison questions it emits a *list* of period-end dates so retrieval can target each year's filing; for multi-company questions it emits a list of tickers.
2. **Gap detection** — Before retrieval, checks whether the detected ticker exists in the corpus and whether any year referenced in the question predates the earliest indexed filing. Returns an explicit, actionable refusal ("I don't have NVDA filings from 2010; the earliest indexed is 2002") rather than silently retrieving the wrong data. An out-of-range *year* for an in-corpus company is reported as such, never as a missing company.
3. **Query rewriting** — Rewrites the user question into a keyword-rich search query optimised for retrieval over SEC filing text.
4. **Hybrid retrieval + reranking** — Candidates are gathered from two indexes in parallel: dense vector search (ChromaDB / MiniLM) and BM25 sparse keyword search over the same chunks. Dense search is fuzzy on exact terms; BM25 reliably catches segment names, "free cash flow", "dividends", and other rare tokens. Ticker, period, and form type are *hard* metadata filters — pinning the wrong company or year would be a correctness failure. The target section is a *soft* signal: the system gathers both an unfiltered candidate pool (a recall safety net if the section guess is wrong) and a section-targeted pool (so compact statement tables aren't drowned out by the much larger risk-factor and MD&A sections). A cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) then scores every candidate. Because the reranker systematically under-ranks dense numeric tables, the final selection reserves a few slots for the guessed section so a line-item value can't sit just below the cut-off. Multi-year questions fan retrieval out across each year's filing; multi-company questions rerank each company separately so one can't crowd the others out of the context.
5. **Generation + verification** — Claude generates a cited answer from the retrieved chunks, then a separate grounding-verification call checks whether every factual claim is supported by the context. Ungrounded answers are flagged; unanswerable questions return a standardised refusal.

**Metrics (`src/metrics.py`)** — Pulls structured financial facts from the SEC EDGAR XBRL API (`/api/xbrl/companyfacts/{CIK}.json`). New companies are registered automatically: the system looks up the CIK from SEC's public company tickers list, fetches the XBRL fact blob, and auto-detects the correct revenue GAAP concept by trying six common tags in priority order. Annual figures are filtered by period duration (≥340 days) to exclude the quarterly breakdowns that Microsoft incorrectly tags as `fp=FY` in their 10-K XBRL submissions. Beyond the curated set of ~10 standard metrics, the system auto-discovers every USD-denominated us-gaap concept with annual 10-K data for each company, adding them as `xbrl__ConceptName` columns. This surfaces company-specific line items (segment revenue, restructuring charges, etc.) that the hardcoded list would miss. Free cash flow subtracts finance lease principal payments in addition to capex, matching Meta's disclosed FCF definition.

**UI (`app.py`)** — Streamlit app with three tabs:
- **Q&A Chat** — RAG chat interface with source citations and grounding status. Starter questions guide new users. The sidebar "Filter by company" pass-through lets users narrow retrieval when they know the target company.
- **Financial Charts** — Interactive line charts for any XBRL metric across all indexed companies, with metric cards showing the latest annual figures. A "View raw data" expander with per-company filtering and a toggle to show all auto-discovered XBRL concepts.
- **Deep Analysis (Quant → Narrative)** — Identifies the biggest year-over-year change in a selected metric and retrieves what management said about it in the corresponding MD&A. Works across the full indexed history (e.g., 25 years of NVDA filings).

The sidebar also shows a live coverage panel (which periods are indexed per company), a filing fetch panel with start-year control, and a one-click "Add company" flow that handles CIK lookup → filing download → indexing → metrics rebuild in sequence.

---

## Evaluation and Hallucination Rate

Every expected answer in every test set was verified directly against the raw JSON filing files before use — the targets are ground truth, not the model's own output. Grading is done by an LLM judge (`evaluation/grade.py`, Claude Sonnet) that compares each answer to its verified expected answer.

### Three test sets, on purpose

To measure whether the pipeline is genuinely accurate rather than tuned to a fixed question list, evaluation uses three separate sets:

- **`test_set_v3.json` (40 questions) — the tuning set.** Used while iterating on retrieval. Factual lookups, multi-year and cross-company comparisons, and unanswerable questions.
- **`test_set_holdout.json` (24 questions) — held-out.** Deliberately drawn from different sections (balance sheet, cash flow, business segments) and years than the tuning set.
- **`test_set_cold.json` (12 questions) — cold validation.** Built by randomly sampling filings across the corpus (2017–2025, varied companies and sections), then run **exactly once with no further changes** to the pipeline.

### Results

| Set | Answerable correct | Refusals correct | Hallucinations |
|---|---|---|---|
| v3 tuning (40Q) | 28/32 (88%) | 8/8 | 0 |
| Held-out (24Q) | 20/21 (95%) | 3/3 | 0 |
| **Cold (12Q)** | **8/10 (80%)** | **2/2** | **0** |

**The honest reading.** The held-out set initially exposed failures (balance-sheet and cash-flow lookups that wrongly refused), which were then fixed — meaning the held-out set was *tuned against*, so its 95% is optimistic. The cold set, run once and never tuned on, is the real generalization signal: **~80% on answerable questions.** The pipeline is strong on mainstream cases (recent filings, headline metrics) and weaker on the long tail (e.g. segment and balance-sheet lookups in older NVIDIA quarterly filings). The gap between the three numbers is itself the lesson: without a set you never touch, "it generalizes" is a hope, not a measurement.

**The property that held across all three sets — including cold data — is a 0% hallucination rate.** Every failure is a *retrieval* failure: the system either surfaces the right figure or refuses ("I don't have enough information…"). It never invents a number. For a financial tool this is the guarantee that matters most, and it survived data the pipeline was never tuned against.

### What the cold set still catches

- **Period pinning on older filings** — "total assets as of October 28, 2018" retrieved ten different NVIDIA filings instead of pinning the single `2018-10-28` 10-Q, so the total-assets line never reached the context.
- **Numeric-table ranking** — the cross-encoder reranker under-ranks dense statement/segment tables relative to prose that merely mentions the topic. The section-reserve mechanism mitigates this for common cases but not the entire long tail.

Both are retrieval-targeting issues, not generation or indexing issues — the answer exists in the index; the pipeline just doesn't always surface it for less-common filings. The next levers (a finance-tuned embedding model; computing comparisons in code rather than trusting the LLM's arithmetic) would lift the tail but were out of scope for this iteration.

### Anti-pattern avoided: eval gaming

An earlier evaluation attempt passed per-question `ticker`/`period` hints straight into the retriever. Scores rose, but those hints encode knowledge you only have *after* reading the answer — the eval was measuring the hints, not the system. That approach was discarded in favour of intent detection that extracts retrieval metadata from the question itself, the way a real user query arrives.

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

**How this evolved:** Applying that lesson too literally — making `sections` a *hard* filter on every question — later proved brittle. When intent detection guessed the section wrong, the answer was excluded with no way to recover (e.g. free-cash-flow questions filtered to `cash_flow_statement`, but the reported figure lives in a reconciliation table in MD&A). The production design therefore treats section as a *soft* signal: it gathers both an unfiltered pool and a section-targeted pool, reranks across both, and reserves a few final slots for the guessed section. This keeps the benefit (compact statement tables surface instead of being buried) without the fragility (a wrong guess is no longer fatal). The Deep Analysis tab still passes explicit `sections`/`period` because there the target filing and section are known for certain.

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
