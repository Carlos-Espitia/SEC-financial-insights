import os
import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "sec_filings"
LLM_MODEL = "claude-haiku-4-5-20251001"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "index"
SEC_FILINGS_DIR = Path(__file__).parent.parent / "data" / "sec-filings"

# How many candidates to pull from each retriever (dense + sparse) before reranking.
CANDIDATE_N = 20


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used for BM25 indexing and querying."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    rewritten_query: str
    is_grounded: bool
    grounding_note: str


class RAGPipeline:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Pull every chunk once to build an in-memory BM25 (sparse) index alongside the
        # dense vector index. Hybrid retrieval lets exact-term matches (segment names,
        # "free cash flow", "dividends") surface even when the embedding model misses them.
        logger.info("Building BM25 sparse index...")
        self._build_bm25_index()

        # Scan the filings directory so intent detection adapts to whatever is indexed
        self._corpus = _scan_corpus(SEC_FILINGS_DIR)
        self._intent_prompt = _build_intent_prompt(self._corpus)

        logger.info(f"RAG pipeline ready. Collection has {self.collection.count()} chunks. "
                    f"BM25 index: {len(self._bm25_docs)} docs. "
                    f"Corpus tickers: {sorted(self._corpus)}")

    def _build_bm25_index(self) -> None:
        """Loads all chunk documents + metadata from ChromaDB and builds a BM25 index."""
        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data["ids"]
        self._bm25_docs = data["documents"]
        self._bm25_meta = data["metadatas"]
        tokenized = [_tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        ticker: str | None = None,
        k: int = 10,
        sections: list[str] | None = None,
        period: str | None = None,
        form_type: str | None = None,
        tickers: list[str] | None = None,
        periods: dict[str, str] | None = None,
    ) -> RAGResult:
        """
        Full RAG pipeline:
          1. Detect retrieval intent (ticker, period, form_type, sections) from the question
          2. Rewrite the question for better retrieval
          3. Retrieve top-k relevant chunks from ChromaDB with detected filters
          4. Generate a cited answer using Claude
          5. Verify the answer is grounded in the sources

        Any explicitly-passed kwargs (ticker, tickers, period, etc.) override intent detection
        for that field only — useful for programmatic callers that already know the target filing.
        """
        # Step 0 — auto-detect retrieval intent unless all filters are explicit
        no_explicit_filters = not any([ticker, tickers, period, periods, form_type, sections])
        if no_explicit_filters:
            intent = self._detect_intent(question)
            ticker    = intent.get("ticker")
            tickers   = intent.get("tickers")
            period    = intent.get("period")
            periods   = intent.get("periods")
            form_type = intent.get("form_type")
            sections  = intent.get("sections")
            logger.info(f"Intent: {intent}")

        # Gap detection — catch requests for companies or years outside the indexed corpus.
        # The not_in_corpus flag only comes from intent detection; when the caller passes
        # explicit filters they already know the ticker is valid, so default to False.
        not_in_corpus = intent.get("not_in_corpus", False) if no_explicit_filters else False
        # Guard: if we identified an in-corpus company, the deterministic ticker/year-range
        # checks handle it — never fire the generic out-of-corpus message (a year being
        # unavailable must not be reported as "company not indexed").
        identified = (tickers or []) + ([ticker] if ticker else [])
        if any(t in self._corpus for t in identified):
            not_in_corpus = False
        gap_msg = self._check_coverage_gap(question, ticker, tickers, not_in_corpus=not_in_corpus)
        if gap_msg:
            return RAGResult(
                answer=gap_msg,
                sources=[],
                rewritten_query=question,
                is_grounded=True,
                grounding_note="Coverage gap detected.",
            )

        # Step 1 — rewrite
        primary_ticker = (tickers[0] if tickers else ticker) if (tickers or ticker) else None
        rewritten = self._rewrite_query(question, primary_ticker)
        logger.info(f"Rewritten query: {rewritten}")

        # Step 2 — retrieve
        if tickers:
            # Multi-ticker: search each company separately, combine by distance
            chunks = self._retrieve_multi_ticker(
                rewritten, tickers=tickers, k_per_ticker=k, sections=sections,
                form_type=form_type, periods=periods,
            )
        else:
            chunks = self._retrieve(
                rewritten, ticker=ticker, k=k, sections=sections,
                period=period, form_type=form_type,
            )
        if not chunks:
            return RAGResult(
                answer="I don't have enough information in the loaded filings to answer this question.",
                sources=[],
                rewritten_query=rewritten,
                is_grounded=True,
                grounding_note="No relevant chunks found.",
            )

        # Step 3 — generate
        answer = self._generate(question, chunks)

        # Step 4 — verify
        is_grounded, grounding_note = self._verify(answer, chunks)

        sources = _deduplicate_sources([
            {
                "ticker": c["metadata"]["ticker"],
                "company": c["metadata"]["company"],
                "form_type": c["metadata"]["form_type"],
                "period": c["metadata"]["period"],
                "section": c["metadata"]["section"],
                "source": c["metadata"]["source"],
            }
            for c in chunks
        ])

        return RAGResult(
            answer=answer,
            sources=sources,
            rewritten_query=rewritten,
            is_grounded=is_grounded,
            grounding_note=grounding_note,
        )

    # ------------------------------------------------------------------ #
    # Gap detection                                                        #
    # ------------------------------------------------------------------ #

    def _check_coverage_gap(
        self,
        question: str,
        ticker: str | None,
        tickers: list[str] | None,
        not_in_corpus: bool = False,
    ) -> str | None:
        """
        Returns an explanatory refusal message when the question references a
        company not in the corpus, or a year clearly before the earliest indexed filing.
        Returns None when everything looks fine and retrieval should proceed.
        """
        if not_in_corpus:
            indexed = ", ".join(sorted(self._corpus.keys()))
            return (
                f"I don't have filings for the company mentioned in your question. "
                f"The indexed companies are: {indexed}. "
                f"Use the sidebar to add more companies."
            )

        targets = tickers if tickers else ([ticker] if ticker else [])

        # Check for companies not in the corpus at all
        for t in targets:
            if t not in self._corpus:
                return (
                    f"I don't have any filings for **{t}** in my database. "
                    f"Use the sidebar to add it."
                )

        # Check for year references that fall before the earliest indexed filing
        year_matches = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', question)
        if not year_matches or not targets:
            return None

        requested_year = min(int(y) for y in year_matches)

        for t in targets:
            all_periods = sorted(
                self._corpus[t].get("10-K", []) + self._corpus[t].get("10-Q", [])
            )
            if not all_periods:
                continue
            earliest_year = int(all_periods[0][:4])
            latest_year = int(all_periods[-1][:4])
            if requested_year < earliest_year:
                company_name = self._corpus[t].get("company", t)
                return (
                    f"I don't have {company_name} filings from **{requested_year}**. "
                    f"The indexed filings for {t} cover **{earliest_year}–{latest_year}**. "
                    f"Use the sidebar to fetch historical filings going further back."
                )

        return None

    # ------------------------------------------------------------------ #
    # Step 0 — Intent detection                                           #
    # ------------------------------------------------------------------ #

    def _detect_intent(self, question: str) -> dict:
        """
        Calls Claude to extract retrieval metadata (ticker, period, form_type, sections)
        from a natural-language question. Returns a dict; falls back to {} on any error.
        The prompt is built at init time from whatever companies are actually indexed,
        so adding a new ticker to the corpus automatically teaches the intent detector.
        """
        try:
            response = self.claude.messages.create(
                model=LLM_MODEL,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"{self._intent_prompt}\n\nQuestion: {question}",
                }],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            intent = json.loads(raw)
            # If only one company was identified, keep it as ticker (not tickers list).
            if isinstance(intent.get("tickers"), list) and len(intent["tickers"]) == 1:
                intent["ticker"] = intent.pop("tickers")[0]
            # For a single-company question, lift any period that came through the
            # periods dict up to `period` (which may be a string or a list of dates),
            # then drop the now-redundant periods dict.
            if intent.get("ticker") and not intent.get("period") \
                    and isinstance(intent.get("periods"), dict):
                p = intent["periods"].get(intent["ticker"])
                if p:
                    intent["period"] = p
            if intent.get("ticker"):
                intent["periods"] = None
            return intent
        except Exception as exc:
            logger.warning(f"Intent detection failed ({exc}); proceeding without filters.")
            return {}

    # ------------------------------------------------------------------ #
    # Step 1 — Query rewriting                                            #
    # ------------------------------------------------------------------ #

    def _rewrite_query(self, question: str, ticker: str | None) -> str:
        """
        Rewrites the user's question into a search query optimised for
        semantic retrieval over SEC filing text.
        """
        ticker_hint = f" Focus on {ticker}." if ticker else ""
        prompt = (
            f"You are a financial search expert.{ticker_hint}\n"
            "Rewrite the following question into a short, keyword-rich search query "
            "that will retrieve the most relevant passages from SEC 10-K and 10-Q filings.\n"
            "Return ONLY the rewritten query — no explanation, no punctuation at the end.\n\n"
            f"Question: {question}"
        )
        response = self.claude.messages.create(
            model=LLM_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------ #
    # Step 2 — Retrieval (hybrid dense + sparse, then cross-encoder rerank) #
    # ------------------------------------------------------------------ #
    #
    # Design:
    # - ticker / period / form_type are HARD filters — they are reliably
    #   extracted and pinning the wrong company or year would be a correctness
    #   failure.
    # - `sections` is a SOFT signal, not on/off. We gather candidates BOTH
    #   unfiltered (a recall safety net, so a wrong section guess can't silently
    #   drop the answer) AND restricted to the guessed section(s) (so compact
    #   statement/table chunks aren't drowned out by huge risk_factors/mda
    #   sections). The cross-encoder reranker then picks the best across the pool.

    def _retrieve(
        self,
        query: str,
        ticker: str | None = None,
        k: int = 6,
        sections: list[str] | None = None,
        period: str | list[str] | None = None,
        form_type: str | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval for one company (or no ticker filter): gather dense +
        sparse candidates across the requested period(s), rerank, return top k.
        When comparing multiple periods, scale k so each filing stays represented.
        """
        candidates = self._gather_candidates(query, ticker, period, form_type, sections)
        if not candidates:
            return []
        n_periods = len(period) if isinstance(period, list) else 1
        ranked = self._rerank(query, candidates)
        return self._select_final(ranked, k * n_periods, sections)

    def _retrieve_multi_ticker(
        self,
        query: str,
        tickers: list[str],
        k_per_ticker: int = 4,
        sections: list[str] | None = None,
        form_type: str | None = None,
        periods: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        Cross-company retrieval: gather + rerank each company's candidates
        separately and take the top k_per_ticker from each, so one company can't
        crowd the others out of the context for a comparison question.
        periods: optional per-ticker period pins, e.g. {"MSFT": "2024-06-30"}.
        """
        all_chunks = []
        for ticker in tickers:
            period = periods.get(ticker) if periods else None
            candidates = self._gather_candidates(query, ticker, period, form_type, sections)
            if candidates:
                ranked = self._rerank(query, candidates)
                all_chunks.extend(self._select_final(ranked, k_per_ticker, sections))
        return all_chunks

    def _gather_candidates(
        self,
        query: str,
        ticker: str | None,
        period: str | list[str] | None,
        form_type: str | None,
        sections: list[str] | None,
    ) -> list[dict]:
        """
        Pulls dense + sparse candidates for a single ticker, fanning out over each
        period when `period` is a list (a multi-year/quarter comparison), and
        deduplicates by chunk id. Gathers both an unfiltered pool (recall) and a
        section-targeted pool (precision for compact statement chunks).
        """
        periods = period if isinstance(period, list) else [period]
        query_vector = self.embedder.encode(query).tolist()

        by_id: dict[str, dict] = {}
        for p in periods:
            # Unfiltered pool — safety net if the section guess is wrong.
            for c in self._dense_search(query_vector, ticker, p, form_type, None, CANDIDATE_N):
                by_id.setdefault(c["id"], c)
            for c in self._sparse_search(query, ticker, p, form_type, None, CANDIDATE_N):
                by_id.setdefault(c["id"], c)
            # Section-targeted pool — guarantees the guessed section's chunks are
            # candidates even when a bigger section would otherwise crowd them out.
            if sections:
                for c in self._dense_search(query_vector, ticker, p, form_type, sections, CANDIDATE_N):
                    by_id.setdefault(c["id"], c)
                for c in self._sparse_search(query, ticker, p, form_type, sections, CANDIDATE_N):
                    by_id.setdefault(c["id"], c)
        return list(by_id.values())

    def _dense_search(
        self,
        query_vector: list[float],
        ticker: str | None,
        period: str | None,
        form_type: str | None,
        sections: list[str] | None,
        n: int,
    ) -> list[dict]:
        """Dense (embedding) similarity query against ChromaDB with metadata filters."""
        filters = []
        if ticker:
            filters.append({"ticker": {"$eq": ticker}})
        if period:
            filters.append({"period": {"$eq": period}})
        if form_type:
            filters.append({"form_type": {"$eq": form_type}})
        if sections:
            filters.append({"section": {"$in": sections}})

        where = None if not filters else (filters[0] if len(filters) == 1 else {"$and": filters})

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            where=where,
            include=["documents", "metadatas"],
        )
        chunks = []
        for cid, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        ):
            chunks.append({"id": cid, "text": doc, "metadata": meta})
        return chunks

    def _sparse_search(
        self,
        query: str,
        ticker: str | None,
        period: str | None,
        form_type: str | None,
        sections: list[str] | None,
        n: int,
    ) -> list[dict]:
        """BM25 (keyword) search over the in-memory index with the same metadata filters."""
        scores = self._bm25.get_scores(_tokenize(query))
        matching = []
        for i, meta in enumerate(self._bm25_meta):
            if ticker and meta["ticker"] != ticker:
                continue
            if period and meta["period"] != period:
                continue
            if form_type and meta["form_type"] != form_type:
                continue
            if sections and meta["section"] not in sections:
                continue
            matching.append(i)
        matching.sort(key=lambda i: scores[i], reverse=True)
        return [
            {"id": self._bm25_ids[i], "text": self._bm25_docs[i], "metadata": self._bm25_meta[i]}
            for i in matching[:n]
        ]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Scores every candidate against the query with the cross-encoder, sorted best-first."""
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates

    def _select_final(
        self, ranked: list[dict], k: int, sections: list[str] | None, reserve: int = 3
    ) -> list[dict]:
        """
        Picks the final k chunks from the reranked list. When intent names specific
        sections, reserve up to `reserve` slots for the best chunks from those
        sections. The cross-encoder (trained on web prose) systematically under-ranks
        dense numeric statement tables — exactly where balance-sheet / cash-flow /
        income-statement answers live — so without this a line-item value can sit just
        below the cut-off and the model answers "not enough information". The remaining
        slots follow the reranker's order, preserving the recall safety net for cases
        where the section guess is wrong.
        """
        if not sections or k <= reserve:
            return ranked[:k]

        chosen, chosen_ids = [], set()
        for c in ranked:
            if len(chosen) >= reserve:
                break
            if c["metadata"]["section"] in sections:
                chosen.append(c)
                chosen_ids.add(c["id"])
        for c in ranked:
            if len(chosen) >= k:
                break
            if c["id"] not in chosen_ids:
                chosen.append(c)
                chosen_ids.add(c["id"])
        return chosen

    # ------------------------------------------------------------------ #
    # Step 3 — Generation                                                  #
    # ------------------------------------------------------------------ #

    def _generate(self, question: str, chunks: list[dict]) -> str:
        """
        Builds a prompt with the retrieved chunks as context and asks Claude
        to answer the question with citations.
        """
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            m = chunk["metadata"]
            header = f"[{i}] {m['company']} | {m['form_type']} {m['period']} | {m['section'].upper()}"
            context_blocks.append(f"{header}\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_blocks)

        system = (
            "You are a financial analyst assistant with access to SEC filing excerpts. "
            "Answer questions using ONLY the provided context. "
            "Cite sources by referencing the numbered blocks (e.g. [1], [2]). "
            "If the answer cannot be found in the context, respond with exactly: "
            "'I don't have enough information in the provided filings to answer this question.' "
            "Never fabricate numbers or facts not present in the context. "
            "When comparing values across companies or time periods, read each value carefully "
            "from its labeled source block and state the exact numbers before drawing conclusions. "
            "If the context contains a figure the company reports and labels directly (for example a "
            "'Free cash flow' line in a reconciliation table), use that reported figure rather than "
            "recomputing it from other line items, since reported non-GAAP measures may include "
            "adjustments you cannot see."
        )

        user = (
            f"Context from SEC filings:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a concise, accurate answer with citations to the numbered sources above. "
            "For comparison questions, quote the specific values from each source before concluding which is higher or lower."
        )

        response = self.claude.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------ #
    # Step 4 — Verification                                                #
    # ------------------------------------------------------------------ #

    def _verify(self, answer: str, chunks: list[dict]) -> tuple[bool, str]:
        """
        Asks Claude to check whether the answer is fully supported by the
        retrieved context. Flags potential hallucinations.
        """
        context = "\n\n".join(c["text"] for c in chunks)

        prompt = (
            "You are a fact-checking assistant.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer to verify:\n{answer}\n\n"
            "Is every factual claim in the answer directly supported by the context above? "
            "Reply with one of:\n"
            "GROUNDED - all claims are supported\n"
            "NOT GROUNDED - <brief reason>\n\n"
            "Reply with ONLY one of those two options."
        )

        response = self.claude.messages.create(
            model=LLM_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = response.content[0].text.strip()
        is_grounded = verdict.upper().startswith("GROUNDED")
        return is_grounded, verdict


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _scan_corpus(filings_dir: Path) -> dict:
    """
    Scans data/sec-filings/{TICKER}/{FORM_TYPE}/{PERIOD}.json and returns:
      {
        "MSFT": {"company": "Microsoft", "10-K": ["2024-06-30", ...], "10-Q": [...]},
        ...
      }
    Works for any set of tickers — no hardcoding required.
    """
    corpus: dict[str, dict] = {}
    if not filings_dir.exists():
        return corpus
    for ticker_dir in sorted(filings_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        corpus[ticker] = {"company": ticker, "10-K": [], "10-Q": []}
        for form_dir in ticker_dir.iterdir():
            form_type = form_dir.name
            if form_type not in ("10-K", "10-Q"):
                continue
            for filing_file in sorted(form_dir.glob("*.json")):
                period = filing_file.stem          # e.g. "2024-06-30"
                corpus[ticker][form_type].append(period)
                # Pull company name from first file we encounter
                if corpus[ticker]["company"] == ticker:
                    try:
                        data = json.loads(filing_file.read_text(encoding="utf-8"))
                        corpus[ticker]["company"] = data.get("company", ticker)
                    except Exception:
                        pass
    return corpus


def _build_intent_prompt(corpus: dict) -> str:
    """
    Builds the intent-detection system prompt from live corpus metadata.
    For each ticker, lists available 10-K and 10-Q periods so the LLM can
    map "fiscal year 2024" or "Q2 2023" to the right period-end date.
    """
    lines = [
        "You are a retrieval-metadata extractor for a financial SEC filing database.",
        "The database contains filings ONLY for the companies listed below.",
        "Each company entry shows its available annual (10-K) and quarterly (10-Q) period-end dates.",
        "",
    ]
    for ticker, info in sorted(corpus.items()):
        annual = sorted(info.get("10-K", []))
        quarterly = sorted(info.get("10-Q", []))
        # Infer fiscal year end from the month-day of 10-K periods
        fy_end = ""
        if annual:
            from datetime import date
            d = date.fromisoformat(annual[0])
            fy_end = f" (fiscal year ends {d.strftime('%B')} {d.day})"
        lines.append(f"  {ticker} ({info['company']}){fy_end}")
        if annual:
            lines.append(f"    10-K periods: {', '.join(annual)}")
        if quarterly:
            lines.append(f"    10-Q periods: {', '.join(quarterly)}")

    lines += [
        "",
        "Given the question below, return a JSON object with exactly these keys:",
        '  "ticker"       : single ticker string if the question is about ONE company, else null',
        '  "tickers"      : list of ALL tickers if the question mentions or compares MULTIPLE companies, else null',
        '  "period"       : for a single-company question, the relevant period-end date "YYYY-MM-DD";',
        '                   if it compares multiple specific years/quarters of that ONE company,',
        '                   a LIST of date strings (e.g. ["2022-12-31","2023-12-31"]); else null',
        '  "periods"      : dict of ticker -> period-end date (string, or list of strings) for',
        '                   multi-company questions, else null',
        '  "form_type"    : "10-K" for annual questions, "10-Q" for quarterly, else null',
        '  "sections"     : list from ["business","risk_factors","mda","income_statement",',
        '                   "balance_sheet","cash_flow_statement"] relevant to the question, else null',
        '  "not_in_corpus": true if the question is about a company NOT listed in the database above, else false',
        "",
        "Rules:",
        "FISCAL YEAR MAPPING — always use the actual period-end dates listed above:",
        "- Each company labels fiscal years differently. Use the period-end dates, not the fiscal year name.",
        "- Example: AAPL 'fiscal year 2024' = 2024-09-28. NVDA 'fiscal year 2024' = 2024-01-28.",
        "",
        "MULTI-YEAR COMPARISONS (same company) — when a question compares TWO OR MORE specific years",
        "or quarters of one company (e.g. 'from 2022 to 2023', 'between FY2023 and FY2024',",
        "'Q3 2023 versus Q3 2022', 'how did X change'):",
        "- Set period to a LIST containing the period-end date of EACH year/quarter mentioned.",
        "- Always include ALL the periods being compared. Some facts (e.g. employee headcount) are stated",
        "  only for the current year of each filing, so both filings must be retrieved.",
        "- Example: 'Alphabet headcount from end of 2022 to end of 2023' -> period=[\"2022-12-31\",\"2023-12-31\"].",
        "",
        "MULTI-COMPANY — when the question explicitly names multiple companies:",
        "- Set tickers to the list of ALL mentioned companies (e.g. ['MSFT','GOOGL','META']).",
        "- Set periods to a dict mapping each ticker to its relevant period-end date (or list of dates).",
        "- NEVER set only one ticker when the question asks to compare several named companies.",
        "",
        "QUARTERLY PERIODS — match the quarter to the correct period-end date:",
        "- 'Q3 2023' for a company with September quarter-end = the period ending 2023-09-30.",
        "- 'Q2 FY2024' for NVDA (July quarter-end) = the period ending 2024-07-28.",
        "",
        "SECTIONS:",
        "- Revenue, margins, expenses, profitability, headcount, segment performance → ['mda','income_statement']",
        "- Risk factors, supply chain, geopolitical, competition → ['risk_factors']",
        "- Free cash flow → ['mda','cash_flow_statement']. Companies report free cash flow as a",
        "  non-GAAP measure in the MD&A liquidity section (a reconciliation table), so MD&A must be",
        "  included to find the company's own reported figure, not just the raw cash flow statement.",
        "- Operating cash flow, capital expenditures → ['cash_flow_statement']",
        "- Balance sheet items (debt, assets, equity) → ['balance_sheet']",
        "",
        "NOT IN CORPUS vs. YEAR UNAVAILABLE — do not confuse these:",
        "- Set not_in_corpus=true ONLY when the COMPANY itself is not in the list above",
        "  (e.g. Tesla, Amazon, Salesforce, Netflix).",
        "- If the company IS in the list but the requested YEAR is not available, set not_in_corpus=false",
        "  and STILL set ticker to that in-corpus company (set period to null). A year being unavailable",
        "  does NOT make a company not_in_corpus.",
        "- Example: 'Meta revenue in 2019' -> ticker=\"META\", not_in_corpus=false.",
        "",
        "Each date must be a string 'YYYY-MM-DD'. Use a list only to hold multiple such dates.",
        "Only output valid JSON. No explanation, no markdown fences.",
    ]
    return "\n".join(lines)


def _deduplicate_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for s in sources:
        key = (s["ticker"], s["form_type"], s["period"], s["section"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ------------------------------------------------------------------ #
# Quick smoke test                                                     #
# ------------------------------------------------------------------ #

# if __name__ == "__main__":
#     pipeline = RAGPipeline()

#     test_questions = [
#         "What was Microsoft's revenue growth in fiscal year 2025?",
#         "How did Meta describe its AI investments?",
#         "What are the main risk factors for Alphabet?",
#         "What was the unemployment rate on Mars in 2023?",  # unanswerable — should refuse
#     ]

#     for q in test_questions:
#         print(f"\n{'='*60}")
#         print(f"Q: {q}")
#         result = pipeline.query(q)
#         print(f"Rewritten: {result.rewritten_query}")
#         print(f"Answer: {result.answer}")
#         print(f"Grounded: {result.is_grounded} — {result.grounding_note}")
#         print(f"Sources: {[s['source'] for s in result.sources]}")
