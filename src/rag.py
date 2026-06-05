import os
import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

import anthropic
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sec_filings"
LLM_MODEL = "claude-haiku-4-5-20251001"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "index"
SEC_FILINGS_DIR = Path(__file__).parent.parent / "data" / "sec-filings"


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

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Scan the filings directory so intent detection adapts to whatever is indexed
        self._corpus = _scan_corpus(SEC_FILINGS_DIR)
        self._intent_prompt = _build_intent_prompt(self._corpus)

        logger.info(f"RAG pipeline ready. Collection has {self.collection.count()} chunks. "
                    f"Corpus tickers: {sorted(self._corpus)}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        ticker: str | None = None,
        k: int = 6,
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

        # Gap detection — catch requests for companies or years outside the indexed corpus
        gap_msg = self._check_coverage_gap(question, ticker, tickers)
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
    ) -> str | None:
        """
        Returns an explanatory refusal message when the question references a
        company not in the corpus, or a year clearly before the earliest indexed filing.
        Returns None when everything looks fine and retrieval should proceed.
        """
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
            # If only one company was identified, keep it as ticker (not tickers list)
            if isinstance(intent.get("tickers"), list) and len(intent["tickers"]) == 1:
                intent["ticker"] = intent.pop("tickers")[0]
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
    # Step 2 — Retrieval                                                   #
    # ------------------------------------------------------------------ #

    def _retrieve(
        self,
        query: str,
        ticker: str | None = None,
        k: int = 6,
        sections: list[str] | None = None,
        period: str | None = None,
        form_type: str | None = None,
    ) -> list[dict]:
        """
        Embeds the query and searches ChromaDB for the top-k most similar chunks.
        Optionally filters by ticker, section, period, and/or form_type.
        """
        query_vector = self.embedder.encode(query).tolist()

        # Build ChromaDB where filter
        filters = []
        if ticker:
            filters.append({"ticker": {"$eq": ticker}})
        if sections:
            filters.append({"section": {"$in": sections}})
        if period:
            filters.append({"period": {"$eq": period}})
        if form_type:
            filters.append({"form_type": {"$eq": form_type}})

        if len(filters) == 0:
            where = None
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = {"$and": filters}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})

        return chunks

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
        Runs _retrieve() once per ticker and merges results sorted by distance.
        This prevents one company's filings from dominating cross-company queries.
        periods: optional per-ticker period pins, e.g. {"MSFT": "2024-06-30"}
        """
        all_chunks = []
        for ticker in tickers:
            period = periods.get(ticker) if periods else None
            chunks = self._retrieve(
                query, ticker=ticker, k=k_per_ticker,
                sections=sections, form_type=form_type, period=period,
            )
            all_chunks.extend(chunks)
        all_chunks.sort(key=lambda x: x["distance"])
        return all_chunks

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
            "Never fabricate numbers or facts not present in the context."
        )

        user = (
            f"Context from SEC filings:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a concise, accurate answer with citations to the numbered sources above."
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
        '  "ticker"    : single ticker string if the question is about ONE company, else null',
        '  "tickers"   : list of tickers if the question compares MULTIPLE companies, else null',
        '  "period"    : period-end date "YYYY-MM-DD" for single-company questions, else null',
        '  "periods"   : dict of ticker->period-end for multi-company questions, else null',
        '  "form_type" : "10-K" for annual questions, "10-Q" for quarterly, else null',
        '  "sections"  : list from ["business","risk_factors","mda","income_statement",',
        '                "balance_sheet","cash_flow_statement"] relevant to the question, else null',
        "",
        "Rules:",
        "- Match fiscal year references to the nearest available 10-K period-end date listed above.",
        "- Match quarter references (Q1/Q2/Q3) to the nearest available 10-Q period-end date.",
        "- For multi-company comparisons, use 'tickers' + 'periods' (not 'ticker'/'period').",
        "- For risk questions: sections=[\"risk_factors\"]. For performance/narrative: sections=[\"mda\"].",
        "- For cash flow questions: include \"cash_flow_statement\" in sections.",
        "- If no specific year is mentioned, set period and periods to null.",
        "- Only output valid JSON. No explanation, no markdown fences.",
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
