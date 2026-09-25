import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Look for other embedding options not sure if this is the best embedding model for our use case
COLLECTION_NAME = "sec_filings"
CHUNK_SIZE = 1200     # chars — keeps chunks within the embedding model's token limit
CHUNK_OVERLAP = 150   # chars — prevents context loss at boundaries

DATA_DIR = Path(__file__).parent.parent / "data" / "sec-filings"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "index"


def build_index() -> None:
    
    # Used to split text for chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], # Finds breaks within text for chunks or counting words
    )

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    # Turns a chunk of words into a chunk of vectors, multiple words per token, 1 token = 1 vector
    # https://www.youtube.com/watch?v=wgfSDrqYMJ4
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Vectors/embeddings are stored in the CHROMA_DIR directory
    client = chromadb.PersistentClient(path=str(CHROMA_DIR)) # ChromaDB client

    # Collection we are storing all our SEC files information
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    json_files = sorted(DATA_DIR.glob("**/*.json")) # Returns a list of all SEC json files

    logger.info(f"Found {len(json_files)} filings to process")

    total_chunks = 0
    for json_file in json_files:
        chunks_added = _process_filing(json_file, splitter, embedder, collection)
        total_chunks += chunks_added

    logger.info(f"\nDone. {total_chunks} chunks added across {len(json_files)} filings")
    logger.info(f"ChromaDB collection total: {collection.count()} chunks")


def _process_filing(
    json_file: Path,
    splitter: RecursiveCharacterTextSplitter,
    embedder: SentenceTransformer,
    collection,
) -> int:
    data = json.loads(json_file.read_text(encoding="utf-8")) # Gets contents of SEC JSON file 

    ticker = data["ticker"]
    form_type = data["form_type"]
    period = data["period"]
    source_id = f"{ticker}_{form_type}_{period}"

    # Skip if already embedded
    existing = collection.get(where={"source_id": source_id}, limit=1)
    if existing["ids"]:
        logger.info(f"  Skipping {source_id} (already in ChromaDB)")
        return 0

    chunks_added = 0
    for section_name, text in data["sections"].items():
        text = text.strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        if not chunks:
            continue

        # This is when chunks of words are converted into vectors. 
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{source_id}__{section_name}__{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "ticker": ticker,
                "company": data["company"],
                "form_type": form_type,
                "period": period,
                "filed_date": data["filed_date"],
                "section": section_name,
                "source_id": source_id,
                "source": f"{ticker}/{form_type}/{period}.json",
            }
            for _ in chunks
        ]

        # Adds the embeddings/vectors into collection
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        chunks_added += len(chunks)

    logger.info(f"  Embedded {source_id}: {chunks_added} chunks")
    return chunks_added


if __name__ == "__main__":
    build_index()
