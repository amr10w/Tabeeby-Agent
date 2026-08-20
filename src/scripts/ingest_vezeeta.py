#!/usr/bin/env python3
"""Vezeeta Doctors — Ingestion & Indexing Pipeline.

A production-ready, memory-efficient script that:

1. Reads and validates ~17K doctor records from CSV via Pydantic models.
2. Builds high-signal semantic text from clinical fields (specialty,
   subspecialties, symptoms, description, about_doctor) — deliberately
   excluding location and fee to avoid noise in symptom similarity.
3. Batch-embeds records using the project's ``LLMProviderFactory`` /
   ``OllamaProvider.embed_text()`` interface.
4. Indexes vectors + full metadata payloads into a Qdrant collection
   (``vezeeta_doctors``) with KEYWORD and INTEGER payload indexes on
   ``address`` and ``fee`` for efficient deterministic filtering.

Usage
-----
::

    # From the src/ directory:
    python -m scripts.ingest_vezeeta --batch-size 64 --reset

    # Or with defaults (batch_size=64, no reset):
    python -m scripts.ingest_vezeeta

Architectural Notes for Future ``search_doctors`` Tool
------------------------------------------------------
1. **Hybrid Vector + Filter Queries**:
   The ``search_doctors`` agent tool should:

   - Embed the user's symptom description via ``embed_text()``.
   - Build a ``models.Filter(must=[...])`` with optional
     ``FieldCondition`` entries for ``address`` (keyword match) and
     ``fee`` (integer range with ``gte`` / ``lte``).
   - Pass both the vector and filter to ``query_points()`` on the
     ``vezeeta_doctors`` collection.

   Example::

       from qdrant_client import models

       conditions = []
       if user_area:
           conditions.append(
               models.FieldCondition(
                   key="address",
                   match=models.MatchValue(value=user_area),
               )
           )
       if max_fee is not None:
           conditions.append(
               models.FieldCondition(
                   key="fee",
                   range=models.Range(lte=max_fee),
               )
           )

       results = client.query_points(
           collection_name="vezeeta_doctors",
           query=symptom_vector,
           query_filter=models.Filter(must=conditions) if conditions else None,
           limit=10,
       )

2. **Handling Loose / Missing Filters**:
   If the user only describes symptoms without specifying an area or budget,
   the ``conditions`` list will be empty and ``query_filter`` will be
   ``None`` — Qdrant returns pure vector similarity results.  The tool
   must *never* require both filters to be present.

3. **Compact Payload Serialization**:
   When formatting results for the LLM agent, extract only the summary
   fields to keep context token usage minimal::

       summary_fields = [
           "name", "specialty", "address", "fee",
           "reviews_count", "profile_url",
       ]
       compact = {k: hit.payload[k] for k in summary_fields if k in hit.payload}
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ── Ensure the src/ package root is on sys.path ─────────────────────────────
# This allows the script to be run as ``python -m scripts.ingest_vezeeta``
# from the src/ directory while still resolving project-internal imports.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv
from helpers.config import get_settings
from models.doctor import DoctorRecord
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

# ── Constants ────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "vezeeta_doctors"
"""Qdrant collection name.  Used consistently across ingestion and the
future ``search_doctors`` tool so that both reference the same index."""

CSV_RELATIVE_PATH: str = os.path.join("assets", "files", "vezeeta.csv")
"""Path to the Vezeeta CSV relative to the ``src/`` directory."""

DEFAULT_BATCH_SIZE: int = 64
"""Default number of records per embedding + insertion batch.  Chosen to
balance memory footprint against per-batch overhead for ~17K records."""

TEST_MEDICAL_PHRASE: str = (
    "cardiology heart disease chest pain hypertension arrhythmia"
)
"""A representative medical phrase used to probe the active embedding
model and dynamically detect the output vector dimension."""

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _configure_logging(level: int = logging.INFO) -> None:
    """Set up structured console logging with timestamp and module name.

    Parameters
    ----------
    level : int
        Logging verbosity (default ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _init_embedding_client(config):
    """Create and configure the embedding client via the LLM factory.

    Uses ``EMBEDDING_BACKEND`` from config (typically ``"OLLAMAE"`` for
    local Ollama) and sets the embedding model + dimension.

    Parameters
    ----------
    config : Settings
        Application settings from ``get_settings()``.

    Returns
    -------
    LLMInterface
        A fully-configured embedding provider.

    Raises
    ------
    RuntimeError
        If the factory returns ``None`` (unknown backend).
    """
    factory = LLMProviderFactory(config=config)
    client = factory.create(config.EMBEDDING_BACKEND)

    if client is None:
        raise RuntimeError(
            f"LLMProviderFactory returned None for backend "
            f"'{config.EMBEDDING_BACKEND}'. Check EMBEDDING_BACKEND in .env."
        )

    client.set_embedding_model(
        model_id=config.EMBEDDING_MODEL_ID,
        embedding_size=config.EMBEDDING_MODEL_SIZE,
    )
    logger.info(
        "Embedding client initialised — model=%s, configured_size=%s",
        config.EMBEDDING_MODEL_ID,
        config.EMBEDDING_MODEL_SIZE,
    )
    return client


def _init_vector_db(config):
    """Create, connect, and return the Qdrant vector DB client.

    Parameters
    ----------
    config : Settings
        Application settings from ``get_settings()``.

    Returns
    -------
    QdrantDBProvider
        A connected vector-database provider.

    Raises
    ------
    RuntimeError
        If the factory returns ``None`` or connection fails.
    """
    factory = VectorDBProviderFactory(config=config)
    db_client = factory.create(config.VECTOR_DB_BACKEND)

    if db_client is None:
        raise RuntimeError(
            f"VectorDBProviderFactory returned None for backend "
            f"'{config.VECTOR_DB_BACKEND}'. Check VECTOR_DB_BACKEND in .env."
        )

    db_client.connect()

    if db_client.client is None:
        raise RuntimeError(
            "Qdrant client failed to connect. "
            "Check VECTOR_DB_PATH and ensure the qdrant-client is installed."
        )

    logger.info("Vector DB connected — backend=%s", config.VECTOR_DB_BACKEND)
    return db_client


def _detect_embedding_dimension(embedding_client) -> int:
    """Embed a test medical phrase and return the detected vector dimension.

    This avoids hard-coding the dimension and instead dynamically reads it
    from the active Ollama model at runtime.

    Parameters
    ----------
    embedding_client : LLMInterface
        A configured embedding provider.

    Returns
    -------
    int
        The length of the embedding vector.

    Raises
    ------
    RuntimeError
        If the embedding call fails or returns an empty vector.
    """
    logger.info("Detecting embedding dimension with test phrase...")
    test_vector = embedding_client.embed_text(text=TEST_MEDICAL_PHRASE)

    if test_vector is None or len(test_vector) == 0:
        raise RuntimeError(
            "Failed to detect embedding dimension — embed_text() returned "
            "None or an empty vector.  Is the Ollama embedding model running?"
        )

    dim = len(test_vector)
    logger.info("Detected embedding dimension: %d", dim)
    return dim


def _load_and_clean_csv(csv_path: str) -> List[DoctorRecord]:
    """Read the Vezeeta CSV and return a list of validated ``DoctorRecord`` objects.

    Processing steps:

    1. Read CSV with pandas (handles encoding, quoting, multiline fields).
    2. Drop the ``page`` column (scraping artefact).
    3. Strip whitespace from all string columns.
    4. Validate each row through ``DoctorRecord.from_csv_row()``.
    5. Log and skip rows that fail validation.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    list[DoctorRecord]
        Successfully validated doctor records.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading CSV from %s ...", csv_path)

    # The Vezeeta CSV may contain Windows-1252 encoded characters.
    # Try common encodings in order of preference.
    df = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                csv_path, dtype=str, keep_default_na=False, encoding=encoding,
            )
            logger.info("CSV decoded successfully with encoding: %s", encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            logger.debug("Encoding '%s' failed, trying next...", encoding)
            continue

    if df is None:
        raise RuntimeError(
            f"Could not decode CSV at {csv_path} with any supported encoding."
        )

    logger.info("Raw CSV rows: %d, columns: %s", len(df), list(df.columns))

    # Drop the `page` column (scraping artefact)
    if "page" in df.columns:
        df = df.drop(columns=["page"])

    # Strip whitespace from all string columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    records: List[DoctorRecord] = []
    skipped: int = 0

    for idx, row in df.iterrows():
        try:
            record = DoctorRecord.from_csv_row(row.to_dict())

            # Skip records that produce no embeddable text
            if not record.build_semantic_text():
                logger.debug(
                    "Row %d skipped — no embeddable clinical fields (name=%s)",
                    idx, row.get("name", "?"),
                )
                skipped += 1
                continue

            records.append(record)
        except Exception as e:
            logger.warning("Row %d validation failed: %s", idx, e)
            skipped += 1

    logger.info(
        "CSV cleaning complete — valid: %d, skipped: %d, total: %d",
        len(records), skipped, len(df),
    )
    return records


def _create_payload_indexes(db_client, collection_name: str) -> None:
    """Create Qdrant payload schema indexes for efficient filtering.

    Creates:
    - ``address`` as **KEYWORD** index → fast exact match filtering by area/city.
    - ``fee`` as **INTEGER** index → efficient range filtering (``gte`` / ``lte``).

    These indexes are consumed by the future ``search_doctors`` tool to
    implement hybrid vector + deterministic filter queries.

    .. note::
       We access ``db_client.client`` (the raw ``QdrantClient``) directly
       because ``create_payload_index`` is a one-time admin operation not
       exposed through the project's ``VectorDBInterface`` abstraction.

    Parameters
    ----------
    db_client : QdrantDBProvider
        A connected vector-database provider.
    collection_name : str
        Target Qdrant collection name.
    """
    try:
        from qdrant_client import models as qdrant_models

        db_client.client.create_payload_index(
            collection_name=collection_name,
            field_name="address",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created KEYWORD payload index on 'address'")

        db_client.client.create_payload_index(
            collection_name=collection_name,
            field_name="fee",
            field_schema=qdrant_models.PayloadSchemaType.INTEGER,
        )
        logger.info("Created INTEGER payload index on 'fee'")

    except Exception as e:
        logger.error("Failed to create payload indexes: %s", e)
        raise


def _embed_batch(
    embedding_client,
    texts: List[str],
) -> List[Optional[list]]:
    """Embed a batch of texts with per-row error handling.

    Each text is embedded individually through ``embed_text()``.  If a
    single row fails, its slot is filled with ``None`` and the error is
    logged — the batch is *not* aborted.

    Parameters
    ----------
    embedding_client : LLMInterface
        A configured embedding provider.
    texts : list[str]
        The semantic texts to embed.

    Returns
    -------
    list[list[float] | None]
        One vector per input text, or ``None`` for failures.
    """
    vectors: List[Optional[list]] = []

    for text in texts:
        try:
            vec = embedding_client.embed_text(text=text)
            vectors.append(vec)
        except Exception as e:
            logger.warning("Embedding failed for text (%.60s...): %s", text, e)
            vectors.append(None)

    return vectors


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_ingestion(
    batch_size: int = DEFAULT_BATCH_SIZE,
    do_reset: bool = False,
) -> None:
    """Orchestrate the full ingestion lifecycle.

    Steps:

    1. Load environment and configuration.
    2. Initialise embedding and vector DB clients.
    3. Detect embedding dimension from the active model.
    4. Load and clean the CSV into validated ``DoctorRecord`` objects.
    5. Create (or reset) the Qdrant collection.
    6. Create payload indexes on ``address`` and ``fee``.
    7. Batch-process: embed → filter failures → insert into Qdrant.
    8. Log final collection statistics.

    Parameters
    ----------
    batch_size : int
        Number of records per embedding + insertion batch.
    do_reset : bool
        If ``True``, drop and recreate the collection before ingesting.
    """
    pipeline_start = time.time()

    # ── 1. Configuration ─────────────────────────────────────────
    load_dotenv()
    config = get_settings()
    logger.info(
        "Pipeline started — app=%s v%s, batch_size=%d, reset=%s",
        config.APP_NAME, config.APP_VERSION, batch_size, do_reset,
    )

    # ── 2. Client initialisation ─────────────────────────────────
    embedding_client = _init_embedding_client(config)
    db_client = _init_vector_db(config)

    # ── 3. Dynamic dimension detection ───────────────────────────
    embedding_dim = _detect_embedding_dimension(embedding_client)

    # ── 4. CSV loading & validation ──────────────────────────────
    csv_path = _SRC_DIR / CSV_RELATIVE_PATH
    records = _load_and_clean_csv(str(csv_path))

    if not records:
        logger.error("No valid records found — aborting ingestion.")
        db_client.disconnect()
        return

    # ── 5. Collection creation / reset ───────────────────────────
    db_client.create_collection(
        collection_name=COLLECTION_NAME,
        embedding_size=embedding_dim,
        do_reset=do_reset,
    )
    logger.info(
        "Collection '%s' ready (dim=%d, reset=%s)",
        COLLECTION_NAME, embedding_dim, do_reset,
    )

    # ── 6. Payload index creation ────────────────────────────────
    _create_payload_indexes(db_client, COLLECTION_NAME)

    # ── 7. Batch embedding & insertion ───────────────────────────
    total_inserted = 0
    total_failed = 0
    batch_count = 0

    progress = tqdm(
        total=len(records),
        desc="Ingesting doctors",
        unit="rec",
        ncols=100,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "batches={postfix}"
        ),
    )
    progress.set_postfix_str(f"{batch_count}")

    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch_count += 1

        # Build semantic texts for embedding
        semantic_texts = [rec.build_semantic_text() for rec in batch_records]

        # Embed the batch
        vectors = _embed_batch(embedding_client, semantic_texts)

        # Collect successful embeddings for insertion
        insert_texts: List[str] = []
        insert_vectors: List[list] = []
        insert_metadata: List[dict] = []

        for rec, text, vec in zip(batch_records, semantic_texts, vectors):
            if vec is not None:
                insert_texts.append(text)
                insert_vectors.append(vec)
                insert_metadata.append(rec.to_qdrant_payload())
            else:
                total_failed += 1
                logger.debug(
                    "Skipping record '%s' — embedding returned None", rec.name
                )

        # Insert successful records into Qdrant
        if insert_texts:
            try:
                db_client.insert_many(
                    collection_name=COLLECTION_NAME,
                    texts=insert_texts,
                    vectors=insert_vectors,
                    metadata=insert_metadata,
                    batch_size=batch_size,
                )
                total_inserted += len(insert_texts)
            except Exception as e:
                logger.error(
                    "Batch %d insertion failed: %s (records lost: %d)",
                    batch_count, e, len(insert_texts),
                )
                total_failed += len(insert_texts)

        progress.update(len(batch_records))
        progress.set_postfix_str(f"{batch_count}")

    progress.close()

    # ── 8. Final statistics ──────────────────────────────────────
    elapsed = time.time() - pipeline_start

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("-" * 60)
    logger.info("Total records processed : %d", len(records))
    logger.info("Successfully inserted   : %d", total_inserted)
    logger.info("Failed / skipped        : %d", total_failed)
    logger.info("Batches executed        : %d", batch_count)
    logger.info("Elapsed time            : %.1f seconds", elapsed)
    logger.info("-" * 60)

    # Query final collection state
    try:
        info = db_client.get_collection_info(COLLECTION_NAME)
        logger.info("Collection info         : %s", info)
    except Exception as e:
        logger.warning("Could not retrieve collection info: %s", e)

    logger.info("=" * 60)

    # Cleanup
    db_client.disconnect()
    logger.info("Vector DB disconnected.  Pipeline finished.")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and launch the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Vezeeta Doctors — Ingestion & Indexing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.ingest_vezeeta --batch-size 64 --reset\n"
            "  python -m scripts.ingest_vezeeta\n"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Records per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    _configure_logging(level=log_level)

    run_ingestion(
        batch_size=args.batch_size,
        do_reset=args.reset,
    )


if __name__ == "__main__":
    main()
