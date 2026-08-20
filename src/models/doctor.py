"""Pydantic validation models for Vezeeta doctor records.

This module defines the ``DoctorRecord`` model used by the ingestion pipeline
to validate, type-cast, and transform raw CSV rows into clean, structured data
ready for vector embedding and Qdrant payload storage.

Architectural Notes for Future ``search_doctors`` Tool
------------------------------------------------------
1. **Hybrid Vector + Filter Queries**:
   The agent tool should embed user symptom text via ``embed_text()`` and query
   Qdrant using ``query_points()`` with both the symptom vector *and* a
   ``models.Filter`` containing ``FieldCondition`` entries:

   - ``address`` → ``models.FieldCondition(key="address",
     match=models.MatchValue(value="Nasr City"))``
   - ``fee`` → ``models.FieldCondition(key="fee",
     range=models.Range(gte=100, lte=500))``

2. **Handling Loose / Missing Filters**:
   Build the ``must`` conditions list dynamically. Only append an ``address``
   condition when the user explicitly mentions an area, and only append a
   ``fee`` range when the user specifies a budget. An empty ``must`` list
   (or omitting the ``query_filter`` entirely) yields a pure vector
   similarity search — the pipeline should never fail due to missing filters.

3. **Compact Payload Serialization**:
   When returning search results to the LLM agent, extract only the fields
   the agent needs to formulate a response:
   ``name``, ``specialty``, ``address``, ``fee``, ``reviews_count``,
   ``profile_url``.
   This keeps the agent's context window lean and avoids wasting tokens on
   large ``about_doctor`` or ``symptoms_text`` blobs.
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


# ── Helper ───────────────────────────────────────────────────────────────────

def _safe_int(value, *, clamp_negative: bool = True) -> Optional[int]:
    """Convert *value* to a non-negative ``int``, or ``None`` on failure.

    Handles:
    - ``None``, ``""``, whitespace-only strings → ``None``
    - ``float("nan")`` / ``math.nan``            → ``None``
    - Negative numbers (clamped to ``0`` when *clamp_negative* is ``True``)
    - Strings like ``"600"`` or ``"3.0"``        → ``600`` / ``3``

    Parameters
    ----------
    value :
        The raw value to convert.
    clamp_negative : bool
        If ``True``, negative values are clamped to ``0`` instead of being
        returned as-is.

    Returns
    -------
    int | None
    """
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except (ValueError, TypeError):
            return None
        if math.isnan(numeric):
            return None
        result = int(numeric)
    else:
        try:
            result = int(value)
        except (ValueError, TypeError):
            return None

    if clamp_negative and result < 0:
        result = 0

    return result


def _clean_str(value) -> str:
    """Return a stripped string, collapsing ``None`` / ``NaN`` to ``""``."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


# ── Model ────────────────────────────────────────────────────────────────────

class DoctorRecord(BaseModel):
    """Validated representation of a single Vezeeta doctor record.

    All string fields are whitespace-stripped; numeric fields are safely
    cast to ``Optional[int]`` with NaN / negative handling.

    The ``page`` column from the CSV is intentionally excluded — it is a
    web-scraping artifact with no clinical or operational relevance.

    Usage
    -----
    >>> row = {"name": " Dr. X ", "fee": "300", "Speciality": "Cardiology", ...}
    >>> doc = DoctorRecord.from_csv_row(row)
    >>> doc.build_semantic_text()
    'Cardiology. ...'
    >>> doc.to_qdrant_payload()
    {'name': 'Dr. X', 'fee': 300, ...}
    """

    # ── Identity & Clinical Profile ──────────────────────────────
    name: str = ""
    description: str = ""
    specialty: str = ""
    about_doctor: str = ""
    symptoms_text: str = ""
    subspecialties_text: str = ""

    # ── Location & Logistics ─────────────────────────────────────
    address: str = ""
    fee: Optional[int] = None
    reviews_count: Optional[int] = None
    waiting_time_min: Optional[int] = None

    # ── External Links ───────────────────────────────────────────
    profile_url: Optional[str] = None
    image_url: Optional[str] = None

    # ── Validators ───────────────────────────────────────────────

    @field_validator(
        "name", "description", "specialty", "about_doctor",
        "symptoms_text", "subspecialties_text", "address",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, v):
        return _clean_str(v)

    @field_validator("profile_url", "image_url", mode="before")
    @classmethod
    def _strip_optional_strings(cls, v):
        cleaned = _clean_str(v)
        return cleaned if cleaned else None

    @field_validator("fee", "reviews_count", "waiting_time_min", mode="before")
    @classmethod
    def _cast_ints(cls, v):
        return _safe_int(v)

    # ── Factory ──────────────────────────────────────────────────

    @classmethod
    def from_csv_row(cls, row: dict) -> "DoctorRecord":
        """Construct a ``DoctorRecord`` from a raw CSV row dictionary.

        Maps the CSV column ``Speciality`` (note the original spelling) to
        the model field ``specialty``.  The ``page`` column is silently
        dropped.

        Parameters
        ----------
        row : dict
            A dictionary keyed by CSV column headers.

        Returns
        -------
        DoctorRecord
        """
        return cls(
            name=row.get("name"),
            description=row.get("description"),
            specialty=row.get("Speciality"),       # CSV uses "Speciality"
            about_doctor=row.get("about_doctor"),
            symptoms_text=row.get("symptoms_text"),
            subspecialties_text=row.get("subspecialties_text"),
            address=row.get("address"),
            fee=row.get("fee"),
            reviews_count=row.get("reviews_count"),
            waiting_time_min=row.get("waiting_time_min"),
            profile_url=row.get("profile_url"),
            image_url=row.get("image_url"),
        )

    # ── Semantic Text Builder ────────────────────────────────────

    def build_semantic_text(self) -> str:
        """Build a high-signal semantic string for vector embedding.

        The text is constructed by concatenating clinical fields in
        **prioritised order**:

        1. **Specialty** (highest priority — anchors the search space)
        2. **Subspecialties** (refines within the specialty)
        3. **Symptoms** (primary match surface for patient complaints)
        4. **Description** (professional title / credentials)
        5. **About doctor** (extended clinical background)

        Location (``address``) and cost (``fee``) are **intentionally
        excluded** — embedding them would introduce noise into symptom
        similarity matching.  These fields are stored in the Qdrant
        payload for deterministic filtering instead.

        Returns
        -------
        str
            A single string suitable for passing to ``embed_text()``.
            An empty string is returned if all clinical fields are blank.
        """
        parts: list[str] = []

        if self.specialty:
            parts.append(f"Specialty: {self.specialty}")
        if self.subspecialties_text:
            parts.append(f"Subspecialties: {self.subspecialties_text}")
        if self.symptoms_text:
            parts.append(f"Symptoms: {self.symptoms_text}")
        if self.description:
            parts.append(f"Title: {self.description}")
        if self.about_doctor:
            parts.append(f"Background: {self.about_doctor}")

        return ". ".join(parts)

    # ── Payload Serializer ───────────────────────────────────────

    def to_qdrant_payload(self) -> dict:
        """Serialize all metadata into a flat dict for Qdrant storage.

        Every raw and cleaned field is included so that the downstream
        ``search_doctors`` tool can:

        - Return any needed field to the agent without a second lookup.
        - Apply Qdrant ``Filter`` conditions on ``address`` (KEYWORD)
          and ``fee`` (INTEGER) for hybrid vector + filter queries.

        Returns
        -------
        dict
        """
        return {
            "name": self.name,
            "description": self.description,
            "specialty": self.specialty,
            "about_doctor": self.about_doctor,
            "symptoms_text": self.symptoms_text,
            "subspecialties_text": self.subspecialties_text,
            "address": self.address,
            "fee": self.fee,
            "reviews_count": self.reviews_count,
            "waiting_time_min": self.waiting_time_min,
            "profile_url": self.profile_url,
            "image_url": self.image_url,
        }
