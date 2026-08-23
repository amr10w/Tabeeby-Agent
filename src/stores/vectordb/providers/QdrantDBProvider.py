from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from typing import Any, List, Optional
import logging

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
    )
except ImportError:
    QdrantClient = None

import uuid


class QdrantDBProvider(VectorDBInterface):
    """Concrete Qdrant implementation of VectorDBInterface.

    Supports two deployment modes controlled at construction time:
    - **Local / Embedded**: pass ``db_path`` (a directory on disk).
    - **Remote / Cloud**:  pass ``db_url`` and optionally ``api_key``.

    If both ``db_path`` and ``db_url`` are supplied, ``db_url`` takes precedence.
    """

   

    def __init__(
        self,
        db_path: Optional[str] = None,
        db_url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_method: str = DistanceMethodEnums.COSINE.value,
    ):
        self.db_path = db_path
        self.db_url = db_url
        self.api_key = api_key

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.client = None
        self.logger = logging.getLogger(__name__)

        if QdrantClient is None:
            self.logger.error("qdrant-client package is not installed")

    # ── connection lifecycle ─────────────────────────────────────

    def connect(self):
        """Establish a connection to the Qdrant instance."""
        if QdrantClient is None:
            self.logger.error("qdrant-client package is not installed")
            return

        try:
            if self.db_url:
                # Remote / Cloud mode
                self.client = QdrantClient(
                    url=self.db_url,
                    api_key=self.api_key,
                )
            else:
                # Local / Embedded disk mode
                self.client = QdrantClient(path=self.db_path)

            self.logger.info("Connected to Qdrant successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def disconnect(self):
        """Close the Qdrant client connection."""
        if self.client:
            try:
                self.client.close()
                self.logger.info("Disconnected from Qdrant")
            except Exception as e:
                self.logger.error(f"Error disconnecting from Qdrant: {e}")
            finally:
                self.client = None

    # ── collection management ────────────────────────────────────

    def is_collection_existed(self, collection_name: str) -> bool:
        """Return True when the named collection already exists."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return False

        try:
            return self.client.collection_exists(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    def list_all_collections(self) -> List:
        """Return a list of all collection names."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return []

        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> dict:
        """Return metadata about the specified collection."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return {}

        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "collection_name": collection_name,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": str(info.status),
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self, collection_name: str):
        """Delete a collection by name."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return

        try:
            self.client.delete_collection(collection_name=collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection '{collection_name}': {e}")

    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        do_reset: bool = False,
    ):
        """Create a collection.  If *do_reset* is True, any existing
        collection with the same name is deleted first."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return

        try:
            if do_reset and self.is_collection_existed(collection_name):
                self.delete_collection(collection_name)

            if not self.is_collection_existed(collection_name):
                qdrant_distance = self.distance_method

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_size,
                        distance=qdrant_distance,
                    ),
                )
                self.logger.info(
                    f"Created collection '{collection_name}' "
                    f"(size={embedding_size}, distance={self.distance_method})"
                )
        except Exception as e:
            self.logger.error(f"Error creating collection '{collection_name}': {e}")

    # ── data insertion ───────────────────────────────────────────

    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id: str = None,
    ):
        """Insert a single point (vector + payload) into a collection."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return

        try:
            point_id = record_id or str(uuid.uuid4())

            payload = metadata.copy() if metadata else {}
            payload["text"] = text

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )

            self.client.upsert(
                collection_name=collection_name,
                points=[point],
            )
        except Exception as e:
            self.logger.error(f"Error inserting point into '{collection_name}': {e}")

    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list = None,
        record_ids: list = None,
        batch_size: int = 50,
    ):
        """Batch-insert multiple points, uploading in chunks of *batch_size*."""
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return

        try:
            points: List[PointStruct] = []
            for idx, (text, vector) in enumerate(zip(texts, vectors)):
                point_id = record_ids[idx] if record_ids else str(uuid.uuid4())

                payload = metadata[idx].copy() if metadata and idx < len(metadata) else {}
                payload["text"] = text

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # Upload in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )

            self.logger.info(
                f"Inserted {len(points)} points into '{collection_name}'"
            )
        except Exception as e:
            self.logger.error(
                f"Error batch-inserting into '{collection_name}': {e}"
            )

    # ── similarity search ────────────────────────────────────────

    def build_filter(
        self,
        area: Optional[str] = None,
        min_fee: Optional[int] = None,
        max_fee: Optional[int] = None,
        specialty: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """Construct a Qdrant `models.Filter` from search constraints.

        Parameters
        ----------
        area : str, optional
            Clinic area / neighborhood keyword to filter on.
        min_fee : int, optional
            Minimum consultation fee.
        max_fee : int, optional
            Maximum consultation fee.
        specialty : str, optional
            Specialty keyword to filter on.

        Returns
        -------
        models.Filter or None
            Populated Qdrant Filter object or None if no conditions exist.
        """
        if models is None:
            self.logger.error("qdrant_client.models not available")
            return None

        conditions = []

        if area and str(area).strip():
            conditions.append(
                models.FieldCondition(
                    key="address",
                    match=models.MatchValue(value=str(area).strip()),
                )
            )

        range_kwargs = {}
        if min_fee is not None:
            range_kwargs["gte"] = min_fee
        if max_fee is not None:
            range_kwargs["lte"] = max_fee

        if range_kwargs:
            conditions.append(
                models.FieldCondition(
                    key="fee",
                    range=models.Range(**range_kwargs),
                )
            )

        if specialty and str(specialty).strip():
            conditions.append(
                models.FieldCondition(
                    key="specialty",
                    match=models.MatchValue(value=str(specialty).strip()),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 10,
        query_filter: Optional[Any] = None,
    ):
        """Return the top-*limit* nearest neighbours for the given query vector.

        Parameters
        ----------
        collection_name : str
            Target Qdrant collection name.
        vector : list
            Query embedding vector.
        limit : int
            Maximum results to return (default 10).
        query_filter : models.Filter, optional
            Optional Qdrant filter object for deterministic metadata filtering.
        """
        if not self.client:
            self.logger.error("Qdrant client is not connected")
            return []

        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
            ).points

            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results
            ]
        except Exception as e:
            self.logger.error(
                f"Error searching in '{collection_name}': {e}"
            )
            return []
