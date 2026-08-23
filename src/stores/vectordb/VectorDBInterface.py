from abc import ABC, abstractmethod
from typing import Any, List, Optional

class VectorDBInterface(ABC):

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def is_collection_existed(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def list_all_collections(self) -> List:
        pass

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> dict:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        do_reset: bool = False,
    ):
        pass

    @abstractmethod
    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id: str = None,
    ):
        pass

    @abstractmethod
    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list = None,
        record_ids: list = None,
        batch_size: int = 50,
    ):
        pass

    @abstractmethod
    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 10,
        query_filter: Optional[Any] = None,
    ):
        """Perform a vector similarity search with optional metadata filtering.

        Parameters
        ----------
        collection_name : str
            The name of the target collection.
        vector : list
            The embedding vector to query against the collection.
        limit : int
            Maximum number of nearest matching records to return.
        query_filter : Any, optional
            Provider-specific filter object (e.g. Qdrant `models.Filter`)
            or structured filter condition for payload filtering.
        """
        pass

    def build_filter(
        self,
        area: Optional[str] = None,
        min_fee: Optional[int] = None,
        max_fee: Optional[int] = None,
        specialty: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """Construct a provider-specific filter object from search constraints.

        Subclasses should override this method to translate query constraints
        (area, fee range, specialty, etc.) into their native filter representations.

        Returns
        -------
        Any or None
            Provider-specific filter object, or None if no conditions are provided.
        """
        return None