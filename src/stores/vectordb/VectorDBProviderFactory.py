from .VectorDBEnums import VectorDBEnums
from .providers.QdrantDBProvider import QdrantDBProvider
import os

class VectorDBProviderFactory:
    """Factory that reads the centralised app config and returns
    a ready-to-use vector-database provider instance.

    Usage mirrors LLMProviderFactory::

        factory = VectorDBProviderFactory(config=config)
        db_client = factory.create(VectorDBEnums.QDRANT.value)
        db_client.connect()
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.database_dir = os.path.join(
            self.base_dir,
            "assets/database"
        )

    def create(self, provider: str):
        """Instantiate and return the requested VectorDB provider.

        Parameters
        ----------
        provider : str
            One of the ``VectorDBEnums`` values (e.g. ``"QDRANT"``).

        Returns
        -------
        VectorDBInterface | None
            A configured provider instance, or ``None`` for an
            unknown identifier.
        """

        database_path = os.path.join(
            self.database_dir, self.config.VECTOR_DB_PATH
        )

        if provider == VectorDBEnums.QDRANT.value:
            return QdrantDBProvider(
                db_path=database_path,
                db_url=getattr(self.config, "VECTOR_DB_URL", None),
                api_key=getattr(self.config, "VECTOR_DB_API_KEY", None),
                distance_method=getattr(
                    self.config, "VECTOR_DB_DISTANCE_METHOD", "cosine"
                ),
            )

        return None
