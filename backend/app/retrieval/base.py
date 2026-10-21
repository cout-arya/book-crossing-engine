"""
Base Retriever abstract class.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_doc_ids: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top-ranked documents for query.
        Returns a list of tuples: (isbn, score).
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of retrieval model."""
        pass

    @property
    def is_available(self) -> bool:
        """Indicates if model artifacts are loaded and ready."""
        return True
