"""
Dense Semantic Retriever using Sentence-Transformers (all-MiniLM-L6-v2).
Pre-computes and caches normalized embeddings for fast cosine similarity search.
"""

import pickle
import logging
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from backend.app.retrieval.base import BaseRetriever
from backend.app.config import (
    DENSE_MODEL_NAME,
    DENSE_EMBEDDINGS_PATH,
    ARTIFACTS_DIR
)

logger = logging.getLogger(__name__)

DENSE_METADATA_PATH = ARTIFACTS_DIR / "dense_metadata.pkl"


class DenseRetriever(BaseRetriever):
    def __init__(self, model_name: str = DENSE_MODEL_NAME):
        self.model_name_str = model_name
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.doc_isbns: List[str] = []
        self._is_available = False
        self._load_error: Optional[str] = None

    @property
    def model_name(self) -> str:
        return "Sentence Transformer"

    @property
    def is_available(self) -> bool:
        return self._is_available and self.embeddings is not None and len(self.doc_isbns) > 0

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_model(self) -> bool:
        """Lazily load SentenceTransformer model into memory."""
        if self.model is not None:
            return True
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name_str}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name_str)
            logger.info("SentenceTransformer model loaded.")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            return False

    def encode_corpus_resumable(
        self,
        texts: List[str],
        isbns: List[str],
        chunk_size: int = 10000,
        batch_size: int = 64
    ) -> bool:
        """
        Resumable batched corpus encoding with per-chunk checkpointing.
        Saves progress after every chunk so interrupted runs resume seamlessly.
        """
        if not self.load_model():
            return False

        checkpoint_dir = ARTIFACTS_DIR / "dense_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        total_docs = len(texts)
        num_chunks = (total_docs + chunk_size - 1) // chunk_size
        logger.info(f"Encoding {total_docs:,} documents across {num_chunks} chunks (chunk_size={chunk_size:,})...")

        all_chunk_embeddings = []
        all_chunk_isbns = []

        for chunk_idx in range(num_chunks):
            start_i = chunk_idx * chunk_size
            end_i = min(start_i + chunk_size, total_docs)
            chunk_npy = checkpoint_dir / f"chunk_{chunk_idx:04d}.npy"
            chunk_meta = checkpoint_dir / f"chunk_{chunk_idx:04d}_meta.pkl"

            if chunk_npy.exists() and chunk_meta.exists():
                logger.info(f"Loading cached checkpoint for chunk {chunk_idx + 1}/{num_chunks} ({start_i:,} to {end_i:,})...")
                chunk_emb = np.load(chunk_npy)
                with open(chunk_meta, "rb") as f:
                    chunk_isbns = pickle.load(f)
            else:
                logger.info(f"Encoding chunk {chunk_idx + 1}/{num_chunks} ({start_i:,} to {end_i:,})...")
                chunk_texts = texts[start_i:end_i]
                chunk_isbns = isbns[start_i:end_i]

                chunk_emb = self.model.encode(
                    chunk_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype(np.float32)

                # Save checkpoint immediately
                np.save(chunk_npy, chunk_emb)
                with open(chunk_meta, "wb") as f:
                    pickle.dump(chunk_isbns, f)

            all_chunk_embeddings.append(chunk_emb)
            all_chunk_isbns.extend(chunk_isbns)

        self.embeddings = np.vstack(all_chunk_embeddings)
        self.doc_isbns = all_chunk_isbns
        self._is_available = True
        self.save()
        logger.info(f"Resumable encoding complete: {len(self.embeddings):,} embeddings assembled.")
        return True

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_doc_ids: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return []

        if not str(query).strip():
            return []

        if self.model is None and not self.load_model():
            return []

        try:
            query_embedding = self.model.encode(
                str(query).strip(),
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)

            # Cosine similarity via dot product (since vectors are L2-normalized)
            scores = np.dot(self.embeddings, query_embedding)

            # Filter candidates if requested
            if filter_doc_ids is not None:
                valid_indices = np.array([idx for idx in range(len(scores)) if idx in filter_doc_ids], dtype=int)
                if len(valid_indices) == 0:
                    return []
                scores_subset = scores[valid_indices]
                actual_limit = min(limit, len(valid_indices))
                sub_part = np.argpartition(scores_subset, -actual_limit)[-actual_limit:]
                top_indices = valid_indices[sub_part]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            else:
                actual_limit = min(limit, len(scores))
                if len(scores) > actual_limit:
                    partition_idx = np.argpartition(scores, -actual_limit)[-actual_limit:]
                    top_indices = partition_idx[np.argsort(-scores[partition_idx])]
                else:
                    top_indices = np.argsort(-scores)

            results = [(self.doc_isbns[idx], float(scores[idx])) for idx in top_indices]
            return results

        except Exception as e:
            logger.error(f"Error during Dense semantic retrieval: {e}")
            return []

    def save(
        self,
        emb_path: Path = DENSE_EMBEDDINGS_PATH,
        meta_path: Path = DENSE_METADATA_PATH
    ):
        """Persist embeddings array and ISBN mappings."""
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
        with open(meta_path, "wb") as f:
            pickle.dump({"doc_isbns": self.doc_isbns, "model_name": self.model_name_str}, f)
        logger.info(f"Saved dense embeddings to {emb_path}")

    def encode_corpus(
        self,
        texts: List[str],
        isbns: List[str],
        batch_size: int = 64
    ) -> bool:
        """Wrapper maintaining backwards-compatibility."""
        return self.encode_corpus_resumable(texts, isbns, chunk_size=10000, batch_size=batch_size)

    def load(
        self,
        emb_path: Path = DENSE_EMBEDDINGS_PATH,
        meta_path: Path = DENSE_METADATA_PATH,
        preload_model: bool = True
    ) -> bool:
        """Load precomputed embeddings, metadata, and warm up model weights."""
        if not (emb_path.exists() and meta_path.exists()):
            self._load_error = "Dense embeddings artifacts not found."
            return False
        try:
            self.embeddings = np.load(emb_path)
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.doc_isbns = data["doc_isbns"]
                self.model_name_str = data.get("model_name", self.model_name_str)
            self._is_available = True
            logger.info(f"Loaded {len(self.embeddings):,} dense embeddings.")
            if preload_model:
                self.load_model()
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load dense embeddings: {e}")
            return False
