import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DocumentStore:
    """Tiny file-backed vector store: one JSON (chunks) + one .npy (embeddings) per document."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for meta_path in self.dir.glob("*.json"):
            emb_path = meta_path.with_suffix(".npy")
            if not emb_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["embeddings"] = np.load(emb_path)
            self._docs[meta["document_id"]] = meta

    def add(self, filename: str, chunks: List[str], embeddings: np.ndarray) -> Dict[str, Any]:
        doc_id = uuid.uuid4().hex[:12]
        meta: Dict[str, Any] = {
            "document_id": doc_id,
            "filename": filename,
            "num_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunks": chunks,
        }
        with self._lock:
            (self.dir / f"{doc_id}.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
            np.save(self.dir / f"{doc_id}.npy", embeddings)
            self._docs[doc_id] = {**meta, "embeddings": embeddings}
        return meta

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._docs.get(doc_id)

    def list(self) -> List[Dict[str, Any]]:
        return [
            {k: d[k] for k in ("document_id", "filename", "num_chunks", "created_at")}
            for d in self._docs.values()
        ]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            del self._docs[doc_id]
            for ext in (".json", ".npy"):
                (self.dir / f"{doc_id}{ext}").unlink(missing_ok=True)
            return True

    def search(self, doc_id: str, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, str, float]]:
        doc = self._docs[doc_id]
        scores = doc["embeddings"] @ query_embedding
        top = np.argsort(-scores)[:top_k]
        return [(int(i), doc["chunks"][i], float(scores[i])) for i in top]
