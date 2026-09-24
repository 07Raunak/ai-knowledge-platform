"""Raw file storage. Local disk here; the same interface maps 1:1 onto S3/GCS/Azure Blob
(put/get/delete by key) for a multi-instance deployment."""

import re
import shutil
from pathlib import Path


class LocalFileStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(filename: str) -> str:
        name = Path(filename).name  # strip any client-supplied directories
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:200] or "upload"

    def put(self, document_id: str, filename: str, data: bytes) -> str:
        folder = self.root / document_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / self._safe_name(filename)
        path.write_bytes(data)
        return path.relative_to(self.root).as_posix()

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def delete_document(self, document_id: str) -> None:
        shutil.rmtree(self.root / document_id, ignore_errors=True)
