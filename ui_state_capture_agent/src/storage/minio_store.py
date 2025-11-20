from __future__ import annotations

import io
import json
import mimetypes
from typing import Optional

from minio import Minio

from src.config import settings
from src.storage.base import StorageBackend

_storage_instance: Optional["MinioStorageBackend"] = None


class MinioStorageBackend(StorageBackend):
    def __init__(self) -> None:
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_endpoint.startswith("https://"),
        )

        if not self.client.bucket_exists(settings.minio_bucket):
            self.client.make_bucket(settings.minio_bucket)

    def save_bytes(self, key: str, data: bytes) -> None:
        content_type, _ = mimetypes.guess_type(key)
        buffer = io.BytesIO(data)
        self.client.put_object(
            settings.minio_bucket,
            key,
            data=buffer,
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def save_json(self, key: str, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.save_bytes(key, data)

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(settings.minio_bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_json(self, key: str) -> dict:
        data = self.get_bytes(key)
        return json.loads(data.decode())


def get_storage() -> MinioStorageBackend:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = MinioStorageBackend()
    return _storage_instance
