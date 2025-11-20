from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def save_bytes(self, key: str, data: bytes) -> None:
        """Persist raw bytes to the backend under the given key."""

    @abstractmethod
    def save_json(self, key: str, obj: dict) -> None:
        """Persist a JSON-serializable object to the backend."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Retrieve raw bytes from the backend for the given key."""

    @abstractmethod
    def get_json(self, key: str) -> dict:
        """Retrieve a JSON object from the backend for the given key."""
