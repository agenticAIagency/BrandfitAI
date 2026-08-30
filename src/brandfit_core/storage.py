from __future__ import annotations

from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from minio import Minio

from brandfit_core.config import Settings, get_settings


class ObjectStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key.get_secret_value(),
            secure=self.settings.minio_secure,
        )
        self.bucket = self.settings.minio_bucket

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def object_exists(self, object_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except Exception:
            return False

    def presigned_upload(self, object_key: str, expires: timedelta = timedelta(hours=1)) -> str:
        self.ensure_bucket()
        return self.client.presigned_put_object(self.bucket, object_key, expires=expires)

    def put_bytes(self, object_key: str, data: bytes, content_type: str) -> None:
        self.ensure_bucket()
        stream = BytesIO(data)
        self.client.put_object(
            self.bucket, object_key, stream, length=len(data), content_type=content_type
        )

    def put_file(self, object_key: str, path: Path, content_type: str) -> None:
        self.ensure_bucket()
        self.client.fput_object(self.bucket, object_key, str(path), content_type=content_type)

    def get_file(self, object_key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, object_key, str(destination))

    def get_bytes(self, object_key: str) -> bytes:
        response = self.client.get_object(self.bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def put_stream(self, object_key: str, stream: BinaryIO, length: int, content_type: str) -> None:
        self.ensure_bucket()
        self.client.put_object(self.bucket, object_key, stream, length, content_type=content_type)
