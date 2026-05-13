from io import BytesIO

from minio import Minio

from config import Config


class StorageService:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.client = Minio(
            endpoint=cfg.MINIO_ENDPOINT,
            access_key=cfg.MINIO_ACCESS_KEY,
            secret_key=cfg.MINIO_SECRET_KEY,
            secure=cfg.MINIO_SECURE,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.config.MINIO_BUCKET):
            self.client.make_bucket(self.config.MINIO_BUCKET)

    def upload_bytes(self, object_name: str, content: bytes, content_type: str | None = None) -> str:
        self.ensure_bucket()
        result = self.client.put_object(
            self.config.MINIO_BUCKET,
            object_name,
            data=BytesIO(content),
            length=len(content),
            content_type=content_type or "application/octet-stream",
        )
        return result.etag
