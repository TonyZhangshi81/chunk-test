"""基于 MinIO 的对象存储集成封装。"""

import logging
from io import BytesIO

from minio import Minio

from config import Config


logger = logging.getLogger(__name__)


class StorageService:
    """负责文档二进制内容的上传与桶管理。"""

    def __init__(self, cfg: Config):
        self.config = cfg
        self.client = Minio(
            endpoint=cfg.MINIO_ENDPOINT,
            access_key=cfg.MINIO_ACCESS_KEY,
            secret_key=cfg.MINIO_SECRET_KEY,
            secure=cfg.MINIO_SECURE,
        )

    def ensure_bucket(self) -> None:
        """在目标桶不存在时自动创建。"""
        # 这个存在性检查成本很低，能让新环境下的上传命令保持幂等。
        if not self.client.bucket_exists(self.config.MINIO_BUCKET):
            logger.info("Creating MinIO bucket name=%s", self.config.MINIO_BUCKET)
            self.client.make_bucket(self.config.MINIO_BUCKET)

    def upload_bytes(self, object_name: str, content: bytes, content_type: str | None = None) -> str:
        """上传原始字节流，并返回对象的 etag。"""
        self.ensure_bucket()
        logger.info(
            "Uploading object to MinIO bucket=%s object_name=%s size=%s",
            self.config.MINIO_BUCKET,
            object_name,
            len(content),
        )
        result = self.client.put_object(
            self.config.MINIO_BUCKET,
            object_name,
            # 借助 BytesIO 可以直接从内存流上传，避免额外创建临时文件。
            data=BytesIO(content),
            length=len(content),
            content_type=content_type or "application/octet-stream",
        )
        return result.etag
