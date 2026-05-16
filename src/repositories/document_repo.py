"""上传文档相关的持久化辅助封装。"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.document import Document


logger = logging.getLogger(__name__)


class DocumentRepository:
    """管理文档对象的创建与查询。"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, document: Document) -> Document:
        """保存新文档，并刷新 ORM 对象上的数据库回填字段。"""
        logger.info("Creating document record id=%s file_name=%s", document.id, document.file_name)
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def get_by_id(self, document_id: str) -> Document | None:
        """按主键查询单个文档。"""
        logger.debug("Loading document by id=%s", document_id)
        stmt = select(Document).where(Document.id == document_id)
        return self.session.execute(stmt).scalar_one_or_none()
