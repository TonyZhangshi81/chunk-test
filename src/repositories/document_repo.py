from sqlalchemy import select
from sqlalchemy.orm import Session

from models.document import Document


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, document: Document) -> Document:
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def get_by_id(self, document_id: str) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        return self.session.execute(stmt).scalar_one_or_none()
