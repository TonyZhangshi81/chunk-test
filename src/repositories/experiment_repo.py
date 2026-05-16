"""实验结果持久化辅助封装。"""

import logging

from sqlalchemy.orm import Session

from models.experiment import Experiment


logger = logging.getLogger(__name__)


class ExperimentRepository:
    """负责保存完整的检索实验记录。"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, experiment: Experiment) -> Experiment:
        """保存一条实验结果记录。"""
        logger.info(
            "Creating experiment id=%s document_id=%s chunk_type=%s",
            experiment.id,
            experiment.document_id,
            experiment.chunk_type,
        )
        self.session.add(experiment)
        self.session.commit()
        self.session.refresh(experiment)
        return experiment
