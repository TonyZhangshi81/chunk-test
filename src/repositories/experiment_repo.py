from sqlalchemy.orm import Session

from models.experiment import Experiment


class ExperimentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, experiment: Experiment) -> Experiment:
        self.session.add(experiment)
        self.session.commit()
        self.session.refresh(experiment)
        return experiment
