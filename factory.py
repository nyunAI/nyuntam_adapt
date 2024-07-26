from algorithm import Algorithm
from factory import Factory as BaseFactory, FactoryTypes
from nyuntam_adapt.utils import AdaptParams, create_instance
from dataclasses import asdict

import os
import sys


class AdaptFactory(BaseFactory):

    _type: FactoryTypes = FactoryTypes.ADAPT

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.kwargs = kwargs

        # Creating Directories
        os.makedirs(self.kwargs.get("OUTPUT_DIR"), exist_ok=True)
        os.makedirs(self.kwargs.get("LOGGING_PATH"), exist_ok=True)
        self.set_logger(self.kwargs.get("LOGGING_PATH"), stream_stdout=True)

        task = self.kwargs.get("TASK", "text_generation")
        subtask = self.kwargs.get("SUBTASK", None)
        adapt_params = create_instance(AdaptParams, self.kwargs)

        loaded_algorithm = self.get_algorithm(task, subtask)
        self.algorithm = loaded_algorithm(**asdict(adapt_params))

    def run(self) -> None:
        return self.algorithm.adapt_model()

    def __call__(self):
        self.algorithm.adapt_model()

    def get_algorithm(self, task: str, subtask: str = None) -> Algorithm:
        from .tasks import initialize_initialization

        loaded_algorithm = initialize_initialization(task, subtask)
        return loaded_algorithm
