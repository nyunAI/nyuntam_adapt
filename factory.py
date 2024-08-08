from algorithm import Algorithm
from factory import Factory as BaseFactory, FactoryTypes
from nyuntam_adapt.utils.params_utils import AdaptParams, create_instance
from dataclasses import asdict

import os


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
        subtask = self.kwargs.get("subtask", None)
        adapt_params = create_instance(AdaptParams, self.kwargs)

        loaded_task = self.get_task(task, subtask)
        self.task = loaded_task(**asdict(adapt_params))

    def run(self) -> None:
        return self.task.adapt_model()

    def __call__(self):
        self.task.adapt_model()

    def get_task(self, task: str, subtask: str = None) -> Algorithm:
        from .tasks import export_task_modules

        loaded_task_module = export_task_modules(task, subtask)
        return loaded_task_module
