from algorithm import AdaptionAlgorithm
from factory import Factory as BaseFactory, FactoryTypes
from nyuntam_adapt.utils.params_utils import AdaptParams, create_instance
from dataclasses import asdict
from typing import Optional
import os


class AdaptFactory(BaseFactory):

    _type: FactoryTypes = FactoryTypes.ADAPT

    @property
    def task(self):
        return self._instance

    @task.setter
    def task(self, instance: AdaptionAlgorithm) -> None:
        self._instance = instance

    def __pre_init__(self, *args, **kwargs):
        kw = args[0]
        task = kw.get("TASK", "text_generation")
        subtask = kw.get("subtask", None)
        self._algorithm_class = self.get_task(task, subtask)
        super().__pre_init__(*args, **kwargs)

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

    def get_algorithm(self, name: Optional[str] = None) -> AdaptionAlgorithm:
        return self._algorithm_class

    def __call__(self):
        self.task.adapt_model()

    def get_task(self, task: str, subtask: str = None) -> AdaptionAlgorithm:
        from .tasks import export_task_modules

        loaded_task_module = export_task_modules(task, subtask)
        return loaded_task_module
