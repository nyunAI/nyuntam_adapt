from transformers import Seq2SeqTrainer
from typing import Dict
from nyuntam_adapt.core.base_trainer import BaseTrainer


class AdaptSeq2SeqTrainer(Seq2SeqTrainer, BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save = super(BaseTrainer, self)._save
        self.log = super(BaseTrainer, self).log

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)

    def log(self, logs: Dict[str, float]) -> None:
        super().log(logs)
