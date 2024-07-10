from .Adapt_Trainer import AdaptTrainer
from transformers import Seq2SeqTrainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


class AdaptSeq2SeqTrainer(Seq2SeqTrainer, AdaptTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save = super(AdaptTrainer, self)._save
        self.log = super(AdaptTrainer, self).log

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)

    def log(self, logs: Dict[str, float]) -> None:
        super().log(logs)
