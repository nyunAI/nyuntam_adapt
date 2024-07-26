from trl import SFTTrainer
from typing import Dict
from itertools import chain
from nyuntam_adapt.core.base_trainer import BaseTrainer


class AdaptSFTTrainer(SFTTrainer, BaseTrainer):
    def __init__(self, **kwargs):
        self.flash_attention = kwargs.pop("flash_attn")
        self.block_size = kwargs.pop("block_size")
        super().__init__(**kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)

    def log(self, logs: Dict[str, float]) -> None:
        super().log(logs)

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_len,
        formatting_func=None,
        remove_unused_columns=True,
        add_special_tokens=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                (
                    element[dataset_text_field]
                    if not use_formatting_func
                    else formatting_func(element)
                ),
                truncation=True,
                padding=False,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        if self.flash_attention:
            tokenized_dataset = tokenized_dataset.map(
                self.group_texts,
                batched=True,
            )

        return tokenized_dataset
