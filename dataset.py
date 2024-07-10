from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path
from typing import List, Optional, Union
import gc
import torch

import logging

logger = logging.getLogger(__name__)


@dataclass
class Dataset:

    dataset_name_or_path: Optional[str] = field(default=None, metadata={
        "help": "Name of the dataset."})
    dataset_subset: Optional[str] = field(default=None, metadata={
        "help": "Name of the dataset subset."})
    format_keys: Optional[str] = field(default="text", metadata={
        "help": "Name of the text column(s). If multiple columns, separate by comma."})
    split: Optional[str] = field(default="train", metadata={
        "help": "Split of the dataset."})
    format_string: Optional[str] = field(default=None, metadata={
        "help": "Format of the dataset."})
    new_text_column: Optional[str] = field(default="text", metadata={
        "help": "Name of the new text column in data"})
    dataset_config: Optional[str] =field(default=None,metadata={
        "help": "Config for translation"
    })

      

    @classmethod
    def set_new_text_column(cls, new_text_column: str):
        cls.new_text_column = new_text_column

    @staticmethod
    def text_columns(format_keys: str) -> List[str]:
        "Return the text columns as a list."
        return format_keys.split(",")

    @classmethod
    def from_name_or_path(
        cls,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
        dataset_subset: Optional[str],
        save_dir: Path,
        split: str = "train",
        format_string: Optional[str] = None,
        format_keys: str = "text",
        new_text_column: str = "text"
    ):
        if (dataset_path is None) \
            or (not dataset_path.exists()) \
            or (len(list(dataset_path.iterdir())) == 0):
            path: Union[str, Path] = dataset_name
        else:
            path: Union[str, Path] = dataset_path
            
        if isinstance(path, str):
            try:
                ds = load_dataset(
                    path=path, name=dataset_subset)
            except Exception as e:
                raise RuntimeError(
                    f"Error while loading dataset `{path}` with subset `{dataset_subset}` and split `{split}`") from e
        elif isinstance(path, Path):
            try:
                ds = load_from_disk(path)
            except Exception as e:
                raise RuntimeError(
                    f"Error while loading dataset from {path}") from e

        text_columns = cls.text_columns(format_keys)
        if len(text_columns) > 1:
            assert format_string is not None, "When multiple text columns are used, `format_string` is required."
        for ds_split in ds.keys():
            for k in text_columns:
                assert k in ds[ds_split][0].keys(
                ), f"The key '{k}' is not present in the dataset. The dataset keys(columns) are {ds[split][0].keys()}"

        cls._format_and_save(ds, text_columns, format_string, save_dir)
        del ds
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return cls(
            dataset_name_or_path=save_dir,
            dataset_subset=dataset_subset,
            format_keys=cls.new_text_column,
            split=split,
            format_string=format_string
        )

    @classmethod
    def _format_dataset(cls, ds: DatasetDict, text_columns: List[str], format_string: str):
        if format_string is None:
            # since no format string is present, we will use the first text column as the new text column
            cls.set_new_text_column(text_columns[0])
            return ds

        try:
            for split in ds.keys():
                fs = format_string.format(
                    **{k: ds[split][0][k] for k in text_columns})
        except KeyError as e:
            raise KeyError(
                f"The format_string expects a key that is not present in {text_columns}: {e}. The dataset keys(columns) are {ds[0].keys()}")
        except Exception as e:
            raise RuntimeError(
                f"Error while formatting the format_string with the dataset columns.") from e

        logger.info(
            f"Found format string.\n"
            f"The dataset will be loaded using the format string formatted with {text_columns}.\n"
            f"Following is a sample of the formatted dataset:\n{fs}.\n"
        )

        new_text_column = cls.new_text_column
        for split in ds.keys():
            ds[split] = ds[split].map(lambda x: {new_text_column: format_string.format(
                **{k: x[k] for k in text_columns})})
        return ds

    @classmethod
    def _format_and_save(cls, ds: DatasetDict, text_columns: List[str], format_string: str, save_dir: Path):
        ds = cls._format_dataset(ds, text_columns, format_string)
        cls._save_dataset(ds, save_dir)

    @staticmethod
    def _save_dataset(ds: DatasetDict, save_dir: Path):
        try:
            ds.save_to_disk(save_dir)
        except Exception as e:
            raise RuntimeError(
                f"Error while saving the dataset to {save_dir}") from e

        logger.info(f"Dataset saved at {save_dir}.")

    # ===== custom loaders =====
        
    def load_calibration_data(self):
        ds = load_from_disk(self.dataset_name_or_path)
        return ds

    # def get_flap_dataloader(self, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    #     from lib.data import TokenizerWrapper
    #     import random
        
    #     ds = load_from_disk(self.dataset_name_or_path)
    #     traindata = ds[self.split]
    #     random.seed(seed)
    #     trainloader = []
    #     for _ in range(nsamples):
    #         while True:
    #             i = random.randint(0, len(traindata) - 1)
    #             trainenc = tokenizer(traindata[i][self.format_keys], return_tensors='pt')
    #             if trainenc.input_ids.shape[1] > seqlen:
    #                 break
    #         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #         j = i + seqlen
    #         inp = trainenc.input_ids[:, i:j]
    #         tar = inp.clone()
    #         tar[:, :-1] = -100
    #         trainloader.append((inp, tar))
        
    #     # TODO: add logic to support testloader when eval is enabled (ref - FLAP.lib.data.get_loaders)
    #     return trainloader
    
    # def get_llmpruner_examples(self, nsamples=128, seed=0, seqlen=128, tokenizer=None):
    #     # ref - https://github.com/nyunAI/LLM-Pruner/blob/2e7872de2e0e5dfe6c1b8972accf1b251f57a086/LLMPruner/datasets/example_samples.py#L8

    #     import random

    #     ds = load_from_disk(self.dataset_name_or_path)
    #     traindata = ds[self.split]
    #     random.seed(seed)
    #     trainloader = []
    #     for _ in range(nsamples):
    #         while True:
    #             i = random.randint(0, len(traindata) - 1)
    #             trainenc = tokenizer(traindata[i][self.format_keys], return_tensors='pt')
    #             if trainenc.input_ids.shape[1] > seqlen:
    #                 break
    #         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #         j = i + seqlen
    #         inp = trainenc.input_ids[:, i:j]
    #         trainloader.append(inp)
        
    #     return torch.cat(trainloader, dim = 0)