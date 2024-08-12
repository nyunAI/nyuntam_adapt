import os
import shutil
import sys
from abc import abstractmethod
from glob import glob
from pathlib import Path

import torch
from transformers import TrainingArguments
from datasets import load_dataset


import logging
from nyuntam_adapt.core.base_algorithm import BaseAlgorithm
from nyuntam_adapt.core.dataset import Dataset
from nyuntam_adapt.utils.algorithm_utils import get_peft_state_dict

from nyuntam_adapt.utils.task_utils import (
    PEFT_TYPE_TO_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
)

from nyuntam_adapt.trainers import AdaptSeq2SeqTrainer, AdaptSFTTrainer
from nyuntam_adapt.core.base_trainer import BaseTrainer


class BaseTask(object):
    """Base Trainer class that defines the structure of each adaptation
    algorithm implemented in this library. Every new tasks is
    expected to directly use or overwrite the template functions defined below.

    The root command to invoke the training  of any model is
    .adapt_model(). Thus, it is required that all algorithms complete this
    template function and use it as the first point of invoking the model
    compression process.

    Parameters
    ----------
        kwargs (object): A yaml safe loaded file with information like cuda_id, log_dir, device, etc.
    """

    def __init__(self, **kwargs):
        self.cuda_id = kwargs.get("cuda_id", "0")
        self.num_gpu = torch.cuda.device_count()
        if self.num_gpu == 1:
            self.device = torch.device("cuda:" + self.cuda_id)
        else:
            self.device = torch.device("cuda")
        self.logging_path = kwargs.get("LOGGING_PATH")
        self.logger = logging.getLogger(__name__)
        self.user_folder = kwargs.get("USER_FOLDER", "user_data")

        # write code to accept from test.yaml file
        self.output_dir = kwargs.get("OUTPUT_DIR", os.getcwd())
        self.overwrite_output_dir = kwargs.get("OVERWRITE_OUTPUT_DIR", False)

        # check if output directory exists or not
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Read dataset arguments
        self.dataset_name = kwargs["DATASET_ARGS"].get("DATASET", "cifar10")
        self.dataset_yaml = kwargs["DATASET_ARGS"].get("DATASET_YAML", None)
        self.custom_dataset_path = kwargs["DATASET_ARGS"].get(
            "CUSTOM_DATASET_PATH", None
        )
        self.format_string = kwargs["DATASET_ARGS"].get("FORMAT_STRING", None)
        self.format_names = kwargs["DATASET_ARGS"].get("FORMAT_NAMES", None)

        if self.custom_dataset_path is not None and not os.path.exists(
            self.custom_dataset_path
        ):
            self.logger.info(self.custom_dataset_path is not None)
            raise FileNotFoundError(
                f"Dataset path {self.custom_dataset_path} not found"
            )

        self.dataset_version = kwargs["DATASET_ARGS"].get("DATA_VERSION", "1.0")
        self.train_val_split = kwargs["DATASET_ARGS"].get("DATA_SPLIT", 0.20)
        self.train_dir = kwargs["DATASET_ARGS"].get("TRAIN_DIR", None)
        self.val_dir = kwargs["DATASET_ARGS"].get("VAL_DIR", None)
        self.test_dir = kwargs["DATASET_ARGS"].get("TEST_DIR", None)
        self.dataset_config = kwargs["DATASET_ARGS"].get("DATASET_CONFIG", None)
        self.max_train_samples = kwargs["DATASET_ARGS"].get("MAX_TRAIN_SAMPLES", 1000)
        self.max_eval_samples = kwargs["DATASET_ARGS"].get("MAX_EVAL_SAMPLES", 1000)
        self.classes_yaml = kwargs["DATASET_ARGS"].get("CLASSES_YAML", None)
        self.dataset_format = kwargs["DATASET_ARGS"].get("DATASET_FORMAT", None)
        self.dataset_text_field = kwargs["dataset_text_field"]
        # Model Arguments
        self.model_name = kwargs["MODEL_ARGS"].get("MODEL", "")
        self.model_path = kwargs["MODEL_ARGS"].get("MODEL_PATH", "")
        self.model_version = kwargs["MODEL_ARGS"].get("MODEL_VERSION", "1.0")
        self.model_checkpoint = kwargs["MODEL_ARGS"].get("MODEL_CHECKPOINT", None)
        self.local_model_path = kwargs["MODEL_ARGS"].get("LOCAL_MODEL_PATH", None)
        self.local_model_path = (
            Path(self.local_model_path) if self.local_model_path else None
        )
        self.fsdp = kwargs.get("FSDP", False)
        self.fsdp = kwargs.get("FSDP", False)

        self.task = kwargs.get("TASK", None)
        self.hf_trainer = None

        # Traning Arguments
        self.num_workers = kwargs["TRAINING_ARGS"].get(
            "NUM_WORKERS", 4
        )  # Data loading from pytorch
        self.seed = kwargs["TRAINING_ARGS"].get("SEED", 42)

        self.training_bool = kwargs["TRAINING_ARGS"].get("DO_TRAIN", True)
        self.eval_bool = kwargs["TRAINING_ARGS"].get("DO_EVAL", True)
        self.batch_size = kwargs["TRAINING_ARGS"].get("BATCH_SIZE", 16)
        self.num_epochs = kwargs["TRAINING_ARGS"].get(
            "EPOCHS", 10
        )  # default epochs for fine tuning
        self.steps = kwargs["TRAINING_ARGS"].get("STEPS", 500)
        self.optimizer_name = kwargs["TRAINING_ARGS"].get("OPTIMIZER", "sgd")
        self.lr = float(kwargs["TRAINING_ARGS"].get("LR", 1e-5))
        self.scheduler_type = kwargs["TRAINING_ARGS"].get("SCHEDULER_TYPE", "linear")
        self.weight_decay = float(kwargs["TRAINING_ARGS"].get("WEIGHT_DECAY", 0.000))
        self.beta1 = float(kwargs["TRAINING_ARGS"].get("BETA1", 0.9))
        self.beta2 = float(kwargs["TRAINING_ARGS"].get("BETA2", 0.999))

        # MMDET specific arguments
        self.t_max = kwargs["TRAINING_ARGS"].get("T_MAX") or 100
        self.milestones = kwargs["TRAINING_ARGS"].get("MILESTONES") or [5, 10]
        self.gamma = kwargs["TRAINING_ARGS"].get("GAMMA") or 0.1
        self.begin = kwargs["TRAINING_ARGS"].get("BEGIN") or 0
        self.end = kwargs["TRAINING_ARGS"].get("END") or self.num_epochs
        self.warmup = kwargs["TRAINING_ARGS"].get("WARMUP") or True
        self.warmup_iters = kwargs["TRAINING_ARGS"].get("WARMUP_ITERS") or 50
        self.warmup_ratio = kwargs["TRAINING_ARGS"].get("WARMUP_RATIO") or 0.1

        self.adam_epsilon = float(kwargs["TRAINING_ARGS"].get("ADAM_EPS", 1e-8))
        self.gradient_accumulation_steps = kwargs["TRAINING_ARGS"].get(
            "GRADIENT_ACCUMULATION_STEPS", 1
        )
        self.gradient_checkpointing = kwargs["TRAINING_ARGS"].get(
            "GRADIENT_CHECKPOINTING", False
        )
        self.interval_strategy = kwargs["TRAINING_ARGS"].get(
            "INTERVAL", "epoch"
        )  # epoch, steps ,no
        self.interval_steps = kwargs["TRAINING_ARGS"].get("INTERVAL_STEPS", 100)
        self.total_checkpoints = kwargs["TRAINING_ARGS"].get("NO_OF_CHECKPOINTS", 5)
        self.mixed_precision = kwargs["TRAINING_ARGS"].get("FP16", False)
        self.resume_from_checkpoint = kwargs["TRAINING_ARGS"].get(
            "RESUME_FROM_CHECKPOINT", False
        )
        self.group_by_length = kwargs["TRAINING_ARGS"].get("GROUP_BY_LENGTH", False)
        self.remove_unused_columns = kwargs["TRAINING_ARGS"].get(
            "REMOVE_UNUSED_COLUMNS", True
        )

        # Bnb Arguments
        self.bnb_config = kwargs["BNB_CONFIG"]

        # Fine-Tuning Arguments
        self.last_linear_tuning = kwargs["FINE_TUNING_ARGS"].get(
            "LAST_LAYER_TUNING", False
        )
        self.full_fine_tuning = kwargs["FINE_TUNING_ARGS"].get(
            "FULL_FINE_TUNING", False
        )
        self.merge_adapter = kwargs.get("MERGE_ADAPTER", True)

        # Peft Config
        self.peft_method = kwargs.get("PEFT_METHOD", None)
        if self.peft_method is not None:
            config_name = f"{self.peft_method}_CONFIG"
            config_dict = kwargs[config_name]

            self.peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[self.peft_method](
                **config_dict
            )

        self.library = kwargs.get("Library", "Huggingface")
        self.save_method = kwargs.get("SAVE_METHOD", "full_torch_model")

        self.auto_select_modules = kwargs.get("auto_select_modules", True)

        self.task_trainer = {
            "generation": AdaptSFTTrainer,
            "Seq2Seq_tasks": AdaptSeq2SeqTrainer,
        }

        self.additional_training_args = {}

        if self.library.lower() == "huggingface":
            self.training_args = TrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=self.overwrite_output_dir,
                do_train=self.training_bool,
                do_eval=self.eval_bool,
                per_device_train_batch_size=self.batch_size,
                evaluation_strategy=(
                    self.interval_strategy if self.eval_bool else "no"
                ),
                num_train_epochs=self.num_epochs,
                learning_rate=self.lr,
                weight_decay=self.weight_decay,
                adam_beta1=self.beta1,
                adam_beta2=self.beta2,
                adam_epsilon=self.adam_epsilon,
                lr_scheduler_type=self.scheduler_type,
                warmup_ratio=self.warmup_ratio,
                warmup_steps=self.warmup_iters,
                optim=self.optimizer_name,
                logging_steps=self.interval_steps,
                save_strategy=self.interval_strategy if self.eval_bool else "no",
                dataloader_num_workers=self.num_workers,
                save_steps=self.interval_steps,
                eval_steps=self.interval_steps,
                save_total_limit=self.total_checkpoints,
                save_safetensors=True,
                fp16=self.mixed_precision,
                remove_unused_columns=self.remove_unused_columns,  # For NLP tasks
                push_to_hub=False,
                # report_to='tensorboard',
                load_best_model_at_end=True,
                torch_compile=False,
                resume_from_checkpoint=self.resume_from_checkpoint,
                seed=self.seed,
                group_by_length=self.group_by_length,
                logging_dir=self.output_dir,
                log_level="info",
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                gradient_checkpointing=self.gradient_checkpointing,
                ddp_find_unused_parameters=True,
            )

        self.logger.info(f"Experiment Arguments: {kwargs}")

    def _load_dataset(self):
        """
        Returns:
            dataset: The preprocessed dataset.
        """

        if self.custom_dataset_path:
            self.dataset_name = None
        else:
            self.logger.info(f"CUSTOM DATA IS NOT GIVEN, DATA IS {self.dataset_name}")
            self.custom_dataset_path = None
            try:
                self.dataset = load_dataset(self.dataset_name)
            except:
                self.dataset = load_dataset(self.dataset_name, self.dataset_config)

            return self.dataset
        if self.task == "image_classification":
            self.dataset = load_dataset(
                "imagefolder", data_dir=self.custom_dataset_path
            )
            return self.dataset
        else:
            self.dataset = Dataset.from_name_or_path(
                dataset_path=(
                    Path(self.custom_dataset_path) if self.custom_dataset_path else None
                ),
                dataset_name=self.dataset_name,
                dataset_subset=None,
                save_dir=Path(f"{self.user_folder}/datasets"),
                format_string=self.format_string,
                format_keys=self.format_names,
                new_text_column=self.dataset_text_field,
            )

        return self.dataset.load_calibration_data()

    # @abstractmethod
    def prepare_dataset(self, dataset, processor):
        """
        Prepare a dataset for training which does transformations and preprocessing.
        """
        return dataset

    def prepare_model(self):
        pass

    def trainer(
        self,
        model,
        training_args,
        dataset,
        compute_metrics,
        processor,
        collate_fn=None,
    ):
        # TODO : Add custom trainer for a custom loss functions or use callback

        if training_args.do_train:
            if "train" not in dataset:
                raise ValueError("--do_train requires a train dataset")

        if len(dataset.keys()) > 1:
            validation_key = list(dataset.keys())[1]

        if training_args.do_eval:
            if "validation" not in dataset and "test" not in dataset:
                raise ValueError("--do_eval requires a validation dataset")

        # Training
        if training_args.do_train:
            selected_trainer = self.task_trainer.get(self.task, BaseTrainer)
            trainer = selected_trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset[validation_key] if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=processor,
                data_collator=collate_fn,
                # logging_path = self.logging_path,
                **self.additional_training_args,
            )

            train_results = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint
            )
            # trainer.save_model()

            trainer.log_metrics("train", train_results.metrics)
            self.logger.info(f"train : {train_results.metrics}")
            trainer.save_metrics("train", train_results.metrics)
            self.hf_trainer = trainer
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            self.logger.info(f"val : {metrics}")
            trainer.save_metrics("eval", metrics)

    @abstractmethod
    def collate_fn(self, *args):
        pass

    @abstractmethod
    def compute_metrics(self, *args):
        pass

    def add_peft_modules(self, model_type):
        if self.peft_method is not None:
            self.peft_model = PEFT_TYPE_TO_MODEL_MAPPING[self.peft_method](
                self.model,
                self.peft_config,
                adapter_name="default",
                model_type=model_type,
                auto_select_modules=self.auto_select_modules,
            )
            self.logger.info(
                f"Adding peft modules {PEFT_TYPE_TO_MODEL_MAPPING[self.peft_method]}"
            )

        # Only last layer tuning, no PEFT
        elif self.last_linear_tuning:
            self.merge_adapter = False
            self.peft_model = BaseAlgorithm(self.model, None, self.logging_path)
            self.peft_model.freeze_model()

        # Full Fine Tuning
        else:
            self.merge_adapter = False
            self.peft_model = BaseAlgorithm(self.model, None, self.logging_path)
            self.logger.info(f"Fine-Tuning all layers")

        # Fine-tuning the classifier layer along with PEFT
        if self.last_linear_tuning:
            self.config = getattr(self, "config", None)
            try:
                self.peft_model.unfreeze_last_layer(self.model, self.config)
            except Exception as e:
                try:
                    # In DDP wrapped models, models are wrapped inside MMDistributedParallel.module
                    self.peft_model.unfreeze_last_layer(self.model.module, self.config)
                except Exception as e:
                    self.logger.exception(e)

        self.peft_model.calc_trainable_params()

    def save_peft_modules(self):
        save_path = os.path.join(self.output_dir, "peft_modules.pth")
        state_dict = get_peft_state_dict(self.model)
        torch.save(state_dict, save_path)

    def sampling_dataset(self):
        if self.max_train_samples is not None:
            self.dataset["train"] = (
                self.dataset["train"]
                .shuffle(seed=self.training_args.seed)
                .select(range(self.max_train_samples))
            )
        if len(self.dataset.keys()) > 1:
            validation_key = list(self.dataset.keys())[1]

            if self.max_eval_samples is not None:
                self.dataset[validation_key] = (
                    self.dataset[validation_key]
                    .shuffle(seed=self.training_args.seed)
                    .select(range(self.max_eval_samples))
                )

    def save_huggingface(self):
        try:
            if self.peft_method is not None:
                self.peft_model.save_pretrained(self.output_dir)
            else:
                self.model.save_pretrained(self.output_dir)
            self.logger.info(f"Saving pretrained model at: {self.output_dir}")
            return True
        except:
            try:
                if self.peft_method is not None:
                    self.peft_model.generation_config.do_sample = True
                    self.peft_model.save_pretrained(self.output_dir)
                else:
                    self.model.generation_config.do_sample = True
                    self.model.save_pretrained(self.output_dir)
                self.logger.info(f"Saving pretrained model at: {self.output_dir}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save HuggingFace model: {e}")
                return False

    def save_state_dict(self):
        try:
            state_dict_path = os.path.join(
                self.output_dir, "merged_model_state_dict.pth"
            )
            if self.peft_method is not None:
                torch.save(self.peft_model.state_dict(), state_dict_path)
            else:
                torch.save(self.model.state_dict(), state_dict_path)
            self.logger.info(f"Saving model state_dict at: {state_dict_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save PyTorch state_dict: {e}")
            return False

    def save_full_torch_model(self):
        try:
            model_path = os.path.join(self.output_dir, "merged_model.pth")
            try:
                if self.peft_method is not None:
                    torch.save(self.peft_model.model, model_path)
                else:
                    torch.save(self.model.model, model_path)
            except:
                if self.peft_method is not None:
                    torch.save(self.peft_model, model_path)
                else:
                    torch.save(self.model, model_path)
            self.logger.info(f"Saving model at: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save PyTorch model: {e}")
            return False

    def save_safetensors(self):
        try:
            safetensors_path = os.path.join(
                self.output_dir, "merged_model_safetensors.safetensors"
            )
            from safetensors.torch import save_file

            if self.peft_method is not None:
                state_dict = self.peft_model.state_dict()
                save_file(state_dict, safetensors_path)
            else:
                state_dict = self.model.state_dict()
                save_file(state_dict, safetensors_path)

            self.logger.info(f"Saving model as SafeTensors at: {safetensors_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save SafeTensors model: {e}")
            return False

    def save_adapted_model(self):
        """
        Save the model in different formats (HuggingFace, PyTorch, and SafeTensors) to the output directory.
        Returns True if at least one saving operation was successful, False otherwise.
        """
        SAVE_MAP = {
            "save_pretrained": self.save_huggingface,
            "state_dict": self.save_state_dict,
            "full_torch_model": self.save_full_torch_model,
            "safetensors": self.save_safetensors,
        }

        if SAVE_MAP[self.save_method]():
            self.logger.info(f"MODEL SAVED USING : {self.save_method}")
            return True
        else:
            for method in SAVE_MAP.values():
                if method():
                    self.logger.info(f"MODEL SAVED USING : {method}")
                    return True

        return False

    def delete_checkpoint_dir(self):
        try:
            for file_name in os.listdir(self.output_dir):
                path = os.path.join(self.output_dir, file_name)
                if "checkpoint" in file_name:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
        except Exception as e:
            self.logger.info("**COULD NOT DELETE CHECKPOINT FILES")

    def adapt_model(self):
        """Template function to be overwritten for each model adaptation
        method."""

        # Load_data

        self.dataset = self._load_dataset()
        self.sampling_dataset()
        preprocessed_dataset = self.prepare_dataset(
            dataset=self.dataset, processor=None
        )

        # Model
        self.model, self.processor, self.config = self.prepare_model()
        self.add_peft_modules(model_type=self.config.model_type)

        # Training and evaluation
        self.trainer(
            self.model,
            self.training_args,
            preprocessed_dataset,
            self.compute_metrics,
            self.processor,
            self.collate_fn,
        )
        if self.merge_adapter:
            try:
                if self.fsdp:
                    state_dict = self.hf_trainer.accelerator.get_state_dict(
                        self.peft_model
                    )
                    from safetensors.torch import save_file

                    save_file(
                        state_dict,
                        os.path.join(self.output_dir, "model_state_dict.safetensors"),
                    )
                    if os.getenv("LOCAL_RANK") == "0":
                        from safetensors import safe_open

                        tensors = {}
                        with safe_open(
                            os.path.join(
                                self.output_dir, "model_state_dict.safetensors"
                            ),
                            framework="pt",
                            device=0,
                        ) as f:
                            for k in f.keys():
                                tensors[k] = f.get_tensor(k)

                        # To save memory, we can delete everything but the LoRA layers:
                        for k in tensors:
                            if ("lora" not in k) and ("ssf" not in k):
                                tensors[k] = None

                        # Load the base model again and add random adapters
                        self.model, self.processor, self.config = self.prepare_model()
                        self.add_peft_modules(model_type=self.config.model_type)

                        new_sd = self.peft_model.state_dict()
                        for k in new_sd:
                            if ("lora" in k) or ("ssf" in k):
                                new_sd[k] = tensors[k]
                        self.peft_model.load_state_dict(new_sd)
                        self.logger.info(self.peft_model.device)
                        self.peft_model.unload_and_merge()
                        self.logger.info(" FSDP PEFT modules merging done.")
                    elif os.getenv("LOCAL_RANK") == "1":
                        sys.exit(0)

                else:
                    self.peft_model.unload_and_merge()
                    self.logger.info("PEFT modules merging done.")
            except Exception as e:
                self.logger.info(
                    f"Following exception happened while attempting to merge : {e}"
                )
                self.logger.info(f"Saving PEFT modules independently")
                self.save_peft_modules()
        assert self.save_adapted_model(), "Failed to save model."
        self.delete_checkpoint_dir()

        self.logger.info("Model Adaptation Completed")
        return __name__
