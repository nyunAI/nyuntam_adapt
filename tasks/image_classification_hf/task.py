import torch
import numpy as np
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
    BitsAndBytesConfig,
)
from nyuntam_adapt.tasks.image_classification_timm import TimmforImageClassification
from nyuntam_adapt.core.custom_model import (
    prepare_custom_image_model_support,
    prepare_timm_model_support,
    CustomModelLoadError,
)
from nyuntam_adapt.utils.task_utils import TimmModelConfig, ModelLoadingError
from nyuntam_adapt.core.base_task import BaseTask
from nyuntam_adapt.utils.task_utils import ModelLoadingError


class ModelLoadingError(RuntimeError):
    """Exception for custom model loading errors."""

    pass


# TODO Create Folder structure to be followed and suggest Augmentations based on heuristics and dataset


class ImageClassification(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.framework = kwargs["MODEL_ARGS"].get("FRAMEWORK", "Huggingface")
        self.timm_model = False
        if self.framework == "Timm":
            self.timm_model = True
            self.model_path = kwargs["MODEL_ARGS"].get("MODEL_PATH", None)
            self.model_type = kwargs["MODEL_ARGS"].get("MODEL_TYPE", None)
            self.img_processor_path = kwargs["MODEL_ARGS"].get(
                "IMAGE_PROCESSOR_PATH", "facebook/convnext-tiny-224"
            )

        self.flash_attention = kwargs.get("flash_attention2", False)
        self.model_args = {}

    def transform(self, example_batch):
        inputs = self.image_processor(
            [x for x in example_batch[img_key]], return_tensors="pt"
        )

        inputs["labels"] = example_batch[label_key]
        return inputs

    def prepare_dataset(self, dataset, processor):
        self.image_processor = self.get_image_processor()

        return self.dataset.with_transform(self.transform)

    def get_image_processor(self):
        if self.timm_model:
            return AutoImageProcessor.from_pretrained(
                self.img_processor_path
            )  # Update the changes in the config of image processor

        try:
            return AutoImageProcessor.from_pretrained(self.local_model_path)
        except:
            return AutoImageProcessor.from_pretrained(self.model_path)

    def prepare_label_mappings(self):
        global label_key, img_key
        label_key = list(self.dataset["train"].features.keys())[-1]
        img_key = list(self.dataset["train"].features.keys())[-2]

        labels = self.dataset["train"].features[label_key].names
        label2id, id2label = {}, {}
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        return label2id, id2label

    def compute_metrics(self, p):
        """Computes accuracy on a batch of predictions"""

        # TODO : Avoid intialization of metric everytime
        metric = load_metric("accuracy")
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    def prepare_model(self):
        use_bnb = False
        if self.bnb_config["USE_4BIT"]["load_in_4bit"]:
            bnb_config_inputs = self.bnb_config["USE_4BIT"]
            use_bnb = True
        elif self.bnb_config["USE_8BIT"]["load_in_8bit"]:
            bnb_config_inputs = self.bnb_config["USE_8BIT"]
            use_bnb = True

        if use_bnb:
            bnb_config = BitsAndBytesConfig(**bnb_config_inputs)
            self.model_args["quantization_config"] = bnb_config
        label2id, id2label = self.prepare_label_mappings()

        if self.local_model_path is not None:
            if self.timm_model:
                try:
                    base_model = TimmforImageClassification(
                        self.model_path, num_classes=len(label2id)
                    )
                    model = prepare_timm_model_support(
                        self.local_model_path, base_model
                    )
                except:
                    try:
                        model = prepare_timm_model_support(self.local_model_path, None)
                    except Exception as e:
                        raise CustomModelLoadError(
                            f"Error while loading custom model from {self.local_model_path}"
                        ) from e

                cfg = model.model_timm.pretrained_cfg
                model_config = TimmModelConfig(self.model_type, **cfg)
                model_config.label2id = label2id
                model_config.id2label = id2label

            else:
                try:
                    model_config = AutoConfig.from_pretrained(self.local_model_path)
                except:
                    model_config = AutoConfig.from_pretrained(self.model_path)
                model_config.num_labels = len(label2id)
                model_config.label2id = label2id
                model_config.id2label = id2label
                model = prepare_custom_image_model_support(
                    self.local_model_path,
                    self.model_path,
                    "HuggingFace",
                    model_config,
                    num_gpu=self.num_gpu,
                    device=self.device,
                    use_bnb=use_bnb,
                    use_flash_attention_2=self.flash_attention,
                    **self.model_args,
                )

        else:
            if self.timm_model:
                model = TimmforImageClassification(
                    self.model_path, num_classes=len(label2id)
                )
                cfg = model.model_timm.pretrained_cfg
                model_config = TimmModelConfig(self.model_type, **cfg)
                model_config.label2id = label2id
                model_config.id2label = id2label

            else:
                model_config = AutoConfig.from_pretrained(self.model_path)
                model_config.num_labels = len(label2id)
                model_config.label2id = label2id
                model_config.id2label = id2label
                try:
                    if self.num_gpu == 1:
                        try:
                            model = AutoModelForImageClassification.from_pretrained(
                                self.model_path,
                                config=model_config,
                                ignore_mismatched_sizes=True,
                                use_flash_attention_2=self.flash_attention,
                                **self.model_args,
                            ).to(self.device)
                        except:
                            model = AutoModelForImageClassification.from_pretrained(
                                self.model_path,
                                config=model_config,
                                ignore_mismatched_sizes=True,
                                use_flash_attention_2=self.flash_attention,
                                **self.model_args,
                            )
                    else:
                        model = AutoModelForImageClassification.from_pretrained(
                            self.model_path,
                            config=model_config,
                            ignore_mismatched_sizes=True,
                            use_flash_attention_2=self.flash_attention,
                            # device_map="auto",
                            **self.model_args,
                        )
                except Exception as e:
                    raise ModelLoadingError(
                        f"Model ({self.model_path}) cannot be loaded due to : {e}, \n Maybe wrong name?"
                    ) from e

        return model, self.image_processor, model_config

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
