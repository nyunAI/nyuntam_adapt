import sys
import torch
import torch.nn as nn
from typing import Any
from collections import deque
from abc import abstractmethod
import logging

import bitsandbytes as bnb
from nyuntam_adapt.utils.algorithm_utils import get_submodules


class BaseAlgorithm(nn.Module):
    """
    Base class for Adapter , LORA and Prompt Tuning methods

    Args :

    model : Transformer Pretrained Model   # Does this needs to be a object of `PreTrainedModel` class
    peft_config : PEFT_CONFIG

    """

    def __init__(
        self,
        model,
        peft_config,
        adapter_name=None,
        logging_path=None,
        auto_select_modules=True,
    ):
        super(BaseAlgorithm, self).__init__()
        self.logging_path = logging_path
        self.logger = logging.getLogger(__name__)

        self.base_model = model
        self.peft_config = peft_config
        self.last_layer_ultralytics = None

    def select_modules(self):
        # This function selects the modules automatically to add PEFT layer

        # Commenting out nn.Embedding because it is not supported in DoRA, SSF.
        module_list = []
        for names, modules in self.base_model.named_modules():

            if (
                isinstance(modules, nn.Linear)
                or isinstance(modules, nn.Conv2d)
                # or isinstance(modules, nn.Embedding)
            ):
                module_list.append(names)

        # Removing the last layer name from the module list

        # Adding a try except to check whether model has model.module (MMDistributedDataParallel)
        try:
            last_layer = self.find_last_layer(self.base_model.module)
            last_layer[:] = [f"module.{item}" for item in last_layer]
        except Exception as e:
            self.logger.warning("Could not find model.module, Trying model")
            try:
                last_layer = self.find_last_layer(self.base_model)
            except Exception as e:
                self.logger.exception(e)
        for items in last_layer:
            if items in module_list:
                module_list.remove(items)

        return module_list

    def freeze_model(self):
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False

        self.logger.info("Freezed Model")

    def initialize_modules(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_peft_modules(self, path):
        loaded_path = torch.load(path)
        self.base_model.load_state_dict(loaded_path, strict=False)

    def calc_trainable_params(self, **kwargs: Any):
        """
        Prints the number of trainable parameters in the model.
        """
        # Reference :- https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L408C9-L408C36

        trainable_params = 0
        all_param = 0
        param_list = []
        trainable_list = []
        for _, param in self.base_model.named_parameters():
            param_list.append(_)
            if param.requires_grad:
                trainable_list.append(_)
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += param.numel()
        self.logger.info(f"ALL PARAMETERS ARE : {param_list}")
        self.logger.info(f"TRAINABLE PARAMETERS ARE : {trainable_list}")
        self.logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def find_last_layer(self, model):
        """
        This function finds the last layer in a model and returns it as a list.
        DETR, YOLOX, RTMDet - It returns the modules in bbox_head
        SEGNEXT - It returns the modules in decode_head
        RTMO - It returns the modules in head
        For Other models (LLM, TIMM) - It used the find classifier function to find the classifier layer
        """
        from mmdet.models.dense_heads.detr_head import DETRHead
        from mmdet.models.dense_heads.yolox_head import YOLOXHead
        from mmdet.models.dense_heads.rtmdet_head import RTMDetSepBNHead
        from mmseg.models.decode_heads.ham_head import LightHamHead
        from mmpose.models.heads.hybrid_heads.rtmo_head import RTMOHead
        from nyuntam_adapt.tasks.image_classification_timm import (
            TimmforImageClassification,
        )

        last_layer_list = []
        if hasattr(model, "bbox_head"):
            # MMDET Models
            detection_head_obj = getattr(model, "bbox_head")
            if isinstance(detection_head_obj, (DETRHead, YOLOXHead, RTMDetSepBNHead)):

                for name, p in model.bbox_head.named_modules():
                    last_layer_list.append("bbox_head." + name)
            else:
                raise AttributeError(
                    "Supported bbox_heads are : DETRHead, YOLOXHead, RTMDetSepBNHead"
                )  # create custom exception name

        # For SEGNEXT
        elif hasattr(model, "decode_head"):
            seg_head_obj = getattr(model, "decode_head")
            if isinstance(seg_head_obj, LightHamHead):
                for name, p in model.decode_head.named_modules():
                    last_layer_list.append("decode_head." + name)

        # For MMPOSE RTMO MODEL
        elif hasattr(model, "head"):
            pose_det_obj = getattr(model, "head")
            final_layer = getattr(model.head, "dcc")
            if isinstance(pose_det_obj, RTMOHead):
                for name, p in model.head.named_modules():
                    last_layer_list.append("head." + name)

        elif isinstance(model, TimmforImageClassification):
            self.config = getattr(self, "config", None)
            classifier_obj = self.find_classifier(model, self.config)
            last_layer_list.append(classifier_obj)
            model_timm = getattr(model, "model_timm")
            classifier = getattr(model_timm, classifier_obj)

        else:
            self.config = getattr(self, "config", None)
            last_layer = self.find_classifier(model, self.config)
            last_layer_list.append(last_layer)

        return last_layer_list

    def unfreeze_last_layer(self, model, config):
        if hasattr(model, "bbox_head"):
            from mmdet.models.dense_heads.detr_head import DETRHead
            from mmdet.models.dense_heads.yolox_head import YOLOXHead
            from mmdet.models.dense_heads.rtmdet_head import RTMDetSepBNHead

            detection_head_obj = getattr(model, "bbox_head")
            if isinstance(detection_head_obj, (DETRHead, YOLOXHead, RTMDetSepBNHead)):

                for name, p in model.bbox_head.named_parameters():
                    p.requires_grad = True
                    self.logger.info(f"Unfreezed {name}")
            else:

                raise AttributeError(
                    "Supported bbox_heads are: DETRHead, YOLOXHead, RTMDetSepBNHead"
                )  # create custom exception name
            return
        if hasattr(model, "decode_head"):
            from mmseg.models.decode_heads.ham_head import LightHamHead
            from mmpose.models.heads.hybrid_heads.rtmo_head import RTMOHead

            seg_head_obj = getattr(model, "decode_head")
            if isinstance(seg_head_obj, LightHamHead):
                for name, p in model.decode_head.named_parameters():
                    p.requires_grad = True
                    self.logger.info(f"Unfreezed {name}")
            return

        # For MMPOSE RTMO MODEL
        if hasattr(model, "head"):
            pose_det_obj = getattr(model, "head")
            if isinstance(pose_det_obj, RTMOHead):
                for name, p in model.head.named_parameters():
                    p.requires_grad = True
                    self.logger.info(f"Unfreezed {name}")
            return
        from nyuntam_adapt.tasks.image_classification_timm import (
            TimmforImageClassification,
        )

        if isinstance(model, TimmforImageClassification):
            classifier_obj = self.find_classifier(model, config)
            model_timm = getattr(model, "model_timm")
            classifier = getattr(model_timm, classifier_obj)

        else:
            classifier_obj = self.find_classifier(model, config)
            classifier = getattr(model, classifier_obj)

        for param in classifier.parameters():
            param.requires_grad = True

        self.logger.info(f"Unfreezed {classifier_obj}")

    def find_classifier(self, model, config=None):
        from mmdet.models.detectors.base import BaseDetector
        from nyuntam_adapt.tasks.image_classification_timm import (
            TimmforImageClassification,
        )

        if isinstance(model, TimmforImageClassification):
            model = model.model_timm

        if isinstance(model, BaseDetector):
            return None

        last_layer = [
            "head",
            "classifier",
            "pooler",
            "cls",
            "fc",  # timm models support ('fc')
            "qa_outputs",  # QA models
            "dec_bbox_head",
            "conv",
        ]
        dd = deque(model.named_modules(), maxlen=1)
        last_element = dd.pop()
        try:
            obj = getattr(model, last_element[0])
            names = [last_element[0]]  # ["model", last_element[0]]

            if isinstance(obj, nn.Linear):  # ViT
                obj_name = ".".join(names)
                return obj_name

            # Bert based models
            for name, _ in model.named_children():  # Distil-BERT
                if name in last_layer:
                    obj = getattr(model, name)
                    names = [name]
                    if isinstance(obj, nn.Linear):
                        obj_name = ".".join(names)
                        return obj_name

            # For multiple linear layers
            if isinstance(obj, nn.Sequential):
                for n, p in obj.named_children():
                    names.append(n)
                    if isinstance(obj, nn.Linear):
                        obj_name = ".".join(names)
                        return obj_name
                    names.pop()

            found = any(s in obj.__class__.__name__.lower() for s in last_layer)
            while found:
                found = any(s in obj.__class__.__name__.lower() for s in last_layer)

                if hasattr(obj, "decoder"):
                    obj_decoder = getattr(obj, "decoder")
                    if (
                        isinstance(obj_decoder, nn.Linear)
                        and obj_decoder.out_features == config.vocab_size
                    ):
                        names.append("decoder")
                        if isinstance(obj, nn.Linear):
                            obj_name = ".".join(names)
                            return obj_name
                try:
                    name, module = next(obj.named_children())
                except StopIteration:
                    break

                obj = getattr(obj, name)
                names.append(name)
                if isinstance(obj, nn.Linear):
                    obj_name = ".".join(names)
                    return obj_name
        except:
            try:
                obj = getattr(model, last_element[0].split(".")[0])
                names = [last_element[0]]  # ["model", last_element[0]]

                # for resnet
                name = str(last_element[0].split(".")[0])

                if name in last_layer:
                    return name

                if isinstance(obj, nn.Linear):  # ViT
                    obj_name = ".".join(names)
                    return obj_name

                # Bert based models
                for name, _ in model.named_children():  # Distil-BERT
                    if name in last_layer:
                        obj = getattr(model, name)
                        names = [name]
                        if isinstance(obj, nn.Linear):
                            obj_name = ".".join(names)
                            return obj_name

                found = any(s in obj.__class__.__name__.lower() for s in last_layer)
                while found:
                    found = any(s in obj.__class__.__name__.lower() for s in last_layer)

                    if hasattr(obj, "decoder"):
                        obj_decoder = getattr(obj, "decoder")
                        if (
                            isinstance(obj_decoder, nn.Linear)
                            and obj_decoder.out_features == config.vocab_size
                        ):
                            names.append("decoder")
                            if isinstance(obj, nn.Linear):
                                obj_name = ".".join(names)
                                return obj_name
                    try:
                        name, module = next(obj.named_children())
                    except StopIteration:
                        break

                    obj = getattr(obj, name)
                    names.append(name)
                    if isinstance(obj, nn.Linear):
                        obj_name = ".".join(names)
                        return obj_name
            except:
                raise AttributeError("Classifier layer not found")

    def _mark_only_adapters_as_trainable(self, adapter_name, skip_list) -> None:
        for n, p in self.base_model.named_parameters():
            if adapter_name not in n:
                # Skips group conv layers which are currently not supported for LoRA, DoRA
                if n.rsplit(".", 1)[0] not in skip_list:
                    p.requires_grad = False
                else:
                    p.required_grad = True

    @abstractmethod
    def _check_target_module_exists(peft_config, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optionnal_kwargs: Any,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overriden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            **optionnal_kwargs (`dict`):
                The optional keyword arguments to pass to deal with particular cases (e.g. 8bit, 4bit quantization)
        """
        ...

    @staticmethod
    def is_exact_substring_at_end(substring, full_string):
        full_string_list = full_string.split(".")
        substring_list = substring.split(".")

        # Compare the end parts of the lists
        if full_string_list[-len(substring_list) :] == substring_list:
            return True
        else:
            return False

    def _generate_target_modules(self, model):
        key_list = []
        supported_classes = [
            nn.Conv2d,
            nn.Linear,
            nn.Embedding,
            bnb.nn.Linear8bitLt,
            bnb.nn.Linear4bit,
        ]
        for name, _ in model.named_modules():
            last_token = name
            if last_token in self.peft_config.target_modules:
                key_list.append(name)
        return key_list

    def inject_module(self, model):
        key_list = self._generate_target_modules(model)
        skip_list = []
        for key in key_list:
            parent, target, target_name = get_submodules(model, key)
            optional_kwargs = {
                "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
                "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
                "current_key": key,
            }

            # To avoid grouped convolutions
            if isinstance(target, nn.Conv2d):
                if target.groups > 1:
                    skip_list.append(key)
                    for param in target.parameters():
                        param.requires_grad = True
                    continue

            self._create_and_replace(
                self.peft_config,
                target,
                target_name,
                parent,
                **optional_kwargs,
            )

        self._mark_only_adapters_as_trainable(
            self.peft_config.peft_type.lower(), skip_list
        )

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        self.base_model.forward(*args, **kwargs)
