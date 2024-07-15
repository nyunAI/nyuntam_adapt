import warnings
import torch.nn as nn
from tqdm import tqdm
import gc
import logging
from dataclasses import dataclass
from .layer import Conv2d, Linear, SSFLayer, LayerNorm, BatchNorm2d
from nyuntam_adapt.algorithms.base_algorithm import BaseAlgorithm
from nyuntam_adapt.algorithms.utils import is_bnb_4bit_available, is_bnb_available
from timm.layers import norm
from nyuntam_adapt.tasks.timm_image_classification import TimmforImageClassification
from nyuntam_adapt.algorithms.utils import (
    _get_submodules,
)

logger = logging.getLogger(__name__)


if is_bnb_available():
    import bitsandbytes as bnb

    from .bnb import Linear8bitLt

if is_bnb_4bit_available():
    from .bnb import Linear4bit


@dataclass
class SSFConfig:
    peft_type: str = "SSF"
    target_modules: list = None


class SSFModel(BaseAlgorithm):
    def __init__(self, model, peft_config, adapter_name = None, model_type=None, auto_select_modules=True):
        super().__init__(model, peft_config)
        self.auto_select_modules = auto_select_modules
        if self.peft_config.target_modules is None:
            self.peft_config.target_modules = self.select_modules()

        print(
            "Default target modules existing in Model are ",
            self.peft_config.target_modules,
        )

        self.inject_module(self.base_model)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _create_and_replace(
        self,
        peft_config,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs,
    ) -> None:
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {"bias": bias}
        kwargs["loaded_in_8bit"] = optional_kwargs.pop("loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = optional_kwargs.pop("loaded_in_4bit", False)

        if not isinstance(target, nn.Embedding):
            new_module = self._create_new_module(peft_config, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(peft_config, target, **kwargs):
        bias = kwargs.pop("bias", False)
        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, nn.Conv2d):
            out_channels, in_channels = target.out_channels, target.in_channels
            groups = target.groups
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups,
                dim=target.out_channels,
                bias=bias,
                **kwargs,
            )
        elif isinstance(target, nn.LayerNorm):
            new_module = LayerNorm(dim=target.normalized_shape, **kwargs)

        elif isinstance(target, nn.BatchNorm2d):
            new_module = BatchNorm2d(dim=target.num_features, **kwargs)

        elif loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                target.in_features,
                target.out_features,
                dim=target.out_features,
                bias=bias,
                **eightbit_kwargs,
            )
        elif (
            loaded_in_4bit
            and is_bnb_4bit_available()
            and isinstance(target, bnb.nn.Linear4bit)
        ):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(
                target.in_features,
                target.out_features,
                dim=target.out_features,
                bias=bias,
                **fourbit_kwargs,
            )
        elif isinstance(target, nn.Linear):
            new_module = Linear(
                target.in_features,
                target.out_features,
                dim=target.out_features,
                bias=bias,
                **kwargs,
            )
        else:
            warnings.warn("Currently unsupported layer type: " + str(type(target)))
        return new_module

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            if child.bias is not None:
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        if hasattr(child, "running_mean"):
            new_module.running_mean = child.running_mean
            new_module.running_var = child.running_var

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "ssf_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def unload_and_merge(self, merge=True, progressbar: bool = False):
        key_list = [
            key for key, _ in self.base_model.named_modules() if "ssf" not in key
        ]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.base_model, key)
            except AttributeError:
                continue
            if isinstance(target, SSFLayer):
                if isinstance(target, nn.Conv2d):
                    new_module = nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                        groups=target.groups,
                    )
                elif isinstance(target, nn.LayerNorm):
                    new_module = nn.LayerNorm(normalized_shape=target.normalized_shape)

                elif isinstance(target, nn.BatchNorm2d):
                    new_module = nn.BatchNorm2d(num_features=target.num_features)

                elif isinstance(target, bnb.nn.Linear4bit):
                    bias = target.bias is not None
                    new_module = bnb.nn.Linear4bit(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        compute_dtype=target.compute_dtype,
                        compress_statistics=target.weight.compress_statistics,
                        quant_type=target.weight.quant_type,
                        device=target.weight.device,
                    )
                elif isinstance(target, bnb.nn.Linear8bitLt):
                    # raise ValueError(
                    #     (
                    #         "Currently merging of 8bit layers is not supported for Linear8bit."
                    #     )
                    # )
                    bias = target.bias is not None
                    new_module = bnb.nn.Linear8bitLt(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        has_fp16_weights=target.state.has_fp16_weights,
                        memory_efficient_backward=target.state.memory_efficient_backward,
                        threshold=target.state.threshold,
                        index=target.index,
                        device=target.weight.device,
                    )
                else:
                    bias = target.bias is not None
                    new_module = nn.Linear(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        device=target.weight.device,
                    ).to(target.weight.device)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)
                del target
                gc.collect()
