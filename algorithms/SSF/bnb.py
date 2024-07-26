import warnings
import bitsandbytes as bnb
import torch
import torch.nn as nn

from .layer import SSFLayer
from nyuntam_adapt.utils import (
    is_bnb_4bit_available,
    is_bnb_available,
    dequantize_8bit,
)


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, SSFLayer):
        def __init__(self, in_features, out_features, dim, **kwargs):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get(
                    "memory_efficient_backward", False
                ),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            SSFLayer.__init__(self, dim)
            self.weight.requires_grad = False

        def merge(self):
            if self.merged:
                warnings.warn("Already merged. Nothing to do.")
                return
            warnings.warn(
                "Merge lora module to 8-bit linear may get different generations due to rounding errors."
            )

            output = dequantize_8bit(self)
            w_data = output.to(self.ssf_scale.dtype).to(
                self.ssf_scale.device
            ) * self.ssf_scale.view(-1, 1)
            self.weight = bnb.nn.Int8Params(
                w_data.to("cpu"),
                requires_grad=False,
                has_fp16_weights=self.weight.has_fp16_weights,
            ).to(self.weight.device)
            if self.bias is None:
                factory_kwargs = {"device": self.weight.device, "dtype": torch.float16}
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, **factory_kwargs)
                )
            self.bias.data = self.ssf_shift + self.bias.data * self.ssf_scale
            self.state.reset_grads()
            self.merged = True

        def forward(self, x):
            result = super().forward(x)

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = self.ssf_scale.dtype
                if result.dtype != compute_dtype:
                    result = result.to(compute_dtype)

            result = self.ssf_ada(result)
            if requires_conversion:
                result = result.to(expected_dtype)
            return result


if is_bnb_4bit_available():

    class Linear4bit(bnb.nn.Linear4bit, SSFLayer):
        def __init__(self, in_features, out_features, dim, **kwargs):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            SSFLayer.__init__(self, dim)
            self.weight.requires_grad = False

        def merge(self):
            if self.merged:
                warnings.warn("Already merged. Nothing to do.")
                return

            warnings.warn(
                "Merge lora module to 4-bit linear may get different generations due to rounding errors."
            )
            kwargs = self.weight.__dict__

            w_data = bnb.functional.dequantize_4bit(
                self.weight.data, self.weight.quant_state
            ) * self.ssf_scale.view(-1, 1)
            self.weight = bnb.nn.Params4bit(
                w_data.to("cpu"), requires_grad=False, **kwargs
            ).to(self.weight.device)
            if self.bias is None:
                factory_kwargs = {"device": self.weight.device, "dtype": torch.float16}
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, **factory_kwargs)
                )
            self.bias.data = self.ssf_shift + self.bias.data * self.ssf_scale
            self.merged = True

        def forward(self, x):
            result = super().forward(x)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = self.ssf_scale.dtype
                if result.dtype != compute_dtype:
                    result = result.to(compute_dtype)

            result = self.ssf_ada(result)
            if requires_conversion:
                result = result.to(expected_dtype)
            return result
