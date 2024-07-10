from typing import Optional, Tuple, Union
import warnings
import torch.nn as nn
import torch
import torch.nn.functional as F

# from transformers.models.llama import modeling_llama as llama

# from transformers.models.llama import modeling_llama as llama


class SSFLayer:
    def __init__(self, dim, dtype=torch.float32) -> None:
        self.ssf_scale = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.ssf_shift = nn.Parameter(torch.zeros(dim, dtype=dtype))

        self.ssf_scale = self.ssf_scale.to(self.weight.device)
        self.ssf_shift = self.ssf_shift.to(self.weight.device)

        self.merged = False
        nn.init.normal_(self.ssf_scale, mean=1, std=0.02)
        nn.init.normal_(self.ssf_shift, std=0.02)

    def ssf_ada(self, x):
        self.to(x.device)
        try:
            if x.dtype == torch.float16:
                self.ssf_scale = self.ssf_scale.half()
                self.ssf_shift = self.ssf_shift.half()
            else:
                self.ssf_scale = self.ssf_scale.float()
                self.ssf_shift = self.ssf_shift.float()
        except:
            # For non distributed training :
            # TypeError: cannot assign 'torch.cuda.HalfTensor' as parameter
            #'ssf_scale' (torch.nn.Parameter or None expected)
            pass
            # self.ssf_scale = self.ssf_scale.to(x.dtype)
            # self.ssf_shift = self.ssf_shift.to(x.dtype)

        self.ssf_scale = nn.Parameter(self.ssf_scale.to(self.weight.device))
        self.ssf_shift = nn.Parameter(self.ssf_shift.to(self.weight.device))
        if x.shape[-1] == self.ssf_scale.shape[0]:
            return x * self.ssf_scale + self.ssf_shift
        elif x.shape[1] == self.ssf_scale.shape[0]:
            return x * self.ssf_scale.view(1, -1, 1, 1) + self.ssf_shift.view(
                1, -1, 1, 1
            )
        else:
            raise ValueError(
                "the input tensor shape does not match the shape of the scale factor."
            )

    def merge(self) -> None:
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.weight.device == self.ssf_scale.device:
            pass
        else:
            self.ssf_scale.to(self.weight.device)

        if self.weight.shape == self.ssf_scale.shape:  # layernorm
            self.weight.data = self.weight.data * self.ssf_scale.to(self.weight.device)

        elif (
            self.weight.shape[0] == self.ssf_scale.shape[0]
            and len(self.weight.shape) == 2
        ):  # linear
            self.weight.data = self.weight.data * self.ssf_scale.to(
                self.weight.device
            ).view(-1, 1)

        elif (
            self.weight.shape[0] == self.ssf_scale.shape[0]
            and len(self.weight.shape) == 4
        ):  # conv
            self.weight.data = self.weight.data * self.ssf_scale.to(
                self.weight.device
            ).view(-1, 1, 1, 1)

        else:
            raise ValueError(
                "the input tensor shape does not match the shape of the scale factor."
            )
        if self.bias is None:
            factory_kwargs = {"device": self.weight.device, "dtype": torch.float32}
            if isinstance(self, Conv2d):
                self.bias = nn.Parameter(
                    torch.zeros(self.out_channels, **factory_kwargs)
                )
            else:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, **factory_kwargs)
                )

        self.bias.data = self.ssf_shift.to(
            self.bias.device
        ) + self.bias.data * self.ssf_scale.to(self.bias.device)

        self.merged = True


class Linear(nn.Linear, SSFLayer):
    def __init__(self, in_features, out_features, dim, **kwargs):
        dtype = kwargs.pop("dtype", torch.float16)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SSFLayer.__init__(self, dim, dtype=dtype)
        self.weight.requires_grad = False

    def forward(self, x):
        x = x.to(self.weight.dtype)
        result = super().forward(x)
        result = self.ssf_ada(result)
        return result


class Conv2d(nn.Conv2d, SSFLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        dim: int = None,
        **kwargs,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=kwargs["bias"],
        )
        SSFLayer.__init__(self, dim)
        self.weight.requires_grad = False

    def forward(self, x):
        result = super().forward(x)
        result = self.ssf_ada(result)
        return result


class LayerNorm(nn.LayerNorm, SSFLayer):
    def __init__(self, dim, **kwargs):
        nn.LayerNorm.__init__(self, dim)
        SSFLayer.__init__(self, dim)
        self.weight.requires_grad = False

    def forward(self, x):
        result = super().forward(x)
        result = self.ssf_ada(result)
        return result


class BatchNorm2d(nn.BatchNorm2d, SSFLayer):
    def __init__(self, dim, **kwargs):
        nn.BatchNorm2d.__init__(self, dim)
        SSFLayer.__init__(self, dim)
        self.weight.requires_grad = False

    def forward(self, x):
        result = super().forward(x)
        result = self.ssf_ada(result)
        return result
