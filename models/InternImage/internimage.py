import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import PreTrainedModel
from timm.layers import trunc_normal_, DropPath
from .layers.dcnv3 import DCNv3_pytorch
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from transformers import PretrainedConfig
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model, generate_default_cfgs


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


class StemLayer(nn.Module):
    r"""Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self, in_chans=3, out_chans=96, act_layer="GELU", norm_layer="BN"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1
        )
        self.norm1 = build_norm_layer(
            out_chans // 2, norm_layer, "channels_first", "channels_first"
        )
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(
            out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1
        )
        self.norm2 = build_norm_layer(
            out_chans, norm_layer, "channels_first", "channels_last"
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r"""Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer="LN"):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = build_norm_layer(
            2 * channels, norm_layer, "channels_first", "channels_last"
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    r"""MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="GELU",
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    r"""Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(
        self,
        core_op,
        channels,
        groups,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer="GELU",
        norm_layer="LN",
        post_norm=False,
        layer_scale=None,
        offset_scale=1.0,
        with_cp=False,
    ):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, "LN")
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(channels, "LN")
        self.mlp = MLPLayer(
            in_features=channels,
            hidden_features=int(channels * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True
            )

    def forward(self, x):
        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    r"""Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(
        self,
        core_op,
        channels,
        depth,
        groups,
        downsample=True,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer="GELU",
        norm_layer="LN",
        post_norm=False,
        offset_scale=1.0,
        layer_scale=None,
        with_cp=False,
    ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList(
            [
                InternImageLayer(
                    core_op=core_op,
                    channels=channels,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    offset_scale=offset_scale,
                    with_cp=with_cp,
                )
                for i in range(depth)
            ]
        )
        if not self.post_norm:
            self.norm = build_norm_layer(channels, "LN")
        self.downsample = (
            DownsampleLayer(channels=channels, norm_layer=norm_layer)
            if downsample
            else None
        )

    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class InternImage(nn.Module):
    r"""InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        num_classes (int): Number of classes. Default: 1000
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
    """

    def __init__(
        self,
        core_op="DCNv3_pytorch",
        channels=64,
        depths=[3, 4, 18, 5],
        groups=[3, 6, 12, 24],
        num_classes=1000,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.2,
        drop_path_type="linear",
        act_layer="GELU",
        norm_layer="LN",
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        cls_scale=1.5,
        with_cp=False,
        **kwargs,
    ):
        super().__init__()
        assert core_op == "DCNv3_pytorch"
        core_op = DCNv3_pytorch

        self.core_op = core_op
        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        print(f"using core type: {core_op}")
        print(f"using activation layer: {act_layer}")
        print(f"using main norm layer: {norm_layer}")
        print(f"using dpr: {drop_path_type}, {drop_path_rate}")

        in_chans = 3
        self.patch_embed = StemLayer(
            in_chans=in_chans,
            out_chans=channels,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if drop_path_type == "uniform":
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = InternImageBlock(
                core_op=core_op,
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
            )
            self.levels.append(level)

        self.conv_head = nn.Sequential(
            nn.Conv2d(
                self.num_features,
                int(self.num_features * cls_scale),
                kernel_size=1,
                bias=False,
            ),
            build_norm_layer(
                int(self.num_features * cls_scale),
                "BN",
                "channels_first",
                "channels_first",
            ),
            build_act_layer(act_layer),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = (
            nn.Linear(int(self.num_features * cls_scale), num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, self.core_op):
            m._reset_parameters()

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = "levels.{}.blocks.{}.".format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios["levels.0.blocks.0."]
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios["levels.1.blocks.0."]
        lr_ratios["levels.0.norm"] = lr_ratios["levels.1.blocks.0."]
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios["levels.2.blocks.0."]
        lr_ratios["levels.1.norm"] = lr_ratios["levels.2.blocks.0."]
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios["levels.3.blocks.0."]
        lr_ratios["levels.2.norm"] = lr_ratios["levels.3.blocks.0."]
        return lr_ratios

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.conv_head(x.permute(0, 3, 1, 2))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_features_seq_out(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
        return seq_out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_internimage(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(InternImage, variant, pretrained, **kwargs)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "source": url,
        "architecture": "internimage",
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.conv1",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "internimage_small": _cfg(
            url="https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_s_1k_224.pth"
        ),
        "internimage_tiny": _cfg(
            url="https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_t_1k_224.pth"
        ),
        "internimage_base": _cfg(
            url="https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_b_1k_224.pth"
        ),
        "internimage_large": _cfg(
            url="https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_l_22kto1k_384.pth"
        ),
        "internimage_xlarge": _cfg(
            url="https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_xl_22kto1k_384.pth"
        ),
        # Add checkpoint urls from HF repo - 'https://huggingface.co/OpenGVLab/InternImage/blob/main/internimage_h_22kto1k_640.pth'
        "internimage_huge": _cfg(url=""),
        # Add checkpoint urls from HF repo - 'https://huggingface.co/OpenGVLab/InternImage/blob/main/internimage_g_22kto1k_512.pth'
        "internimage_giga": _cfg(url=""),
    }
)


@register_model
def internimage_tiny(pretrained=False, **kwargs) -> InternImage:
    model_args = dict(
        channels=64, depths=(4, 4, 18, 4), groups=(4, 8, 16, 32), drop_path_rate=0.1
    )
    model = _create_internimage(
        "internimage_tiny", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


@register_model
def internimage_small(pretrained=False, **kwargs) -> InternImage:
    model_args = dict(
        channels=80, depths=(4, 4, 21, 4), groups=(5, 10, 20, 40), drop_path_rate=0.4
    )
    model = _create_internimage(
        "internimage_small", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


@register_model
def internimage_base(pretrained=False, **kwargs) -> InternImage:
    model_args = dict(
        channels=112,
        depths=(4, 4, 21, 4),
        groups=(7, 14, 28, 56),
        drop_path_rate=0.5,
        layer_scale=1e-5,
    )
    model = _create_internimage(
        "internimage_base", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


@register_model
def internimage_large(pretrained=False, **kwargs) -> InternImage:
    model_args = dict(
        channels=160,
        depths=(5, 5, 22, 5),
        groups=(10, 20, 40, 80),
        drop_path_rate=0.1,
        offset_scale=2.0,
        layer_scale=1e-5,
    )
    model = _create_internimage(
        "internimage_large", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


@register_model
def internimage_xlarge(pretrained=False, **kwargs) -> InternImage:
    model_args = dict(
        channels=192,
        depths=(5, 5, 24, 5),
        groups=(12, 24, 48, 96),
        drop_path_rate=0.2,
        offset_scale=2.0,
        layer_scale=1e-5,
    )
    model = _create_internimage(
        "internimage_xlarge", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


# Config
"""
"act_layer": "GELU", -- 
  "architectures": [
    "InternImageModel"
  ],"channels": 80, -- 
  "cls_scale": 1.5, 
  "layer_scale": 1e-05,
   "model_type": "intern_image",
"""

# Transformation
"""
input_size=config.DATA.IMG_SIZE, -- 224
# is_training=True,
color_jitter=config.AUG.COLOR_JITTER --0.4
auto_augment=config.AUG.AUTO_AUGMENT -- 'rand-m9-mstd0.5-inc1'
if config.AUG.AUTO_AUGMENT != 'none' else None,
re_prob=config.AUG.REPROB, -- 0.25
re_mode=config.AUG.REMODE, -- pixel
re_count=config.AUG.RECOUNT, -- 1
_C.AUG.MIXUP = 0.8 
interpolation=config.DATA.INTERPOLATION, -- 'bicubic'
_C.AUG.RANDOM_RESIZED_CROP = False
_C.AUG.MEAN = (0.485, 0.456, 0.406)
_C.AUG.STD = (0.229, 0.224, 0.225)
"""

# Different Models
"""
1. tiny
2. small
3. base
4. Large
5. Xtra large
6. Huge
"""

# class InternImageModel(PreTrainedModel):
#     config_class = InternImageConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = InternImage(
#             core_op=config.core_op,
#             channels=config.channels,
#             depths=config.depths,
#             groups=config.groups,
#             num_classes=config.num_classes,
#             mlp_ratio=config.mlp_ratio,
#             drop_rate=config.drop_rate,
#             drop_path_rate=config.drop_path_rate,
#             drop_path_type=config.drop_path_type,
#             act_layer=config.act_layer,
#             norm_layer=config.norm_layer,
#             layer_scale=config.layer_scale,
#             offset_scale=config.offset_scale,
#             post_norm=config.post_norm,
#             cls_scale=config.cls_scale,
#             with_cp=config.with_cp,
#         )

#     def forward(self, tensor):
#         return self.model.forward_features(tensor)

# class InternImageModelForImageClassification(PreTrainedModel):
#     config_class = InternImageConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = InternImage(
#             core_op=config.core_op,
#             channels=config.channels,
#             depths=config.depths,
#             groups=config.groups,
#             num_classes=config.num_classes,
#             mlp_ratio=config.mlp_ratio,
#             drop_rate=config.drop_rate,
#             drop_path_rate=config.drop_path_rate,
#             drop_path_type=config.drop_path_type,
#             act_layer=config.act_layer,
#             norm_layer=config.norm_layer,
#             layer_scale=config.layer_scale,
#             offset_scale=config.offset_scale,
#             post_norm=config.post_norm,
#             cls_scale=config.cls_scale,
#             with_cp=config.with_cp,
#         )

#     def forward(self, tensor, labels=None):
#         logits = self.model(tensor)

#         if labels is not None:
#             loss = F.cross_entropy(logits, labels)
#             return {'loss': loss, 'logits': logits}

#         return {'logits': logits}
