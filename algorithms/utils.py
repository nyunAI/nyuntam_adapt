import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import bitsandbytes as bnb


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "distilbert": ["q_lin", "v_lin"],
    "distilbert": ["q_lin", "v_lin"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "vit": ["query", "value"],
    "resnet": ["convolution"],
    "resnet_timm": ["conv1", "conv2"],
    "convnext": ["pwconv1", "pwconv2"],  # pwconv1,pwconv2 ,"dwconv",
    "convnextv2": ["dwconv"],  # pwconv1,pwconv2
    "swin": ["query", "value"],  # reduction
    "swinv2": ["query", "value"],  # reduction
    "efficientnet_timm": ["conv_pwl", "conv_dw", "conv_reduce", "conv_pw"],
    "internImage_timm": ["fc1", "fc2", "conv"],  # Add dwconv.0
    "densenet_timm": ["conv0", "conv1", "conv2"],
    "yolo": ["conv"],
    # "yolo-nas" : [],
    "rt-detr": [
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "linear1",
        "linear2",
    ],
    "rtmdet": ["conv", "fc"],
    "detr": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "out_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
    ],  # donot add bbox head layers
    "deformable_detr": [
        "conv1",
        "conv2",
        "conv3",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
    ],
    "rtmdet": ["conv", "fc"],
    "glip": [
        "qkv",
        "proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "reduction",
        "conv",
    ],  # WIP
    "grounding_dino": [
        "qkv",
        "proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "reduction",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "v_proj",
        "l_proj",
        "values_v_proj",
        "values_l_proj",
        "out_v_proj",
        "out_l_proj",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
        "query",
        "key",
        "value",
        "dense",
    ],  # contains lang model layers , embedding layer - query embedding and dn_query_embedding ?
    "dino": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "co_detr": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "align_detr": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "ddq": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "marian": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],

    "rtmo": ["conv"],

    "segnext": ["conv0_1", "conv0_2", "conv1_2", "conv1_2", "conv2_2"],

}

TRANSFORMERS_MODELS_TO_DORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "distilbert": ["q_lin", "v_lin"],
    "distilbert": ["q_lin", "v_lin"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "vit": ["query", "value"],
    "resnet": ["convolution"],
    "resnet_timm": ["conv1", "conv2"],
    "convnext": ["pwconv1", "pwconv2"],  # pwconv1,pwconv2 ,"dwconv",
    "convnextv2": ["dwconv"],  # pwconv1,pwconv2
    "swin": ["query", "value"],  # reduction
    "swinv2": ["query", "value"],  # reduction
    "efficientnet_timm": ["conv_pwl", "conv_dw", "conv_reduce", "conv_pw"],
    "internImage_timm": ["fc1", "fc2", "conv"],  # Add dwconv.0
    "densenet_timm": ["conv0", "conv1", "conv2"],
    "yolo": ["conv"],
    # "yolo-nas" : [],
    "rt-detr": [
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "linear1",
        "linear2",
    ],
    "rtmdet": ["conv", "fc"],
    "detr": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "out_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
    ],  # donot add bbox head layers
    "deformable_detr": [
        "conv1",
        "conv2",
        "conv3",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
    ],
    "glip": [
        "qkv",
        "proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "reduction",
        "conv",
    ],  # WIP
    "grounding_dino": [
        "qkv",
        "proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "reduction",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "v_proj",
        "l_proj",
        "values_v_proj",
        "values_l_proj",
        "out_v_proj",
        "out_l_proj",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
        "query",
        "key",
        "value",
        "dense",
    ],  # contains lang model layers , embedding layer - query embedding and dn_query_embedding ?
    "dino": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "co_detr": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "align_detr": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "ddq": [
        "conv1",
        "conv2",
        "conv3",
        "downsample.0",
        "conv",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "ffn.layers.0.0",
        "ffn.layers.1",
        "out_proj",
        "ref_point_head.layers.0",
        "ref_point_head.layers.1",
    ],
    "marian": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],

}


def Transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_bnb_4bit_available():
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb

    return hasattr(bnb.nn, "Linear4bit")


def is_auto_gptq_available():
    return importlib.util.find_spec("auto_gptq") is not None


def is_optimum_available():
    return importlib.util.find_spec("optimum") is not None


def dequantize_8bit(module):
    if module.state.SCB is None:
        module.state.SCB = module.weight.SCB
    im = (
        torch.eye(module.weight.data.shape[-1])
        .contiguous()
        .half()
        .to(module.weight.device)
    )
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")

    if module.state.CxB is None:
        module.state.CxB, module.state.SB = bnb.functional.transform(
            module.weight.data, to_order=module.state.formatB
        )
    out32, Sout32 = bnb.functional.igemmlt(im, module.state.CxB, Sim, module.state.SB)
    output = bnb.functional.mm_dequant(
        out32, Sout32, SCim, module.state.SCB, bias=None
    ).t()
    return output
