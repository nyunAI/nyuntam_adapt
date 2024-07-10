import torch
import inspect
import warnings
from algorithms.SSF import SSFModel, SSFConfig
from algorithms.LoRA_PEFT import LoraModel_PEFT
from algorithms.LoRA_PEFT.config import LoraConfig_PEFT
from transformers.configuration_utils import PretrainedConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device, offload_weight
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers
from transformers.integrations import (
    get_keys_to_not_convert,
    replace_with_bnb_linear,
    set_module_quantized_tensor_to_device,
)
import gc

PEFT_TYPE_TO_MODEL_MAPPING = {
    "LoRA": LoraModel_PEFT,
    "SSF": SSFModel,
    "DoRA": LoraModel_PEFT,
}

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "LoRA": LoraConfig_PEFT,
    "SSF": SSFConfig,
    "DoRA": LoraConfig_PEFT,
}




class TimmModelConfig(PretrainedConfig):
    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        for key, value in kwargs.items():
            self.key = value


# If in case the checkpoint is sharded then changes wrt to loaded keys is required
def get_start_prefix(model, loaded_keys):

    start_prefix = ""
    cls = model.__class__
    model_to_load = model
    expected_keys = list(model.state_dict().keys())  # complete set of keys
    prefix = model.base_model_prefix

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    if remove_prefix_from_model:
        _prefix = f"{prefix}."
        expected_keys_not_prefixed = [
            s for s in expected_keys if not s.startswith(_prefix)
        ]
        expected_keys = [
            s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys
        ]

    if (
        len(cls.base_model_prefix) > 0
        and not hasattr(model, cls.base_model_prefix)
        and has_prefix_module
    ):
        start_prefix = cls.base_model_prefix + "."
    if (
        len(cls.base_model_prefix) > 0
        and hasattr(model, cls.base_model_prefix)
        and not has_prefix_module
    ):
        model_to_load = getattr(model, cls.base_model_prefix)
        base_model_expected_keys = list(model_to_load.state_dict().keys())
        if any(
            key in expected_keys_not_prefixed and key not in base_model_expected_keys
            for key in loaded_keys
        ):
            raise ValueError(
                "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                "properly saved?"
            )
        if device_map is not None:
            device_map = {
                k.replace(f"{cls.base_model_prefix}.", ""): v
                for k, v in device_map.items()
            }

    return start_prefix


def load_model_from_checkpoint(checkpoint_path, flash_attention, quantization_config):
    model_orig = torch.load(checkpoint_path)

    state_dict = model_orig.state_dict()

    model_args = {}
    model_kwargs = {}
    use_flash_attention_2 = flash_attention
    model_orig.quantization_method = quantization_config

    device_map = None
    _fast_init = True
    keys_list = list(state_dict.keys())  # if sharded then load keys of each shard
    start_prefix = get_start_prefix(model_orig, keys_list)

    if device_map is None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
            low_cpu_mem_usage = True

    load_in_8bit = quantization_config.load_in_8bit
    load_in_4bit = quantization_config.load_in_4bit
    config = model_orig.config

    if load_in_8bit or load_in_4bit:
        torch_dtype = torch.float16

    # create dummy class to find modules_to_skip
    init_contexts = [no_init_weights(_enable=_fast_init)]
    cls = model_orig.__class__
    if load_in_8bit or load_in_4bit or low_cpu_mem_usage:
        init_contexts.append(init_empty_weights())

    if use_flash_attention_2:
        warnings.warn("Flash attention 2 is not supported for current architecture")
        # config = cls._check_and_enable_flash_attn_2(config, torch_dtype=model_orig.dtype)

    with ContextManagers(init_contexts):
        model2 = cls(config, *model_args, **model_kwargs)

    modules_to_not_convert = []
    llm_int8_skip_modules = quantization_config.llm_int8_skip_modules
    if llm_int8_skip_modules is None:
        modules_to_not_convert = get_keys_to_not_convert(model2)
    else:
        modules_to_not_convert = llm_int8_skip_modules
    use_keep_in_fp32_modules = model_orig._keep_in_fp32_modules
    if use_keep_in_fp32_modules:
        keep_in_fp32_modules = model_orig._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []
    modules_to_not_convert.extend(keep_in_fp32_modules)

    model = replace_with_bnb_linear(
        model_orig,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )
    model._is_quantized_training_enabled = True
    model.is_8bit_serializable = True
    model.is_loaded_in_8bit = load_in_8bit
    model.is_loaded_in_4bit = load_in_4bit

    is_safetensors = False
    is_quantized = True
    offload_folder = None
    offload_index = None
    state_dict_folder = None
    state_dict_index = None
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    values_list = list(state_dict.values())
    num_of_keys = len(keys_list)
    index = 0

    del state_dict
    del model_orig
    gc.collect()
    torch.cuda.empty_cache()

    # for param_name, param in state_dict.items():
    while num_of_keys > 0:
        num_of_keys = num_of_keys - 1
        # First part of the test is always true as load_state_dict_keys always contains state_dict keys.
        param_name = keys_list[index]
        param = values_list[index]
        # if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
        #     continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        module_name = param_name
        set_module_kwargs = {}

        # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
        # in int/uint/bool and not cast them.
        if torch_dtype is not None and torch.is_floating_point(param):
            if (
                keep_in_fp32_modules is not None
                and any(
                    module_to_keep_in_fp32 in param_name.split(".")
                    for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
                and torch_dtype == torch.float16
            ):
                param = param.to(torch.float32)

                # For backward compatibility with older versions of `accelerate`
                # TODO: @sgugger replace this check with version check at the next `accelerate` release
                if "dtype" in list(
                    inspect.signature(set_module_tensor_to_device).parameters
                ):
                    set_module_kwargs["dtype"] = torch.float32
            else:
                param = param.to(torch_dtype)

        set_module_kwargs["value"] = param

        if device_map is None:
            param_device = "cpu"
        else:
            # find next higher level module that is defined in device_map:
            # bert.lm_head.weight -> bert.lm_head -> bert -> ''
            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            if module_name == "" and "" not in device_map:
                # TODO: group all errors and raise at the end.
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]

        if param_device == "disk":
            if not is_safetensors:
                offload_index = offload_weight(
                    param, param_name, offload_folder, offload_index
                )
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(
                param, param_name, state_dict_folder, state_dict_index
            )
        elif not is_quantized:
            # For backward compatibility with older versions of `accelerate`
            set_module_tensor_to_device(
                model, param_name, param_device, **set_module_kwargs
            )
        else:
            if (
                param.dtype == torch.int8
                and param_name.replace("weight", "SCB") in state_dict.keys()
            ):
                fp16_statistics = state_dict[param_name.replace("weight", "SCB")]
            else:
                fp16_statistics = None

            if "SCB" not in param_name:
                set_module_quantized_tensor_to_device(
                    model,
                    param_name,
                    param_device,
                    value=param,
                    fp16_statistics=fp16_statistics,
                )
        index = index + 1
        if index == 10:  # deleting in chunks of 10
            del keys_list[0:index], values_list[0:index]
            gc.collect()
            torch.cuda.empty_cache()
            index = 0

    return model


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    # is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    # for name, param in model.named_parameters():
    #     # freeze base model's layers
    #     param.requires_grad = False

    # if not is_gptq_quantized:
    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if (loaded_in_kbit) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model
