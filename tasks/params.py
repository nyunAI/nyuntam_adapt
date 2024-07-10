from dataclasses import dataclass, field, asdict, fields, is_dataclass, MISSING
from typing import Dict, Union, Optional, Literal
from __future__ import annotations

@dataclass
class BNBConfigUse4Bit:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_storage: str = "float16"


@dataclass
class BNBConfigUse8Bit:
    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[str] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False


@dataclass
class BNBConfig:
    USE_4BIT: BNBConfigUse4Bit = field(default_factory=BNBConfigUse4Bit)
    USE_8BIT: BNBConfigUse8Bit = field(default_factory=BNBConfigUse8Bit)


@dataclass
class LoRA_PEFT:
    r: int = field(default=8)
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(default=False)
    bias: Literal["none", "all", "lora_only"] = field(default="none")
    use_rslora: bool = field(default=False)
    modules_to_save: Optional[list[str]] = field(default=None)
    init_lora_weights: bool | Literal["gaussian", "loftq"] = field(default=True)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)
    rank_pattern: Optional[dict] = field(default_factory=dict)
    alpha_pattern: Optional[dict] = field(default_factory=dict)
    megatron_config: Optional[dict] = field(default=None)
    megatron_core: Optional[str] = field(default="megatron.core")
    # dict type is used when loading config.json
    use_dora: bool = field(default=False)
    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(default=None)
    peft_type: str = "LoRA"


@dataclass
class DoRA_PEFT:
    r: int = field(default=8)
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(default=False)
    bias: Literal["none", "all", "lora_only"] = field(default="none")
    use_rslora: bool = field(default=False)
    modules_to_save: Optional[list[str]] = field(default=None)
    init_lora_weights: bool | Literal["gaussian", "loftq"] = field(default=True)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)
    rank_pattern: Optional[dict] = field(default_factory=dict)
    alpha_pattern: Optional[dict] = field(default_factory=dict)
    megatron_config: Optional[dict] = field(default=None)
    megatron_core: Optional[str] = field(default="megatron.core")
    # dict type is used when loading config.json
    use_dora: bool = field(default=True)
    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(default=None)
    peft_type: str = "DoRA"

@dataclass
class SSFConfig:
    peft_type: str = "SSF"


@dataclass
class DatasetConfig:
    DATASET: Optional[str] = None
    DATA_VERSION: str = "1.0"
    DATA_SPLIT: float = 0.2
    TRAIN_DIR: str = ""
    VAL_DIR: str = ""
    TEST_DIR: str = ""
    MAX_TRAIN_SAMPLES: Optional[int] = None
    MAX_EVAL_SAMPLES: Optional[int] = None
    CUSTOM_DATASET_PATH: str = None
    DATASET_CONFIG: str = ""
    DATASET_FORMAT: Optional[str] = None
    FORMAT_NAMES: str = "text"
    FORMAT_STRING: Optional[str] = None
    # QnA specific parameters
    input_column: str = "context"
    input_question_column: str = "question"
    target_column: str = "answers"
    squad_v2_format: bool = False  # True


@dataclass
class ModelArgs:
    MODEL: str = ""
    MODEL_PATH: str = "luodian/llama-7b-hf"
    MODEL_VERSION: str = "1.0"
    CACHE_BOOL: bool = False
    MODEL_CHECKPOINT: str = ""
    LOCAL_MODEL_PATH: str = None  # "abc/jobs/model.pt"


@dataclass
class TrainingArgs:
    SEED: int = 56
    DO_TRAIN: bool = True
    DO_EVAL: bool = False
    NUM_WORKERS: int = 4
    BATCH_SIZE: int = 4
    EPOCHS: float = 0.001
    STEPS: int = 1
    OPTIMIZER: str = "paged_adamw_32bit"
    LR: float = 0.0002
    SCHEDULER_TYPE: str = "constant"
    WEIGHT_DECAY: float = 0.001
    BETA1: float = 0.9
    BETA2: float = 0.999
    ADAM_EPS: str = "1e-8"
    INTERVAL: str = "steps"
    INTERVAL_STEPS: int = 50
    NO_OF_CHECKPOINTS: int = 5
    FP16: bool = False
    REMOVE_UNUSED_COLUMNS: bool = False
    RESUME_FROM_CHECKPOINT: bool = False
    GRADIENT_ACCUMULATION_STEPS: int = 1
    GRADIENT_CHECKPOINTING: bool = True
    GROUP_BY_LENGTH: bool = True
    T_MAX: int = 100
    MILESTONES: list = None
    GAMMA: float = 0.1
    WARMUP: bool = True
    WARMUP_ITERS: int = 50
    WARMUP_RATIO: float = 0.1
    BEGIN: int = 0
    END: int = 10


@dataclass
class FineTuningArgs:
    LAST_LAYER_TUNING: bool = False
    FULL_FINE_TUNING: bool = False


@dataclass
class AdaptParams:
    TASK: str = "text_generation"
    subtask: str = None
    packing: bool = True
    dataset_text_field: str = "text"
    max_seq_length: int = 512
    flash_attention2: bool = False
    blocksize: int = 128
    cuda_id: str = "0"
    auto_select_modules: bool = True
    OUTPUT_DIR: str = "abc/jobs/1/"
    OVERWRITE_OUTPUT_DIR: bool = False
    LOGGING_PATH: str = "abc/logs/1/log.log"
    MERGE_ADAPTERS: bool = False
    DATASET_ARGS: DatasetConfig = field(default_factory=DatasetConfig)
    MODEL_ARGS: ModelArgs = field(default_factory=ModelArgs)
    TRAINING_ARGS: TrainingArgs = field(default_factory=TrainingArgs)
    FINE_TUNING_ARGS: FineTuningArgs = field(default_factory=FineTuningArgs)
    PEFT_METHOD: str = None
    DoRA_CONFIG: DoRA_PEFT = field(default_factory=DoRA_PEFT)
    SSF_CONFIG: SSFConfig = field(default_factory=SSFConfig)
    LoRA_CONFIG: LoRA_PEFT = field(default_factory=LoRA_PEFT)
    BNB_CONFIG: BNBConfig = field(default_factory=BNBConfig)
    SAVE_METHOD: str = "state_dict"
    # Seq2Seq specific arguments
    max_input_length: int = 128
    max_target_length: int = 128
    eval_metric: str = "rogue"
    source_lang: str = ""
    target_lang: str = ""
    PREFIX: str = ""
    # QnA specific
    max_answer_length: int = 30
    doc_stride: int = 128
    max_length: int = 384
    Library: str = "Huggingface"
    #DDP args
    DDP:bool = False
    num_nodes:int = 1
    #FSDP args 
    FSDP:bool = False

def create_instance(data_class, flat_dict):
    instance_args = {}
    for field in fields(data_class):
        field_name = field.name
        field_type = field.type
        field_value = flat_dict.get(field_name, MISSING)

        if is_dataclass(field_type):
            instance_args[field_name] = create_instance(field_type, flat_dict)
        else:
            if field_value is None or field_value == "" or field_value == MISSING:
                instance_args[field_name] = field.default
            else:
                instance_args[field_name] = field_value

    dc = data_class(**instance_args)
    return dc
