import torch
import os
import sys
import warnings
from packaging import version
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.training_args import ParallelMode, TrainingArguments
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

from transformers.trainer_utils import (
    EvalPrediction,
    FSDPOption,
    TrainerMemoryTracker,
    set_seed,
    enable_full_determinism,
    has_length,
)

from transformers.trainer_pt_utils import LabelSmoother

sys.path.append("/workspace/Adapt/logging_adapt")
from logging_adapt import define_logger

from transformers.utils import (
    # logging,
    can_return_loss,
    find_labels,
    is_torch_tpu_available,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_apex_available,
    is_safetensors_available,
)
from transformers.utils.quantization_config import QuantizationMethod
from algorithms.base_algorithm import get_peft_state_dict
from safetensors.torch import save_file as safe_save_file


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback
TRAINING_ARGS_NAME = "training_args.bin"

if is_safetensors_available():
    import safetensors.torch
# if is_sagemaker_mp_enabled():
#     import smdistributed.modelparallel.torch as smp
#     from smdistributed.modelparallel import __version__ as SMP_VERSION

#     IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

#     from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
# else:
#     IS_SAGEMAKER_MP_POST_1_10 = False

# logger = logging.get_logger(__name__)


class AdaptTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        logging_path: Optional[str] = None,
    ):
        self.logging_path = logging_path
        self.logger = define_logger(__name__, self.logging_path)

        if args is None:
            output_dir = "tmp_trainer"
            self.logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        (
            enable_full_determinism(self.args.seed)
            if self.args.full_determinism
            else set_seed(self.args.seed)
        )
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        # log_level = args.get_process_log_level()
        # logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError(
                    "`Trainer` requires either a `model` or `model_init` argument"
                )
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        if (
            hasattr(model, "is_parallelizable")
            and model.is_parallelizable
            and model.model_parallel
        ):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        if getattr(model, "hf_device_map", None) is not None:
            devices = [
                device
                for device in set(model.hf_device_map.values())
                if device not in ["cpu", "disk"]
            ]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])
            else:
                self.is_model_parallel = False

            # warn users
            if self.is_model_parallel:
                self.logger.info(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

        # Changes added 16/11/2023 - Commenting below code -- Start
        # _is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        # _is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
        #     model, "_hf_peft_config_loaded", False
        # )

        # At this stage the model is already loaded
        # if _is_quantized_and_base_model and not _is_peft_model:
        #     raise ValueError(
        #         "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
        #         " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
        #         " for more details"
        #     )
        # elif _is_quantized_and_base_model and not getattr(model, "_is_quantized_training_enabled", False):
        #     raise ValueError(
        #         "The model you want to train is loaded in 8-bit precision.  if you want to fine-tune an 8-bit"
        #         " model, please make sure that you have installed `bitsandbytes>=0.37.0`. "
        #     )
        #  Changes added 16/11/2023 -- END

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        self.is_fsdp_xla_v2_enabled = args.fsdp_config["xla"]

        self.fsdp = None
        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if (
                not args.fsdp_config["xla"]
                and args.parallel_mode != ParallelMode.DISTRIBUTED
            ):
                raise ValueError("Using fsdp only works in distributed training.")

            # dep_version_check("torch>=1.12.0")
            # Would have to update setup.py with torch>=1.12.0
            # which isn't ideally given that it will force people not using FSDP to also use torch>=1.12.0
            # below is the current alternative.
            if version.parse(
                version.parse(torch.__version__).base_version
            ) < version.parse("1.12.0"):
                raise ValueError("FSDP requires PyTorch >= 1.12.0")

            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                BackwardPrefetch,
                ShardingStrategy,
            )

            if FSDPOption.FULL_SHARD in args.fsdp:
                self.fsdp = ShardingStrategy.FULL_SHARD
            elif FSDPOption.SHARD_GRAD_OP in args.fsdp:
                self.fsdp = ShardingStrategy.SHARD_GRAD_OP
            elif FSDPOption.NO_SHARD in args.fsdp:
                self.fsdp = ShardingStrategy.NO_SHARD

            self.backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            if (
                "backward_prefetch" in self.args.fsdp_config
                and "backward_post"
                in self.args.fsdp_config.get("backward_prefetch", [])
            ):
                self.backward_prefetch = BackwardPrefetch.BACKWARD_POST

            self.limit_all_gathers = False
            if self.args.fsdp_config.get("limit_all_gathers", False):
                self.limit_all_gathers = True

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or (self.fsdp is not None)
            or self.is_fsdp_enabled
        ):
            self.place_model_on_device = False

        default_collator = (
            default_data_collator
            if tokenizer is None
            else DataCollatorWithPadding(tokenizer)
        )
        self.data_collator = (
            data_collator if data_collator is not None else default_collator
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # Bnb Quantized models doesn't support `.to` operation.
        if (
            self.place_model_on_device
            and not getattr(model, "quantization_method", None)
            == QuantizationMethod.BITS_AND_BYTES
        ):
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_tpu_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        if (self.is_deepspeed_enabled or (self.fsdp is not None)) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if Deepspeed or PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        # self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(
            getattr(self.data_collator, "collate_batch", None)
        ):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`)."
            )

        if args.max_steps > 0:
            self.logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        if (
            train_dataset is not None
            and not has_length(train_dataset)
            and args.max_steps <= 0
        ):
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            args.group_by_length = False
            self.logger.warn(
                "the `--group_by_length` option is only available for `Dataset`, not `IterableDataset."
                "Changing GROUP_BY_LENGTH to FALSE"
            )

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Mixed precision setup for SageMaker Model Parallel
        # if is_sagemaker_mp_enabled():
        #     # BF16 + model parallelism in SageMaker: currently not supported, raise an error
        #     if args.bf16:
        #         raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")

        #     if IS_SAGEMAKER_MP_POST_1_10:
        #         # When there's mismatch between SMP config and trainer argument, use SMP config as truth
        #         if args.fp16 != smp.state.cfg.fp16:
        #             logger.warning(
        #                 f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
        #                 f"but FP16 provided in trainer argument is {args.fp16}, "
        #                 f"setting to {smp.state.cfg.fp16}"
        #             )
        #             args.fp16 = smp.state.cfg.fp16
        #     else:
        #         # smp < 1.10 does not support fp16 in trainer.
        #         if hasattr(smp.state.cfg, "fp16"):
        #             logger.warning(
        #                 f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
        #                 "but SageMaker Model Parallelism < 1.10 does not support FP16 in trainer."
        #             )
        if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16:
                    raise ValueError(
                        "Tried to use `fp16` but it is not supported on cpu"
                    )
                else:
                    args.half_precision_backend = "cpu_amp"
            self.logger.info(
                f"Using {args.half_precision_backend} half precision backend"
            )

        if (args.fp16 or args.bf16) and not (
            self.is_deepspeed_enabled or is_sagemaker_mp_enabled()
        ):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor
            )
        else:
            self.label_smoother = None

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = find_labels(self.model.__class__)
        self.label_names = (
            default_label_names
            if self.args.label_names is None
            else self.args.label_names
        )
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control
        )

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        state_dict = get_peft_state_dict(unwrap_model(self.model))
        os.makedirs(output_dir, exist_ok=True)
        if getattr(self.model, "is_loaded_in_4bit", False):
            if self.args.save_safetensors:
                try:
                    safe_save_file(
                        state_dict, os.path.join(output_dir, "model.safetensors")
                    )
                except:
                    print("Could not save safetensors, hence skipping")
                    torch.save(state_dict, os.path.join(output_dir, "peft_modules.pt"))
            else:
                torch.save(state_dict, os.path.join(output_dir, "peft_modules.pt"))
        else:
            super()._save(output_dir, state_dict)

        self.model._keys_to_ignore_on_save = None

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )
        self.logger.info(str(output))
