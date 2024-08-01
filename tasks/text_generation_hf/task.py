import torch
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from nyuntam_adapt.core.base_task import BaseTask
from nyuntam_adapt.utils.task_utils import (
    prepare_model_for_kbit_training,
)
from nyuntam_adapt.tasks.custom_model import prepare_custom_model_support
from nyuntam_adapt.utils.task_utils import ModelLoadingError


class CausalLLM(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.processor = None
        self.config = None
        self.task = "generation"
        self.additional_training_args = {  # For SFT trainer
            "dataset_text_field": kwargs.get("dataset_text_field", None),
            "packing": kwargs.get(
                "packing", False
            ),  # Packing of dataset to gain Flash attention speedup in case of ConstantLengthDataset
            "max_seq_length": kwargs.get("max_seq_length", 2048),
        }

        self.flash_attention = kwargs.get("flash_attention2", False)
        self.additional_training_args["flash_attn"] = self.flash_attention
        self.additional_training_args["block_size"] = kwargs.get("block_size", 256)

        self.model_args = {}

    def alpaca_formatting_func(self, examples):
        output_text = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            response = examples["output"][i]

            if len(input_text) >= 2:
                text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                
                ### Instruction:
                {instruction}
                
                ### Input:
                {input_text}
                
                ### Response:
                {response}
                """
            else:
                text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                
                ### Instruction:
                {instruction}
                
                ### Response:
                {response}
                """
            output_text.append(text)

        return output_text

    def prepare_model(self):
        # Quantization

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

        # Custom-Model Path
        if self.local_model_path is not None:
            model, tokenizer, config = prepare_custom_model_support(
                self.local_model_path,
                "text_generation",
                self.model_path,
                num_gpu=self.num_gpu,
                device=self.device,
                use_bnb=use_bnb,
                use_flash_attention_2=self.flash_attention,
                **self.model_args,
            )
            # if use_bnb:
            #     model = load_model_from_checkpoint(
            #         model, self.flash_attention, bnb_config
            #     )

        # Hugging Face Model path
        else:
            config = AutoConfig.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            try:
                if self.num_gpu == 1:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            use_flash_attention_2=self.flash_attention,
                            **self.model_args,
                        ).to(self.device)
                    except:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            use_flash_attention_2=self.flash_attention,
                            **self.model_args,
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        use_flash_attention_2=self.flash_attention,
                        # device_map="auto",
                        torch_dtype=torch.float16,
                        **self.model_args,
                    )

            except Exception as e:
                raise ModelLoadingError(f"Following Error Happened : {e}") from e
            if use_bnb:
                model = prepare_model_for_kbit_training(
                    model, self.gradient_checkpointing
                )
            elif self.gradient_checkpointing:
                model.gradient_checkpointing_enable()

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        return model, tokenizer, config

    def compute_metrics(self, *args):
        # TODO Add evaluation metrics
        pass
