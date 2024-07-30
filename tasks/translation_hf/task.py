import numpy as np
from evaluate import load
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)
from nyuntam_adapt.core.base_task import BaseTask
from nyuntam_adapt.utils.task_utils import prepare_model_for_kbit_training, ModelLoadingError
from nyuntam_adapt.core.custom_model import prepare_custom_model_support


class Translation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtask = kwargs.get("subtask", None)
        self.eval_metric = kwargs.get("eval_metric", None)
        self.max_input_length = kwargs.get("max_input_length", 128)
        self.source_lang = kwargs.get("source_lang", "en")
        self.target_lang = kwargs.get("target_lang", "ro")
        self.column_name = kwargs.get("FORMAT_NAMES", "translation")
        self.max_target_length = kwargs.get("max_target_length", 128)
        self.input_column = kwargs["DATASET_ARGS"].get("input_column", "english-text")
        self.target_column = kwargs["DATASET_ARGS"].get("target_column", "roman-text")
        predict_with_generate = kwargs["TRAINING_ARGS"].get(
            "predict_with_generate", True
        )
        generation_max_length = kwargs["TRAINING_ARGS"].get(
            "generation_max_length", 128
        )
        self.training_args = Seq2SeqTrainingArguments(
            **self.training_args.to_dict(),
            predict_with_generate=predict_with_generate,
            generation_max_length=generation_max_length,
        )
        self.prefix = kwargs.get("PREFIX", "")
        self.flash_attention = kwargs.get("flash_attention2", False)
        self.model_args = {}

    def prepare_model(self):

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

        if self.local_model_path is not None:
            model, self.tokenizer, model_config = prepare_custom_model_support(
                self.local_model_path,
                "translation",
                self.model_path,
                num_gpu=self.num_gpu,
                device=self.device,
                use_bnb=use_bnb,
                use_flash_attention_2=self.flash_attention,
                **self.model_args,
            )

        else:
            model_config = AutoConfig.from_pretrained(self.model_path)
            try:
                if self.num_gpu == 1:
                    try:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.model_path,
                            config=model_config,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            **self.model_args,
                            use_flash_attention_2=self.flash_attention,
                        ).to(self.device)
                    except:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.model_path,
                            config=model_config,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            **self.model_args,
                            use_flash_attention_2=self.flash_attention,
                        )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_path,
                        config=model_config,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True,
                        **self.model_args,
                        use_flash_attention_2=self.flash_attention,
                    )
            except Exception as e:
                raise ModelLoadingError(f"Following Error Happened : {e}") from e

        if use_bnb:
            model = prepare_model_for_kbit_training(model, self.gradient_checkpointing)
        elif self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        self.collate_fn = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)

        return model, self.tokenizer, model_config

    def preprocess_function(self, examples):
        inputs = [
            self.prefix + ex[self.source_lang] for ex in examples[self.column_name]
        ]

        targets = [ex[self.target_lang] for ex in examples[self.column_name]]

        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, max_length=self.max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def postprocess(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        metric = load(self.eval_metric)
        predictions, labels = eval_preds

        predictions = np.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        decoded_preds, decoded_labels = self.postprocess(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # Additional result processing
        if self.eval_metric == "sacrebleu":
            result = {"sacrebleu": result["score"]}
            prediction_lens = [
                np.count_nonzero(pred != self.tokenizer.pad_token_id)
                for pred in predictions
            ]
            result["gen_len"] = np.mean(prediction_lens)
            return {k: round(v, 4) for k, v in result.items()}
        else:
            result = {key: value * 100 for key, value in result.items()}
            prediction_lens = [
                np.count_nonzero(pred != self.tokenizer.pad_token_id)
                for pred in predictions
            ]
            result["gen_len"] = np.mean(prediction_lens)
            return {k: round(v, 4) for k, v in result.items()}

    def prepare_dataset(self, dataset, processor):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        return dataset.map(self.preprocess_function, batched=True)
