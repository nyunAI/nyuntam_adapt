import numpy as np
from datasets import load_metric
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from nyuntam_adapt.tasks.custom_model import prepare_custom_model_support
from nyuntam_adapt.core.base_task import BaseTask


class ModelLoadingError(RuntimeError):
    """Exception for custom model loading errors."""

    pass


class SequenceClassification(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = "Classification"
        self.sub_task = kwargs.get("subtask", None)  # ner , pos, chunk
        self.text_key = kwargs["DATASET_ARGS"].get("input_column", "text")
        self.label_key = kwargs["DATASET_ARGS"].get("target_column", "labels")
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
        label2id, id2label = self.prepare_label_mappings()

        if self.sub_task is not None:
            # For subtasks like NER, POS, CHUNK
            try:
                model_config = AutoConfig.from_pretrained(self.local_model_path)
            except:
                model_config = AutoConfig.from_pretrained(self.model_path)

            model_config.num_labels = len(label2id)
            model_config.label2id = label2id
            model_config.id2label = id2label
            if self.local_model_path is not None:
                model, self.tokenizer, config = prepare_custom_model_support(
                    self.local_model_path,
                    self.sub_task,
                    self.model_path,
                    model_config,
                    num_gpu=self.num_gpu,
                    device=self.device,
                    use_bnb=use_bnb,
                    use_flash_attention_2=self.flash_attention,
                    **self.model_args,
                )
            else:
                try:
                    if self.num_gpu == 1:
                        # if not use_bnb:
                        try:
                            model = AutoModelForTokenClassification.from_pretrained(
                                self.model_path,
                                config=model_config,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                **self.model_args,
                                use_flash_attention_2=self.flash_attention,
                            ).to(self.device)
                        # else:
                        except:
                            model = AutoModelForTokenClassification.from_pretrained(
                                self.model_path,
                                config=model_config,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                **self.model_args,
                                use_flash_attention_2=self.flash_attention,
                            )
                    else:
                        model = AutoModelForTokenClassification.from_pretrained(
                            self.model_path,
                            config=model_config,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            **self.model_args,
                            # device_map="auto",
                            use_flash_attention_2=self.flash_attention,
                        )
                except Exception as e:
                    raise ModelLoadingError(f"Following Error Happened : {e}") from e

            self.collate_fn = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer
            )
            self.compute_metrics = self.compute_metrics_token
        #######################################################################################################
        #######################################################################################################
        else:
            # For Normal Text Classification
            try:
                model_config = AutoConfig.from_pretrained(self.local_model_path)
            except:
                model_config = AutoConfig.from_pretrained(self.model_path)
            model_config.num_labels = len(label2id)
            model_config.label2id = label2id
            model_config.id2label = id2label

            if self.local_model_path is not None:
                model, self.tokenizer, config = prepare_custom_model_support(
                    self.local_model_path,
                    "text_classification",
                    self.model_path,
                    model_config,
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
                            model = AutoModelForSequenceClassification.from_pretrained(
                                self.model_path,
                                config=model_config,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                **self.model_args,
                                use_flash_attention_2=self.flash_attention,
                            ).to(self.device)
                        except:
                            model = AutoModelForSequenceClassification.from_pretrained(
                                self.model_path,
                                config=model_config,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                **self.model_args,
                                use_flash_attention_2=self.flash_attention,
                            )
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            self.model_path,
                            config=model_config,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            **self.model_args,
                            use_flash_attention_2=self.flash_attention,
                            # device_map="auto",
                        )

                except Exception as e:
                    raise ModelLoadingError(f"Following Error Happened : {e}") from e

            self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.compute_metrics = self.compute_metrics_sequence

        return model, self.tokenizer, model_config

    def preprocess_function_token(self, examples):
        # Assuming input key is "tokens"
        # tokenized_inputs = self.tokenizer(
        #     examples["tokens"], truncation=True, is_split_into_words=True
        # )
        tokenized_inputs = self.tokenizer(
            examples[self.text_key], truncation=True, is_split_into_words=True
        )
        label_all_tokens = True

        labels = []
        # for i, label in enumerate(examples[f"{self.sub_task}_tags"]):
        for i, label in enumerate(examples[self.label_key]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_dataset(self, dataset, processor):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path, add_prefix_space=True
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, add_prefix_space=True
            )

        if self.sub_task is None:
            return dataset.map(self.preprocess_function, batched=True)
        else:
            return dataset.map(self.preprocess_function_token, batched=True)

    def prepare_label_mappings(self):
        if self.sub_task:
            self.labels = self.dataset["train"].features[self.label_key].feature.names
        else:
            self.labels = self.dataset["train"].features[self.label_key].names

        label2id, id2label = {}, {}
        for i, label in enumerate(self.labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        return label2id, id2label

    def preprocess_function(self, examples):
        encodings = self.tokenizer(examples[self.text_key], truncation=True)

        return encodings

    def compute_metrics_token(self, p):
        metric = load_metric("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def compute_metrics_sequence(self, p):
        """Computes accuracy on a batch of predictions"""

        # TODO : Avoid intialization of metric everytime and add it under utils.py
        metric = load_metric("accuracy")
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )
