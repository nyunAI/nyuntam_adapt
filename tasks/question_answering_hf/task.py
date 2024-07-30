# SQUAd v2 format
import collections
import numpy as np
from tqdm.auto import tqdm
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    default_data_collator,
    BitsAndBytesConfig,
)
from nyuntam_adapt.tasks.custom_model import prepare_custom_model_support
from nyuntam_adapt.core.base_task import BaseTask
from nyuntam_adapt.utils.task_utils import prepare_model_for_kbit_training, ModelLoadingError


class QuestionAnswering(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_length = kwargs.get("max_length", 384)
        self.doc_stride = kwargs.get("doc_stride", 128)
        self.input_column = kwargs["DATASET_ARGS"].get("input_column", "document")
        self.input_question_column = kwargs["DATASET_ARGS"].get(
            "input_question_column", "document"
        )
        self.target_column = kwargs["DATASET_ARGS"].get("target_column", "answer")
        self.max_answer_length = kwargs.get("max_answer_length", 30)
        self.eval_metric = kwargs.get("eval_metric", None)
        self.squad_v2_format = kwargs["DATASET_ARGS"].get("squad_v2_format", False)
        self.model_args = {}
        self.flash_attention = kwargs.get("flash_attention2", False)

    def prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        pad_on_right = self.tokenizer.padding_side == "right"

        examples[self.input_question_column] = [
            q.lstrip() for q in examples[self.input_question_column]
        ]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.input_question_column if pad_on_right else self.input_column],
            examples[self.input_column if pad_on_right else self.input_question_column],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.target_column][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        pad_on_right = self.tokenizer.padding_side == "right"

        examples[self.input_question_column] = [
            q.lstrip() for q in examples[self.input_question_column]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.input_question_column if pad_on_right else self.input_column],
            examples[self.input_column if pad_on_right else self.input_question_column],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

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
                "question_answering",
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
                        model = AutoModelForQuestionAnswering.from_pretrained(
                            self.model_path,
                            use_flash_attention_2=self.flash_attention,
                            **self.model_args,
                        ).to(self.device)
                    except:
                        model = AutoModelForQuestionAnswering.from_pretrained(
                            self.model_path,
                            use_flash_attention_2=self.flash_attention,
                            **self.model_args,
                        )
                else:
                    model = AutoModelForQuestionAnswering.from_pretrained(
                        self.model_path,
                        use_flash_attention_2=self.flash_attention,
                        # device_map = "auto",
                        **self.model_args,
                    )
            except Exception as e:
                raise ModelLoadingError(f"Following Error Happened : {e}") from e

        if use_bnb:
            model = prepare_model_for_kbit_training(model, self.gradient_checkpointing)
        elif self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        self.collate_fn = default_data_collator

        if not self.squad_v2_format:
            self.compute_metrics = None

        return model, self.tokenizer, model_config

    def prepare_dataset(self, dataset, processor):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return dataset.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

    def postprocess_qa_predictions(
        self, examples, features, raw_predictions, n_best_size=20, max_answer_length=30
    ):
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example[self.input_column]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(
                    self.tokenizer.cls_token_id
                )
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "text": context[start_char:end_char],
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(
                    valid_answers, key=lambda x: x["score"], reverse=True
                )[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not self.squad_v2_format:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = (
                    best_answer["text"] if best_answer["score"] > min_null_score else ""
                )
                predictions[example["id"]] = answer

        return predictions

    def compute_metrics(self, predictions):
        metric = load_metric("squad_v2" if self.squad_v2_format else "squad")
        validation_features = self.dataset["validation"].map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=self.dataset["validation"].column_names,
        )
        # raw_predictions = trainer.predict(validation_features)

        final_predictions = self.postprocess_qa_predictions(
            self.dataset["validation"], validation_features, predictions.predictions
        )

        if self.squad_v2_format:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in final_predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in final_predictions.items()
            ]
        references = [
            {"id": ex["id"], self.target_column: ex[self.target_column]}
            for ex in self.dataset["validation"]
        ]

        return metric.compute(predictions=formatted_predictions, references=references)
