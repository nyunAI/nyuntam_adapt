import torch
import warnings
from collections import OrderedDict
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    DataCollatorForLanguageModeling,
)

ALLOWED_EXTENSIONS = {".pt", ".bin", ".pth"}


# =========== Exceptions ===========
class CustomModelLoadError(RuntimeError):
    """Exception for custom model loading errors."""

    pass


class FlashAttentionError(ValueError):
    """Error for flash attention"""

    pass


class NamedModelLoadError(RuntimeError):
    """Exception for named model loading errors."""

    pass


def prepare_timm_model_support(custom_model_path, model):
    try:
        if not custom_model_path.exists():
            raise FileNotFoundError(f"Model path {custom_model_path} does not exist.")

        if custom_model_path.is_dir():
            for file_name in custom_model_path.iterdir():
                if str(file_name).split("/")[-1] in ["wds.pt", "wds.pth"]:
                    weights = torch.load(file_name)
                    break
                else:
                    raise FileNotFoundError(
                        "wds.pt(h) not found in file path. Please ensure that your weight file is named as wds.pt/wds.pth."
                    )
            if isinstance(weights, torch.nn.Module):
                print("Found a full model")
                loaded_model = weights
                return loaded_model
            elif isinstance(weights, OrderedDict):
                print("Found a state dictionary.")
                loaded_model = model.load_state_dict(weights)
                return loaded_model

    except Exception as e:
        raise CustomModelLoadError(f"Following Error Happened : {e}") from e


def prepare_mm_model_support(custom_model_path, model):
    try:
        if not custom_model_path.exists():
            raise FileNotFoundError(f"Model path {custom_model_path} does not exist.")

        if custom_model_path.is_dir():
            for file_name in custom_model_path.iterdir():
                if str(file_name).split("/")[-1] in ["wds.pt", "wds.pth"]:
                    weights = torch.load(file_name)
                    break
                else:
                    raise FileNotFoundError(
                        "wds.pt(h) not found in file path. Please ensure that your weight file is named as wds.pt/wds.pth."
                    )
            if isinstance(weights, OrderedDict):
                print("Found a state_dict.")
                loaded_model = model.load_state_dict(weights)
                return loaded_model
            elif isinstance(weights, torch.nn.Module):
                print("Found a full model file.")
                return weights
        else:
            raise FileNotFoundError(
                f"Model path {custom_model_path} should be a folder and not a file."
            )
    except Exception as e:
        raise CustomModelLoadError(f"Following Error Happened : {e}") from e


def prepare_custom_image_model_support(
    custom_model_path,
    model_name,
    framework,
    model_config="",
    num_gpu=1,
    device=None,
    use_bnb=False,
    use_flash_attention_2=False,
    **model_args,
):

    if framework.lower() == "huggingface":
        if custom_model_path.is_dir():
            for file in custom_model_path.iterdir():
                if file.name == "config.json":
                    config = file.name
                    custom_model_path = file.parent
                    break
                if file.suffix in ALLOWED_EXTENSIONS:
                    custom_model_path = file
                    break
        if custom_model_path.is_dir():
            # try:
            # For attention based model
            try:
                if num_gpu == 1:
                    try:
                        model = AutoModelForImageClassification.from_pretrained(
                            custom_model_path,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        ).to(device)
                    except:
                        model = AutoModelForImageClassification.from_pretrained(
                            custom_model_path,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )
                else:
                    model = AutoModelForImageClassification.from_pretrained(
                        custom_model_path,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True,
                        config=model_config,
                        use_flash_attention_2=use_flash_attention_2,
                        **model_args,
                    )
                config = AutoConfig.from_pretrained(custom_model_path)
            except Exception as e:
                raise CustomModelLoadError(
                    f"Error while loading custom model from {custom_model_path}"
                ) from e
        elif custom_model_path.is_file():
            if not custom_model_path.suffix in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Invalid file extension for model file {custom_model_path}. Allowed extensions: {ALLOWED_EXTENSIONS}"
                )

            try:
                load = torch.load(custom_model_path)
                if isinstance(load, OrderedDict):
                    model = AutoModelForImageClassification.from_pretrained(
                        model_name, state_dict=load
                    )
                elif isinstance(load, torch.nn.Module):  # load is a complete model
                    model = load
            except Exception as e:
                raise CustomModelLoadError(f"Following Error Happened : {e}") from e

    else:
        load = torch.load(custom_model_path)
        try:
            if isinstance(load, torch.nn.Module):
                model = load
            elif isinstance(load, OrderedDict):
                pass

        except Exception as e:
            raise CustomModelLoadError(
                f"Error while loading custom model from {custom_model_path}"
            ) from e

    return model


def prepare_custom_model_support(
    custom_model_path,
    task,
    model_name,
    model_config="",
    num_gpu=1,
    device=None,
    use_bnb=False,
    use_flash_attention_2=False,
    **model_args,
):

    if not custom_model_path.exists():
        raise FileNotFoundError(f"Model path {custom_model_path} does not exist.")

    # custom model path is always a directory. Check if config.json exists otherwise change model path to .pt or .bin file inside the directory

    if custom_model_path.is_dir():
        flag = False
        for file in custom_model_path.iterdir():
            if file.name == "config.json":
                config = file.name
                custom_model_path = file.parent
                flag = True
                break
        if not flag:
            for file in custom_model_path.iterdir():
                if file.suffix in ALLOWED_EXTENSIONS:
                    custom_model_path = file
                    break

    if custom_model_path.is_dir():
        # Load model from a directory saved with `.save_pretrained()`
        try:
            if task in ["summarization", "translation"]:
                if num_gpu == 1:
                    try:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            custom_model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        ).to(device)
                    except:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            custom_model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        custom_model_path,
                        use_flash_attention_2=use_flash_attention_2,
                        # device_map="auto",
                        **model_args,
                    )

            elif task == "question_answering":
                if num_gpu == 1:
                    try:
                        model = AutoModelForQuestionAnswering.from_pretrained(
                            custom_model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        ).to(device)
                    except:
                        model = AutoModelForQuestionAnswering.from_pretrained(
                            custom_model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )
                else:
                    model = AutoModelForQuestionAnswering.from_pretrained(
                        custom_model_path,
                        use_flash_attention_2=use_flash_attention_2,
                        # device_map="auto",
                        **model_args,
                    )

            elif task in ["ner", "pos", "chunk"]:
                if num_gpu == 1:
                    try:
                        model = AutoModelForTokenClassification.from_pretrained(
                            custom_model_path,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        ).to(device)
                    except:
                        model = AutoModelForTokenClassification.from_pretrained(
                            custom_model_path,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )
                else:
                    model = AutoModelForTokenClassification.from_pretrained(
                        custom_model_path,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True,
                        config=model_config,
                        use_flash_attention_2=use_flash_attention_2,
                        # device_map="auto",
                        **model_args,
                    )

            elif task == "text_classification":
                if num_gpu == 1:
                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            custom_model_path,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        ).to(device)
                    except:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            custom_model_path,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        custom_model_path,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True,
                        config=model_config,
                        use_flash_attention_2=use_flash_attention_2,
                        # device_map="auto",
                        **model_args,
                    )

            elif task == "text_generation":
                if num_gpu == 1:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            custom_model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        ).to(device)
                    except:
                        model = AutoModelForCausalLM.from_pretrained(
                            custom_model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        custom_model_path,
                        use_flash_attention_2=use_flash_attention_2,
                        **model_args,
                    )

            config = AutoConfig.from_pretrained(custom_model_path)
        except Exception as e:
            raise CustomModelLoadError(f"Following Error Happened : {e}") from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
        except Exception as e:
            warnings.warn(
                f"No tokenizer found at {custom_model_path} Loading instead from {model_name}"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif custom_model_path.is_file():

        if not custom_model_path.suffix in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension for model file {custom_model_path}. Allowed extensions: {ALLOWED_EXTENSIONS}"
            )

        try:
            load = torch.load(custom_model_path)
            if isinstance(load, OrderedDict):  # load is a state_dict
                if task in ["summarization", "translation"]:
                    if num_gpu == 1:
                        try:
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            ).to(device)
                        except:
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            )
                    else:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            custom_model_path,
                            state_dict=load,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )

                elif task == "question_answering":
                    if num_gpu == 1:
                        try:
                            model = AutoModelForQuestionAnswering.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            ).to(device)
                        except:
                            model = AutoModelForQuestionAnswering.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            )
                    else:
                        model = AutoModelForQuestionAnswering.from_pretrained(
                            custom_model_path,
                            state_dict=load,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )

                elif task in ["ner", "pos", "chunk"]:
                    if num_gpu == 1:
                        try:
                            model = AutoModelForTokenClassification.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                config=model_config,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            ).to(device)
                        except:
                            model = AutoModelForTokenClassification.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                config=model_config,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            )
                    else:
                        model = AutoModelForTokenClassification.from_pretrained(
                            custom_model_path,
                            state_dict=load,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )

                elif task == "text_classification":
                    if num_gpu == 1:
                        try:
                            model = AutoModelForSequenceClassification.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                config=model_config,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            ).to(device)
                        except:
                            model = AutoModelForSequenceClassification.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True,
                                config=model_config,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            )
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            custom_model_path,
                            state_dict=load,
                            ignore_mismatched_sizes=True,
                            trust_remote_code=True,
                            config=model_config,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )

                elif task == "text_generation":
                    if num_gpu == 1:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            ).to(device)
                        except:
                            model = AutoModelForCausalLM.from_pretrained(
                                custom_model_path,
                                state_dict=load,
                                use_flash_attention_2=use_flash_attention_2,
                                **model_args,
                            )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            custom_model_path,
                            state_dict=load,
                            use_flash_attention_2=use_flash_attention_2,
                            **model_args,
                        )

            elif isinstance(load, torch.nn.Module):  # load is a complete model
                model = load

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
        except Exception as e:
            raise CustomModelLoadError(f"Following Error Happened : {e}") from e

    return model, tokenizer, config
