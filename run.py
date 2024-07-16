# NOTE - This main file should be outside the Adapt folder while testing/ running
# this is because it is outside the Adapt folder in production
# and is written with to follow paths accordingly.


import os
import sys

# NOTE - Uncomment line 10, Comment line 10 for production.
# sys.path.append("/workspace/Adapt")
sys.path.append("/wspace/Adapt")
import argparse
import yaml
from yaml_json_flattened import execute_yaml_creation

# os.chdir("../Adapt")

from utils import ErrorMessageHandler

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", default="", type=str)
parser.add_argument("--json_path", default="", type=str)
parser.add_argument("--local_rank", type=int, default=-1)
input_mode = parser.parse_args()


# Removing /Adapt from the given path
def remove_adapt_prefix(path):
    if path.startswith("Adapt/"):
        return path[6:]  # Remove the "Adapt/" prefix
    else:
        return path


if input_mode.yaml_path == "":
    input_mode.json_path = remove_adapt_prefix(input_mode.json_path)
    yaml_path = execute_yaml_creation(input_mode.json_path)
    file = yaml_path
else:
    input_mode.yaml_path = remove_adapt_prefix(input_mode.yaml_path)
    file = input_mode.yaml_path

from dataclasses import asdict


with open(file, "r") as f:
    args = yaml.safe_load(f)

from logging_adapt import define_logger
import logging

logging_path=args['LOGGING_PATH']
os.makedirs(logging_path, exist_ok=True)


# cuda_env = CudaDeviceEnviron(cuda_device_ids_str=args["cuda_id"])

from tasks import AdaptParams, create_instance

adapt_params = create_instance(AdaptParams, args)


# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args["cuda_id"]))


try:
    # Loading appropriate module based on input arg
    if args["TASK"] == "text_generation":
        from tasks import CausalLLM

        train_llama = CausalLLM(**asdict(adapt_params))
        name = train_llama.adapt_model()

    elif args["TASK"] == "text_classification":
        from tasks import SequenceClassification

        train_detector = SequenceClassification(**asdict(adapt_params))
        name = train_detector.adapt_model()

    elif args["TASK"] == "Seq2Seq_tasks":
        if args["subtask"] == "translation":
            from tasks import Translation

            train_detector = Translation(**asdict(adapt_params))

        elif args["subtask"] == "summarization":
            from tasks import Seq2Seq

            train_detector = Seq2Seq(**asdict(adapt_params))

        name = train_detector.adapt_model()

    elif args["TASK"] in ["question_answering", "Question Answering"]:
        from tasks import QuestionAnswering

        train_detector = QuestionAnswering(**asdict(adapt_params))
        name = train_detector.adapt_model()

    elif args["TASK"] == "image_classification":
        from tasks import ImageClassification

        train_classifier = ImageClassification(**asdict(adapt_params))
        name = train_classifier.adapt_model()

    logger = logging.getLogger(name)
    logger.info("JOB COMPLETED")

except Exception as e:
    logger = define_logger("failure", args["LOGGING_PATH"])
    logger.exception(e)
    error_handler = ErrorMessageHandler()
    custom_err_message = error_handler.get_custom_error_message(e)
    if custom_err_message != "not_found":
        logger.info(custom_err_message)
    else:
        pass

    logger.info("JOB FAILED")
