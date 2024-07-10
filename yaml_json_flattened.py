import yaml
import json
import os
import copy
import shutil
import zipfile
import tarfile
import ast
from utils import CudaDeviceEnviron

task_map = {
    "Text Generation": "text_generation",
    "Text Classification": "text_classification",
    "Summarization": "summarization",
    "Translation": "translation",
    "Question Answering": "question_answering",
    "Image Classification": "image_classification",
}


def fill_data(training_recipe, json_data):
    yaml_recipe = copy.deepcopy(training_recipe)
    if json_data["task"] in ["Summarization", "Translation"]:
        yaml_recipe["TASK"] = "Seq2Seq_tasks"
        yaml_recipe["subtask"] = task_map[json_data["task"]]
    else:
        yaml_recipe["TASK"] = task_map[json_data["task"]]

    yaml_recipe["JOB_SERVICE"] = json_data["job_service"]
    yaml_recipe["USER_FOLDER"] = "user_data"
    yaml_recipe["OVERWRITE_OUTPUT_DIR"] = False
    yaml_recipe["PEFT_METHOD"] = json_data["method"]
    yaml_recipe["JOB_ID"] = json_data["id"]
    yaml_recipe["OUTPUT_DIR"] = (
        f"{yaml_recipe['USER_FOLDER']}/jobs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}"
    )
    yaml_recipe["DATASET_ID"] = json_data["dataset"]["id"]
    yaml_recipe["LOGGING_PATH"] = (
        f"{yaml_recipe['USER_FOLDER']}/logs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}"
    )
    yaml_recipe["peft_type"] = json_data["method"]

    # MODEL HYPERPARAMETERS
    for model_hyperparameters in json_data["model"]:
        if model_hyperparameters not in [
            "model_subname",
            "model_name",
            "model_relative_folder_path",
            "framework",
        ]:
            yaml_recipe[model_hyperparameters.upper()] = json_data["model"][
                model_hyperparameters
            ]
        elif model_hyperparameters == "model_subname":
            yaml_recipe["MODEL_PATH"] = json_data["model"][model_hyperparameters]
        elif model_hyperparameters == "model_name":
            yaml_recipe["MODEL"] = json_data["model"][model_hyperparameters]
        elif model_hyperparameters == "model_relative_folder_path":
            if json_data["model"][model_hyperparameters]:
                if json_data["model"][model_hyperparameters] != "":
                    yaml_recipe["LOCAL_MODEL_PATH"] = os.path.join(
                        "/wspace/Adapt/custom_data",
                        json_data["model"][model_hyperparameters],
                    )
                else:
                    yaml_recipe["LOCAL_MODEL_PATH"] = None
        elif model_hyperparameters == "framework":
            yaml_recipe["Library"] = json_data["model"][model_hyperparameters]

    # DATASET HYPERPARAMETERS
    for dataset_hyperparams in json_data["dataset"]:
        if dataset_hyperparams == "dataset_name":
            yaml_recipe["DATASET"] = json_data["dataset"][dataset_hyperparams]
        elif dataset_hyperparams == "dataset_relative_folder_path":
            if json_data["dataset"][dataset_hyperparams]:
                if json_data["dataset"][dataset_hyperparams] != "":
                    yaml_recipe["CUSTOM_DATASET_PATH"] = os.path.join(
                        "/wspace/Adapt/custom_data",
                        json_data["dataset"][dataset_hyperparams],
                    )
                else:
                    yaml_recipe["CUSTOM_DATASET_PATH"] = None
        else:
            yaml_recipe[dataset_hyperparams.upper()] = json_data["dataset"][
                dataset_hyperparams
            ]

    # ALL THE REST OF THE PARAMETERS FROM METHOD HYPERPARAMETERS
    for method_hyperparams in json_data["method_hyperparameters"]:
        if method_hyperparams == "DATASET_SUBNAME":
            yaml_recipe["DATASET_CONFIG"] = json_data["method_hyperparameters"][
                method_hyperparams
            ]

        elif method_hyperparams in ["class_list", "palette", "MILESTONES"]:
            # convert string_list to list
            yaml_recipe[method_hyperparams] = ast.literal_eval(
                json_data["method_hyperparameters"][method_hyperparams]
            )
        else:
            yaml_recipe[method_hyperparams] = json_data["method_hyperparameters"][
                method_hyperparams
            ]

    return yaml_recipe


def execute_yaml_creation(json_path):
    f = open(json_path)
    json_data = json.load(f)
    task_name = task_map[json_data["task"]]
    if task_name == "text_generation":
        with open(
            os.path.join("testing_scripts/flattened_yml", "text_generation.yml")
        ) as file:
            training_recipe = yaml.full_load(file)
    elif task_name == "text_classification":
        with open(
            os.path.join("testing_scripts/flattened_yml", "text_classification.yml")
        ) as file:
            training_recipe = yaml.full_load(file)
    elif task_name == "summarization":
        with open(
            os.path.join("testing_scripts/flattened_yml", "summarization.yml")
        ) as file:
            training_recipe = yaml.full_load(file)
    elif task_name == "question_answering":
        with open(
            os.path.join("testing_scripts/flattened_yml", "question_answering.yml")
        ) as file:
            training_recipe = yaml.full_load(file)
    elif task_name == "image_classification":
        with open(
            os.path.join("testing_scripts/flattened_yml", "image_classification.yml")
        ) as file:
            training_recipe = yaml.full_load(file)
    elif task_name == "translation":
        with open(
            os.path.join("testing_scripts/flattened_yml", "translation.yml")
        ) as file:
            training_recipe = yaml.full_load(file)
    else:
        print("Invalid Task")
    yaml_recipe = fill_data(training_recipe, json_data)
    create_folders(yaml_recipe)
    with open(os.path.join(yaml_recipe["OUTPUT_DIR"], "config.yaml"), "w") as file:
        documents = yaml.dump(yaml_recipe, file)

    with open(os.path.join(yaml_recipe["OUTPUT_DIR"], "config.yaml"), "r") as f:
        args = yaml.safe_load(f)

    cuda_env = CudaDeviceEnviron(cuda_device_ids_str=args["cuda_id"])
    return os.path.join(yaml_recipe["OUTPUT_DIR"], "config.yaml")


def create_folders(
    yaml_recipe, original_data_location=None, original_model_location=None
):
    folder_name = yaml_recipe["USER_FOLDER"]
    if not os.path.exists(f"{folder_name}"):
        os.makedirs(f"{folder_name}", exist_ok=True)
        os.makedirs(f"{folder_name}/models", exist_ok=True)
        os.makedirs(f"{folder_name}/datasets", exist_ok=True)
        os.makedirs(f"{folder_name}/jobs", exist_ok=True)
        os.makedirs(f"{folder_name}/logs", exist_ok=True)
        os.makedirs(f"{folder_name}/.cache", exist_ok=True)
    os.makedirs(f"{folder_name}/datasets/{yaml_recipe['DATASET_ID']}", exist_ok=True)
    os.makedirs(
        f"{folder_name}/jobs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}",
        exist_ok=True,
    )
    os.makedirs(
        f"{folder_name}/logs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}",
        exist_ok=True,
    )
