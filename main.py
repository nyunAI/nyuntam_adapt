# NOTE - This main file should be outside the Adapt folder while testing/ running
# this is because it is outside the Adapt folder in production
# and is written with to follow paths accordingly.


import os
import sys
import subprocess
import yaml
import copy
import json

# NOTE - Uncomment line 10, Comment line 10 for production.
# sys.path.append("/workspace/Adapt")
sys.path.append("/wspace/Adapt")
import argparse
import yaml
from yaml_json_flattened import execute_yaml_creation

#os.chdir("./Adapt")

from utils import ErrorMessageHandler

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", default="", type=str)
parser.add_argument("--json_path", default="", type=str)
# parser.add_argument("--local-rank", type=int, default=0)
input_mode = parser.parse_args()

# if "LOCAL_RANK" not in os.environ:
#     os.environ["LOCAL_RANK"] = str(input_mode.local_rank)


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
from logging_adapt import define_logger
import logging


with open(file, "r") as f:
    args = yaml.safe_load(f)


# cuda_env = CudaDeviceEnviron(cuda_device_ids_str=args["cuda_id"])

from tasks import AdaptParams, create_instance

adapt_params = create_instance(AdaptParams, args)


# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args["cuda_id"]))

if args["DDP"]:
    # Subprocess to call torchrun for DDP training
    num_gpu = len(args["cuda_id"].split(","))
    num_nodes = args["num_nodes"]
    if input_mode.json_path:
        arg_string = f"torchrun --nnodes={num_nodes} --nproc-per-node={num_gpu} ../run.py --json_path={input_mode.json_path}"
    elif input_mode.yaml_path:
        arg_string = f"torchrun --nnodes={num_nodes} --nproc-per-node={num_gpu} ../run.py --yaml_path={input_mode.yaml_path}"

    os.system(arg_string)

elif args["FSDP"]:
    # Subprocess to call accelerate launch for FSDP training
    # YAML Creation
    config_yaml_path = "/workspace/Adapt/accelerate_config.yaml"
    final_accelerate_config = (
        "/workspace/Adapt/user_data/jobs/Adapt/accelerate_config.yaml"
    )
    with open(config_yaml_path, "r") as config_file:
        cfg = yaml.full_load(config_file)
    accelerate_config = copy.deepcopy(cfg)
    if input_mode.json_path:
        acc_json_file = open(input_mode.json_path)
        acc_json = json.load(acc_json_file)
        job_id = acc_json['id']
        final_accelerate_config = f"/workspace/Adapt/user_data/jobs/Adapt/{job_id}/accelerate_config.yaml"
        acc_json = acc_json["method_hyperparameters"]
        for args in accelerate_config:
            if args in list(acc_json.keys()):
                accelerate_config[args] = acc_json[args]
            if args == "fsdp_config":
                for fsdp_args in accelerate_config[args]:
                    if fsdp_args in list(acc_json.keys()):
                        accelerate_config["fsdp_config"][fsdp_args] = acc_json[
                            fsdp_args
                        ]
        arg_string = f"accelerate launch --config_file={final_accelerate_config} ../run.py --json_path={input_mode.json_path}"

    if input_mode.yaml_path:
        acc_yaml_file = open(input_mode.yaml_path)
        acc_yaml = yaml.full_load(acc_yaml_file)
        for args in accelerate_config:
            if args in list(acc_yaml.keys()):
                accelerate_config[args] = acc_yaml[args]
            if args == "fsdp_config":
                for fsdp_args in accelerate_config[args]:
                    if fsdp_args in list(acc_yaml.keys()):
                        accelerate_config["fsdp_config"][fsdp_args] = acc_yaml[
                            fsdp_args
                        ]
        arg_string = f"accelerate launch --config_file={final_accelerate_config} ../run_dist.py --json_path={input_mode.json_path}"

    with open(final_accelerate_config, "w") as file:
        yaml.dump(accelerate_config, file)

    os.system(arg_string)


else:
    # Subprocess to call main.py for single GPU training
    if input_mode.json_path:
        args_string = f"python run.py --json_path={input_mode.json_path}"
    elif input_mode.yaml_path:
        args_string = f"python run.py --yaml_path={input_mode.yaml_path}"
    os.system(args_string)
