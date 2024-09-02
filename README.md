# Nyuntam Adapt
A robust adaptation module that effortlessly fine-tunes and performs transfer learning on a diverse range of deep learning tasks and models. With ADAPT, users can adapt large and even medium-sized vision and language models seamlessly, enabling the creation of custom models with just a few clicks. This module incorporates state-of-the-art adaptation methods such as (Q)-LoRA, SSF, etc., allowing users to strike the optimal balance across various model metrics, including parameter count and training throughput.

## Installation
Installation can be performed either via installing requirements in a virtual environment or by utilizing our docker images. To quickly run Adapt for experimentation and usage, utilize Nyun CLI to get Adapt running in a few clicks. For contributing to Adapt build docker containers from the available docker image or create a virtual enviroment from the provided requirements.txt.

### Nyunzero CLI
The recommended method to install and use nyuntam-adapt is via the nyunzero-cli. Further details can be found here : [NyunZero CLI](https://github.com/nyunAI/nyunzero-cli)

### Git + Docker
Nyuntam-Adapt can also be used by cloning the repository and pulling hhe required docker. 

1. **Git Clone** : First, clone the repository to your local machine:
    ```bash
    $ git clone --recursive https://github.com/nyunAI/nyuntam_adapt.git
    $ cd nyuntam_adapt
    ```

2. **Docker Pull**: Next, pull the docker container and run it : 


    ```bash 
    $ docker pull nyunadmin/nyunzero_adapt:v0.1

    $ docker run -it -d --gpus all -v /dev/shm:/dev/shm -v $(pwd):/workspace --name {CONTAINER_NAME} --network=host nyunadmin/nyunzero_adapt:v0.1 bash 
    ```

<span style="color:red">**NOTE:**</span> nvidia-container-toolkit is expected to be installed before the execution of this

3. **Docker run**: Next, start the docker. 

    ```sh
    $ docker exec -it {CONTAINER_NAME} /bin/bash
    ```

Now you are all set to finetune models using nyuntam adapt.



### Git + virtual environment

Nyuntam Aadpt can also be used by cloning the repository and setting up a virtual environment. 

1. **Git Clone** : First, clone the repository to your local machine:
    ```bash
    $ git clone --recursive https://github.com/nyunAI/nyuntam_adapt.git
    $ cd nyuntam_adapt
    ```

2. **Create a virtual environment using Venv**
   ```sh
   python3 -m venv {ENVIRONMENT_NAME}
   source {ENVIRONMENT_NAME}/bin/activate
   ```

3. **Pip install requirements**
   ```sh
   pip install -r requirements.txt
   ```

## Usage 

### Setting up the YAML files
If the dataset and models weights exist online (huggingface hub) then the experiments can be started easily. Adapt requires a yaml file for all the required hyperparameters to finetune a model. The yaml files for all the tasks are here : [YAML FILES](/testing_scripts/flattened_yml)

The major hyperparameters are metioned below : 

- ***Dataset***
```yaml
DATASET: teknium/openhermes
DATASET_TYPE: LLM - Openhermes
DATASET_CONFIG : {}
CUSTOM_DATASET_PATH :
```
<ol>
<li> "DATASET" refers to the name of the dataset as can be found on the hugginface hub. eg - 'mlabonne/guanaco-llama2-1k'.</li>

<li> The "DATASET_TYPE" refers to the user given name for the dataset. </li>

<li> Some datasets like opus-books have subsets like - "en-fr". These are used while downloading the dataset and can be mentioned in "DATASET_CONFIG". </li>

<li> "CUSTOM_DATASET_PATH" refers to the dataset path is user has a custom dataset that is not on huggingface hub. </li>
</ol>

Further information about using custom datasets can be found [here](https://nyunai.github.io/nyun-docs/dataset/)

- ***Model***

```yaml
Library: Huggingface
MODEL: Llama-3
MODEL_PATH: meta-llama/Meta-Llama-3-70B
LOCAL_MODEL_PATH : 
```
<ol>
<li> "Library" is self explanatory and refers to the library which is used for the model. for all the language tasks, Adapt supports Huggingface and Timm and Huggingface are supported for vision tasks. </li>

<li> "MODEL" refers to the user given name of the model and can be chosen by the user. this DOES NOT refer to the model details. </li>

<li> "MODEL_PATH" refers to the huggingface model id and needs to be the same as that in the HF hub. </li>

<li> "LOCAL_MODEL_PATH" refers to the path of the model weights if the user has custom model weights. </li>

</ol>

Further information about using custom model weights can be found [here](https://nyunai.github.io/nyun-docs/model/#using-model-relative-folder-path)
- ***Training Arguments***

```yaml
SEED : 56
DO_TRAIN : True
DO_EVAL : False
NUM_WORKERS : 4
BATCH_SIZE : 4
EPOCHS : 1
STEPS : 100
OPTIMIZER : 'paged_adamw_32bit'
LR : 0.0002
SCHEDULER_TYPE : 'constant'
WEIGHT_DECAY : 0.001
BETA1 : 0.9
BETA2 : 0.999
ADAM_EPS : 1e-8              
INTERVAL : 'steps'
INTERVAL_STEPS : 50
NO_OF_CHECKPOINTS : 5
FP16 : True
RESUME_FROM_CHECKPOINT : False
GRADIENT_ACCUMULATION_STEPS : 1
GRADIENT_CHECKPOINTING : True
GROUP_BY_LENGTH : True
REMOVE_UNUSED_COLUMNS : True
```

The training arguments can be changed to alter the training parameters like learning rate, scheduler, weight decay, etc. 

- ***Layers to tune***

```yaml
LAST_LAYER_TUNING : False
FULL_FINE_TUNING :  False
```
<ol>
<li> "LAST_LAYER_TUNING" should be set to True when only the last layer is to be fine tuned of the model. When PEFT methods are used, LAST_LAYER_TUNING ensures that the layers that are to be finetuned are the PEFT modules and the last layer of the network. </li>

<li> "FULL_FINE_TUNING" should be set to True when the all the layers of the model weights are to be finetuned. This utilises lot of computational resources. </li>
</ol>

- ***PEFT arguments***
```yaml 
PEFT_METHOD : 'LoRA'
r : 2
alpha : 16
dropout : 0.1
peft_type : 'LoRA'
target_modules :   
fan_in_fan_out : False
init_lora_weights : True  
```
<ol>
<li> "PEFT_METHOD" and "peft_type" refers to the type of PEFT method used to create the adapters. Currently supported PEFT methods are -  "LoRA", "DoRA" or "SSF". </li>

<li> "r" and "alpha" are parameters for LoRA and DoRA and theyr details can be found here : [LoRA/DoRA](https://arxiv.org/pdf/2106.09685). </li>

<li> "target_modules" are the layers where LoRA adapters are to be added. By default they will be added to every Linear/Conv2D layers in the model. </li>
</ol>


- ***Quantization arguments***
```yaml
load_in_4bit : True
bnb_4bit_compute_dtype : "float16"
bnb_4bit_quant_type : "nf4"
bnb_4bit_use_double_quant : False 

load_in_8bit : False 
llm_int8_threshold : 6.0
llm_int8_skip_modules : 
llm_int8_enable_fp32_cpu_offload : False
llm_int8_has_fp16_weight : False
```
<ol>
<li> "load_in_4bit" should be set to True is the model is to be loaded in 4bit and simillarly "load_in_8bit" should be set to True if the model is to be loaded in 8bit. </li>

<li> Details of the other parameteres can be found here : [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index)</li>
</ol>

- ***Distributed Training Arguments***
```yaml 
DDP: False
num_nodes: 1
FSDP: False
```

To train the model using [Full Sharded Data Parallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), FSDP parameter has to be set to True. The FSDP arguments are mentioned in accelerate_config.yaml. 

```yaml
compute_environment: LOCAL_MACHINE                             
debug: true                        
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: NO_PREFETCH
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: 
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
<ol>
<li> "distributed_type" should be set to FSDP to ensure that model being trained on multiple GPUs (if they are available)</li>

<li> "fsdp_sharding_strategy" refers to the sharding strategy to be used while training the model. If "NO_SHARD" is selected, then it behaves like  [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)</li>

<li> num_processes NUM_PROCESSES (int) — The total number of processes to be launched in parallel.</li>

<li> num_machines NUM_MACHINES (int) — The total number of machines used in this training.</li>
</ol>

Further information about the FSDP parameters can be found [here](https://huggingface.co/docs/accelerate/en/package_reference/cli)


<span style="color:red">**NOTE**</span> : FURTHER INFORMATION ABOUT THE YAML FILES CAN BE FOUND IN OUR [OFFICIAL DOCUMENTATION](https://nyunai.github.io/nyun-docs/)


### Run command
```sh
python main.py --yaml_path {path to the yaml file}
```

This command runs the main file with the configuration setup in the yaml file.

## Acknowledgments and Citations

This repository utilizes various state-of-the-art methods and algorithms developed by the research community. We acknowledge the following works that have contributed to the development and performance of the Nyuntam Text Generation module:

- [**LoRA: Low-Rank Adaptation of Large Language Models**](https://arxiv.org/abs/2106.09685)  
  *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.* arXiv preprint, 2021.
  ```bibtex
  @misc{hu2021loralowrankadaptationlarge,
      title={LoRA: Low-Rank Adaptation of Large Language Models},
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.09685},
  }
  ```

- [**DoRA: Weight-Decomposed Low-Rank Adaptation**](https://arxiv.org/abs/2402.09353)  
  *Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen.* arXiv preprint, 2024.
  ```bibtex
  @misc{liu2024doraweightdecomposedlowrankadaptation,
      title={DoRA: Weight-Decomposed Low-Rank Adaptation},
      author={Shih-Yang Liu and Chien-Yi Wang and Hongxu Yin and Pavlo Molchanov and Yu-Chiang Frank Wang and Kwang-Ting Cheng and Min-Hung Chen},
      year={2024},
      eprint={2402.09353},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.09353},
  }
  ```

- [**Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning**](https://arxiv.org/abs/2210.08823)  
  *Dongze Lian, Daquan Zhou, Jiashi Feng, Xinchao Wang.* arXiv preprint, 2023.
  ```bibtex
  @misc{lian2023scalingshiftingfeatures,
      title={Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning},
      author={Dongze Lian and Daquan Zhou and Jiashi Feng and Xinchao Wang},
      year={2023},
      eprint={2210.08823},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2210.08823},
  }
  ```

- [**QLoRA: Efficient Finetuning of Quantized LLMs**](https://arxiv.org/abs/2305.14314)  
  *Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer.* arXiv preprint, 2023.
  ```bibtex
  @misc{dettmers2023qloraefficientfinetuningquantized,
      title={QLoRA: Efficient Finetuning of Quantized LLMs},
      author={Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer},
      year={2023},
      eprint={2305.14314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.14314},
  }
  ```

- [**PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel**](https://arxiv.org/abs/2304.11277)  
  *Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, Shen Li.* arXiv preprint, 2023.
  ```bibtex
  @misc{zhao2023pytorchfsdpexperiencesscaling,
      title={PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel},
      author={Yanli Zhao and Andrew Gu and Rohan Varma and Liang Luo and Chien-Chin Huang and Min Xu and Less Wright and Hamid Shojanazeri and Myle Ott and Sam Shleifer and Alban Desmaison and Can Balioglu and Pritam Damania and Bernard Nguyen and Geeta Chauhan and Yuchen Hao and Ajit Mathews and Shen Li},
      year={2023},
      eprint={2304.11277},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2304.11277},
  }
  ```

- [**PyTorch Distributed: Experiences on Accelerating Data Parallel Training**](https://arxiv.org/abs/2006.15704)  
  *Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, Soumith Chintala.* arXiv preprint, 2020.
  ```bibtex
  @misc{li2020pytorchdistributedexperiencesaccelerating,
      title={PyTorch Distributed: Experiences on Accelerating Data Parallel Training},
      author={Shen Li and Yanli Zhao and Rohan Varma and Omkar Salpekar and Pieter Noordhuis and Teng Li and Adam Paszke and Jeff Smith and Brian Vaughan and Pritam Damania and Soumith Chintala},
      year={2020},
      eprint={2006.15704},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2006.15704},
  }
  ```

- [**Accelerate: Training and inference at scale made simple, efficient and adaptable**](https://github.com/huggingface/accelerate)  
  *Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, Benjamin Bossan.* 2022.
  ```bibtex
  @Misc{accelerate,
      title = {Accelerate: Training and inference at scale made simple, efficient and adaptable.},
      author = {Sylvain Gugger and Lysandre Debut and Thomas Wolf and Philipp Schmid and Zachary Mueller and Sourab Mangrulkar and Marc Sun and Benjamin Bossan},
      howpublished = {\url{https://github.com/huggingface/accelerate}},
      year = {2022}
  }
  ```

- [**PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods**](https://github.com/huggingface/peft)  
  *Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, Benjamin Bossan.* 2022.
  ```bibtex
  @Misc{peft,
      title = {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
      author = {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
      howpublished = {\url{https://github.com/huggingface/peft}},
      year = {2022}
  }
  ```
