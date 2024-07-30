import sys
import torch
import os
import os.path as osp
from typing import Sequence
from pycocotools.coco import COCO

from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.model.base_module import BaseModule
from mim.commands.search import get_model_info
from mim.utils import download_from_file


from mmdet.utils import setup_cache_size_limit_of_dynamo
from nyuntam_adapt.core.base_task import BaseTask
from nyuntam_adapt.utils.task_utils import MMDETECTION_DEFAULT_MODEL_WEIGHT_MAPPING
from nyuntam_adapt.tasks.custom_model import prepare_mm_model_support

import logging


class Obj_detection_mmdet(BaseTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        setup_cache_size_limit_of_dynamo()
        print(kwargs)
        args = kwargs["MMLABS_ARGS"]
        # load config
        self.logging_path = kwargs.get("LOGGING_PATH")
        self.logger = logging.getLogger(__name__)
        self.config_file_name = self.model_path
        checkpoint_url, config_file = self.get_config_details(self.config_file_name)

        self.cfg = Config.fromfile(config_file)

        self.cfg.launcher = args["launcher"]
        self.cfg.train_ann_file = args["train_ann_file"]
        self.cfg.val_ann_file = args["val_ann_file"]
        self.cfg.dest_root = args["dest_root"]
        self.checkpoint_intervals = args["checkpoint_interval"]

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args["work_dir"] is not None:
            # update configs according to CLI args if args.work_dir is not None
            self.cfg.work_dir = args["work_dir"]
        elif self.cfg.get("work_dir", None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            self.cfg.work_dir = osp.join(
                "./work_dirs", osp.splitext(osp.basename(config_file))[0]
            )

        # enable automatic-mixed-precision training
        if args["amp"] is True:
            self.cfg.optim_wrapper.type = "AmpOptimWrapper"
            self.cfg.optim_wrapper.loss_scale = "dynamic"

        # enable automatically scaling LR
        if args["auto_scale_lr"]:
            if (
                "auto_scale_lr" in self.cfg
                and "enable" in self.cfg.auto_scale_lr
                and "base_batch_size" in self.cfg.auto_scale_lr
            ):
                self.cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError(
                    'Can not find "auto_scale_lr" or '
                    '"auto_scale_lr.enable" or '
                    '"auto_scale_lr.base_batch_size" in your'
                    " configuration file."
                )

        # resume is determined in this priority: resume from > auto_resume
        if args["resume"] == "auto":
            self.cfg.resume = True
            self.cfg.load_from = None
        elif args["resume"] == False:
            self.cfg.resume = False
        elif args["resume"] is not None:
            self.cfg.resume = True
            self.cfg.load_from = args["resume"]

        abs_filename = self.download_checkpoint(args["dest_root"], checkpoint_url)
        if not self.local_model_path:
            self.cfg.load_from = abs_filename

        self.update_config()

    @staticmethod
    def get_config_details(config_file_name):

        model_info = get_model_info(
            "mmdet", shown_fields=["weight", "config", "model"], to_dict=True
        )
        # config_key = MMdetection_Default_Model_mapping[model_name]
        config_key = config_file_name

        if osp.isfile(config_key):
            config_file = config_key
            weights = config_file.split("/")[-1]
            checkpoint_url = MMDETECTION_DEFAULT_MODEL_WEIGHT_MAPPING[weights[:-3]]
        else:
            config_file = model_info[config_key]["config"]
            config_file = osp.join("nyuntam_adapt/tasks/object_detection_mmdet/mmdetection", config_file)
            checkpoint_url = model_info[config_key]["weight"]

        return checkpoint_url, config_file

    @staticmethod
    def download_checkpoint(dest_root, checkpoint_url):
        os.makedirs(dest_root, exist_ok=True)
        filename = checkpoint_url.split("/")[-1]
        checkpoint_path = osp.join(dest_root, filename)
        absolute_path = osp.abspath(checkpoint_path)
        if not osp.exists(absolute_path):
            download_from_file(checkpoint_url, checkpoint_path, check_certificate=True)

        return absolute_path

    @staticmethod
    def get_classes(annotation_file):
        classes = []
        coco = COCO(annotation_file)
        categories = coco.loadCats(coco.getCatIds())
        for cat in categories:
            classes.append(cat["name"])

        return classes, len(classes)  # get class names

    def update_config(self):

        self.cfg.data_root = self.custom_dataset_path

        # Unused parameters give error in DDP
        try:
            self.cfg.find_unused_parameters = True
        except:
            pass

        if self.cfg.data_root:
            abs_path_annotation_file = osp.abspath(
                self.cfg.data_root
                + "/"
                + self.train_dir
                + "/"
                + self.cfg.train_ann_file
            )
        else:
            abs_path_annotation_file = osp.abspath(
                self.train_dir + "/" + self.cfg.train_ann_file
            )
        classes_list, num_classes = self.get_classes(abs_path_annotation_file)
        self.cfg.metainfo = {"classes": classes_list}

        # update annotations file name as annotation_coco.json

        if self.model_name == "yolox":
            self.cfg.train_dataloader.dataset.dataset.ann_file = (
                self.train_dir + "/" + self.cfg.train_ann_file
            )
            self.cfg.train_dataloader.dataset.dataset.data_root = self.cfg.data_root
            self.cfg.train_dataloader.dataset.dataset.data_prefix.img = self.train_dir
            self.cfg.train_dataloader.dataset.dataset.metainfo = self.cfg.metainfo

        else:
            self.cfg.train_dataloader.dataset.ann_file = (
                self.train_dir + "/" + self.cfg.train_ann_file
            )
            self.cfg.train_dataloader.dataset.data_root = self.cfg.data_root
            self.cfg.train_dataloader.dataset.data_prefix.img = self.train_dir
            self.cfg.train_dataloader.dataset.metainfo = self.cfg.metainfo

        self.cfg.val_dataloader.dataset.ann_file = (
            self.val_dir + "/" + self.cfg.val_ann_file
        )
        self.cfg.val_dataloader.dataset.data_root = self.cfg.data_root
        self.cfg.val_dataloader.dataset.data_prefix.img = self.val_dir
        self.cfg.val_dataloader.dataset.metainfo = self.cfg.metainfo

        self.cfg.test_dataloader = self.cfg.val_dataloader

        # Modify metric config
        self.cfg.val_evaluator.ann_file = (
            self.cfg.data_root + "/" + self.val_dir + "/" + self.cfg.val_ann_file
        )
        self.cfg.test_evaluator = self.cfg.val_evaluator

        self.cfg.seed = self.seed
        self.cfg.train_cfg.val_interval = self.interval_steps
        self.cfg.default_hooks.checkpoint.interval = self.checkpoint_intervals
        self.cfg.train_cfg.max_epochs = self.num_epochs
        self.cfg.train_dataloader.batch_size = self.batch_size

        # Changing optimizer parameters:
        self.cfg.optim_wrapper["accumulative_counts"] = self.gradient_accumulation_steps
        self.cfg.optim_wrapper["optimizer"]["type"] = self.optimizer_name
        self.cfg.optim_wrapper["optimizer"]["lr"] = self.lr
        self.cfg.optim_wrapper["optimizer"]["weight_decay"] = self.weight_decay
        if "adam" in self.optimizer_name.lower():
            self.cfg.optim_wrapper["optimizer"]["eps"] = self.adam_epsilon
            self.cfg.optim_wrapper["optimizer"]["betas"] = (self.beta1, self.beta2)

        # Changing scheduler parameters:
        scheduler_args = [
            dict(type="MultiStepLR", milestones=self.milestones, gamma=self.gamma),
            dict(type="LinearLR"),
            dict(type="CosineAnnealingLR", T_max=self.t_max),
        ]
        for scheduler in scheduler_args:
            if self.scheduler_type == scheduler["type"]:
                self.cfg.param_scheduler = scheduler
        self.cfg.param_scheduler["begin"] = self.begin
        self.cfg.param_scheduler["end"] = self.end

        if self.warmup:
            self.cfg.lr_config = dict(policy=self.scheduler_type)
            self.cfg.lr_config["warmup_iters"] = self.warmup_iters
            self.cfg.lr_config["warmup_ratio"] = self.warmup_ratio

        if self.local_model_path:
            try:
                for file in self.local_model_path.iterdir():
                    if str(file).split("/")[-1] in ["wds.pt", "wds.pth"]:
                        self.cfg.load_from = str(file)
                        self.logger.info(f"MODEL WEIGHTS LOADED FROM {file}")
            except:
                pass

        try:
            self.cfg.model.bbox_head.num_classes = num_classes
        except AttributeError:
            if isinstance(self.cfg.model.bbox_head, Sequence):
                self.cfg.model.bbox_head[0]["num_classes"] = num_classes

    def get_checkpointing_modules(self, model):

        checkpointing_modules = []
        for name, param in model.named_children():
            if isinstance(param, BaseModule):
                checkpointing_modules.append(name)

        return checkpointing_modules

    def adapt_model(self):

        # build the runner from config
        if "runner_type" not in self.cfg:
            # build the default runner
            runner = Runner.from_cfg(self.cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(self.cfg)

        model_type = self.model_name.lower()
        self.model = runner.model
        if self.local_model_path:
            self.model = prepare_mm_model_support(self.local_model_path, self.model)
            print(f"Model weights are loaded from : {self.local_model_path}")

        self.add_peft_modules(model_type)
        # runner.cfg.activation_checkpointing = self.get_checkpointing_modules(self.model)

        # TODO - Add an if statement for GRADIENT_CHECKPOINTING
        runner.cfg.activation_checkpointing = ["backbone", "neck"]

        runner.train()
        if self.eval_bool:
            runner.test()

        self.save_peft_modules()

        if self.merge_adapter:
            try:
                self.peft_model.unload_and_merge()
                self.logger.info("PEFT modules merging done.")
            except Exception as e:
                self.logger.info(
                    f"Following exception happened while attempting to merge : {Exception}"
                )

        self.save_adapted_model()

        torch.cuda.empty_cache()  # clears Adapter modules from GPU memory and frees up memory
        self.logger.info("Model Adaptation Completed")

        return __name__