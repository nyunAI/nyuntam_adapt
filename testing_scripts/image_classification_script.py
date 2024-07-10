import yaml
import sys
import os
import datetime


sys.path.append('/workspace/Adapt')

file = datetime.datetime.now().strftime("%b-%d %H:%M:%S")

from tasks import ImageClassification

import logging

# logger = logging.getLogger(__name__)

logging.basicConfig(
    filename=f"test_runs/{file} img_classification_test.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logging.info("Logging started")

import logging

# logger = logging.getLogger(__name__)

logging.basicConfig(
    filename=f"test_runs/{file} img_classification_test.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logging.info("Logging started")

with open(os.path.join("image_classification.yml"), "r") as stream:
    data_loaded = yaml.safe_load(stream)


train_classifier = ImageClassification(**data_loaded)
train_classifier.adapt_model()
