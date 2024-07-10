import yaml
import os

from tasks import SequenceClassification

with open(os.path.join("testing_scripts/text_classification.yml"), "r") as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded)

train_detector = SequenceClassification(**data_loaded)
train_detector.adapt_model()
