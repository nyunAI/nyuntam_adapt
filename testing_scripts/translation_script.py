import yaml
import os
import sys
sys.path.append('/workspace/Adapt')
from tasks import Translation
with open(os.path.join("testing_scripts/translation.yml"), "r") as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded)
train_detector = Translation(**data_loaded)
train_detector.adapt_model()
