import yaml
import os
import sys 



sys.path.append('/workspace/Adapt')
from tasks import QuestionAnswering

with open(os.path.join("testing_scripts/question_answering.yml"), "r") as stream:
    data_loaded = yaml.safe_load(stream)
    


train_detector = QuestionAnswering(**data_loaded)
train_detector.adapt_model()
