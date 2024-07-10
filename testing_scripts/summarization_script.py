import yaml
import os
import torch
from pickle import dump


from tasks import Seq2Seq

with open(os.path.join("testing_scripts/summarization.yml"), "r") as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded)

# torch.cuda.memory._record_memory_history(enabled = True)

train_detector = Seq2Seq(**data_loaded)
train_detector.adapt_model()

# snapshot = torch.cuda.memory._snapshot()
# dump(snapshot, open('oom_snapshot.pickle', 'wb'))
