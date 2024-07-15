import timm
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from nyuntam_adapt.models import internimage  # TO register model and config


class TimmforImageClassification(nn.Module):
    def __init__(self, model_path, num_classes):
        super(
            TimmforImageClassification, self
        ).__init__()  # check does this do multiple inheritance
        # TODO check if model has head or not if not add one
        self.model_timm = timm.create_model(
            model_path, pretrained=True, num_classes=num_classes
        )

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        logits = self.model_timm(pixel_values)  # When model contains head also
        # output_hidden_states = self.model.forward_features(pixel_values)
        loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutputWithNoAttention(
            loss=loss, logits=logits, hidden_states=None
        )
