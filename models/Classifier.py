"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

Partially based on https://github.com/zhmiao/OpenLongTailRecognition-OLTR
@author: Zhongqi Miao
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""

import torch.nn as nn
from models.SquashingCosineClassifier import SquashingCosine_Classifier

from utils import *

class Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000):
        super(Classifier, self).__init__()
        print("Initializing the Meta Embedding Classifier", feat_dim, num_classes)
        self.num_classes = num_classes
        self.cosnorm_classifier = SquashingCosine_Classifier(in_dims=feat_dim, out_dims=num_classes, scale=16, margin=0.5)
        self.margin = 0.0

    def forward(self, x, labels=None, use_labels=False):
        logits = self.cosnorm_classifier(x)
        if use_labels and self.margin > 0:
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            margin_logits = 1 * (logits - one_hot * self.margin)
            other = margin_logits
        else:
            other = logits
        return logits, other

    
def create_model(feat_dim=2048, num_classes=1000, *args):
    print('Loading Meta Embedding Classifier.')
    clf = Classifier(feat_dim, num_classes)
    return clf
