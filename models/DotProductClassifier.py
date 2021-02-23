"""
@author: Zhongqi Miao https://github.com/zhmiao/OpenLongTailRecognition-OLTR
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""

import torch.nn as nn
from utils import *

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=512, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, test=False, device='cuda:1', *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc = init_weights(model=clf.fc, weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset, classifier=True, device=device)
        else:
            print('Random initialized classifier weights.')

    return clf