"""
@author: Zhongqi Miao https://github.com/zhmiao/OpenLongTailRecognition-OLTR
Copyright (c) 2019, Zhongqi Miao
# Licensed under the BSD 3-Clause License.
"""

import torch.nn as nn

def create_loss ():
    print('Loading Softmax Loss.')
    return nn.CrossEntropyLoss()

