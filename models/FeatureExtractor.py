"""
# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.
"""

from models.ResNetFeature import *
from utils import *
import torchvision
import torch
from efficientnet_pytorch import EfficientNet

class res(torch.nn.Module):
    def __init__(self, model, size=512, vgg=False, efficientnet=False):
        super(res, self).__init__()
        print("Creating different ResNet model")
        self.model = model
        self.efficientnet = efficientnet
        self.vgg = vgg

        if not self.efficientnet and not self.vgg:
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.vgg:
            self.model = self.model.features

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(size, 512, bias=True)


    def forward(self, x):
        if not self.efficientnet:
            x = self.model(x)
        else:
            x = self.model.extract_features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

        
def create_model(use_selfatt=False, stage1_weights=False, dataset=None, caffe=False, test=False, backbone='res10', device='cuda:1', *args):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    print(backbone)
    if backbone == 'res10':
        print("Creating res10 weights")
        model = ResNet(BasicBlock, [1, 1, 1, 1])
    elif backbone == 'res18':
        print("Creating res18 weights")
        model = torchvision.models.resnet18(pretrained=caffe)
        model = res(model=model)
    elif backbone == 'res34':
        print("Creating res34 weights")
        model = torchvision.models.resnet34(pretrained=caffe)
        model = res(model=model)
    elif backbone == 'res50':
        print("Creating res50 weights")
        model = torchvision.models.resnet50(pretrained=caffe)
        model = res(model=model, size = 2048)
    elif backbone == 'res101':
        print("Creating res101 weights")
        model = torchvision.models.resnet101(pretrained=caffe)
        model = res(model=model, size = 2048)
    elif backbone == 'shufflenet':
        print("Creating shufflenet weights")
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=caffe)
        model = res(model=model, size = 464)
    elif backbone == 'incv3':
        print("Creating Inception v3 weights")
        model = torchvision.models.inception_v3(pretrained=caffe)
        model = res(model=model, size=2048)
    elif backbone == 'densenet':
        print("Creating densenet weights")
        model = torchvision.models.densenet121(pretrained=caffe)
        model = res(model=model, size=1024, vgg=True)
    elif backbone == 'vgg16':
        print("Creating VGG16 weights")
        model = torchvision.models.vgg16(pretrained=caffe)
        model = res(model=model, size=1000, vgg=True)
    elif backbone == 'rsx':
        print("Creating ResNetx weights")
        model = torchvision.models.resnext50_32x4d(pretrained=caffe)
        model = res(model=model, size=2048)
    elif backbone == 'efficientnetb0':
        print("Creating EfficientNet-b0 weights")
        model = EfficientNet.from_name('efficientnet-b0')
        model = res(model=model, size = 1280, efficientnet=True)
    elif backbone == 'efficientnetb3':
        print("Creating EfficientNet-b3 weights")
        model = EfficientNet.from_name('efficientnet-b3')
        model = res(model=model, size = 1536, efficientnet=True)
    elif backbone == 'efficientnetb5':
        print("Creating EfficientNet-b5 weights")
        model = EfficientNet.from_name('efficientnet-b5')
        model = res(model=model, size = 2048, efficientnet=True)
    elif backbone == 'efficientnetb7':
        print("Creating EfficientNet-b7 weights")
        model = EfficientNet.from_name('efficientnet-b7')
        model = res(model=model, size = 2560, efficientnet=True)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            model = init_weights(model=model, weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset, device=device)
        else:
            print('No Pretrained Weights For Feature Model.')
    return model
