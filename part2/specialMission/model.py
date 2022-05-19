from tkinter import N
from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import numpy as np
"""
사용가능 모델 
사용방법 --model 모델명
BaseModel,
BaseModel2,
FCNResnet101,
Deeplabv3_Resnet50,
Deeplabv3_Resnet101,
UnetPlusPlus_Resnet50,
DeconvNet_VGG16,
FCN8_VGG16,
UNet,
UnetPlusPlus_Efficient4,
UnetPlusPlus_Efficient_b5,

"""


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.fcn_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)

class BaseModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.fcn_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, 11, kernel_size=1)


    def forward(self, x):
        return self.model(x)

class FCNResnet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.fcn_resnet101(pretrained=True)
    
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)

class Deeplabv3_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)

class Deeplabv3_Resnet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, 11, kernel_size=1)
  
    def forward(self, x):
        return self.model(x)

# use smp
class UNet_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "resnet50"      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.Unet(         # choose architecture 
            encoder_name=encoder_name,     
            encoder_weights="imagenet",   
            in_channels=3,              
            classes=11,                 
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UNet_Resnet101(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "resnet101"  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.Unet(      # choose architecture 
            encoder_name=encoder_name,         
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=11,                      
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "resnet50"        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(   # choose architecture 
            encoder_name=encoder_name,     
            encoder_weights="imagenet", 
            in_channels=3,              
            classes=11,                
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus_Efficient4(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b4"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)
    
class UnetPlusPlus_Efficient7(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b7"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)    

class UnetPlusPlus_Efficient5_N(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "timm-efficientnet-b5"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="noisy-student",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='noisy-student')
    
    def forward(self, x):
        return self.model(x)    

class UnetPlusPlus_Efficient_b5(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b5"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus_Efficient_b7(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b7"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)
        
class Deeplabv3Plus_Resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "resnet101"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class Deeplabv3Plus_Resnext50(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "resnext50_32x4d"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class Deeplabv3Plus_Resnext101(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "resnext101_32x8d"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class Deeplabv3Plus_SEResnet152(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "se_resnet152"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UNetPlusPlus_HRNet30(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "tu-hrnet_w30"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(    # choosqe architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)