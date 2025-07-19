""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts import *

class featureAttentionModule(nn.Module):
    def __init__(self, dim, reduction=16):  
        super(featureAttentionModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
        
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([x_avg, x_max], dim=1)
        out = self.ca(out)

        return out
        
    
    
        
class locationAttentionModule(nn.Module):
    def __init__(self):  
        super(locationAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.sa = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7, stride=1, padding=3)        
   
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        out = self.sa(x2)

        return out
     
# FLAB
class MRAB(nn.Module):
    def __init__(self, feature_attention, location_attention): 
        super(MRAB, self).__init__()
        self.feature_attention = featureAttentionModule()
        self.location_attention = locationAttentionModule()
 
    def forward(self, x):
        out = self.sigmoid(self.conv2d(out))
        out = self.feature_attention(x) * x
        out = self.location_attention(out) * out
        return out   
    
    
class FLA_UNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(FLA_UNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = Down(ch_in=img_ch, ch_out=64)
        self.Conv2 = Down(ch_in=64, ch_out=128)
        self.Conv3 = Down(ch_in=128, ch_out=256)
        self.Conv4 = Down(ch_in=256, ch_out=512)
        self.Conv5 = Down(ch_in=512, ch_out=1024)
 
 
        self.Up5 = Up(ch_in=1024, ch_out=512)
        self.MRAB = MRAB(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = OutConv(ch_in=1024, ch_out=512)
 
        self.Up4 = Up(ch_in=512, ch_out=256)
        self.MRAB = MRAB(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = OutConv(ch_in=512, ch_out=256)
        
        self.Up3 = Up(ch_in=256, ch_out=128)
        self.MRAB = MRAB(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = OutConv(ch_in=256, ch_out=128)
        
        self.Up2 = Up(ch_in=128, ch_out=64)
        self.MRAB = MRAB(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = OutConv(ch_in=128, ch_out=64)
 
 
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self,x):
        # encoder
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
 
 
        # decoder
        d5 = self.Up5(x5)
        
        d5 = torch.cat((x4,d5),dim=1)  
        x4 = self.MRAB(x4)
        d5 = self.Up_conv5(d5)        
        d4 = self.Up4(d5)
        
        d4 = torch.cat((x3,d4),dim=1)
        x3 = self.MRAB(x3)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        
        
        d3 = torch.cat((x2,d3),dim=1)
        x2 = self.MRAB(x2)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        
        
        d2 = torch.cat((x1,d2),dim=1)
        x1 = self.MRAB(x1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1

