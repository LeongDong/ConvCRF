import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(double_conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inconv, self).__init__()
        self.conv=double_conv(in_ch,out_ch)
    def forward(self,x):
        x=self.conv(x)
        return x

class down(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(down,self).__init__()
        self.conv=nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch,out_ch)
        )

    def forward(self,x):
        x=self.conv(x)
        return x

class up(nn.Module):
    def __init__(self,in_ch,out_ch,TransConv=True):
        super(up,self).__init__()
        if TransConv:
            self.up=nn.ConvTranspose2d(in_ch,in_ch//2,2,stride=2)
        else:
            self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=double_conv(in_ch,out_ch)
    def forward(self,xl,xr):
        xr=self.up(xr)
        diffH=xr.size()[2] - xl.size()[2]
        diffW=xr.size()[3] - xl.size()[3]
        xl=F.pad(xl,(diffW//2,diffW-diffW//2,diffH//2,diffH-diffH//2))
        x=torch.cat([xr,xl],dim=1)
        x=self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(outconv, self).__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=1)

    def forward(self,x):
        x=self.conv(x)
        return x