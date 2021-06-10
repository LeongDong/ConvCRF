import torch.nn.functional as F
from unet_parts import *

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,TransConv=True):
        super(UNet,self).__init__()

        self.inc=inconv(n_channels,64)
        self.down1=down(64,128)
        self.down2=down(128,256)
        self.down3=down(256,512)
        factor=1 if TransConv else 2
        self.down4=down(512,1024//factor)
        self.up1=up(1024,512//factor,TransConv)
        self.up2=up(512,256//factor,TransConv)
        self.up3=up(256,128//factor,TransConv)
        self.up4=up(128,64,TransConv)
        self.outc=outconv(64,n_classes)

    def forward(self,x):
        x1=self.inc(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x=self.up1(x4,x5)
        x=self.up2(x3,x)
        x=self.up3(x2,x)
        x=self.up4(x1,x)
        x=self.outc(x)

        return F.sigmoid(x)