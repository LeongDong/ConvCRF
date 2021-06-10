import torch
import torch.nn as nn
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()

    def forward(self, pred, target):
        N = target.size()[0]
        loss_sum = torch.FloatTensor(1).cuda().zero_()

        pre_flat = pred.view(N,-1)
        target_flat = target.view(N,-1)

        inter = pre_flat * target_flat
        dice_eff = (2 * inter.sum(1) + 1e-5) / (pre_flat.sum(1) + target_flat.sum(1) + 1e-5)
        loss = 1 - dice_eff
        num = 0
        for multi_loss in loss:
                loss_sum = loss_sum + multi_loss
                num = num + 1
        return loss_sum / num

