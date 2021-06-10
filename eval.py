import torch
import numpy as np

def eval_net(segnet,crfnet,val_img):
    segnet.eval()
    crfnet.eval()
    dice_sum=0.0
    i_mark = 1

    for i,t in enumerate(val_img):
        img = t[0]
        true_mask = t[1]

        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        true_mask = torch.from_numpy(true_mask)

        img = img.cuda()
        true_mask = true_mask.cuda()

        with torch.no_grad():
            label_pre = segnet(img)
            pre_mask = crfnet(img,label_pre)

        mask_flat = true_mask.view(-1)
        pre_flat = pre_mask.view(-1)

        inter = torch.dot(mask_flat,pre_flat)
        union = torch.sum(pre_flat)+torch.sum(mask_flat)
        t = (2.*inter.float()+1e-5) / (union.float()+1e-5)
        dice_sum = dice_sum+t
        i_mark = i

    segnet.train()
    return dice_sum / (i_mark+1)

