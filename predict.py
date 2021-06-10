import numpy as np
import torch
torch.set_printoptions(profile="full")
from PIL import Image

from unet_para import UNet
from ConvCRF import ConvCRF
from pre_processed import crop_image, hwc_to_chw,img_Normalize
from data_loader import get_ids
import torchvision
from torchvision import utils as ImgSave

def mask_to_image(image):
    return image * 255

def predict_img(segnet, crfnet, origin_img, set_height = 256, set_width = 256, threshold = 0.5, gpu = True):
    segnet.eval()
    crfnet.eval()
    img = crop_image(origin_img,set_height = set_height, set_width = set_width)

    img = img_Normalize(img)
    trans_img = hwc_to_chw(img)
    trans_gpu = torch.from_numpy(trans_img).unsqueeze(0)

    if gpu:
        trans_gpu = trans_gpu.cuda()

    with torch.no_grad():
        out_pred = segnet(trans_gpu)
        out_img = crfnet(trans_gpu,out_pred)

    return out_img

if __name__=="__main__":
    segnet = UNet(n_channels=1, n_classes=1,TransConv=True)
    crfnet = ConvCRF()
    dir_checkpoint_seg = '/home/liang/Data/SKI10/processed/unet.pth'
    dir_checkpoint_crf = '/home/liang/Data/SKI10/processed/crfnet.pth'
    dir_img = '/home/liang/Data/SKI10/processed/test/image/'
    dir_mask = '/home/liang/Data/SKI10/processed/test/calabel/'
    dir_result = '/home/liang/Data/SKI10/processed/predict_results/'
    img_suf = '.png'

    segnet.cuda()
    crfnet.cuda()
    segnet.load_state_dict(torch.load(dir_checkpoint_seg))
    crfnet.load_state_dict(torch.load(dir_checkpoint_crf))
    dice_sum = 0
    num = 0.0

    ids = get_ids(dir_img)
    for id in ids:
        img_origin = Image.open(dir_img + id + img_suf)
        mask_origin = Image.open(dir_mask + id + img_suf)
        img_gray = img_origin.convert('L')
        mask_gray = mask_origin.convert('L')
        mask_crop_gray = crop_image(mask_gray,set_height=256,set_width=256)
        mask_crop_gray = img_Normalize(mask_crop_gray)

        mask_trans = torch.from_numpy(mask_crop_gray)
        mask_gpu = mask_trans.cuda()

        pre_result = predict_img(segnet,crfnet,origin_img=img_gray,set_height=256,set_width=256,threshold=0.5,gpu=True)

        mask_flat = mask_gpu.view(-1)
        result_flat = pre_result.view(-1)

        inter = torch.dot(mask_flat, result_flat.float())
        union = torch.sum(mask_flat) + torch.sum(result_flat)

        dice_coef = (2 * inter + 1e-5) / (union + 1e-5)
        dice_sum = dice_sum + dice_coef
        num = num + 1

       # print('The Dice of {} th image is {}'.format(id,dice_coef))

        # result = mask_to_image(pre_result)
        # ImgSave.save_image(result,dir_result+id+img_suf,normalize=True)
        # result = mask_to_image(mask_trans)
        # ImgSave.save_image(result,dir_result+id+img_suf,normalize=True)

    print('Dice coefficient of the whole test dataset is {}'.format(dice_sum / num))