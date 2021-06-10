import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']
import numpy as np
import random
import torch
import torch.nn as nn
from ConvCRF import ConvCRF
from torch import optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

from dice_loss import DiceLoss
from eval import eval_net
from unet_para import UNet
from pre_processed import split_dataset, batch
from data_loader import get_ids, get_images
def train_net(segnet,crfnet,epochs=50,batch_size=1,lr=0.00001,val_percent=0.05,save_cp=True,gpu=True,set_height=256,set_width=256):
    dir_img = '/home/liang/Data/SKI10/processed/train/image/'
    dir_mask = '/home/liang/Data/SKI10/processed/train/calabel/'
    dir_checkpoint = '/home/liang/Data/SKI10/processed/'

    ids = get_ids(dir_img)
    id_dataset = split_dataset(ids, val_percent)
    train_data = []
    for t in id_dataset['train']:
        train_data.append(t)

    print('Training parameters:Epochs:{} Batch size:{} Learning rate:{} Training size:{} validation size:{}'.format(epochs,batch_size,lr,len(id_dataset['train']),len(id_dataset['val'])))
    train_num = len(id_dataset['train'])
    optimizer = optim.Adam(crfnet.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=False)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.5)
    criterion = DiceLoss()

    max_val_dice = -1
    epochs_x = []
    train_diceloss_y = []
    vali_dicecoef_y = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1,epochs))

        segnet.eval()
        crfnet.train()
        random.shuffle(train_data)
        train_img = get_images(train_data,dir_img,dir_mask,set_height,set_width)
        val_imgs = get_images(id_dataset['val'],dir_img,dir_mask,set_height,set_width)
        sum_loss = 0

        for j,b in enumerate(batch(train_img,batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks=true_masks.cuda()
            label_pred = segnet(imgs)
            masks_pred = crfnet(imgs,label_pred)
            loss = criterion(masks_pred,true_masks)
            epoch_loss = loss.item()
            sum_loss = sum_loss + epoch_loss
            # print('Training process {}/{}-----loss{}'.format(j*batch_size,train_num,loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochs_x.append((epoch + 1))
        train_diceloss_y.append((sum_loss / train_num))
        val_dice = eval_net(segnet,crfnet,val_imgs)###########################################################
        vali_dicecoef_y.append(val_dice)
        print('Epoch {} finished---loss:{}'.format((epoch+1),sum_loss / train_num))
        print('The Dice coefficient of validation is: {}; learning rate is {}'.format(val_dice, optimizer.state_dict()['param_groups'][0]['lr']))

        scheduler.step()

        if max_val_dice < val_dice and save_cp == True:
            max_val_dice = val_dice
            torch.save(crfnet.state_dict(),dir_checkpoint+'crfif.pth')
            print('Checkpoint{} with the best validation result is saved'.format(epoch+1))

    line_train, = plt.plot(epochs_x,train_diceloss_y,color='red',linewidth = 1.0, linestyle = '--')
    line_val, = plt.plot(epochs_x,vali_dicecoef_y,color = 'green', linewidth = 1.0, linestyle = ':')
    plt.title("The training curves")
    plt.legend(handles = [line_train, line_val], labels = ['Dice loss of training set', 'Dice coefficient of validation set'])
    plt.show()
if __name__=='__main__':
    segnet = UNet(n_channels=1, n_classes=1, TransConv=True)
    crfnet = ConvCRF()
    model_path = '/home/liang/Data/SKI10/processed/unet.pth'
    segnet.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        segnet.cuda()
        crfnet.cuda()

    epochs = 100
    batch_size = 4
    lr = 0.00001
    gpu = True
    set_height = 256
    set_width = 256
    try:
        train_net(segnet,crfnet,epochs=epochs,batch_size=batch_size,lr=lr,val_percent=0.05,save_cp=True,gpu=True,set_height=set_height,set_width=set_width)
    except KeyboardInterrupt:
        torch.save(segnet.state_dict(),'Interrupt.pth')
        print('Interruption occurred')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)