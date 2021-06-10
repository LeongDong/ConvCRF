import random
import numpy as np

def hwc_to_chw(img):
    if img.ndim == 2:
        img = np.expand_dims(img,axis=0)
    return img

def crop_image(pilimg, set_height=256, set_width=256):
    w=pilimg.size[0]
    h=pilimg.size[1]

    centerx=int(w*0.5)
    centery=int(h*0.5)

    left=centerx-set_width//2
    right=set_width+centerx-set_width//2
    upper=centery-set_height//2
    lower=set_height+centery-set_height//2
    pilimg = pilimg.crop((left,upper,right,lower))

    return np.array(pilimg,dtype=np.float32)

def batch(iterable,batch_size):
    b=[]
    for i,t in enumerate(iterable):
        b.append(t)
        if (i+1)%batch_size==0:
            yield b
            b=[]
    if len(b)>0:
        yield b

def split_dataset(dataset,val_percent=0.05):
    dataset=list(dataset)
    length = len(dataset)
    n = int(length*val_percent)
    #random.shuffle(dataset)
    return {'train':dataset[:-n],'val':dataset[-n:]}

def img_Normalize(x):
    return x/255