from os.path import splitext
from os import listdir
from PIL import Image
from pre_processed import crop_image, img_Normalize, hwc_to_chw

'''
return file names of all images in the same directory in order
'''
def get_ids(dir):
    filename_sort = listdir(dir)
    filename_sort.sort(key=lambda x:int(x[:-4]))
    return (splitext(file)[0] for file in filename_sort)

'''
return cropped images by pre-set size
'''
def to_crop_image(ids,dir,suffix,set_height,set_width):
    for id in ids:
        img = Image.open(dir+id+suffix)
        image=crop_image(img.convert('L'),set_height,set_width)
        yield image

def get_images(ids,dir_img,dir_mask,set_height,set_width):
    imgs=to_crop_image(ids,dir_img,'.png',set_height,set_width)
    img_change=map(hwc_to_chw,imgs)
    img_normalzied=map(img_Normalize,img_change)

    masks=to_crop_image(ids,dir_mask,'.png',set_height,set_width)
    mask_normalized=map(img_Normalize,masks)

    return zip(img_normalzied,mask_normalized)

def get_images_ids(ids,dir_img,set_height,set_width):
    imgs = to_crop_image(ids,dir_img,'.png',set_height,set_width)
    img_change = map(hwc_to_chw,imgs)
    img_normalized = map(img_Normalize,img_change)

    return zip(ids,img_normalized)