from __future__ import division
import numpy as np
from scipy.ndimage import interpolation
#import random
import copy

def resize_n(old, new_shape):
    new_f, new_t = new_shape
    old_f, old_t = old.shape
    scale_f, scale_t = new_f/old_f, new_t/old_t
    new = interpolation.zoom(old, (scale_f, scale_t))
    return new 

def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

#    img = cv2.imread(img_data_aug['filepath'])
    print(img_data_aug['filepath'])
    img_o = np.loadtxt(img_data_aug['filepath'])
    sd = 2126.5
    img = img_o/sd
    img = resize_n(img, (224, 224))
    #img = np.stack((img, img, img), axis=2)

    if augment:
        rows, cols = img.shape[:2]
        print(img.shape)

        if config.use_freq_mask and np.random.randint(0, 2) == 0:
            print('freq_mask')
            mid = np.random.randint(0, img.shape[0])
            length = np.random.randint(0, 10)
            img_new = img
            start = mid -length
            stop = mid + length
            if start < 0:
                start = 0
            if stop >= img.shape[0]:
                stop = img.shape[0]-1
            img_new[:, start:stop] = np.mean(img[:, start:stop])

        if config.use_time_mask and np.random.randint(0, 2) == 0:
            print('time_mask')
            mid = np.random.randint(0, img.shape[1])
            length = np.random.randint(0, 10)
            img_new = img
            start = mid -length
            stop = mid + length
            if start < 0:
                start = 0
            if stop >= img.shape[1]:
                stop = img.shape[1]-1
            img_new[start:stop, :] = np.mean(img[start:stop, :])
            
        img = np.stack((img_new, img_new, img_new), axis=2)
        if config.use_horizontal_flips and np.random.randint(0, 2) == 5:
            #img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 5:
            #img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:
            angle = 0#np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                #img = cv2.flip(img, 0)
            elif angle == 180:
                img = np.transpose(img, (1,0,2))
                #img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                #img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img
