#import cv2
import numpy as np
from scipy.ndimage import interpolation

def resize_n(old, new_shape):
    new_f, new_t = new_shape
    old_f, old_t = old.shape
    scale_f, scale_t = new_f/old_f, new_t/old_t
    new = interpolation.zoom(old, (scale_f, scale_t))
    return new 

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path,'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            filename = '/home/LORIEN+ravali.nalla/Txt_Data/' + filename
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = np.loadtxt(filename)
                sd = 2126.5
                img = img/sd
                img = resize_n(img)
                img = np.stack((img, img, img), axis=2)
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                set_n = filename.split('/')[4]
                if set_n == "Train" or set_n == "Validate":
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


