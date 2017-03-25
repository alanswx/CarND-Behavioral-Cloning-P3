
from __future__ import print_function
import os
import numpy as np
import fnmatch
from PIL import Image
import conf

#some images are getting cut off
min_image_size = 1 * 1024

def get_files(filemask):
    path, mask = os.path.split(filemask)
    #print(path, mask)
    matches = []
    for root, dirnames, filenames in os.walk(path):
        #print(root, dirnames, filenames)
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    return matches

def clean_zero_len_files(filemask):
    img_paths = get_files(filemask)
    for f in img_paths:
        if os.path.getsize(f) < min_image_size:
            os.unlink(f)

def parse_img_filepath(filepath):
    f = filepath.split('/')[-1]
    f = f.split('.')[0]
    f = f.split('_')

    '''
    The neural network seems to train well on values that are not too large or small.
    We recorded the raw axis values. So we normalize them and then apply a STEERING_NN_SCALE
    that puts them roughly in units of degrees +- 30 or so.
    '''
    steering = float(f[3]) / float(conf.js_axis_scale) * conf.STEERING_NN_SCALE
    throttle = float(f[5]) / float(conf.js_axis_scale) * conf.STEERING_NN_SCALE
    
    data = {'steering':steering, 'throttle':throttle }
    return data

def get_data(file_path):
        with Image.open(file_path) as img:
            img_arr = np.array(img)

            #if just grey..
            if conf.GREY_SCALE:
                img_arr = img_arr[:,:,:1]

        data = parse_img_filepath(file_path)
        #return img_arr.transpose(), data
        return img_arr, data

def load_dataset(filemask):
    clean_zero_len_files(filemask)
    img_paths = get_files(filemask)
    img_count = len(img_paths)
    gen = load_generator(filemask)
    print( "found", img_count, "images.")

    X = [] #images
    Y = [] #velocity (angle, speed)
    for _ in range(img_count):
        x, y = next(gen)
        X.append(x)
        Y.append(y)
        
    X = np.array(X) #image array [[image1],[image2]...]
    Y = np.array(Y) #array [[angle1, speed1],[angle2, speed2] ...]

    return X, Y


def load_generator(filemask):
    ''' 
    Return a generator that will loops through image arrays and data labels.
    ''' 
    img_paths = get_files(filemask)

    while True:
        for f in img_paths:
            img_arr, data = get_data(f)
            
            #only steering for now
            data_arr = np.array([data['steering'], data['throttle']])

            yield img_arr, data_arr



def batch_generator(filemask, batch_size):
    clean_zero_len_files(filemask)
    img_paths = get_files(filemask)
    img_count = len(img_paths)
    gen = load_generator(filemask)
    print("found", img_count, "images.")

    num_batches = img_count / batch_size
    print(num_batches, "batches")

    for b in range(num_batches):
        X = [] #images
        Y = [] #velocity (angle, speed)
        for _ in range(batch_size):
            x, y = next(gen)
            X.append(x)
            Y.append(y)
            
        X = np.array(X) #image array [[image1],[image2]...]
        Y = np.array(Y) #array [[angle1, speed1],[angle2, speed2] ...]

        yield X, Y
