import scipy.misc
import random
import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
#from keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageEnhance, ImageOps
import load_data

# some settings to tweak the way the training data is loaded

# do we want to load left/right images?
leftright = True
# 0.15 seemed to work as the best offset for the right/left cameras
steering_camera_offset = 0.15
#steering_camera_offset = 0.27
#steering_camera_offset = 0.10
#steering_camera_offset = 0.08

#
#  this function opens the training data, and returns two
#  lists, one with the image paths, and one with the steering angles
#
import os
def getFiles(name):
    files = os.listdir(name)
    files = [f for f in files if f[-3:] =='jpg']
    files.sort()
    file_paths = [os.path.join(name, f) for f in files]
    return file_paths

def parse_img_filepath(filepath):
        f = filepath.split('/')[-1]
        f = f.split('.')[0]
        f = f.split('_')

        throttle = int(f[3])
        angle = int(f[5])
        milliseconds = int(f[7])

        data = {'name':filepath,'throttle':throttle, 'angle':angle, 'milliseconds': milliseconds}
        return data

    
def parse_img_filepaths(filepaths):
    result = [parse_img_filepath(f) for f in filepaths]
    return result

def loadTraining():
    inputs='/home/alans/shark/log/*.jpg'
    xs, ys = load_data.load_dataset(inputs)

    new_xs=[]
    new_ys=[]
    remove = 0

    print('orig ',len(xs))
    for i in range(len(xs)):
        #if (abs(ys[i][0] < 1) or (abs(ys[i][0] > 29))):
        if (abs(float(ys[i][0])) < 1e-5):
        #if (abs(ys[i][0] < 1)):
            remove=remove+1
            if (remove==10):
                remove=0
                # add 10% of the frames where the steering angle is close to 0
                new_xs.append(xs[i])
                new_ys.append(ys[i][0])
        else:
            new_xs.append(xs[i])
            new_ys.append(ys[i][0])

    xs=new_xs
    ys=new_ys
    print('after ',len(xs))
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    print(len(xs))
    return xs,ys

def loadTraining2():
    #fp1=getFiles("/home/alans/donkey_data/sessions/2017_02_15__07_02_07_PM")
    #fp2=getFiles("/home/alans/donkey_data/sessions/2017_02_15__07_24_32_PM")
    #fp=fp1+fp2
    fp1=getFiles("/home/alans/mydonkey/sessions/2017_02_17__08_08_30_AM")
    fp2=getFiles("/home/alans/mydonkey/sessions/2017_02_17__12_56_56_PM")
    fp3=getFiles("/home/alans/mydonkey/sessions/2017_02_18__07_55_03_AM")
    fp4=getFiles("/home/alans/mydonkey/sessions/2017_02_18__01_20_31_PM")
    fp=fp1+fp2+fp3+fp4
    #fp=fp1+fp2
    print(len(fp))
    r=parse_img_filepaths(fp)

    xs=[]
    ys=[]
    i=0
    remove=0
    for entry in r:
        if (entry['throttle'] >=5 and abs(float(entry['angle']))<89 ):
           i=i+1
           if (abs(float(entry['angle'])) < 2):
               remove=remove+1
               if (remove==10):
                   remove=0
                   xs.append(np.array(Image.open(entry['name'])))
                   ys.append(entry['angle'])
           else:
               xs.append(np.array(Image.open(entry['name'])))
               ys.append(entry['angle'])


    ys = np.asarray(ys)
    ys = ys / 90.0

    #shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    print(len(xs))
    return xs,ys


def  loadTraining_old():
    xs = []
    ys = []

    #file_path='/home/alans/donkey_data/sessions/warehouse_lane.pkl'
    file_path='/home/alans/donkey_data/sessions/warehouseRGB.pkl'
    with open(file_path, 'rb') as f:
        X, Y = pickle.load(f)

    #
    # do we want to remove the center data?
    #   it seems reasomable

    # xs now has the image paths
    # ys now has the steering angles
    Y = Y / 90
    #shuffle list of images
    c = list(zip(X, Y))
    random.shuffle(c)
    xs, ys = zip(*c)

    return xs,ys

rotation = True
#cropImage = False


def add_boxes(image):
    (h, w) = image.shape[:2]
    rect_w = 25
    rect_h = 25
    rect_count = 30
    for i in range (rect_count):
        pt1 = (random.randint (0, w), random.randint (0, h))
        pt2 = (pt1[0] + rect_w, pt1[1] + rect_h)
        cv2.rectangle(image, pt1, pt2, (-0.5, -0.5, -0.5), -1)
    return image


def augment_brightness_camera_images(image):
      (height, width) = image.shape[:2]
      #correctionVal= .25+np.random.uniform()
      correctionVal= np.random.uniform(0.1,0.6)
      #correctionVal = 0.20 # fraction of white to add to the main image
      if (random.uniform (0, 1) > 0.5):
        img_file_white = Image.new("RGB", (width, height), "white")
      else:
        img_file_white = Image.new("RGB", (width, height), "black")
      img = Image.fromarray(image) 
      img_blended = Image.blend(img, img_file_white, correctionVal)
      image = np.array(img_blended)
      return image 

#
#  this brightness function taken from:
#  https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc
def augment_brightness_camera_images_old(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RGB again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


#
# This function crops the image - it didn't seem to help so it isn't used
#
def cropImage(image):
    if cropImage:
      top_crop = 55
      bottom_crop = 135
      image= image[top_crop:bottom_crop, :, :]
    return image
    


#
# this function rotates and scales the image 
#
# it randomly scales it +/- 1.02 and +/- 1 degree

# from http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
def rotateAndScaleImage(image):
    (rows, cols) = image.shape[:2]
    rotation_degrees = 1
    scale = 0.02
    scale = random.uniform(1.0 - scale, 1.0 + scale)
    rotation = random.uniform(-rotation_degrees, rotation_degrees)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,scale)
    image = cv2.warpAffine(image,M,(cols,rows))
    return image

#
#  This function translates the image randmonly between -2 and 2 pixels in each direction
#

def translateImage(image):
    trans = 4
    transx= random.randint (-trans, trans);
    transy= random.randint (-trans, trans);
    (rows, cols) = image.shape[:2]
    M = np.float32([[1,0,transx],[0,1,transy]])
    image = cv2.warpAffine(image,M,(cols,rows))
    return image

def processImagePixels(image):
    image = np.copy (image)
    # cropping the image didn't help
    #image = cropImage(image)

    #randomize brightness
    image = augment_brightness_camera_images(image)
    #image = add_boxes(image)

    #rotation and scaling
    image = rotateAndScaleImage(image)
    image = translateImage(image)
    # resize for the nvidia model 200x66
    #image = cv2.resize(image, (160, 120) )
    return image

# open image image from disk
def openImage(name):
   image = np.array(name)
   #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

   return image

# open the image, and call the image pipeline
def processImage(name):
   image = np.array(name)
   #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   return processImagePixels(image)

# open the image, resize, don't augment 
def processImageValidation(name):
   image = openImage(name)
   # resize for the nvidia model 200x66
   #image = cv2.resize(image, (160, 120) )
   return image.transpose()

# adjust the steering angle if needed
# this is adding some random noise 
def yFunc(y):
   y= y+ np.random.normal (0, 0.005)
   return y 
 
#
# the generator takes the image paths, steering angles, and two functions
# the functions process the steering angle, and the image pipeline
# 
# because the X_items are image paths, this function will open and augment
# the image each time it needs to fill the returning batch array with an image
# we might be able to speed this up by passing in an array of preloaded images
# but that would use more memory
def generator(X_items,y_items,batch_size,x_func=processImage,y_func=yFunc):
  #print("inside generator")
  num_items = len(X_items) -1
  while 1:
    y = []
    X = []
    for i in range (batch_size):
      # grab a random image, and run it through the augmentation
      # to get a unique image
      this_image_index = random.randint (0, num_items)
      image = x_func(X_items[this_image_index])
      steering = y_func(y_items[this_image_index])
      # flip the image and steering angle half the time
      if (random.uniform (0, 1) > 0.5):
            image = cv2.flip (image, 1)
            steering = - steering
      y.append(steering)
      X.append(image.transpose())
    yield np.asarray(X), np.asarray(y)

#
# This function returns an array of images and steering angles. It is passed in
# an array of image paths and steering angles
# it probably should call the steering function to alter it if needed - this code
# doesn't alter the steering angle much
def getValidationDataset(val_xs,val_ys,func=processImageValidation):
    images= [func(x) for x in val_xs]
    return np.array(images), np.array(val_ys)

