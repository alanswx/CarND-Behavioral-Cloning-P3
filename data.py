import scipy.misc
import random
import cv2
import numpy as np
from sklearn.utils import shuffle
#from keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageEnhance, ImageOps

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

def  loadTraining():
    xs = []
    ys = []



    path = '/data1/udacity/simulator/data'
    img_path = path +'/IMG'
    csv_file = path +'/driving_log.csv'

    # open the CSV file and loop through each line
    # load the CSV so we can have labels
    csv_data=np.recfromcsv(csv_file, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')

    i  = 0
    remove = 0
    for line in csv_data:
        # remove 90% of straight scenes
        i = i + 1
        if (abs(float(line[3])) < 1e-5):
          remove=remove+1
          if (remove==10):
            remove=0
            # add 10% of the frames where the steering angle is close to 0
            xs.append( path+'/'+line[0].decode('UTF-8').strip())
            ys.append(float(line[3]))
            if leftright:
                # add the left image
                xs.append( path+'/'+line[1].decode('UTF-8').strip())
                ys.append(float(line[3])+steering_camera_offset)
                # add the right image
                xs.append( path+'/'+line[0].decode('UTF-8').strip())
                ys.append(float(line[3])-steering_camera_offset)
        else:
          # add all non straight frames
          # add center image
          xs.append( path+'/'+line[0].decode('UTF-8').strip())
          ys.append(float(line[3]))
          if leftright:
            # add the left image
            xs.append( path+'/'+line[1].decode('UTF-8').strip())
            ys.append(float(line[3])+steering_camera_offset)
            # add the right image
            xs.append( path+'/'+line[0].decode('UTF-8').strip())
            ys.append(float(line[3])-steering_camera_offset)

    # xs now has the image paths
    # ys now has the steering angles

    #shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    return xs,ys

rotation = True
#cropImage = False


#
#  this brightness function taken from:
#  https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc
def augment_brightness_camera_images(image):
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
    trans = 2
    transx= random.randint (-trans, trans);
    transy= random.randint (-trans, trans);
    (rows, cols) = image.shape[:2]
    M = np.float32([[1,0,transx],[0,1,transy]])
    image = cv2.warpAffine(image,M,(cols,rows))
    return image

def processImagePixels(image):
    #grab the height and width of the image
    (h, w) = image.shape[:2]

    # cropping the image didn't help
    #image = cropImage(image)

    #randomize brightness
    image = augment_brightness_camera_images(image)

    #rotation and scaling
    image = rotateAndScaleImage(image)
    image = translateImage(image)
    # resize for the nvidia model 200x66
    image = cv2.resize(image, (200, 66) )
    return image

# open image image from disk
def openImage(name):
   image = np.array(Image.open(name))
   return image

# open the image, and call the image pipeline
def processImage(name):
   image = openImage(name)
   return processImagePixels(image)

# open the image, resize, don't augment 
def processImageValidation(name):
   image = openImage(name)
   # resize for the nvidia model 200x66
   image = cv2.resize(image, (200, 66) )
   return image

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
      X.append(image)
    yield np.asarray(X), np.asarray(y)

#
# This function returns an array of images and steering angles. It is passed in
# an array of image paths and steering angles
# it probably should call the steering function to alter it if needed - this code
# doesn't alter the steering angle much
def getValidationDataset(val_xs,val_ys,func=processImageValidation):
    images= [func(x) for x in val_xs]
    return np.array(images), np.array(val_ys)

