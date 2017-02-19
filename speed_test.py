from keras.models import *
from keras.models import load_model
from keras.callbacks import *
from keras.layers import Lambda, Convolution2D, Activation, Dropout, Flatten, Dense
import keras.backend as K
import cv2
import argparse
import data
import pickle

import time

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print ('elapsed time: %f ms' % self.msecs)

def speed_test(model):
  # loop through images and run them through predict
 
  xs,ys = data.loadTraining()
  with Timer() as tout:
   for image in xs:
     image_array=image
     with Timer() as t:
         steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
     print ("=> model.predict : %s s" % t.secs)
     print(steering_angle )

  print ("=> total: %d items in  %s s" % (len(xs),tout.secs))
  print ("=> %d fps" % (len(xs)/float(tout.secs)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    model = load_model(args.model)
    print (model.summary())
    speed_test(model)

