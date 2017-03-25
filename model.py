from keras.models import *
from keras.callbacks import *
from keras.layers import Lambda, Convolution2D, Activation, Dropout, Flatten, Dense
from keras.layers import Dense, Lambda, ELU
import keras.backend as K
import cv2
import argparse
import data
import pickle
import conf


# This model is an NVIDIA Variant. I used SullyChen's ideas, with 
# jacobgil's keras implementation. Small changes (based on Comma AI) to
# add in the Lambda and move the regularization step into the model, makes
# debugging the images much easier
#http://github.com/SullyChen/Autopilot-TensorFlow/
#https://github.com/jacobgil/keras-steering-angle-visualizations.git
#  -- needed to turn dropout back on!

def get_nvidia_model2():
    row, col, ch = conf.row, conf.col, conf.ch
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dense(1))

    return model


def nvidia_net():
    model = Sequential()
    #p=0.33
    p=0.5
    # this lambda function normalizes the values (0-255 to -1 to 1 of each pixel)
    # SullyChen used 66x200 color images. I tried a different model with grayscale, and that worked well too
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120,160,3), output_shape=(120,160,3)))
    model.add(Convolution2D(24, 5, 5, init = 'normal', subsample= (2, 2), name='conv1_1', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv2_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(48, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv3_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_2'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))

    model.add(Flatten())

    model.add(Dense(1164, init = 'normal', name = "dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(100, init = 'normal',  name = "dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(50, init = 'normal', name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = 'normal', name = "dense_3"))
    model.add(Activation('tanh'))
    model.add(Dense(1, init = 'normal', name = "dense_4"))

    return model
def nvidia_small_net():
    model = Sequential()
    #p=0.33
    p=0.5
    # this lambda function normalizes the values (0-255 to -1 to 1 of each pixel)
    # SullyChen used 66x200 color images. I tried a different model with grayscale, and that worked well too
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120,160,3), output_shape=(120,160,3)))
    model.add(Convolution2D(24, 5, 5, init = 'normal', subsample= (2, 2), name='conv1_1', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv2_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(48, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv3_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    #model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_2'))
    #model.add(Activation('relu'))
    #model.add(Dropout(p))

    model.add(Flatten())

    #model.add(Dense(1164, init = 'normal', name = "dense_0"))
    model.add(Dense(512, init = 'normal', name = "dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    #model.add(Dense(100, init = 'normal',  name = "dense_1"))
    model.add(Dense(256, init = 'normal',  name = "dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    #model.add(Dense(50, init = 'normal', name = "dense_2"))
    model.add(Dense(128, init = 'normal', name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = 'normal', name = "dense_3"))
    model.add(Activation('tanh'))
    model.add(Dense(1, init = 'normal', name = "dense_4"))

    return model
    
def nvidia_smaller_net():
    model = Sequential()
    #p=0.33
    p=0.5
    # this lambda function normalizes the values (0-255 to -1 to 1 of each pixel)
    # SullyChen used 66x200 color images. I tried a different model with grayscale, and that worked well too
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120,160,3), output_shape=(120,160,3)))
    model.add(Convolution2D(24, 5, 5, init = 'normal', subsample= (2, 2), name='conv1_1', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv2_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(48, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv3_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv4_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    #model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_2'))
    #model.add(Activation('relu'))
    #model.add(Dropout(p))

    model.add(Flatten())

    #model.add(Dense(1164, init = 'normal', name = "dense_0"))
    model.add(Dense(512, init = 'normal', name = "dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    #model.add(Dense(100, init = 'normal',  name = "dense_1"))
    model.add(Dense(256, init = 'normal',  name = "dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    #model.add(Dense(50, init = 'normal', name = "dense_2"))
    model.add(Dense(128, init = 'normal', name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = 'normal', name = "dense_3"))
    model.add(Activation('tanh'))
    model.add(Dense(1, init = 'normal', name = "dense_4"))

    return model
    
def nvidia_smallest_net():
    model = Sequential()
    #p=0.33
    p=0.5
    # this lambda function normalizes the values (0-255 to -1 to 1 of each pixel)
    # SullyChen used 66x200 color images. I tried a different model with grayscale, and that worked well too
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120,160,3), output_shape=(120,160,3)))
    model.add(Convolution2D(24, 5, 5, init = 'normal', subsample= (2, 2), name='conv1_1', border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv2_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(48, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv3_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv4_1'))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    #model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_2'))
    #model.add(Activation('relu'))
    #model.add(Dropout(p))

    model.add(Flatten())
    model.add(Dropout(0.2))
    #model.add(Dense(1164, init = 'normal', name = "dense_0"))
    model.add(Dense(512, init = 'normal', name = "dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    #model.add(Dense(100, init = 'normal',  name = "dense_1"))
    #model.add(Dense(256, init = 'normal',  name = "dense_1"))
    #model.add(Activation('relu'))
    #model.add(Dropout(p))
    #model.add(Dense(50, init = 'normal', name = "dense_2"))
    model.add(Dense(128, init = 'normal', name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    #model.add(Dense(10, init = 'normal', name = "dense_3"))
    #model.add(Activation('tanh'))
    model.add(Dense(1, init = 'normal', name = "dense_4"))

    return model
    

def get_model():
    #model = nvidia_net()
    #model = nvidia_smallest_net()
    #model = get_nvidia_model()
    model = get_nvidia_model2()
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model

def load_model(path):
    #model = nvidia_net()
    #model = nvidia_smallest_net()
    #model = get_nvidia_model()
    model = get_nvidia_model2()
    model.load_weights(path)
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model

#epochs=75
#epochs=20
epochs=20
#epochs=12
def train():
        #
        #  SaveModel is a CallBack class that we can use to save the model for each epoch
        #  This allows us to easily test each epoch on the simulator. The simulator seems like
        #  a better validation than just the validation data set
        class SaveModel(Callback):
           def on_epoch_end(self, epoch, logs={}):
             epoch += 1
             if (epoch>0):
                 #with open ('model-' + str(epoch) + '.json', 'w') as file:
                 #    file.write (model.to_json ())
                 #    file.close ()
                 #model.save_weights ('model-' + str(epoch) + '.h5')
                 model.save('model-'+str(epoch)+'.h5')

        #
        #  load the model
        #
        model = get_model()
     
        #  Keras has a nice tool to create an image of our network
        from keras.utils.visualize_util import plot
        plot(model, to_file='car_model.png',show_shapes=True)

        print ("Loaded model")

        # load the data 
        xs,ys = data.loadTraining()


        # split the dataset into training and validation  80% / 20%
        train_xs = xs[:int(len(xs) * 0.8)]
        train_ys = ys[:int(len(xs) * 0.8)]

        val_xs = xs[-int(len(xs) * 0.2):]
        val_ys = ys[-int(len(xs) * 0.2):]

        # load the validation dataset, it is better not generate an image each time - process them once
        # Use the validation process function, it doesn't augment the image, just resizes it
        X, y = data.getValidationDataset(val_xs,val_ys,data.processImageValidation)


        print (model.summary())
        print ("Loaded validation datasetset")
        print ("Total of", len(train_ys),"out of")
        print ("Training..")


        checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')

	# I tried using the earlystopping callback, but now I run it for a fixed number of epochs and test to see which is best
        earlystopping =  EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

        res=model.fit_generator(data.generator(train_xs,train_ys,256), validation_data = (X, y), samples_per_epoch = 125*256, nb_epoch=epochs, verbose = 1  ,callbacks = [ SaveModel()])

        # pickle and dump the history so we can graph it in a notebook
        history=res.history
        with open('history.p','wb') as f:
           pickle.dump(history,f)

if __name__ == "__main__":
    train()
