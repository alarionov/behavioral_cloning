import json
import math
import random
import numpy as np
import pandas as pd
import cv2
import click

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocessing(image):
    """ Returns cropped, resized and normalized image """
    image = image[65:135,:]
    image = cv2.resize(image, None, fx = 0.125, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    image = (image.astype(np.float32) - 128.0) / 128.0
    return image

def data_generator(df, batch_size):
    """ The generator goes through the dataframe
    and yields lists of images and angles of set batch size.
    The generator also flip the image along x axis with 50% chance.
    """
    while 1:
        images = []
        angles = []
        for index, row in df.sample(batch_size).iterrows():
            image = cv2.imread(row['image'].strip(), -1)
            image = preprocessing(image)
            angle = row['angle']

            if random.randrange(2) > 0:
                image = cv2.flip(image, 1)
                angle = -angle

            images.append(image)
            angles.append(angle)

        images = np.array(images)
        angles = np.array(angles)
        yield (images, angles)

def read_csv(path, short_image_path = False):
    """ Returns dataframe of driving logs from csv file """
    df = pd.read_csv(path + 'driving_log.csv').ix[:,0:4]
    df.columns = ['center', 'left', 'right', 'angle']

    if short_image_path:
        df['center'] = df['center'].apply(lambda x: path + '/' +  x.strip())
        df['left']   = df['left'].apply(lambda x: path + '/' +  x.strip())
        df['right']  = df['right'].apply(lambda x: path + '/' +  x.strip())

    return df

def left_angle(angle, angle_factor):
    """ Returns the angle for the left image based on the angle for the central image"""
    return angle + abs(angle * angle_factor) + 7 * math.pi / 180.0

def right_angle(angle, angle_factor):
    """ Returns the angle for the right image based on the angle for the central image"""
    return angle - abs(angle * angle_factor) - 7 * math.pi / 180.0

def extract_data(df):
    """ Returns dataframe of images and steering angles extracted from driving logs dataframe
    """
    angle_factor = 0.25
    rows = []

    for index, row in df.iterrows():
        c_angle = row['angle']
        l_angle = left_angle(c_angle, angle_factor)
        r_angle = right_angle(c_angle, angle_factor)
        rows.append({'image': row['center'], 'angle': c_angle})
        rows.append({'image': row['left'],   'angle': l_angle})
        rows.append({'image': row['right'],  'angle': r_angle})

    return pd.DataFrame(rows)

def build_model(input_shape, dropout = 0.5):
    """ Returns the model """
    model = Sequential()
    model.add(Convolution2D( 3, 1, 1, border_mode = 'same', subsample = (1, 1), init = 'uniform', input_shape = input_shape))
    model.add(Convolution2D(36, 6, 6, border_mode = 'same', subsample = (1, 1), init = 'uniform'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(dropout))
    model.add(Convolution2D(42, 6, 6, border_mode = 'valid', subsample = (2, 2), init = 'uniform'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 3, 3, border_mode = 'valid', subsample = (2, 2), init = 'uniform'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(300, init = 'normal'))
    model.add(PReLU())
    model.add(Dense(150, init = 'normal'))
    model.add(PReLU())
    model.add(Dense( 50, init = 'normal'))
    model.add(PReLU())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='tanh'))
    return model

@click.command()
@click.argument('path')
@click.option('--weights', default = None, help = 'Path to a file with default weights.')
@click.option('--batch',   default = 512,  help = 'Batch size.')
@click.option('--epochs',  default = 10,   help = 'Number of epochs.')
@click.option('--lr',      default = 1e-4, help = 'Learning rate.')
def model(path, weights, batch, epochs, lr):
    """ Main function to train the model """

    # read logs and extracting images and steering angles
    logs = read_csv(path, True)
    data = extract_data(logs)

    # split data into training set(66%) and validation set(33%)
    train, valid = train_test_split(data, test_size=0.33, random_state = 1)

    # build and compile the model
    model = build_model((35, 40, 3))
    model.compile(loss = 'mse', optimizer = Adam(lr))

    # set predefined weights if required
    if weights:
        model.load_weights(weights)

    # save weights after each epoch if accuracy increased on validation set
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks  = [checkpoint]

    # train the model
    model.fit_generator(
        generator         = data_generator(train, batch),
        samples_per_epoch = train.shape[0],
        max_q_size        = batch,
        nb_epoch          = epochs,
        callbacks         = callbacks,
        validation_data   = data_generator(valid, batch),
        nb_val_samples    = valid.shape[0]
    )

    # save the model into JSON file
    model_json = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(model_json, outfile)

if __name__ == '__main__':
    model()
