import json
import math
import numpy as np
import pandas as pd
import cv2
import click

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def preprocessing(image):
    image = image[65:135,:]
    image = cv2.resize(image, None, fx = 0.125, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    image = (image.astype(np.float32) - 128.0) / 128.0
    return image

def get_data(df):
    images = []
    angles = []

    for index, row in df.iterrows():
        image = cv2.imread(row['image'].strip(), -1)
        image = preprocessing(image)
        angle = row['angle']

        images.append(image)
        angles.append(angle)

        image = cv2.flip(image, 1)
        angle = -angle

        images.append(image)
        angles.append(angle)

    return (np.array(images), np.array(angles))

def read_csv(path, short_image_path = False):
    df = pd.read_csv(path + 'driving_log.csv').ix[:,0:4]
    df.columns = ['center', 'left', 'right', 'angle']

    if short_image_path:
        df['center'] = df['center'].apply(lambda x: path + '/' +  x.strip())
        df['left']   = df['left'].apply(lambda x: path + '/' +  x.strip())
        df['right']  = df['right'].apply(lambda x: path + '/' +  x.strip())

    return df

def left_angle(angle, angle_factor):
    return angle + abs(angle * angle_factor) + 7 * math.pi / 180.0

def right_angle(angle, angle_factor):
    return angle - abs(angle * angle_factor) - 7 * math.pi / 180.0

def extract_data(df):
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
    model = Sequential()
    model.add(Convolution2D( 3, 1, 1, border_mode = 'same',  subsample = (1, 1), init = 'uniform', input_shape = input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode = 'valid', subsample = (2, 2), init = 'uniform'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(dropout))
    model.add(Flatten())
    model.add(Dense(50, init = 'normal'))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='tanh'))
    return model

@click.command()
@click.argument('path')
@click.option('--weights', default = None, help = 'Path to a file with default weights.')
@click.option('--batch',   default = 512,  help = 'Batch size.')
@click.option('--epochs',  default = 10,   help = 'Number of epochs.')
def model(path, weights, batch, epochs):
    udacity_data = read_csv(path, True)
    driving_data = extract_data(udacity_data)

    features, labels = get_data(driving_data)
    features, labels = shuffle(features, labels, random_state = 1)

    train_features, valid_features, train_labels, valid_labels = \
        train_test_split(features, labels, test_size=0.33, random_state = 1)

    model = build_model((35, 40, 3))
    model.compile(loss = 'mse', optimizer = Adam(1e-5))

    if weights:
        model.load_weights(weights)

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks  = [checkpoint]

    model.fit(
        train_features,
        train_labels,
        batch_size = batch,
        nb_epoch = epochs,
        callbacks         = callbacks,
        validation_data = (
            valid_features,
            valid_labels
        )
    )

    model_json = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(model_json, outfile)

if __name__ == '__main__':
    model()
