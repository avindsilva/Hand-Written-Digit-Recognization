from PIL import Image
import pandas as pd

import PIL.ImageOps
import numpy as np
import pickle
import sys
from scipy.ndimage import interpolation

# !/usr/bin/env python
## -*- coding: utf-8 -*-

# to avoid string encoding error.app
# reload(sys)
# sys.setdefaultencoding("utf-8")

def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]  # A trick in numPy to create a mesh grid
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0 * image) / totalImage  # mu_x
    m1 = np.sum(c1 * image) / totalImage  # mu_y
    m00 = np.sum((c0 - m0) ** 2 * image) / totalImage  # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / totalImage  # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
    mu_vector = np.array([m0, m1])  # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00, m01], [m01, m11]])  # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

# Function used for deskewing the image which internally first calls the moment function described above
def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)

# Function for scaling the data between 0 and 1
def scale(vect):
    return (vect - vect.min()) / (vect.max() - vect.min())

def create_vec(image_path):
    im = Image.open(image_path).convert('L')
    avg_color_per_row = np.average(im, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if avg_color > 125:
        im = PIL.ImageOps.invert(im)
    img = im.resize((28, 28), Image.ANTIALIAS)
    #pixels = np.array(img).flatten()
    return img

def get_image_pixels(image_path):
    im = Image.open(image_path).convert('L')
    avg_color_per_row = np.average(im, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if avg_color > 125:
        im = PIL.ImageOps.invert(im)
    img = im.resize((28, 28), Image.ANTIALIAS)
    image = np.array(img).flatten()
    return image

def load_model(model_path):
    filename = open(model_path, 'rb')
    return pickle.load(filename)

def classify_image(pixel_array, clf):
    df = pd.Series(pixel_array)
    return clf.predict(df.values.reshape(1, -1))


if __name__ == '__main__':
    try:
        data_path = '/home/venky/PycharmProjects/digit-recognization/data/upload/images.jpeg'
    except:
        print('Error: Please enter system path')
    df = create_vec(data_path)
    #Deskewing the data
    df = deskew(np.array(df)).flatten()
    # Scaling the data
    images = scale(df)
    # fillting model
    clf = load_model("../data/model.pickle")
    label = classify_image(images, clf)
    print(" The image is of {} ".format(label))
