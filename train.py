import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import warnings
import os
import csv
import numpy as np
import glob
from scipy.ndimage import interpolation
from PIL import Image
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, \
    accuracy_score, f1_score

warnings.filterwarnings("ignore")


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


def make_csv():
    csvData = ['pixel label']
    for i in range(784):
        csvData.append('pixel' + str(i))
    with open('train2.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(csvData)
    csvFile.close()


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def fill_me_in(pixels, label):
    row = [label]
    for i in range(len(pixels)):
        row.append(pixels[i])

    with open('train2.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


def create_vec(image_path, label):
    im_names = glob.glob(os.path.join(image_path, '*.png')) + \
               glob.glob(os.path.join(image_path, '*.jpg')) + \
               glob.glob(os.path.join(image_path, '*.jpeg'))

    for im_name in im_names:
        im = Image.open(im_name).convert('L')
        img = im.resize((28, 28), Image.ANTIALIAS)
        pixels = np.array(img).flatten()
        fill_me_in(pixels, label)


def train(directory, file_name, model_file="model.pickle"):
    print("Going to read data from {}".format(file_name))
    df = pd.read_csv(directory + file_name)
    print(df.shape)
    df_X = df.drop("pixel label", axis=1)
    # Deskewing the data
    df_X = df_X.apply(lambda x: deskew(x.values.reshape(28, 28)).flatten(), axis=1)
    # Scaling the data
    images = df_X.apply(scale)
    # Dropping all the columns with only NaN values
    # Saving the label data as the target variable
    labels = df["pixel label"]
    images.head()
    '''
    labels = df.iloc[0:500000, :1]
    images = df.iloc[0:500000, 1:]
    '''
    images.fillna(0.0).values
    print("Data read , Going to spli into train and test sets ..... ")
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.3,
                                                                            random_state=42, stratify=labels)
    print("Data split done, Going to train.... ")

    clf = svm.SVC(C=1, kernel='linear')
    N = max([len(i) for i in train_images])
    train_images = [np.r_[a, np.zeros((N - a.shape[0]), dtype=a.dtype)] for a in train_images]
    test_images = [np.r_[a, np.zeros((N - a.shape[0]), dtype=a.dtype)] for a in test_images]
    clf = clf.fit(train_images, train_labels)
    print("Training completed ")
    print("Going to save the model....")

    filename = open(directory + model_file, "wb")
    pickle.dump(clf, filename)
    filename.close()
    print("Model saved in {}".format(model_file))

    print('Going to evaluate model....')
    predicted = clf.predict(test_images)
    print(confusion_matrix(test_labels, predicted, range(0, 10)))
    print('accuracy_score:   ', accuracy_score(test_labels, predicted))
    print('f1_score:   ', f1_score(test_labels, predicted, average=None))
    print('recall_score:   ', recall_score(test_labels, predicted, average=None))
    print('precision_score:', precision_score(test_labels, predicted, average=None))


if __name__ == '__main__':
    '''
    make_csv()
    path = 'data/'
    for i in range(10):
        label = i
        path_to = path + 'images/trainingSet/' + str(i) + '/'
        print(path_to)
        create_vec(path_to,label)
    '''
    path = 'data/'
    train(directory=path, file_name='train2.csv')