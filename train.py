import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import warnings
warnings.filterwarnings("ignore")


def train(file_name, model_file="model.pickle"):
    print("Going to read data from {}".format(file_name))
    df = pd.read_csv(file_name)
    labels = df.iloc[0:100, :1]
    images = df.iloc[0:100, 1:]
    print("Data read , Going to split into train and test sets ..... ")
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.1,
                                                                            random_state=0)
    print("Data split done, Going to train.... ")

    clf = svm.SVC(kernel='linear')
    clf = clf.fit(train_images, train_labels.values.ravel())
    print("Training completed ")
    print("Going to save the model....")

    filename = open(model_file, "wb")
    pickle.dump(clf, filename)
    filename.close()
    print("Model saved in {}".format(model_file))


if __name__ == '__main__':
    train(file_name='/home/venky/Downloads/train.csv')
