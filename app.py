import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np

from infer.infer import create_vec, deskew, scale, load_model, classify_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/upload/'


@app.route('/')
def upload_file():
    return render_template('upload.html')


def get_label(data_path):
    # try:
    #     data_path = '/home/venky/PycharmProjects/digit-recognization/data/upload/images.jpeg'
    # except:
    #     print('Error: Please enter system path')
    df = create_vec(data_path)
    # Deskewing the data
    df = deskew(np.array(df)).flatten()
    # Scaling the data
    images = scale(df)
    # fillting model
    clf = load_model("data/model.pickle")
    label = classify_image(images, clf)
    return label


@app.route('/uploader', methods=['POST', 'GET'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)

        label = get_label(path)
        return " The image is of {} ".format(label[0])


if __name__ == '__main__':
    app.run(debug=True)
