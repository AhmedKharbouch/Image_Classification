from flask import Flask, render_template, request
import os
from keras.models import load_model
import keras.utils as image
import numpy as np
import PIL as load_img

UPLOAD_FOLDER = 'static/uploads'
app = Flask(__name__)
model = load_model(
    r'C:\Users\hp\Downloads\AIClothingRecognistionTF-master\AIClothingRecognistionTF-master\Lab3\Deploy_Model\static\model\AhmedModelV1.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model.make_predict_function()


def predict_label(img_path):

    array = image.load_img(img_path,grayscale=True, target_size=(28, 28))
    array = image.img_to_array(array)
    array = array.reshape(1, 28, 28,1)
    array = array.astype('float32')
    #array = array / 255.0
    array = 255 - array
    prediction = model.predict(array)
    print(prediction)

    return class_names[np.argmax(prediction[0])]


@app.route('/')
def index():
    return (render_template('index.html'))


@app.route('/upload', methods=['POST'])
def upload_file():
    img = request.files['file']
    img.save(os.path.join(UPLOAD_FOLDER, img.filename))

    img_path = "static/uploads/" + img.filename
    p = predict_label(img_path)
    return render_template("index.html", prediction = p, img_path = img_path)
    #return 'file uploaded successfully'


app.run()
