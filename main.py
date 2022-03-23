#this is Mca mini project

from __future__ import division, print_function
# coding=utf-8
import os.path
import numpy as np
from keras.models import Model,load_model
from keras.preprocessing import image
#Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

#defining Flask app
app = Flask(__name__)

#path of model saved with keras.save()
MODEL_PATH ="assets/model_inception.h5"

#loading our trained model
model = load_model(MODEL_PATH)


#Prediction fuction for the webapp
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path,target_size=(224,224))

    #preprocessing the image
    x= image.img_to_array(img)
    x=x/225 #scaling
    x = np.expand_dims(x, axis=0)

    #predicting the output
    preds = model.predict(x)
    preds =np.argmax(preds, axis = 1)
    print(preds)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"

    return preds

@app.route('/', methods=['GET'])
def front():
    return render_template('front.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods =['GET','POST'])
def upload():
    if request.method=='POST':
        f =request.files['file']#get the file from the post request
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path,model)
        result =preds
        return result
    return None

if __name__ == '__main__':
    app.run(port=5001,debug=True)


