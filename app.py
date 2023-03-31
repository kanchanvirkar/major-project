
import subprocess
from flask import Flask, flash, render_template, json, request, jsonify, Response, redirect, url_for
from werkzeug.utils import secure_filename
import json
import requests
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


UPLOAD_FOLDER = '/temp'

app = Flask(__name__)

new_model = tf.keras.models.load_model('lang_model.h5') 
new_model.summary()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods = ['POST'])
def predict():    
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join("temp", filename))
    path = 'temp/'+filename    
    X = []        
    img = image.load_img(path,grayscale='true',color_mode='rgb',target_size=(300,140,3))
    img = image.img_to_array(img)
    img = img/255.0
    X.append(img)
    X = np.array(X)
    y_result=new_model.predict_classes(X)    
    return str(y_result[0])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


    
