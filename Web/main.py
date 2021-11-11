import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
from skimage import io

ALLOWED_EXTENSIONS = ['jpg','jpeg','tif','png']
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

#Defining dice coefficiat metric
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + keras.backend.epsilon()) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + keras.backend.epsilon())

model = keras.models.load_model("assets/BrainTumorSegModel.h5", custom_objects={'dice_coef': dice_coef})


def check_file(filename):
    if filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS:
        return True
    return False

def predict(path):
    filename = path.rsplit('/',1)[1].split('.')[0]
    img = filename+'.png'
    result = make_prediction(path)
    if result==False:
        return render_template('upload.html', message='No Tumor', img=img)
    else:
        mask = filename+'_mask.png'
        imgWithMask = filename+"_withMask.png"
        return render_template('upload.html', img=img, mask=mask, imgWithMask=imgWithMask)

def make_prediction(path):

    filename = path.rsplit('/',1)[1].split('.')[0]

    X = np.empty((1,256,256,3))
    img = io.imread(path)
    cv2.imwrite(os.path.join(app.config['IMAGE_FOLDER'],filename+'.png'),img)
    img = cv2.resize(img, (256,256))
    img = np.array(img, dtype=np.float64)

    #Standardizing    
    img -= img.mean()
    img /= img.std()
    X[0,] = img
    
    predict = model.predict(X)
    
    # if sum of predicted mask is 0 then there is no tumour
    if predict.round().astype(int).sum()==0:
        return False
    else:
        mask = predict.squeeze().round()
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'],filename+'_mask.png'),mask*255)
        img_ = io.imread(path)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_[mask==1] = (255,0,0)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'],filename+'_withMask.png'),img_)
    return True

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No File part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    elif file and check_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        flash('Image uploaded')
        return predict(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    else:
        flash("Allowed image types are jpg,jpeg,tif")
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='images/' + filename))

@app.route('/results/<filename>')
def display_results(filename):
    return redirect(url_for('static', filename='results/' + filename))

if __name__ == '__main__':
    app.run(debug=False)