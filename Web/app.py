from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
IMAGE_FOLDER = 'static/images/'

app = Flask(__name__)
app.secret_key = b'boom'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024