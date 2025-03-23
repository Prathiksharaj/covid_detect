from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Load models
xray_model = load_model('xray_model.keras')
ct_model = load_model('ct_model.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_single_image(file_path, img_size=(150, 150)):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Image not found or cannot be loaded: {file_path}")
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image(model, image_path, img_size=(150, 150), threshold=0.5):
    image = preprocess_single_image(image_path, img_size)
    prediction = model.predict(image)[0][0]
    if prediction > threshold:
        return "COVID-19", prediction
    else:
        return "Non-COVID-19", prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'xray_image' not in request.files or 'ct_image' not in request.files:
        return redirect(request.url)
    
    xray_file = request.files['xray_image']
    ct_file = request.files['ct_image']
    
    xray_image_url = None
    ct_image_url = None
    
    if xray_file and allowed_file(xray_file.filename):
        xray_filename = secure_filename(xray_file.filename)
        xray_file_path = os.path.join(app.config['UPLOAD_FOLDER'], xray_filename)
        xray_file.save(xray_file_path)
        xray_classification, xray_prediction = classify_image(xray_model, xray_file_path)
        xray_image_url = url_for('uploaded_file', filename=xray_filename)
    
    if ct_file and allowed_file(ct_file.filename):
        ct_filename = secure_filename(ct_file.filename)
        ct_file_path = os.path.join(app.config['UPLOAD_FOLDER'], ct_filename)
        ct_file.save(ct_file_path)
        ct_classification, ct_prediction = classify_image(ct_model, ct_file_path)
        ct_image_url = url_for('uploaded_file', filename=ct_filename)
    
    # Compare models
    if xray_prediction > ct_prediction:
        comparison = "X-ray model has higher confidence in detecting COVID-19."
    elif ct_prediction > xray_prediction:
        comparison = "CT scan model has higher confidence in detecting COVID-19."
    else:
        comparison = "Both models have similar confidence in detecting COVID-19."
    
    return render_template('result.html', 
                           xray_classification=xray_classification, xray_prediction=xray_prediction,
                           ct_classification=ct_classification, ct_prediction=ct_prediction,
                           comparison=comparison,
                           xray_image_url=xray_image_url,
                           ct_image_url=ct_image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
