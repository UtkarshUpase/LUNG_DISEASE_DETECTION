from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from datetime import datetime
import uuid
import locale
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import json
import sys
import io
import io
from flask_ngrok import run_with_ngrok
from flask import Flask, Response
import matplotlib.pyplot as plt


# Set locale to UTF-8 to avoid encoding issues
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Set UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define class names for the lung cancer model
LUNG_CANCER_CLASSES = ['Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

# Load the three models
def load_lung_cancer_model():
    try:
        model = tf.keras.models.load_model('models/lung_cancer_model.h5')
        print("Lung cancer model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading lung cancer model: {e}")
        return None

def load_tuberculosis_model():
    try:
        model = tf.keras.models.load_model('models/tuberculosis_model.h5')
        print("Tuberculosis model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading tuberculosis model: {e}")
        return None

def load_pneumonia_model():
    try:
        model_path = 'models/pneumonia_model.h5'
        print(f"Checking if pneumonia model exists at: {model_path}")
        if not os.path.exists(model_path):
            print(f"Error: Pneumonia model file not found at {model_path}")
            return None
        print("Model file found, attempting to load...")
        model = tf.keras.models.load_model(model_path)
        print("Pneumonia model loaded successfully!")
        return model
    except Exception as e:
        print(f"Exception while loading pneumonia model: {e}")
        return None

# Load models at startup
lung_cancer_model = load_lung_cancer_model()
tuberculosis_model = load_tuberculosis_model()
pneumonia_model = load_pneumonia_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Remove non-ASCII characters from the filename and ensure it's safe."""
    # Remove non-ASCII characters
    filename = re.sub(r'[^\x00-\x7F]+', '', filename)
    # Replace spaces and special characters with underscores
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    return filename

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input (used for lung cancer and tuberculosis models)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)  # Resize to 224x224
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

def preprocess_image_pneumonia(image_path, target_size=(256, 256)):
    """Preprocess image for pneumonia model input."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)  # Resize to 256x256
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

def generate_occlusion_map(model, preprocessed_img, file_path, class_idx, patch_size=20, stride=8):
    """
    Generate an occlusion sensitivity map to visualize which parts of the image
    are most important for the model's prediction.
    """
    print(f"Running occlusion sensitivity analysis (patch_size={patch_size}, stride={stride})...")
    
    # Get original image dimensions
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image at {file_path}")
    
    original_img = cv2.resize(img, (224, 224))  # Resize to match model input
    img_array = preprocessed_img.copy()
    
    # Get baseline prediction without occlusion
    baseline_pred = model.predict(img_array, verbose=0)[0][class_idx]
    
    # Create empty heatmap
    heatmap = np.zeros((224, 224), dtype=np.float32)
    
    # Track number of patches processed
    total_patches = ((224 - patch_size) // stride + 1) * ((224 - patch_size) // stride + 1)
    patches_done = 0
    
    # Process each patch
    for y in range(0, 224 - patch_size + 1, stride):
        for x in range(0, 224 - patch_size + 1, stride):
            # Create a copy of the image
            occluded_img = np.copy(img_array)
            
            # Occlude the region (replace with gray value of 0.5)
            occluded_img[0, y:y+patch_size, x:x+patch_size, :] = 0.5
            
            # Predict with occlusion
            occluded_pred = model.predict(occluded_img, verbose=0)[0][class_idx]
            
            # Calculate drop in confidence (importance of this region)
            diff = baseline_pred - occluded_pred
            
            # Update heatmap - higher values mean more important regions
            heatmap[y:y+patch_size, x:x+patch_size] += diff
            
            # Update progress
            patches_done += 1
            if patches_done % 20 == 0:
                print(f"Processed {patches_done}/{total_patches} patches ({(patches_done/total_patches)*100:.1f}%)")
    
    # Normalize heatmap to 0-1 range
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Generate visualization
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)
    
    # Save visualization
    heatmap_filename = file_path.replace('.', '_heatmap.')
    cv2.imwrite(heatmap_filename, superimposed_img)
    
    return os.path.basename(heatmap_filename)

def generate_pdf_report(result, output_path):
    """Generate a PDF report for the given result."""
    try:
        # Create a PDF canvas
        c = canvas.Canvas(output_path, pagesize=letter)
        
        # Set up some basic formatting
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, "Lung Disease Detection Report")
        c.setFont("Helvetica", 12)
        
        # Add patient details
        c.drawString(72, 730, f"Patient Name: {result['patient_name']}")
        c.drawString(72, 710, f"Age: {result['patient_age']}")
        c.drawString(72, 690, f"Gender: {result['patient_gender']}")
        c.drawString(72, 670, f"Contact Number: {result['patient_contact']}")
        
        # Add timestamp
        c.drawString(72, 650, f"Date: {result['timestamp']}")
        
        # Add prediction result
        c.drawString(72, 630, f"Detection Result: {result['prediction']}")
        c.drawString(72, 610, f"Confidence Level: {result['confidence']}%")
        
        # Add detailed findings
        c.drawString(72, 590, "Detailed Findings:")
        y_position = 570
        for detail in result['details']:
            c.drawString(72, y_position, f"- {detail}")
            y_position -= 20
        
        # Add images (original and annotated)
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['original_image'])
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['annotated_image'])
        
        # Draw original image
        c.drawString(72, y_position - 20, "Original Image:")
        c.drawImage(ImageReader(original_image_path), 72, y_position - 220, width=200, height=200)
        
        # Draw annotated image
        c.drawString(72, y_position - 240, "Annotated Image (Occlusion Sensitivity Map):")
        c.drawImage(ImageReader(annotated_image_path), 72, y_position - 440, width=200, height=200)
        
        # Save the PDF
        c.save()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise

@app.after_request
def add_headers(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lung-cancer')
def lung_cancer_page():
    return render_template('lung-cancer.html')

@app.route('/tuberculosis')
def tuberculosis_page():
    return render_template('tuberculosis.html')

@app.route('/tuberculosis-results')
def tuberculosis_results():
    return render_template('tuberculosis-results.html')


@app.route('/pneumonia')
def pneumonia_page():
    return render_template('pneumonia.html')

# @app.route('/detect/<disease_type>', methods=['POST'])
# def detect_disease(disease_type):
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part'}), 400
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         # Sanitize filename to remove non-ASCII characters
#         filename = sanitize_filename(secure_filename(file.filename))
#         unique_filename = f"{uuid.uuid4()}_{filename}"
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
#         try:
#             # Save the file with UTF-8 encoding
#             file.save(file_path)
#         except Exception as e:
#             print(f"Error saving file: {e}")
#             return jsonify({'error': 'Error saving file. Please check the file name and try again.'}), 500
        
#         try:
#             # Preprocess the image based on disease type
#             if disease_type == 'pneumonia':
#                 preprocessed_img = preprocess_image_pneumonia(file_path)
#             else:
#                 preprocessed_img = preprocess_image(file_path)  # Use the existing function for other models
            
#             print(f"Preprocessed image shape: {preprocessed_img.shape}")  # Debugging
            
#             # Select the appropriate model based on disease_type
#             if disease_type == 'lung-cancer':
#                 model = lung_cancer_model
#             elif disease_type == 'tuberculosis':
#                 model = tuberculosis_model
#             elif disease_type == 'pneumonia':
#                 model = pneumonia_model
#             else:
#                 return jsonify({'error': 'Invalid disease type'}), 400
            
#             if model is None:
#                 print(f"Model for {disease_type} is not available. Check if the model file exists and is valid.")
#                 return jsonify({'error': f'Model for {disease_type} is not available'}), 500
            
#             # Make prediction
#             if disease_type == 'lung-cancer':
#                 # Multi-class prediction for lung cancer (3 classes)
#                 predictions = model.predict(preprocessed_img)[0]
#                 predicted_class_idx = np.argmax(predictions)
#                 confidence_percent = round(float(predictions[predicted_class_idx]) * 100, 2)
#                 predicted_class = LUNG_CANCER_CLASSES[predicted_class_idx]
                
#                 # Generate heatmap using occlusion sensitivity
#                 heatmap_filename = generate_occlusion_map(
#                     model=model,
#                     preprocessed_img=preprocessed_img,
#                     file_path=file_path,
#                     class_idx=predicted_class_idx,
#                     patch_size=20,
#                     stride=8
#                 )
#             elif disease_type == 'tuberculosis':
#                 # Binary prediction for tuberculosis
#                 prediction = model.predict(preprocessed_img)[0][0]
#                 print(f"Raw TB prediction value: {prediction}")  # Debug output
                
#                 # Adjust threshold for TB detection (may need tuning)
#                 is_positive = prediction > 0.4  # Lowered from 0.5
#                 confidence = float(prediction) if is_positive else float(1 - prediction)
#                 confidence_percent = round(confidence * 100, 2)
#                 predicted_class = "positive" if is_positive else "negative"
                
#                 # Generate proper heatmap for TB using occlusion sensitivity
#                 heatmap_filename = generate_occlusion_map(
#                     model=model,
#                     preprocessed_img=preprocessed_img,
#                     file_path=file_path,
#                     class_idx=0,  # For binary classification
#                     patch_size=24,  # Larger patch size for TB
#                     stride=12
#                 )
#             else:  # pneumonia
#                 # Binary prediction for pneumonia
#                 prediction = model.predict(preprocessed_img)[0][0]
#                 is_positive = prediction > 0.5
#                 confidence = float(prediction) if is_positive else float(1 - prediction)
#                 confidence_percent = round(confidence * 100, 2)
#                 predicted_class = "Positive" if is_positive else "Negative"
                
#                 # Generate basic heatmap for pneumonia
#                 original_img = cv2.imread(file_path)
#                 # Generate a simple placeholder heatmap
#                 heatmap = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
#                 center_x, center_y = original_img.shape[1] // 2, original_img.shape[0] // 2
#                 cv2.circle(heatmap, (center_x, center_y), radius=min(center_x, center_y) // 2, color=255, thickness=-1)
#                 heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#                 overlay = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)
#                 heatmap_path = file_path.replace('.', '_heatmap.')
#                 cv2.imwrite(heatmap_path, overlay)
#                 heatmap_filename = os.path.basename(heatmap_path)
            
#             # Get user details from the form
#             patient_name = request.form.get('patientName')
#             patient_age = request.form.get('patientAge')
#             patient_gender = request.form.get('patientGender')
#             patient_contact = request.form.get('patientContact')
            
#             # Create result object
#             result = {
#                 'original_image': unique_filename,
#                 'annotated_image': heatmap_filename,
#                 'prediction': predicted_class,
#                 'confidence': confidence_percent,
#                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Simplified timestamp format
#                 'details': [],
#                 'disease_type': disease_type,  # Add disease type for the download report
#                 'patient_name': patient_name,
#                 'patient_age': patient_age,
#                 'patient_gender': patient_gender,
#                 'patient_contact': patient_contact
#             }
            
#             # Add findings based on prediction
#             if disease_type == 'lung-cancer':
#                 if predicted_class == LUNG_CANCER_CLASSES[0]:  # Lung Adenocarcinoma
#                     result['details'] = [
#                         "Irregular nodular patterns detected in lung tissue",
#                         "Ground-glass opacity patterns observed",
#                         "Predominant peripheral distribution in the lungs",
#                         "Consistent with adenocarcinoma histopathology"
#                     ]
#                 elif predicted_class == LUNG_CANCER_CLASSES[1]:  # Lung Benign Tissue
#                     result['details'] = [
#                         "Normal lung parenchyma texture",
#                         "No suspicious nodules or masses detected",
#                         "Clear airway passages visible",
#                         "Healthy tissue patterns throughout the scan"
#                     ]
#                 elif predicted_class == LUNG_CANCER_CLASSES[2]:  # Lung Squamous Cell Carcinoma
#                     result['details'] = [
#                         "Central mass detected in bronchial region",
#                         "Cavitation signs present",
#                         "Thickened bronchial walls observed",
#                         "Pattern consistent with squamous cell histopathology"
#                     ]
#             elif disease_type == 'tuberculosis':
#                 if predicted_class == "positive":
#                     result['details'] = [
#                         "Infiltrates detected in upper lobes",
#                         "Fibrotic changes present",
#                         "Cavity formation observed",
#                         "Possible signs of active tuberculosis"
#                     ]
#                 else:
#                     result['details'] = [
#                         "Normal lung parenchyma pattern",
#                         "No significant abnormalities detected",
#                         "No signs of cavitation or fibrosis",
#                         "No indicators of active tuberculosis"
#                     ]
#             elif disease_type == 'pneumonia' and predicted_class == "Positive":
#                 result['details'] = [
#                     "Consolidation present in lower lobes",
#                     "Air bronchograms visible",
#                     "Pleural effusion detected",
#                     "Possible bacterial pneumonia"
#                 ]
#             else:  # pneumonia negative
#                 result['details'] = [
#                     "Clear lung fields",
#                     "No significant consolidation",
#                     "Normal bronchial patterns",
#                     "No signs of pneumonia detected"
#                 ]
            
#             # Redirect to results page with the session ID
#             return jsonify({
#                 'success': True,
#                 'redirect': f'/{disease_type}-results',
#                 'result': result
#             })
        
#         except Exception as e:
#             print(f"Error processing image: {e}")
#             print(f"File path: {file_path}")
#             print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#             return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
#     return jsonify({'error': 'File type not allowed'}), 400

@app.route('/detect/<disease_type>', methods=['POST'])
def detect_disease(disease_type):
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Sanitize filename to remove non-ASCII characters
        filename = sanitize_filename(secure_filename(file.filename))
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # Save the file with UTF-8 encoding
            file.save(file_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': 'Error saving file. Please check the file name and try again.'}), 500
        
        try:
            # Preprocess the image based on disease type
            if disease_type == 'pneumonia':
                preprocessed_img = preprocess_image_pneumonia(file_path)
            else:
                preprocessed_img = preprocess_image(file_path)
            
            print(f"Preprocessed image shape: {preprocessed_img.shape}")  # Debugging
            
            # Select the appropriate model based on disease_type
            if disease_type == 'lung-cancer':
                model = lung_cancer_model
            elif disease_type == 'tuberculosis':
                model = tuberculosis_model
            elif disease_type == 'pneumonia':
                model = pneumonia_model
            else:
                return jsonify({'error': 'Invalid disease type'}), 400
            
            if model is None:
                print(f"Model for {disease_type} is not available. Check if the model file exists and is valid.")
                return jsonify({'error': f'Model for {disease_type} is not available'}), 500
            
            # Make prediction
            if disease_type == 'lung-cancer':
                # Multi-class prediction for lung cancer (3 classes)
                predictions = model.predict(preprocessed_img)[0]
                predicted_class_idx = np.argmax(predictions)
                confidence_percent = round(float(predictions[predicted_class_idx]) * 100, 2)
                predicted_class = LUNG_CANCER_CLASSES[predicted_class_idx]
                
                # Generate heatmap using occlusion sensitivity
                heatmap_filename = generate_occlusion_map(
                    model=model,
                    preprocessed_img=preprocessed_img,
                    file_path=file_path,
                    class_idx=predicted_class_idx,
                    patch_size=20,
                    stride=8
                )
            elif disease_type == 'tuberculosis' or disease_type == 'pneumonia':
                # Binary prediction for tuberculosis and pneumonia
                prediction = model.predict(preprocessed_img)[0][0]
                print(f"Raw {disease_type} prediction value: {prediction}")  # Debug output
                
                # Adjust threshold based on disease type
                threshold = 0.4 if disease_type == 'tuberculosis' else 0.45  # Lower thresholds
                is_positive = prediction > threshold
                confidence = float(prediction) if is_positive else float(1 - prediction)
                confidence_percent = round(confidence * 100, 2)
                predicted_class = "positive" if is_positive else "negative"
                
                # Generate proper heatmap using occlusion sensitivity
                heatmap_filename = generate_occlusion_map(
                    model=model,
                    preprocessed_img=preprocessed_img,
                    file_path=file_path,
                    class_idx=0,  # For binary classification
                    patch_size=24 if disease_type == 'tuberculosis' else 28,  # Adjust patch size
                    stride=12 if disease_type == 'tuberculosis' else 14
                )
            
            # Get user details from the form
            patient_name = request.form.get('patientName')
            patient_age = request.form.get('patientAge')
            patient_gender = request.form.get('patientGender')
            patient_contact = request.form.get('patientContact')
            
            # Create result object
            result = {
                'original_image': unique_filename,
                'annotated_image': heatmap_filename,
                'prediction': predicted_class,
                'confidence': confidence_percent,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'details': [],
                'disease_type': disease_type,
                'patient_name': patient_name,
                'patient_age': patient_age,
                'patient_gender': patient_gender,
                'patient_contact': patient_contact
            }
            
            # Add findings based on prediction
            if disease_type == 'lung-cancer':
                if predicted_class == LUNG_CANCER_CLASSES[0]:  # Lung Adenocarcinoma
                    result['details'] = [
                        "Irregular nodular patterns detected in lung tissue",
                        "Ground-glass opacity patterns observed",
                        "Predominant peripheral distribution in the lungs",
                        "Consistent with adenocarcinoma histopathology"
                    ]
                elif predicted_class == LUNG_CANCER_CLASSES[1]:  # Lung Benign Tissue
                    result['details'] = [
                        "Normal lung parenchyma texture",
                        "No suspicious nodules or masses detected",
                        "Clear airway passages visible",
                        "Healthy tissue patterns throughout the scan"
                    ]
                elif predicted_class == LUNG_CANCER_CLASSES[2]:  # Lung Squamous Cell Carcinoma
                    result['details'] = [
                        "Central mass detected in bronchial region",
                        "Cavitation signs present",
                        "Thickened bronchial walls observed",
                        "Pattern consistent with squamous cell histopathology"
                    ]
            elif disease_type == 'tuberculosis':
                if predicted_class == "positive":
                    result['details'] = [
                        "Infiltrates detected in upper lobes",
                        "Fibrotic changes present",
                        "Cavity formation observed",
                        "Possible signs of active tuberculosis"
                    ]
                else:
                    result['details'] = [
                        "Normal lung parenchyma pattern",
                        "No significant abnormalities detected",
                        "No signs of cavitation or fibrosis",
                        "No indicators of active tuberculosis"
                    ]
            elif disease_type == 'pneumonia':
                if predicted_class == "positive":
                    result['details'] = [
                        "Consolidation present in lower lobes",
                        "Air bronchograms visible",
                        "Pleural effusion detected",
                        "Possible bacterial pneumonia"
                    ]
                else:
                    result['details'] = [
                        "Clear lung fields",
                        "No significant consolidation",
                        "Normal bronchial patterns",
                        "No signs of pneumonia detected"
                    ]
            
            # Redirect to results page with the session ID
            return jsonify({
                'success': True,
                'redirect': f'/{disease_type}-results',
                'result': result
            })
        
        except Exception as e:
            print(f"Error processing image: {e}")
            print(f"File path: {file_path}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400
@app.route('/download-report/<disease_type>', methods=['GET'])
def download_report(disease_type):
    # Retrieve the result from sessionStorage (passed as a query parameter)
    result_json = request.args.get('result')
    if not result_json:
        return jsonify({'error': 'No result data provided'}), 400
    
    # Parse the result JSON
    result = json.loads(result_json)
    
    # Generate a unique filename for the PDF
    pdf_filename = f"{disease_type}_report_{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    try:
        # Generate the PDF report
        generate_pdf_report(result, pdf_path)
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({'error': 'Failed to generate PDF report'}), 500
    
    # Serve the PDF for download
    return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)

@app.route('/<disease_type>-results')
def results_page(disease_type):
    # In a real app, you would retrieve results from database/cache using a session ID
    # For this example, we'll render the template with placeholder data
    return render_template(f'{disease_type}-results.html')

run_with_ngrok(app)  # This enables a public URL

if __name__ == '__main__':
    app.run()