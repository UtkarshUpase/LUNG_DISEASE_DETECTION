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

# Load the three models (placeholder functions)
def load_lung_cancer_model():
    # Replace with your actual model loading code
    try:
        model = tf.keras.models.load_model('models/lung_cancer_model.h5')
        print("Lung cancer model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading lung cancer model: {e}")
        return None

def load_tuberculosis_model():
    # Replace with your actual model loading code
    try:
        model = tf.keras.models.load_model('models/tuberculosis_model.h5')
        print("Tuberculosis model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading tuberculosis model: {e}")
        return None

def load_pneumonia_model():
    # Replace with your actual model loading code
    try:
        model = tf.keras.models.load_model('models/pneumonia_model.h5')
        print("Pneumonia model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading pneumonia model: {e}")
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
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

def generate_heatmap(model, preprocessed_img, original_img_path):
    """Generate a heatmap showing model focus areas (simplified implementation)"""
    # This is a placeholder. You would implement your own heatmap generation
    # based on techniques like Grad-CAM
    
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        raise ValueError(f"Could not read original image at {original_img_path}")
    
    height, width = original_img.shape[:2]
    
    # Create a simple heatmap (replace with actual implementation)
    heatmap = np.zeros((height, width), dtype=np.uint8)
    
    # Add a sample "hot" area - replace with actual model interpretation
    center_x, center_y = width // 2, height // 2
    cv2.circle(heatmap, (center_x, center_y), radius=50, color=255, thickness=-1)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)
    
    # Save the overlay
    overlay_path = original_img_path.replace('.', '_heatmap.')
    cv2.imwrite(overlay_path, overlay)
    
    return os.path.basename(overlay_path)

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
        c.drawString(72, 630, f"Detection Result: {result['prediction'].capitalize()}")
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
        c.drawString(72, y_position - 240, "Annotated Image:")
        c.drawImage(ImageReader(annotated_image_path), 72, y_position - 440, width=200, height=200)
        
        # Save the PDF
        c.save()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lung-cancer')
def lung_cancer_page():
    return render_template('lung-cancer.html')

@app.route('/tuberculosis')
def tuberculosis_page():
    return render_template('tuberculosis.html')

@app.route('/pneumonia')
def pneumonia_page():
    return render_template('pneumonia.html')

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
            # Preprocess the image
            preprocessed_img = preprocess_image(file_path)
            
            # Select the appropriate model based on disease_type
            if disease_type == 'lung-cancer':
                model = lung_cancer_model
                threshold = 0.5  # Set appropriate threshold
            elif disease_type == 'tuberculosis':
                model = tuberculosis_model
                threshold = 0.5
            elif disease_type == 'pneumonia':
                model = pneumonia_model
                threshold = 0.5
            else:
                return jsonify({'error': 'Invalid disease type'}), 400
            
            if model is None:
                print(f"Model for {disease_type} is not available. Check if the model file exists and is valid.")
                return jsonify({'error': f'Model for {disease_type} is not available'}), 500
            
            # Make prediction
            prediction = model.predict(preprocessed_img)[0][0]
            is_positive = prediction > threshold
            confidence = float(prediction) if is_positive else float(1 - prediction)
            confidence_percent = round(confidence * 100, 2)
            
            # Generate heatmap
            heatmap_filename = generate_heatmap(model, preprocessed_img, file_path)
            
            # Get user details from the form
            patient_name = request.form.get('patientName')
            patient_age = request.form.get('patientAge')
            patient_gender = request.form.get('patientGender')
            patient_contact = request.form.get('patientContact')
            
            # Create result object
            result = {
                'original_image': unique_filename,
                'annotated_image': heatmap_filename,
                'prediction': 'positive' if is_positive else 'negative',
                'confidence': confidence_percent,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Simplified timestamp format
                'details': [],
                'disease_type': disease_type,  # Add disease type for the download report
                'patient_name': patient_name,
                'patient_age': patient_age,
                'patient_gender': patient_gender,
                'patient_contact': patient_contact
            }
            
            # Add disease-specific findings
            if disease_type == 'lung-cancer' and is_positive:
                result['details'] = [
                    "Nodule detected in lung tissue",
                    "Irregular border patterns observed",
                    "Density analysis indicates solid mass",
                    "No signs of metastasis to surrounding tissues"
                ]
            elif disease_type == 'tuberculosis' and is_positive:
                result['details'] = [
                    "Infiltrates detected in upper lobes",
                    "Fibrotic changes present",
                    "Cavity formation observed",
                    "Possible signs of active tuberculosis"
                ]
            elif disease_type == 'pneumonia' and is_positive:
                result['details'] = [
                    "Consolidation present in lower lobes",
                    "Air bronchograms visible",
                    "Pleural effusion detected",
                    "Possible bacterial pneumonia"
                ]
            
            # Store result in session
            session_id = str(uuid.uuid4())
            
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

if __name__ == '__main__':
    app.run(debug=True)