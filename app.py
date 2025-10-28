from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
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
import tensorflow as tf
import gdown
from tensorflow.keras.models import load_model

# Force CPU (Render has no GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === Configuration ===
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive IDs for each model (replace with your IDs if different)
MODEL_IDS = {
    # mapping uses underscore filenames, keep IDs you provided earlier
    "lung_cancer_model.h5": "1IuBs4zJjDRt-r5ershRjWxn3u4_aQhb7",
    "pneumonia_model.h5": "1tayAtpf4i2xEWbsRR4wOsQXoEjV_VzPj",
    "tuberculosis_model.h5": "1RgH32TxcvPZQUDn3QCb-FsKadkJjUkMK"
}

# Map disease_type from URL to actual model filename
DISEASE_TO_FILENAME = {
    "lung-cancer": "lung_cancer_model.h5",
    "pneumonia": "pneumonia_model.h5",
    "tuberculosis": "tuberculosis_model.h5"
}

# cached models
loaded_models = {}

# Allowed upload types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# === Helpers: locale / logging ===
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')
    print("⚠️ Locale 'en_US.UTF-8' not available. Using default locale.")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === Flask app ===
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_to_a_random_string")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB uploads (adjust if needed)

# Class names
LUNG_CANCER_CLASSES = ['Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

# === Utility functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    filename = re.sub(r'[^\x00-\x7F]+', '', filename)
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    return filename

def download_model_file(model_filename):
    """Download a model from Google Drive into MODEL_DIR if missing."""
    path = os.path.join(MODEL_DIR, model_filename)
    if os.path.exists(path):
        print(f"✅ Model {model_filename} already present.")
        return path

    file_id = MODEL_IDS.get(model_filename)
    if not file_id:
        raise ValueError(f"No Drive ID found for {model_filename}")

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"🔽 Downloading {model_filename} from {url}")
    gdown.download(url, path, quiet=False)
    return path

def get_model(disease_type):
    """Return loaded model for disease_type, downloading & loading lazily."""
    if disease_type not in DISEASE_TO_FILENAME:
        raise ValueError(f"Invalid disease type: {disease_type}")

    model_filename = DISEASE_TO_FILENAME[disease_type]
    if model_filename in loaded_models:
        print(f"✅ Using cached model: {model_filename}")
        return loaded_models[model_filename]

    # Ensure file exists (download if needed) and load
    model_path = download_model_file(model_filename)
    print(f"🧠 Loading model from {model_path} ...")
    model = load_model(model_path)
    loaded_models[model_filename] = model
    print(f"✅ Model loaded and cached: {model_filename}")
    return model

# === Image preprocessing / enhancement / visualization ===
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_pneumonia(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def enhance_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img_equalized = cv2.equalizeHist(img)
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img_equalized, -1, sharpening_kernel)
        enhanced_filename = image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_filename, img_sharpened)
        return os.path.basename(enhanced_filename)
    except Exception as e:
        print(f"Error enhancing image: {e}")
        raise

def generate_occlusion_map(model, preprocessed_img, file_path, class_idx, patch_size=20, stride=8):
    """Generate occlusion sensitivity heatmap and save annotated image in the uploads folder."""
    print(f"Running occlusion sensitivity analysis (patch_size={patch_size}, stride={stride})...")
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image at {file_path}")

    original_img = cv2.resize(img, (224, 224))
    img_array = preprocessed_img.copy()

    baseline_pred = model.predict(img_array, verbose=0)[0][class_idx]

    heatmap = np.zeros((224, 224), dtype=np.float32)
    total_patches = ((224 - patch_size) // stride + 1) * ((224 - patch_size) // stride + 1)
    patches_done = 0

    for y in range(0, 224 - patch_size + 1, stride):
        for x in range(0, 224 - patch_size + 1, stride):
            occluded_img = np.copy(img_array)
            occluded_img[0, y:y+patch_size, x:x+patch_size, :] = 0.5
            occluded_pred = model.predict(occluded_img, verbose=0)[0][class_idx]
            diff = baseline_pred - occluded_pred
            heatmap[y:y+patch_size, x:x+patch_size] += diff
            patches_done += 1
            if patches_done % 20 == 0:
                print(f"Processed {patches_done}/{total_patches} patches ({(patches_done/total_patches)*100:.1f}%)")

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)

    heatmap_filename = file_path.replace('.', '_heatmap.')
    cv2.imwrite(heatmap_filename, superimposed_img)
    return os.path.basename(heatmap_filename)

def generate_pdf_report(result, output_path):
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, "Lung Disease Detection Report")
        c.setFont("Helvetica", 12)
        c.drawString(72, 730, f"Patient Name: {result.get('patient_name','N/A')}")
        c.drawString(72, 710, f"Age: {result.get('patient_age','N/A')}")
        c.drawString(72, 690, f"Gender: {result.get('patient_gender','N/A')}")
        c.drawString(72, 670, f"Contact Number: {result.get('patient_contact','N/A')}")
        c.drawString(72, 650, f"Date: {result.get('timestamp','N/A')}")
        c.drawString(72, 630, f"Detection Result: {result.get('prediction','N/A')}")
        c.drawString(72, 610, f"Confidence Level: {result.get('confidence','N/A')}%")
        c.drawString(72, 590, "Detailed Findings:")
        y_position = 570
        for detail in result.get('details', []):
            c.drawString(72, y_position, f"- {detail}")
            y_position -= 20

        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['original_image'])
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['annotated_image'])

        c.drawString(72, y_position - 20, "Original Image:")
        c.drawImage(ImageReader(original_image_path), 72, y_position - 220, width=200, height=200)
        c.drawString(72, y_position - 240, "Annotated Image (Occlusion Sensitivity Map):")
        c.drawImage(ImageReader(annotated_image_path), 72, y_position - 440, width=200, height=200)
        c.save()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise

# === Routes ===
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

@app.route('/pneumonia')
def pneumonia_page():
    return render_template('pneumonia.html')

@app.route('/enhance')
def enhance_page():
    return render_template('enhance.html')

@app.route('/enhance-image', methods=['POST'])
def enhance_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = sanitize_filename(secure_filename(file.filename))
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(file_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': 'Error saving file.'}), 500
        try:
            enhanced_filename = enhance_image(file_path)
            return jsonify({
                'success': True,
                'original_image': unique_filename,
                'enhanced_image': enhanced_filename
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/detect/<disease_type>', methods=['POST'])
def detect_disease(disease_type):
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = sanitize_filename(secure_filename(file.filename))
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    try:
        file.save(file_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file.'}), 500

    try:
        # Preprocess
        if disease_type == 'pneumonia':
            preprocessed_img = preprocess_image_pneumonia(file_path)
        else:
            preprocessed_img = preprocess_image(file_path)

        print(f"Preprocessed image shape: {preprocessed_img.shape}")

        # Load model lazily
        model = get_model(disease_type)

        if model is None:
            return jsonify({'error': 'Model not available'}), 500
        

        # Predict and generate heatmap
        if disease_type == 'lung-cancer':
            predictions = model.predict(preprocessed_img)[0]
            predicted_class_idx = int(np.argmax(predictions))
            confidence_percent = round(float(predictions[predicted_class_idx]) * 100, 2)
            predicted_class = LUNG_CANCER_CLASSES[predicted_class_idx]

            heatmap_filename = generate_occlusion_map(
                model=model,
                preprocessed_img=preprocessed_img,
                file_path=file_path,
                class_idx=predicted_class_idx,
                patch_size=20,
                stride=8
            )

        elif disease_type in ('tuberculosis', 'pneumonia'):
            prediction = model.predict(preprocessed_img)[0][0]
            threshold = 0.4 if disease_type == 'tuberculosis' else 0.45
            is_positive = prediction > threshold
            confidence = float(prediction) if is_positive else float(1 - prediction)
            confidence_percent = round(confidence * 100, 2)
            predicted_class = "positive" if is_positive else "negative"

            heatmap_filename = generate_occlusion_map(
                model=model,
                preprocessed_img=preprocessed_img,
                file_path=file_path,
                class_idx=0,
                patch_size=24 if disease_type == 'tuberculosis' else 28,
                stride=12 if disease_type == 'tuberculosis' else 14
            )
        else:
            return jsonify({'error': 'Invalid disease type'}), 400

        # Collect extra info from form (if any)
        patient_name = request.form.get('patientName', '')
        patient_age = request.form.get('patientAge', '')
        patient_gender = request.form.get('patientGender', '')
        patient_contact = request.form.get('patientContact', '')

        # Build details based on predicted class (same logic as before)
        details = []
        if disease_type == 'lung-cancer':
            if predicted_class == LUNG_CANCER_CLASSES[0]:
                details = [
                    "Irregular nodular patterns detected in lung tissue",
                    "Ground-glass opacity patterns observed",
                    "Predominant peripheral distribution in the lungs",
                    "Consistent with adenocarcinoma histopathology"
                ]
            elif predicted_class == LUNG_CANCER_CLASSES[1]:
                details = [
                    "Normal lung parenchyma texture",
                    "No suspicious nodules or masses detected",
                    "Clear airway passages visible",
                    "Healthy tissue patterns throughout the scan"
                ]
            else:
                details = [
                    "Central mass detected in bronchial region",
                    "Cavitation signs present",
                    "Thickened bronchial walls observed",
                    "Pattern consistent with squamous cell histopathology"
                ]
        elif disease_type == 'tuberculosis':
            details = [
                "Infiltrates detected in upper lobes",
                "Fibrotic changes present",
                "Cavity formation observed",
                "Possible signs of active tuberculosis"
            ] if predicted_class == "positive" else [
                "Normal lung parenchyma pattern",
                "No significant abnormalities detected",
                "No signs of cavitation or fibrosis",
                "No indicators of active tuberculosis"
            ]
        elif disease_type == 'pneumonia':
            details = [
                "Consolidation present in lower lobes",
                "Air bronchograms visible",
                "Pleural effusion detected",
                "Possible bacterial pneumonia"
            ] if predicted_class == "positive" else [
                "Clear lung fields",
                "No significant consolidation",
                "Normal bronchial patterns",
                "No signs of pneumonia detected"
            ]

        result = {
            'original_image': unique_filename,
            'annotated_image': heatmap_filename,
            'prediction': predicted_class,
            'confidence': confidence_percent,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': details,
            'disease_type': disease_type,
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_contact': patient_contact
        }

        # Save result to session and redirect to the proper results page
        session['last_result'] = result
        return jsonify({
    'success': True,
    'redirect': url_for(f"{disease_type.replace('-', '_')}_results"),
    'result': result
})

    except Exception as e:
        print(f"Error processing image: {e}")
        print(f"File path: {file_path}")
        return jsonify({'error': str(e)}), 500

# Results routes (render templates and pass result)
@app.route('/lung-cancer-results')
def lung_cancer_results():
    result = session.get('last_result')
    if not result:
        return redirect(url_for('index'))
    return render_template('lung-cancer-results.html', result=result)

@app.route('/tuberculosis-results')
def tuberculosis_results():
    result = session.get('last_result')
    if not result:
        return redirect(url_for('index'))
    return render_template('tuberculosis-results.html', result=result)

@app.route('/pneumonia-results')
def pneumonia_results():
    result = session.get('last_result')
    if not result:
        return redirect(url_for('index'))
    return render_template('pneumonia-results.html', result=result)

# Download PDF report based on last_result in session
@app.route('/download-report/<disease_type>', methods=['GET'])
def download_report(disease_type):
    result = session.get('last_result')
    if not result:
        return jsonify({'error': 'No result data provided'}), 400

    # Ensure result corresponds to requested disease (optional)
    if result.get('disease_type') != disease_type:
        return jsonify({'error': 'Result mismatch'}), 400

    pdf_filename = f"{disease_type}_report_{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

    try:
        generate_pdf_report(result, pdf_path)
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({'error': 'Failed to generate PDF report'}), 500

    return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)

# Serve uploaded files if needed (static serves /static/uploads already)
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Use the Flask dev server only locally. On Render, use gunicorn start command.
    app.run(host="0.0.0.0", port=port)
