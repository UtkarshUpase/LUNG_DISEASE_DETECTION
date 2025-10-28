# from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session
# import os
# import numpy as np
# import cv2
# from werkzeug.utils import secure_filename
# from datetime import datetime
# import uuid
# import locale
# import re
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# import json
# import sys
# import io
# import tensorflow as tf
# import gdown
# from tensorflow.keras.models import load_model

# # Force CPU (Render has no GPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # === Configuration ===
# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Google Drive IDs for each model (replace with your IDs if different)
# MODEL_IDS = {
#     # mapping uses underscore filenames, keep IDs you provided earlier
#     "lung_cancer_model.h5": "1IuBs4zJjDRt-r5ershRjWxn3u4_aQhb7",
#     "pneumonia_model.h5": "1tayAtpf4i2xEWbsRR4wOsQXoEjV_VzPj",
#     "tuberculosis_model.h5": "1RgH32TxcvPZQUDn3QCb-FsKadkJjUkMK"
# }

# # Map disease_type from URL to actual model filename
# DISEASE_TO_FILENAME = {
#     "lung-cancer": "lung_cancer_model.h5",
#     "pneumonia": "pneumonia_model.h5",
#     "tuberculosis": "tuberculosis_model.h5"
# }

# # cached models
# loaded_models = {}

# # Allowed upload types
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # === Helpers: locale / logging ===
# try:
#     locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
# except locale.Error:
#     locale.setlocale(locale.LC_ALL, '')
#     print("⚠️ Locale 'en_US.UTF-8' not available. Using default locale.")

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# # === Flask app ===
# app = Flask(__name__, static_folder='static', template_folder='templates')
# app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_to_a_random_string")

# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB uploads (adjust if needed)

# # Class names
# LUNG_CANCER_CLASSES = ['Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

# # === Utility functions ===
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def sanitize_filename(filename):
#     filename = re.sub(r'[^\x00-\x7F]+', '', filename)
#     filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
#     return filename

# def download_model_file(model_filename):
#     """Download a model from Google Drive into MODEL_DIR if missing."""
#     path = os.path.join(MODEL_DIR, model_filename)
#     if os.path.exists(path):
#         print(f"✅ Model {model_filename} already present.")
#         return path

#     file_id = MODEL_IDS.get(model_filename)
#     if not file_id:
#         raise ValueError(f"No Drive ID found for {model_filename}")

#     url = f"https://drive.google.com/uc?id={file_id}"
#     print(f"🔽 Downloading {model_filename} from {url}")
#     gdown.download(url, path, quiet=False)
#     return path

# def get_model(disease_type):
#     """Return loaded model for disease_type, downloading & loading lazily."""
#     if disease_type not in DISEASE_TO_FILENAME:
#         raise ValueError(f"Invalid disease type: {disease_type}")

#     model_filename = DISEASE_TO_FILENAME[disease_type]
#     if model_filename in loaded_models:
#         print(f"✅ Using cached model: {model_filename}")
#         return loaded_models[model_filename]

#     # Ensure file exists (download if needed) and load
#     model_path = download_model_file(model_filename)
#     print(f"🧠 Loading model from {model_path} ...")
#     model = load_model(model_path)
#     loaded_models[model_filename] = model
#     print(f"✅ Model loaded and cached: {model_filename}")
#     return model

# # === Image preprocessing / enhancement / visualization ===
# def preprocess_image(image_path, target_size=(224, 224)):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Could not read image at {image_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, target_size)
#     img = img / 255.0
#     return np.expand_dims(img, axis=0)

# def preprocess_image_pneumonia(image_path, target_size=(256, 256)):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Could not read image at {image_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, target_size)
#     img = img / 255.0
#     return np.expand_dims(img, axis=0)

# def enhance_image(image_path):
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise ValueError(f"Could not read image at {image_path}")
#         img_equalized = cv2.equalizeHist(img)
#         sharpening_kernel = np.array([[-1, -1, -1],
#                                       [-1,  9, -1],
#                                       [-1, -1, -1]])
#         img_sharpened = cv2.filter2D(img_equalized, -1, sharpening_kernel)
#         enhanced_filename = image_path.replace('.', '_enhanced.')
#         cv2.imwrite(enhanced_filename, img_sharpened)
#         return os.path.basename(enhanced_filename)
#     except Exception as e:
#         print(f"Error enhancing image: {e}")
#         raise

# def generate_occlusion_map(model, preprocessed_img, file_path, class_idx, patch_size=20, stride=8):
#     """Generate occlusion sensitivity heatmap and save annotated image in the uploads folder."""
#     print(f"Running occlusion sensitivity analysis (patch_size={patch_size}, stride={stride})...")
#     img = cv2.imread(file_path)
#     if img is None:
#         raise ValueError(f"Could not read image at {file_path}")

#     original_img = cv2.resize(img, (224, 224))
#     img_array = preprocessed_img.copy()

#     baseline_pred = model.predict(img_array, verbose=0)[0][class_idx]

#     heatmap = np.zeros((224, 224), dtype=np.float32)
#     total_patches = ((224 - patch_size) // stride + 1) * ((224 - patch_size) // stride + 1)
#     patches_done = 0

#     for y in range(0, 224 - patch_size + 1, stride):
#         for x in range(0, 224 - patch_size + 1, stride):
#             occluded_img = np.copy(img_array)
#             occluded_img[0, y:y+patch_size, x:x+patch_size, :] = 0.5
#             occluded_pred = model.predict(occluded_img, verbose=0)[0][class_idx]
#             diff = baseline_pred - occluded_pred
#             heatmap[y:y+patch_size, x:x+patch_size] += diff
#             patches_done += 1
#             if patches_done % 20 == 0:
#                 print(f"Processed {patches_done}/{total_patches} patches ({(patches_done/total_patches)*100:.1f}%)")

#     heatmap = np.maximum(heatmap, 0)
#     if np.max(heatmap) > 0:
#         heatmap = heatmap / np.max(heatmap)

#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)

#     heatmap_filename = file_path.replace('.', '_heatmap.')
#     cv2.imwrite(heatmap_filename, superimposed_img)
#     return os.path.basename(heatmap_filename)

# def generate_pdf_report(result, output_path):
#     try:
#         c = canvas.Canvas(output_path, pagesize=letter)
#         c.setFont("Helvetica-Bold", 16)
#         c.drawString(72, 750, "Lung Disease Detection Report")
#         c.setFont("Helvetica", 12)
#         c.drawString(72, 730, f"Patient Name: {result.get('patient_name','N/A')}")
#         c.drawString(72, 710, f"Age: {result.get('patient_age','N/A')}")
#         c.drawString(72, 690, f"Gender: {result.get('patient_gender','N/A')}")
#         c.drawString(72, 670, f"Contact Number: {result.get('patient_contact','N/A')}")
#         c.drawString(72, 650, f"Date: {result.get('timestamp','N/A')}")
#         c.drawString(72, 630, f"Detection Result: {result.get('prediction','N/A')}")
#         c.drawString(72, 610, f"Confidence Level: {result.get('confidence','N/A')}%")
#         c.drawString(72, 590, "Detailed Findings:")
#         y_position = 570
#         for detail in result.get('details', []):
#             c.drawString(72, y_position, f"- {detail}")
#             y_position -= 20

#         original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['original_image'])
#         annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['annotated_image'])

#         c.drawString(72, y_position - 20, "Original Image:")
#         c.drawImage(ImageReader(original_image_path), 72, y_position - 220, width=200, height=200)
#         c.drawString(72, y_position - 240, "Annotated Image (Occlusion Sensitivity Map):")
#         c.drawImage(ImageReader(annotated_image_path), 72, y_position - 440, width=200, height=200)
#         c.save()
#     except Exception as e:
#         print(f"Error generating PDF: {e}")
#         raise

# # === Routes ===
# @app.after_request
# def add_headers(response):
#     response.headers["ngrok-skip-browser-warning"] = "true"
#     return response

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/lung-cancer')
# def lung_cancer_page():
#     return render_template('lung-cancer.html')

# @app.route('/tuberculosis')
# def tuberculosis_page():
#     return render_template('tuberculosis.html')

# @app.route('/pneumonia')
# def pneumonia_page():
#     return render_template('pneumonia.html')

# @app.route('/enhance')
# def enhance_page():
#     return render_template('enhance.html')

# @app.route('/enhance-image', methods=['POST'])
# def enhance_image_route():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part'}), 400
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if file and allowed_file(file.filename):
#         filename = sanitize_filename(secure_filename(file.filename))
#         unique_filename = f"{uuid.uuid4()}_{filename}"
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#         try:
#             file.save(file_path)
#         except Exception as e:
#             print(f"Error saving file: {e}")
#             return jsonify({'error': 'Error saving file.'}), 500
#         try:
#             enhanced_filename = enhance_image(file_path)
#             return jsonify({
#                 'success': True,
#                 'original_image': unique_filename,
#                 'enhanced_image': enhanced_filename
#             })
#         except Exception as e:
#             print(f"Error processing image: {e}")
#             return jsonify({'error': str(e)}), 500
#     return jsonify({'error': 'File type not allowed'}), 400

# @app.route('/detect/<disease_type>', methods=['POST'])
# def detect_disease(disease_type):
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part'}), 400
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'File type not allowed'}), 400

#     filename = sanitize_filename(secure_filename(file.filename))
#     unique_filename = f"{uuid.uuid4()}_{filename}"
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     try:
#         file.save(file_path)
#     except Exception as e:
#         print(f"Error saving file: {e}")
#         return jsonify({'error': 'Error saving file.'}), 500

#     try:
#         # Preprocess
#         if disease_type == 'pneumonia':
#             preprocessed_img = preprocess_image_pneumonia(file_path)
#         else:
#             preprocessed_img = preprocess_image(file_path)

#         print(f"Preprocessed image shape: {preprocessed_img.shape}")

#         # Load model lazily
#         model = get_model(disease_type)

#         if model is None:
#             return jsonify({'error': 'Model not available'}), 500
        

#         # Predict and generate heatmap
#         if disease_type == 'lung-cancer':
#             predictions = model.predict(preprocessed_img)[0]
#             predicted_class_idx = int(np.argmax(predictions))
#             confidence_percent = round(float(predictions[predicted_class_idx]) * 100, 2)
#             predicted_class = LUNG_CANCER_CLASSES[predicted_class_idx]

#             heatmap_filename = generate_occlusion_map(
#                 model=model,
#                 preprocessed_img=preprocessed_img,
#                 file_path=file_path,
#                 class_idx=predicted_class_idx,
#                 patch_size=20,
#                 stride=8
#             )

#         elif disease_type in ('tuberculosis', 'pneumonia'):
#             prediction = model.predict(preprocessed_img)[0][0]
#             threshold = 0.4 if disease_type == 'tuberculosis' else 0.45
#             is_positive = prediction > threshold
#             confidence = float(prediction) if is_positive else float(1 - prediction)
#             confidence_percent = round(confidence * 100, 2)
#             predicted_class = "positive" if is_positive else "negative"

#             heatmap_filename = generate_occlusion_map(
#                 model=model,
#                 preprocessed_img=preprocessed_img,
#                 file_path=file_path,
#                 class_idx=0,
#                 patch_size=24 if disease_type == 'tuberculosis' else 28,
#                 stride=12 if disease_type == 'tuberculosis' else 14
#             )
#         else:
#             return jsonify({'error': 'Invalid disease type'}), 400

#         # Collect extra info from form (if any)
#         patient_name = request.form.get('patientName', '')
#         patient_age = request.form.get('patientAge', '')
#         patient_gender = request.form.get('patientGender', '')
#         patient_contact = request.form.get('patientContact', '')

#         # Build details based on predicted class (same logic as before)
#         details = []
#         if disease_type == 'lung-cancer':
#             if predicted_class == LUNG_CANCER_CLASSES[0]:
#                 details = [
#                     "Irregular nodular patterns detected in lung tissue",
#                     "Ground-glass opacity patterns observed",
#                     "Predominant peripheral distribution in the lungs",
#                     "Consistent with adenocarcinoma histopathology"
#                 ]
#             elif predicted_class == LUNG_CANCER_CLASSES[1]:
#                 details = [
#                     "Normal lung parenchyma texture",
#                     "No suspicious nodules or masses detected",
#                     "Clear airway passages visible",
#                     "Healthy tissue patterns throughout the scan"
#                 ]
#             else:
#                 details = [
#                     "Central mass detected in bronchial region",
#                     "Cavitation signs present",
#                     "Thickened bronchial walls observed",
#                     "Pattern consistent with squamous cell histopathology"
#                 ]
#         elif disease_type == 'tuberculosis':
#             details = [
#                 "Infiltrates detected in upper lobes",
#                 "Fibrotic changes present",
#                 "Cavity formation observed",
#                 "Possible signs of active tuberculosis"
#             ] if predicted_class == "positive" else [
#                 "Normal lung parenchyma pattern",
#                 "No significant abnormalities detected",
#                 "No signs of cavitation or fibrosis",
#                 "No indicators of active tuberculosis"
#             ]
#         elif disease_type == 'pneumonia':
#             details = [
#                 "Consolidation present in lower lobes",
#                 "Air bronchograms visible",
#                 "Pleural effusion detected",
#                 "Possible bacterial pneumonia"
#             ] if predicted_class == "positive" else [
#                 "Clear lung fields",
#                 "No significant consolidation",
#                 "Normal bronchial patterns",
#                 "No signs of pneumonia detected"
#             ]

#         result = {
#             'original_image': unique_filename,
#             'annotated_image': heatmap_filename,
#             'prediction': predicted_class,
#             'confidence': confidence_percent,
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'details': details,
#             'disease_type': disease_type,
#             'patient_name': patient_name,
#             'patient_age': patient_age,
#             'patient_gender': patient_gender,
#             'patient_contact': patient_contact
#         }

#         # Save result to session and redirect to the proper results page
#         session['last_result'] = result
#         return jsonify({
#     'success': True,
#     'redirect': url_for(f"{disease_type.replace('-', '_')}_results"),
#     'result': result
# })

#     except Exception as e:
#         print(f"Error processing image: {e}")
#         print(f"File path: {file_path}")
#         return jsonify({'error': str(e)}), 500

# # Results routes (render templates and pass result)
# @app.route('/lung-cancer-results')
# def lung_cancer_results():
#     result = session.get('last_result')
#     if not result:
#         return redirect(url_for('index'))
#     return render_template('lung-cancer-results.html', result=result)

# @app.route('/tuberculosis-results')
# def tuberculosis_results():
#     result = session.get('last_result')
#     if not result:
#         return redirect(url_for('index'))
#     return render_template('tuberculosis-results.html', result=result)

# @app.route('/pneumonia-results')
# def pneumonia_results():
#     result = session.get('last_result')
#     if not result:
#         return redirect(url_for('index'))
#     return render_template('pneumonia-results.html', result=result)

# # Download PDF report based on last_result in session
# @app.route('/download-report/<disease_type>', methods=['GET'])
# def download_report(disease_type):
#     result = session.get('last_result')
#     if not result:
#         return jsonify({'error': 'No result data provided'}), 400

#     # Ensure result corresponds to requested disease (optional)
#     if result.get('disease_type') != disease_type:
#         return jsonify({'error': 'Result mismatch'}), 400

#     pdf_filename = f"{disease_type}_report_{uuid.uuid4()}.pdf"
#     pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

#     try:
#         generate_pdf_report(result, pdf_path)
#     except Exception as e:
#         print(f"Error generating PDF: {e}")
#         return jsonify({'error': 'Failed to generate PDF report'}), 500

#     return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)

# # Serve uploaded files if needed (static serves /static/uploads already)
# @app.route('/uploads/<path:filename>')
# def uploads(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# # Run
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     # Use the Flask dev server only locally. On Render, use gunicorn start command.
#     app.run(host="0.0.0.0", port=port)

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

# Google Drive IDs for each model
MODEL_IDS = {
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

# === Layer Names Found by check_model.py ===
# Using names found in your output. Pneumonia is left as placeholder to disable annotation.
LUNG_CANCER_CONV_LAYER = "conv2d_3"
PNEUMONIA_CONV_LAYER = "YOUR_PNEUMONIA_LAYER_NAME_HERE" # Placeholder - check_model failed
TUBERCULOSIS_CONV_LAYER = "conv2d_3"
# ============================================

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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

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
    path = os.path.join(MODEL_DIR, model_filename)
    if os.path.exists(path):
        print(f"✅ Model {model_filename} already present.")
        return path
    file_id = MODEL_IDS.get(model_filename)
    if not file_id: raise ValueError(f"No Drive ID for {model_filename}")
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"🔽 Downloading {model_filename}...")
    gdown.download(url, path, quiet=False)
    return path

def get_model(disease_type):
    model_filename = DISEASE_TO_FILENAME.get(disease_type)
    if not model_filename: raise ValueError(f"Invalid disease type: {disease_type}")
    if model_filename in loaded_models:
        print(f"✅ Using cached model: {model_filename}")
        return loaded_models[model_filename]
    model_path = download_model_file(model_filename)
    print(f"🧠 Loading model from {model_path}...")
    try:
        # Load model with compile=False if you are only doing inference
        model = load_model(model_path, compile=False)
        loaded_models[model_filename] = model
        print(f"✅ Model loaded: {model_filename}")
        return model
    except Exception as e:
        print(f"❌ ERROR loading model {model_filename}: {e}")
        return None # Return None if loading fails

# === Image preprocessing / enhancement / visualization ===
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_pneumonia(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def enhance_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Could not read image at {image_path}")
        img_equalized = cv2.equalizeHist(img)
        sharpening_kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img_equalized, -1, sharpening_kernel)
        enhanced_filename = image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_filename, img_sharpened)
        return os.path.basename(enhanced_filename)
    except Exception as e: print(f"Error enhancing image: {e}"); raise

# =================================================================
# === GRAD-CAM FUNCTION (with enhanced logging AND FIX) ===
# =================================================================
def generate_grad_cam(model, preprocessed_img, file_path, last_conv_layer_name):
    """Generates Grad-CAM, returns heatmap filename or original on failure."""
    print(f"--- Running Grad-CAM ---")
    print(f"Model: {model.name if hasattr(model, 'name') else 'N/A'}")
    print(f"Target Layer: {last_conv_layer_name}")
    original_basename = os.path.basename(file_path) # Keep original name for fallback

    img = cv2.imread(file_path)
    if img is None: print(f"❌ ERROR: Could not read image {file_path}"); return original_basename
    target_size = (preprocessed_img.shape[1], preprocessed_img.shape[2])
    original_img = cv2.resize(img, target_size)

    try:
        # === FIX: Ensure model is built by running a dummy prediction ===
        # This forces Keras to define output shapes before creating grad_model
        _ = model(preprocessed_img, training=False)
        print("Model called once to ensure it's built.")
        # === END FIX ===

        target_layer = model.get_layer(last_conv_layer_name); conv_layer_output = target_layer.output
        print(f"Successfully found layer: {target_layer.name}")
    except ValueError: print(f"❌ FATAL ERROR: Layer '{last_conv_layer_name}' not found."); return original_basename
    except Exception as e: print(f"❌ UNEXPECTED ERROR getting layer or running dummy prediction: {e}"); return original_basename

    if isinstance(conv_layer_output, list): conv_layer_output = conv_layer_output[0]

    try: grad_model = tf.keras.models.Model([model.inputs], [conv_layer_output, model.output])
    except Exception as e: print(f"❌ ERROR creating Grad-CAM model: {e}"); return original_basename

    heatmap_np = None
    try:
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(preprocessed_img, training=False)
            idx = tf.argmax(preds[0]) if preds.shape[1] > 1 else 0
            class_output = preds[:, idx]
        grads = tape.gradient(class_output, last_conv_layer_output)
        if grads is None: print(f"❌ ERROR: Gradient calculation None."); return original_basename
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        if last_conv_layer_output is None or last_conv_layer_output.shape[0]==0: print(f"❌ ERROR: conv output empty."); return original_basename
        last_conv_layer_output = last_conv_layer_output[0]
        if tf.reduce_any(tf.math.is_nan(pooled_grads)|tf.math.is_inf(pooled_grads)): print(f"❌ ERROR: pooled_grads invalid."); return original_basename
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]; heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0); max_val = tf.math.reduce_max(heatmap)
        if max_val == 0 or tf.math.is_nan(max_val) or tf.math.is_inf(max_val): print("⚠️ WARNING: Heatmap empty/invalid."); heatmap_np = np.zeros(target_size, dtype=np.uint8)
        else: heatmap = heatmap / max_val; heatmap_np = heatmap.numpy()
    except Exception as e: import traceback; print(f"❌ ERROR during Grad-CAM calc: {e}"); traceback.print_exc(); return original_basename

    try:
        heatmap_resized = cv2.resize(heatmap_np, target_size)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)
        heatmap_filename_base = original_basename.replace('.', '_heatmap.')
        heatmap_filepath = os.path.join(os.path.dirname(file_path), heatmap_filename_base)
        save_success = cv2.imwrite(heatmap_filepath, superimposed_img)
        if save_success: print(f"✅ Saved Grad-CAM: {heatmap_filename_base}"); return heatmap_filename_base
        else: print(f"❌ ERROR: Failed saving heatmap {heatmap_filepath}"); return original_basename
    except Exception as e: print(f"❌ ERROR saving/overlaying heatmap: {e}"); return original_basename
# =================================================================

# =================================================================
# === PDF Report Generation ===
# =================================================================
def generate_pdf_report(result, output_path):
    # ... (generate_pdf_report function remains the same as previous version) ...
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter # Get page dimensions

        # --- Header ---
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 50, "Lung Disease Detection Report")
        c.line(72, height - 55, width - 72, height - 55)

        # --- Patient Info ---
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, height - 80, "Patient Information")
        c.setFont("Helvetica", 11)
        textobject = c.beginText(72, height - 100)
        textobject.textLine(f"Name: {result.get('patient_name','N/A')}")
        textobject.textLine(f"Age: {result.get('patient_age','N/A')}")
        textobject.textLine(f"Gender: {result.get('patient_gender','N/A')}")
        textobject.textLine(f"Contact: {result.get('patient_contact','N/A')}")
        textobject.textLine(f"Report Date: {result.get('timestamp','N/A')}")
        c.drawText(textobject)

        # --- Detection Summary ---
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, height - 180, "Detection Summary")
        c.setFont("Helvetica", 11)
        textobject = c.beginText(72, height - 200)
        textobject.textLine(f"Analysis Type: {result.get('disease_type','N/A').replace('-', ' ').title()}")
        textobject.textLine(f"Result: {result.get('prediction','N/A')}")
        textobject.textLine(f"Confidence: {result.get('confidence','N/A')}%")
        c.drawText(textobject)

        # --- Detailed Findings ---
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, height - 260, "Detailed Findings")
        c.setFont("Helvetica", 10)
        y_position = height - 280
        details = result.get('details', [])
        if details:
            for detail in details:
                max_line_width = 75
                lines = [f"- {detail[i:i+max_line_width]}" for i in range(0, len(detail), max_line_width)]
                for line in lines:
                    if y_position < 100:
                         c.showPage(); y_position = height - 50; c.setFont("Helvetica", 10)
                    c.drawString(85, y_position, line)
                    y_position -= 15
        else:
             c.drawString(85, y_position, "- No specific findings listed."); y_position -= 15

        # --- Images ---
        img_y_start = y_position - 30
        max_img_height = 180
        max_img_width = (width - 144) / 2 - 20
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['original_image'])
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['annotated_image'])
        if img_y_start - max_img_height < 72: c.showPage(); img_y_start = height - 80
        c.setFont("Helvetica-Bold", 12); c.drawString(72, img_y_start, "Images"); img_y_start -= 20

        # Draw Original Image
        c.setFont("Helvetica", 10); c.drawString(72, img_y_start, "Original Image:")
        img_y_current = img_y_start - (max_img_height + 10)
        if os.path.exists(original_image_path):
            try:
                img_reader = ImageReader(original_image_path); img_width, img_height = img_reader.getSize()
                aspect = img_height / float(img_width) if img_width else 1; display_width = max_img_width
                display_height = display_width * aspect
                if display_height > max_img_height: display_height = max_img_height; display_width = display_height / aspect if aspect else max_img_width
                c.drawImage(img_reader, 72, img_y_current, width=display_width, height=display_height, preserveAspectRatio=True, anchor='n')
            except Exception as img_e: print(f"Err draw orig PDF: {img_e}"); c.drawString(72, img_y_current + max_img_height/2, "[Err loading orig img]")
        else: c.drawString(72, img_y_current + max_img_height/2, "[Orig img not found]")

        # Draw Annotated Image
        img_x_start_annotated = 72 + max_img_width + 20
        img_y_start_ann_title = img_y_start
        img_y_current_ann = img_y_start - (max_img_height + 10)
        c.setFont("Helvetica", 10)
        if result['original_image'] != result['annotated_image']:
            c.drawString(img_x_start_annotated, img_y_start_ann_title, "Annotated Image (Grad-CAM):")
            if os.path.exists(annotated_image_path):
                try:
                    img_reader_ann = ImageReader(annotated_image_path); img_width_ann, img_height_ann = img_reader_ann.getSize()
                    aspect_ann = img_height_ann / float(img_width_ann) if img_width_ann else 1; display_width_ann = max_img_width
                    display_height_ann = display_width_ann * aspect_ann
                    if display_height_ann > max_img_height: display_height_ann = max_img_height; display_width_ann = display_height_ann / aspect_ann if aspect_ann else max_img_width
                    c.drawImage(img_reader_ann, img_x_start_annotated, img_y_current_ann, width=display_width_ann, height=display_height_ann, preserveAspectRatio=True, anchor='n')
                except Exception as img_ann_e: print(f"Err draw ann PDF: {img_ann_e}"); c.drawString(img_x_start_annotated, img_y_current_ann + max_img_height/2, "[Err loading ann img]")
            else: c.drawString(img_x_start_annotated, img_y_current_ann + max_img_height/2, "[Ann img not found]")
        else:
             reason = "Skipped (Placeholder Name)" if "YOUR_" in PNEUMONIA_CONV_LAYER and result.get('disease_type') == 'pneumonia' else "Failed / Skipped"
             c.drawString(img_x_start_annotated, img_y_start_ann_title, f"Annotation: {reason}")
             c.rect(img_x_start_annotated, img_y_current_ann, max_img_width, max_img_height, stroke=1, fill=0)
             c.drawCentredString(img_x_start_annotated + max_img_width / 2, img_y_current_ann + max_img_height / 2, "[No Annotation Available]")

        # Footer Disclaimer
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(72, 50, "Disclaimer: AI analysis for informational purposes only. Not a substitute for professional medical advice.")
        c.drawString(72, 35, "Consult a qualified healthcare provider.")
        c.save()
        print(f"✅ PDF report generated: {output_path}")
    except Exception as e: import traceback; print(f"❌ Error generating PDF: {e}"); traceback.print_exc(); raise
# =================================================================

# === Routes ===
@app.after_request
def add_headers(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index(): return render_template('index.html')
@app.route('/lung-cancer')
def lung_cancer_page(): return render_template('lung-cancer.html')
@app.route('/tuberculosis')
def tuberculosis_page(): return render_template('tuberculosis.html')
@app.route('/pneumonia')
def pneumonia_page(): return render_template('pneumonia.html')
@app.route('/enhance')
def enhance_page(): return render_template('enhance.html')

@app.route('/enhance-image', methods=['POST'])
def enhance_image_route():
    # ... (enhance_image_route remains the same) ...
    if 'image' not in request.files: return jsonify({'error': 'No image part'}), 400
    file = request.files['image'];
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = sanitize_filename(secure_filename(file.filename))
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try: file.save(file_path)
        except Exception as e: print(f"Error saving enhance file: {e}"); return jsonify({'error': 'Error saving file.'}), 500
        try:
            enhanced_filename = enhance_image(file_path)
            return jsonify({'success': True, 'original_image': unique_filename, 'enhanced_image': enhanced_filename})
        except Exception as e: print(f"Error processing enhance image: {e}"); return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File type not allowed'}), 400

# =================================================================
# === detect_disease FUNCTION (Uses Fixed Grad-CAM) ===
# =================================================================
@app.route('/detect/<disease_type>', methods=['POST'])
def detect_disease(disease_type):
    if 'image' not in request.files: return jsonify({'error': 'No image part'}), 400
    file = request.files['image'];
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename): return jsonify({'error': 'File type not allowed'}), 400

    filename = sanitize_filename(secure_filename(file.filename))
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    try: file.save(file_path); print(f"Saved uploaded file: {file_path}")
    except Exception as e: print(f"Error saving file: {e}"); return jsonify({'error': 'Error saving file.'}), 500

    heatmap_filename_base = unique_filename # Default to original

    try:
        # Preprocess & Determine Layer Name
        if disease_type == 'pneumonia':
            preprocessed_img = preprocess_image_pneumonia(file_path)
            layer_name_constant = PNEUMONIA_CONV_LAYER
        elif disease_type == 'tuberculosis':
             preprocessed_img = preprocess_image(file_path)
             layer_name_constant = TUBERCULOSIS_CONV_LAYER
        elif disease_type == 'lung-cancer':
             preprocessed_img = preprocess_image(file_path)
             layer_name_constant = LUNG_CANCER_CONV_LAYER
        else: return jsonify({'error': 'Invalid disease type'}), 400
        print(f"Preprocessed image shape: {preprocessed_img.shape}")

        # Load model
        model = get_model(disease_type)
        if model is None: return jsonify({'error': f"Model for {disease_type} failed to load."}), 500

        # Prediction
        print(f"--- Running Prediction for {disease_type} ---")
        if disease_type == 'lung-cancer':
            predictions = model.predict(preprocessed_img)[0]
            idx = int(np.argmax(predictions))
            conf = round(float(predictions[idx]) * 100, 2)
            pred_class = LUNG_CANCER_CLASSES[idx]
            print(f"Lung Cancer Prediction Raw: {predictions}")
            print(f"Lung Cancer Predicted Class: {pred_class} (Index: {idx}), Conf: {conf}%")
        else: # TB or Pneumonia
            pred_val = model.predict(preprocessed_img)[0][0]
            thresh = 0.4 if disease_type == 'tuberculosis' else 0.45
            is_pos = pred_val > thresh
            conf = round(float(pred_val if is_pos else 1 - pred_val) * 100, 2)
            pred_class = "positive" if is_pos else "negative"
            print(f"{disease_type.title()} Prediction Raw: {pred_val}, Threshold: {thresh}")
            print(f"{disease_type.title()} Predicted Class: {pred_class}, Conf: {conf}%")

        # Attempt Grad-CAM
        print(f"\n--- Attempting Grad-CAM for {disease_type} ---")
        if not layer_name_constant or "YOUR_" in layer_name_constant:
             print(f"⚠️ SKIPPING Grad-CAM: Invalid/placeholder layer name '{layer_name_constant}'.")
             # heatmap_filename_base remains unique_filename (original)
        else:
             try:
                 # Call the updated Grad-CAM function
                 heatmap_filename_base = generate_grad_cam(model, preprocessed_img, file_path, layer_name_constant)
                 if heatmap_filename_base == unique_filename: print(f"⚠️ Grad-CAM failed/fallback for {disease_type}.")
                 else: print(f"✅ Grad-CAM successful for {disease_type}: {heatmap_filename_base}")
             except Exception as grad_e:
                 import traceback
                 print(f"❌ Uncaught Error during Grad-CAM call for {disease_type}: {grad_e}")
                 traceback.print_exc()
                 heatmap_filename_base = unique_filename # Ensure fallback

        # Details based on prediction
        details = []
        # (Your existing details logic)
        if disease_type == 'lung-cancer':
             if pred_class == LUNG_CANCER_CLASSES[0]: details = ["Cellular patterns suggest glandular origin.", "Potential adenocarcinoma features observed.", "Further cytological analysis recommended."]
             elif pred_class == LUNG_CANCER_CLASSES[1]: details = ["Normal cellular structures observed.", "No significant signs of malignancy detected.", "Appears consistent with benign tissue."]
             else: details = ["Keratinization patterns noted.", "Features suggestive of squamous cell origin.", "Biopsy and histological confirmation advised."]
        elif disease_type == 'tuberculosis':
             details = ["Presence of granulomas or caseous necrosis suspected.", "Patterns potentially indicative of TB infection.", "Clinical correlation and further testing needed."] if pred_class == "positive" else ["No clear signs of granulomatous inflammation.", "Lung tissue appears within normal limits for TB.", "Tuberculosis unlikely based on this image."]
        elif disease_type == 'pneumonia':
             details = ["Alveolar spaces appear filled with exudate.", "Inflammatory cell infiltration suggested.", "Findings consistent with pneumonia patterns."] if pred_class == "positive" else ["Lung fields appear clear.", "No significant signs of alveolar consolidation.", "Pneumonia unlikely based on this image."]


        result = {
            'original_image': unique_filename,
            'annotated_image': heatmap_filename_base,
            'prediction': pred_class,
            'confidence': conf,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': details,
            'disease_type': disease_type,
            'patient_name': request.form.get('patientName', ''),
            'patient_age': request.form.get('patientAge', ''),
            'patient_gender': request.form.get('patientGender', ''),
            'patient_contact': request.form.get('patientContact', '')
        }

        session['last_result'] = result
        print(f"✅ Analysis complete. Result: {pred_class} ({conf}%)")
        return jsonify({'success': True, 'redirect': url_for(f"{disease_type.replace('-', '_')}_results"), 'result': result})

    except Exception as e:
        import traceback
        print(f"❌ Unhandled Error in detect_disease: {e}"); traceback.print_exc()
        return jsonify({'error': 'Internal server error during analysis.'}), 500

# Results routes
@app.route('/lung-cancer-results')
def lung_cancer_results():
    result = session.get('last_result');
    if not result or result.get('disease_type') != 'lung-cancer': print("Redirect: No/Wrong result."); return redirect(url_for('index'))
    print(f"Render LC results: {result.get('prediction')}"); return render_template('lung-cancer-results.html', result=result)

@app.route('/tuberculosis-results')
def tuberculosis_results():
    result = session.get('last_result');
    if not result or result.get('disease_type') != 'tuberculosis': print("Redirect: No/Wrong result."); return redirect(url_for('index'))
    print(f"Render TB results: {result.get('prediction')}"); return render_template('tuberculosis-results.html', result=result)

@app.route('/pneumonia-results')
def pneumonia_results():
    result = session.get('last_result');
    if not result or result.get('disease_type') != 'pneumonia': print("Redirect: No/Wrong result."); return redirect(url_for('index'))
    print(f"Render Pneu results: {result.get('prediction')}"); return render_template('pneumonia-results.html', result=result)

# Download PDF route
@app.route('/download-report/<disease_type>', methods=['GET'])
def download_report(disease_type):
    result = session.get('last_result')
    if not result: print("DL Error: No session data."); return jsonify({'error': 'No result data.'}), 404
    if result.get('disease_type') != disease_type: print(f"DL Error: Type mismatch."); return jsonify({'error': 'Mismatch report type.'}), 400

    pdf_filename = f"{disease_type}_report_{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    try: print(f"Generating PDF: {pdf_path}"); generate_pdf_report(result, pdf_path)
    except Exception as e: print(f"❌ Error generating PDF: {e}"); return jsonify({'error': 'Failed generating PDF.'}), 500

    try:
         print(f"Sending PDF: {pdf_filename}")
         if not os.path.exists(pdf_path): print(f"❌ Send Error: PDF not found {pdf_path}"); return jsonify({'error': 'Generated report not found.'}), 404
         return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path=pdf_filename, as_attachment=True)
    except Exception as send_e: print(f"❌ Error sending PDF: {send_e}"); return jsonify({'error': 'Failed sending report.'}), 500
    finally:
         if os.path.exists(pdf_path):
             try: os.remove(pdf_path); print(f"Cleaned up PDF: {pdf_path}")
             except OSError as e_rem: print(f"Error removing PDF {pdf_path}: {e_rem}")

# Uploads route (for direct access - ensure security)
@app.route('/uploads/<path:filename>')
def uploads(filename):
    if '..' in filename or filename.startswith('/'): return jsonify({'error': 'Invalid path'}), 400
    # Consider adding more checks here if needed
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # app.run(host="0.0.0.0", port=port, debug=True) # Debug locally
    app.run(host="0.0.0.0", port=port) # For Render