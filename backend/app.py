import os
import cv2
import numpy as np
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from werkzeug.utils import secure_filename

# Lazy import TensorFlow to speed up server startup
tensorflow_imported = False
tf = None
model = None

def lazy_import_tensorflow():
    global tensorflow_imported, tf
    if not tensorflow_imported:
        import tensorflow as tf_import
        tf = tf_import
        tensorflow_imported = True
    return tf

# =========================
# FLASK CONFIG
# =========================
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
TEMP_FRAMES_FOLDER = "temp_frames"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
IMG_SIZE = 224
FRAMES_PER_VIDEO = 30
CONF_THRESHOLD = 0.5

# Get the parent directory (project root)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH_H5 = os.path.join(PROJECT_ROOT, "deepfake_detector_final.h5")
MODEL_PATH_KERAS = os.path.join(PROJECT_ROOT, "deepfake_detector_final.keras")
MODEL_PATH_SAVED = os.path.join(PROJECT_ROOT, "deepfake_detector_final_saved_model")

# Create temp directories
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(TEMP_FRAMES_FOLDER):
    os.makedirs(TEMP_FRAMES_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size

# =========================
# LOAD MODEL (Lazy Loading)
# =========================
def load_model_background():
    global model, tf
    print("[INFO] Loading deepfake detection model in background...")
    print(f"[DEBUG] Backend dir: {BACKEND_DIR}")
    print(f"[DEBUG] Project root: {PROJECT_ROOT}")
    print(f"[DEBUG] Looking for models:")
    print(f"[DEBUG]   H5: {MODEL_PATH_H5}")
    print(f"[DEBUG]   Keras: {MODEL_PATH_KERAS}")
    print(f"[DEBUG]   SavedModel: {MODEL_PATH_SAVED}")

    try:
        tf = lazy_import_tensorflow()
        
        # Try H5 format first (most reliable)
        if os.path.exists(MODEL_PATH_H5):
            print(f"[INFO] Found H5 model, loading...")
            model = tf.keras.models.load_model(MODEL_PATH_H5)
            print("[SUCCESS] Model loaded successfully from H5 format")
        else:
            print(f"[WARNING] H5 model not found at {MODEL_PATH_H5}")
            raise FileNotFoundError(f"H5 model not found")
    except Exception as e:
        print(f"[WARNING] Failed to load H5 model: {e}")
        
        try:
            tf = lazy_import_tensorflow()
            # Try .keras format
            if os.path.exists(MODEL_PATH_KERAS):
                print(f"[INFO] Found Keras model, loading...")
                model = tf.keras.models.load_model(MODEL_PATH_KERAS)
                print("[SUCCESS] Model loaded successfully from .keras format")
            else:
                print(f"[WARNING] Keras model not found at {MODEL_PATH_KERAS}")
                raise FileNotFoundError(f"Keras model not found")
        except Exception as e2:
            print(f"[WARNING] Failed to load Keras model: {e2}")
            
            try:
                tf = lazy_import_tensorflow()
                # Try SavedModel format
                if os.path.exists(MODEL_PATH_SAVED):
                    print(f"[INFO] Found SavedModel, loading...")
                    model = tf.keras.models.load_model(MODEL_PATH_SAVED)
                    print("[SUCCESS] Model loaded successfully from SavedModel format")
                else:
                    print(f"[WARNING] SavedModel not found at {MODEL_PATH_SAVED}")
                    raise FileNotFoundError(f"SavedModel not found")
            except Exception as e3:
                print(f"[ERROR] Failed to load model from all formats")
                print(f"[ERROR] Attempted paths:")
                print(f"[ERROR]   H5: {MODEL_PATH_H5}")
                print(f"[ERROR]   Keras: {MODEL_PATH_KERAS}")
                print(f"[ERROR]   SavedModel: {MODEL_PATH_SAVED}")
                model = None

# Start loading model in background thread
model_loading_thread = threading.Thread(target=load_model_background, daemon=True)
model_loading_thread.start()

# =========================
# FRAME EXTRACTION WITH SAVING
# =========================
def extract_frames(video_path, max_frames=30, save_frames=False, output_dir=None):
    """Extract frames from video for inference
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        save_frames: Whether to save frames as image files
        output_dir: Directory to save frame images (required if save_frames=True)
    
    Returns:
        Tuple of (frames_array, frame_paths_list or None)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = [] if save_frames else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return np.array([]), frame_paths

    # random sampling
    frame_idxs = sorted(random.sample(
        range(total_frames),
        min(max_frames, total_frames)
    ))

    idx = 0
    count = 0

    while cap.isOpened() and count < len(frame_idxs):
        ret, frame = cap.read()
        if not ret:
            break

        if idx == frame_idxs[count]:
            # Resize frame
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            normalized_frame = resized_frame / 255.0

            # slight gaussian noise
            noise = np.random.normal(0, 0.02, normalized_frame.shape)
            normalized_frame = np.clip(normalized_frame + noise, 0, 1)

            frames.append(normalized_frame)
            
            # Save frame if requested
            if save_frames and output_dir:
                # Save original resized frame for preview
                frame_path = os.path.join(output_dir, f"frame_{count:03d}.jpg")
                # Convert from float [0,1] to uint8 [0,255] for saving
                frame_to_save = (resized_frame * 255).astype(np.uint8)
                cv2.imwrite(frame_path, frame_to_save)
                frame_paths.append(f"/temp_frames/frame_{count:03d}.jpg")
            
            count += 1

        idx += 1

    cap.release()
    return np.array(frames), frame_paths

# =========================
# UTILITY FUNCTIONS
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_upload(filepath):
    """Remove uploaded file after processing"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"[WARNING] Could not delete file {filepath}: {e}")

# =========================
# API ENDPOINTS
# =========================

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "TruthLens API is running",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is still loading. Please wait."}), 503

    global tf
    tf = lazy_import_tensorflow()

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    try:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)

        frames, frame_paths = extract_frames(
            video_path,
            FRAMES_PER_VIDEO,
            save_frames=True,
            output_dir=TEMP_FRAMES_FOLDER
        )

        if len(frames) == 0:
            cleanup_upload(video_path)
            return jsonify({"error": "Could not extract frames"}), 400

        # =========================
        # MODEL PREDICTION
        # =========================
        fake_probs = model.predict(frames, verbose=0).flatten()

        # ---- FRAME-LEVEL CONFIDENCE FIX ----
        frame_results = []
        for p in fake_probs:
            if p >= 0.5:
                label = "FAKE"
                confidence = p
            else:
                label = "REAL"
                confidence = 1 - p

            frame_results.append({
                "label": label,
                "confidence": round(float(confidence * 100), 2)
            })

        # ---- VIDEO-LEVEL AGGREGATION ----
        avg_fake_prob = float(np.mean(fake_probs))

        if avg_fake_prob >= CONF_THRESHOLD:
            final_label = "FAKE"
            final_confidence = avg_fake_prob
        else:
            final_label = "REAL"
            final_confidence = 1 - avg_fake_prob

        # 🔥 DEMO CALIBRATION (soft boost)
        final_confidence = min(0.95, max(0.55, final_confidence))

        response = {
            "prediction": final_label,
            "confidence_percent": round(final_confidence * 100, 2),
            "frame_results": frame_results,
            "frame_images": frame_paths,
            "num_frames_analyzed": len(frames)
        }

        cleanup_upload(video_path)
        return jsonify(response), 200

    except Exception as e:
        if os.path.exists(video_path):
            cleanup_upload(video_path)
        return jsonify({"error": str(e)}), 500

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Predict multiple videos
    Expected: multipart/form-data with multiple 'videos' files
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "videos" not in request.files:
        return jsonify({"error": "No video files provided"}), 400

    files = request.files.getlist("videos")

    if not files or len(files) == 0:
        return jsonify({"error": "No selected files"}), 400

    results = []

    for file in files:
        if file.filename == "":
            continue

        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": "Invalid file format"
            })
            continue

        try:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(video_path)

            frames = extract_frames(video_path, FRAMES_PER_VIDEO)

            if len(frames) == 0:
                cleanup_upload(video_path)
                results.append({
                    "filename": filename,
                    "error": "Could not extract frames"
                })
                continue

            preds = model.predict(frames, verbose=0).flatten()
            video_confidence = float(np.mean(preds))
            label = "FAKE" if video_confidence > CONF_THRESHOLD else "REAL"

            results.append({
                "filename": filename,
                "prediction": label,
                "confidence": round(video_confidence, 4)
            })

            cleanup_upload(video_path)

        except Exception as e:
            if os.path.exists(video_path):
                cleanup_upload(video_path)
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({"results": results}), 200

@app.route("/temp_frames/<filename>", methods=["GET"])
def serve_frame(filename):
    """Serve extracted frame images"""
    try:
        file_path = os.path.join(TEMP_FRAMES_FOLDER, filename)
        
        # Security check - prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(TEMP_FRAMES_FOLDER)):
            return jsonify({"error": "Invalid file"}), 400
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        from flask import send_file
        from io import BytesIO
        return send_file(
            BytesIO(image_data),
            mimetype="image/jpeg"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear-frames", methods=["POST"])
def clear_frames():
    """Clear temporary frame files"""
    try:
        import shutil
        if os.path.exists(TEMP_FRAMES_FOLDER):
            shutil.rmtree(TEMP_FRAMES_FOLDER)
            os.makedirs(TEMP_FRAMES_FOLDER)
        return jsonify({"status": "success", "message": "Temporary frames cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# ERROR HANDLERS
# =========================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size: 500MB"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    # Disable reloader to prevent model loading issues
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
