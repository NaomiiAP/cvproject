# TruthLens - Deepfake Video Detection

TruthLens is a deep learning–based system designed to detect manipulated images and videos (deepfakes) by analysing spatial and temporal inconsistencies in facial regions. The system uses a ResNet-based convolutional architecture and is deployed through a lightweight Streamlit interface for real-time analysis.

## 🎯 Project Overview

The rapid advancement of deepfake generation techniques has made it increasingly difficult to distinguish between authentic and manipulated visual media. TruthLens addresses this challenge by providing an accessible and reliable deepfake detection framework that focuses on image and video forgery detection using deep neural networks.
The proposed system is intended for academic research, media forensics, and educational purposes.
**TruthLens** is a comprehensive deepfake detection system with:
- **Backend API** (Flask) - Processes video uploads and runs deepfake detection using TensorFlow
- **Frontend Interface** (React) - User-friendly web interface for uploading and analyzing videos
- **ResNet-based Model** - Deep learning model built on ResNet architecture trained to classify video frames as real or fake

## 📋 Features

- ✅ **Video Upload** - Support for multiple video formats (MP4, AVI, MOV, MKV)
- ✅ **Frame Analysis** - Analyzes 30 frames per video for comprehensive detection
- ✅ **Confidence Scoring** - Provides confidence levels for predictions
- ✅ **Batch Processing** - Analyze multiple videos simultaneously
- ✅ **Real-time Preview** - Video preview before analysis
- ✅ **Lazy Model Loading** - Optimized server startup with background model loading
- ✅ **CORS Support** - Cross-origin requests enabled for frontend-backend communication

## 🏗️ Project Structure

```
cvproj/
├── backend/                          # Flask API server
│   ├── app.py                       # Main Flask application with prediction endpoints
│   ├── requirements.txt             # Python dependencies
│   ├── README.md                    # Backend documentation
│   ├── uploads/                     # Temporary video upload storage
│   └── temp_frames/                 # Temporary frame extraction folder
├── frontend/                         # React web application
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   ├── App.css                 # Application styling
│   │   └── index.js                # React entry point
│   ├── public/
│   │   └── index.html              # HTML template
│   ├── package.json                # Node dependencies
│   └── README.md                    # Frontend documentation
├── deepfake_detector_final.h5      # model (H5 format)
├── deepfake_detector_final.keras   # model (Keras format)
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 14+** (for frontend)
- **Git** (optional, for cloning)

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the trained model files exist:**
   - The trained model files (`deepfake_detector_final.h5` or `deepfake_detector_final.keras`) should be in the project root directory
   - If not present, place them there or update `MODEL_PATH` in `app.py`

4. **Run the Flask server:**
   ```bash
   python app.py
   ```
   - The API will be available at `http://localhost:5000`
   - Model loads in the background during startup

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   - The application will open at `http://localhost:3000`

## 📡 API Endpoints

### 1. Health Check
```
GET /
```
Returns API status and model loading status.

**Response Example:**
```json
{
  "status": "OK",
  "model_loaded": true
}
```

### 2. Single Video Prediction
```
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `video` (file): Video file (mp4, avi, mov, mkv)

**Response Example:**
```json
{
  "prediction": "REAL",
  "confidence": 0.8532,
  "frame_confidences": [0.85, 0.86, 0.84, 0.85, ...],
  "num_frames_analyzed": 30
}
```

### 3. Batch Video Prediction
```
POST /predict-batch
Content-Type: multipart/form-data
```

**Parameters:**
- `videos` (files): Multiple video files

**Response Example:**
```json
{
  "results": [
    {
      "filename": "video1.mp4",
      "prediction": "FAKE",
      "confidence": 0.92
    },
    {
      "filename": "video2.mp4",
      "prediction": "REAL",
      "confidence": 0.78
    }
  ]
}
```

## ⚙️ Configuration

### Backend Configuration (app.py)

Edit the following constants in `backend/app.py` to customize behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size for the model |
| `FRAMES_PER_VIDEO` | 30 | Number of frames to extract from each video |
| `CONF_THRESHOLD` | 0.5 | Confidence threshold for predictions |
| `MAX_CONTENT_LENGTH` | 500MB | Maximum upload file size |
| `ALLOWED_EXTENSIONS` | {mp4, avi, mov, mkv} | Supported video formats |

### Frontend Configuration (App.js)

- `API_URL` - Backend API endpoint (default: `http://localhost:5000`)

## 🔧 Technology Stack

### Backend
- **Flask** - Web framework for Python
- **TensorFlow/Keras** - Deep learning framework
- **ResNet** - Residual Neural Network architecture for feature extraction
- **OpenCV** - Video processing and frame extraction
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning utilities
- **Flask-CORS** - Cross-Origin Resource Sharing

### Frontend
- **React** - JavaScript UI library
- **HTML5/CSS3** - Markup and styling
- **Fetch API** - HTTP requests to backend

## 📊 How It Works

1. **Video Upload** - User selects a video through the web interface
2. **Frame Extraction** - Backend extracts frames from the video using OpenCV
3. **Model Inference** - Each frame is passed through the ResNet-based detection model
4. **Confidence Calculation** - ResNet model outputs confidence scores for each frame
5. **Aggregation** - Confidence scores are averaged across all frames
6. **Result Display** - Prediction (REAL/FAKE) and confidence displayed to user

## 📝 Notes

- Videos are processed frame-by-frame for comprehensive analysis
- Temporary files are cleaned up after processing
- The model uses lazy loading to optimize initial server startup
- CORS is enabled to allow cross-domain frontend requests
- Maximum upload size is configurable (default: 500MB)



