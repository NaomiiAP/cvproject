# TruthLens Backend API

Flask-based API for deepfake video detection using TensorFlow.

## Setup

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Ensure the model file exists:**
   - Place `deepfake_detector_final.keras` in the parent directory (`../deepfake_detector_final.keras`)
   - Or update `MODEL_PATH` in `app.py` to point to your model

3. **Run the server:**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /
```
Returns the status of the API and whether the model is loaded.

### Single Video Prediction
```
POST /predict
Content-Type: multipart/form-data
```
**Parameters:**
- `video` (file): Video file (mp4, avi, mov, mkv)

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 0.8532,
  "frame_confidences": [0.85, 0.86, 0.84, ...],
  "num_frames_analyzed": 30
}
```

### Batch Prediction
```
POST /predict-batch
Content-Type: multipart/form-data
```
**Parameters:**
- `videos` (files): Multiple video files

**Response:**
```json
{
  "results": [
    {
      "filename": "video1.mp4",
      "prediction": "FAKE",
      "confidence": 0.92
    },
    ...
  ]
}
```

## Configuration

Edit `app.py` to modify:
- `IMG_SIZE`: Image size for model input (default: 224)
- `FRAMES_PER_VIDEO`: Number of frames to extract (default: 30)
- `CONF_THRESHOLD`: Confidence threshold (default: 0.5)
- `MAX_CONTENT_LENGTH`: Maximum upload size (default: 500MB)

## Notes

- The model predicts on a per-frame basis and averages the confidence across frames
- Videos are temporarily stored in the `uploads/` folder and deleted after processing
- CORS is enabled to allow frontend requests
