import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const API_URL = "http://localhost:5000";

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setError(null);
      
      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      setVideoPreviewUrl(previewUrl);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setVideoPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
      setError("Please select a video file");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("video", selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "An error occurred");
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (prediction) => {
    return prediction === "FAKE" ? "#ff4444" : "#44ff44";
  };

  const getConfidenceBar = (confidence) => {
    return (
      <div style={styles.confidenceContainer}>
        <div
          style={{
            ...styles.confidenceBar,
            width: `${Math.max(confidence * 100, 5)}%`,
            backgroundColor: confidence > 0.5 ? "#ff4444" : "#44ff44",
          }}
        />
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>🎥 TruthLens</h1>
        <p>Upload a video to check if it's real or fake</p>
      </div>

      <div style={styles.card}>
        <form onSubmit={handleSubmit}>
          <div style={styles.uploadArea} onClick={handleUploadClick} onDragOver={(e) => e.preventDefault()} onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
              const file = e.dataTransfer.files[0];
              setSelectedFile(file);
              setError(null);
              const previewUrl = URL.createObjectURL(file);
              setVideoPreviewUrl(previewUrl);
            }
          }}>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              disabled={loading}
              style={styles.fileInput}
            />
            <label style={styles.fileLabel}>
              {selectedFile ? `✓ ${selectedFile.name}` : "📁 Click or drag video here"}
            </label>
          </div>

          {videoPreviewUrl && (
            <div style={styles.previewSection}>
              <h3 style={styles.previewTitle}>📹 Video Preview</h3>
              <video 
                src={videoPreviewUrl} 
                style={styles.videoPreview} 
                controls 
                controlsList="nodownload"
              />
              <div style={styles.videoInfo}>
                <p><strong>File:</strong> {selectedFile.name}</p>
                <p><strong>Size:</strong> {(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
          )}

          <div style={styles.buttonGroup}>
            <button
              type="submit"
              disabled={loading || !selectedFile}
              style={{
                ...styles.button,
                ...styles.analyzeButton,
                opacity: loading || !selectedFile ? 0.5 : 1,
              }}
            >
              {loading ? "⏳ Analyzing..." : "▶️ Analyze Video"}
            </button>
            {selectedFile && (
              <button
                type="button"
                onClick={handleClearFile}
                style={{
                  ...styles.button,
                  ...styles.clearButton,
                }}
              >
                🗑️ Clear
              </button>
            )}
          </div>
        </form>
      </div>

      {error && (
        <div style={styles.errorMessage}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={styles.card}>
          <h2 style={styles.resultTitle}>Analysis Result</h2>

          <div style={styles.resultSection}>
            <div style={styles.predictionBox}>
              <div style={styles.predictionLabel}>Prediction</div>
              <div
                style={{
                  ...styles.predictionText,
                  color: getPredictionColor(result.prediction),
                }}
              >
                {result.prediction}
              </div>
            </div>

            <div style={styles.confidenceSection}>
              <div style={styles.confidenceLabel}>
                Confidence: {result.confidence_percent}%
              </div>
              {getConfidenceBar(result.confidence_percent / 100)}
            </div>
          </div>

          <div style={styles.statsSection}>
            <p>
              <strong>Frames Analyzed:</strong> {result.num_frames_analyzed}
            </p>
          </div>

          <details style={styles.details}>
            <summary>📸 Frame-by-frame analysis (click to expand)</summary>
            <div style={styles.frameContainer}>
              {result.frame_results && result.frame_results.map((frame, idx) => (
                <div key={idx} style={styles.frameCard}>
                  {result.frame_images && result.frame_images[idx] && (
                    <img 
                      src={`http://localhost:5000${result.frame_images[idx]}`}
                      alt={`Frame ${idx + 1}`}
                      style={styles.frameImage}
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                  )}
                  <div style={styles.frameInfo}>
                    <strong>Frame {idx + 1}</strong>
                    <div style={styles.frameConfidenceBar}>
                      <div 
                        style={{
                          ...styles.frameConfidenceBarFill,
                          width: `${frame.confidence}%`,
                          backgroundColor: frame.label === "FAKE" ? "#ff4444" : "#44ff44",
                        }}
                      />
                    </div>
                    <div style={styles.frameConfidenceText}>
                      {frame.confidence}% - {frame.label}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <button 
              type="button"
              onClick={async () => {
                try {
                  await fetch("http://localhost:5000/clear-frames", { method: "POST" });
                  alert("Temporary frame files cleared");
                } catch (err) {
                  alert("Could not clear frames: " + err.message);
                }
              }}
              style={styles.clearFramesButton}
            >
              🗑️ Clear Temporary Frames
            </button>
          </details>
        </div>
      )}

      <div style={styles.footer}>
        <p>Make sure the backend server is running on http://localhost:5000</p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "800px",
    margin: "0 auto",
    padding: "20px",
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#f5f5f5",
    minHeight: "100vh",
  },
  header: {
    textAlign: "center",
    marginBottom: "30px",
    color: "#333",
  },
  card: {
    backgroundColor: "white",
    borderRadius: "8px",
    padding: "20px",
    marginBottom: "20px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
  },
  uploadArea: {
    border: "2px dashed #ccc",
    borderRadius: "8px",
    padding: "20px",
    textAlign: "center",
    marginBottom: "15px",
    backgroundColor: "#fafafa",
  },
  fileInput: {
    display: "none",
  },
  fileLabel: {
    cursor: "pointer",
    color: "#666",
    fontSize: "16px",
  },
  button: {
    width: "100%",
    padding: "12px",
    backgroundColor: "#4CAF50",
    color: "white",
    border: "none",
    borderRadius: "4px",
    fontSize: "16px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  errorMessage: {
    backgroundColor: "#f8d7da",
    color: "#721c24",
    padding: "12px",
    borderRadius: "4px",
    marginBottom: "20px",
  },
  resultTitle: {
    marginTop: "0",
    color: "#333",
  },
  resultSection: {
    display: "flex",
    gap: "20px",
    marginBottom: "20px",
    flexWrap: "wrap",
  },
  predictionBox: {
    flex: 1,
    minWidth: "150px",
    border: "2px solid #ddd",
    borderRadius: "8px",
    padding: "15px",
    textAlign: "center",
  },
  predictionLabel: {
    fontSize: "14px",
    color: "#666",
    marginBottom: "10px",
  },
  predictionText: {
    fontSize: "32px",
    fontWeight: "bold",
  },
  confidenceSection: {
    flex: 1,
    minWidth: "200px",
  },
  confidenceLabel: {
    fontSize: "14px",
    color: "#666",
    marginBottom: "8px",
  },
  confidenceContainer: {
    width: "100%",
    height: "30px",
    backgroundColor: "#eee",
    borderRadius: "4px",
    overflow: "hidden",
  },
  confidenceBar: {
    height: "100%",
    transition: "width 0.3s ease",
  },
  statsSection: {
    backgroundColor: "#f9f9f9",
    padding: "15px",
    borderRadius: "4px",
    marginBottom: "15px",
    color: "#666",
  },
  details: {
    cursor: "pointer",
    color: "#666",
  },
  frameContainer: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
    gap: "15px",
    marginTop: "15px",
    padding: "15px 0",
  },
  frameCard: {
    backgroundColor: "#f9f9f9",
    borderRadius: "8px",
    padding: "10px",
    border: "1px solid #e0e0e0",
    overflow: "hidden",
  },
  frameImage: {
    width: "100%",
    height: "120px",
    objectFit: "cover",
    borderRadius: "4px",
    marginBottom: "8px",
    backgroundColor: "#000",
  },
  frameInfo: {
    fontSize: "12px",
  },
  frameConfidenceBar: {
    width: "100%",
    height: "8px",
    backgroundColor: "#eee",
    borderRadius: "4px",
    overflow: "hidden",
    margin: "6px 0",
  },
  frameConfidenceBarFill: {
    height: "100%",
    transition: "width 0.3s ease",
  },
  frameConfidenceText: {
    fontSize: "11px",
    color: "#555",
    marginTop: "4px",
  },
  clearFramesButton: {
    marginTop: "15px",
    padding: "10px 15px",
    backgroundColor: "#f44336",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "14px",
  },
  frameConfidences: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
    gap: "10px",
    marginTop: "10px",
    padding: "10px 0",
  },
  frameConfidence: {
    display: "flex",
    justifyContent: "space-between",
    padding: "8px",
    backgroundColor: "#f5f5f5",
    borderRadius: "4px",
    fontSize: "14px",
  },
  footer: {
    textAlign: "center",
    color: "#999",
    marginTop: "30px",
    fontSize: "14px",
  },
  previewSection: {
    backgroundColor: "#f9f9f9",
    borderRadius: "8px",
    padding: "15px",
    marginBottom: "15px",
    border: "1px solid #e0e0e0",
  },
  previewTitle: {
    marginTop: "0",
    marginBottom: "10px",
    color: "#333",
    fontSize: "18px",
  },
  videoPreview: {
    width: "100%",
    maxHeight: "400px",
    borderRadius: "6px",
    marginBottom: "12px",
    backgroundColor: "#000",
  },
  videoInfo: {
    fontSize: "14px",
    color: "#666",
    lineHeight: "1.6",
  },
  videoInfo_p: {
    margin: "5px 0",
  },
  buttonGroup: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
  },
  analyzeButton: {
    flex: 1,
    minWidth: "200px",
    backgroundColor: "#4CAF50",
  },
  clearButton: {
    minWidth: "100px",
    backgroundColor: "#f44336",
  },
};

export default App;
