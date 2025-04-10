import React, { useState } from "react";
import Papa from "papaparse";
import { useNavigate } from "react-router-dom";

function PredictionPage() {
  const [predictionFile, setPredictionFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPredictionFile(file);
  };

  const handlePredict = async () => {
    if (!predictionFile) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError("");
    
    const formData = new FormData();
    formData.append("file", predictionFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const result = await response.json();
      setPredictions(result.predictions);
      setLoading(false);
    } catch (err) {
      console.error("Error making prediction:", err);
      setError("Failed to get predictions: " + err.message);
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Make Predictions</h2>
      <div>
        <button onClick={() => navigate("/")}>Back to Upload</button>
      </div>
      
      <div>
        <h3>Upload file for prediction</h3>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Processing..." : "Make Predictions"}
        </button>
      </div>

      {error && <div style={{ color: "red" }}>{error}</div>}

      {predictions && (
        <div>
          <h3>Prediction Results</h3>
          <pre>{JSON.stringify(predictions, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default PredictionPage;