import React, { useState } from "react";
import Papa from "papaparse";
import { useNavigate } from "react-router-dom";

function UploadPage() {
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [taskType, setTaskType] = useState("regression");
  const [modelReady, setModelReady] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function (results) {
        setColumns(results.meta.fields);
      },
    });
  };

  const handleSubmit = async () => {
    if (!selectedFile || !selectedColumn || !taskType) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("target_column", selectedColumn);
    formData.append("task_type", taskType);

    try {
      const response = await fetch("http://127.0.0.1:8000/send_training", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Training success:", result);
      setModelReady(true);
    } catch (err) {
      console.error("Error uploading:", err);
    }
  };

  return (
    <div>
      <h2>Upload CSV & Train</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      {columns.length > 0 && (
        <>
          <select onChange={(e) => setSelectedColumn(e.target.value)}>
            <option value="">Select Target Column</option>
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <select onChange={(e) => setTaskType(e.target.value)} value={taskType}>
            <option value="regression">Regression</option>
            <option value="classification">Classification</option>
          </select>
          <button onClick={handleSubmit}>Train Model</button>
        </>
      )}

      {modelReady && (
        <button onClick={() => navigate("/predict")}>Go to Prediction</button>
      )}
    </div>
  );
}

export default UploadPage;
