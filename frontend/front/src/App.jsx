import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./UploadPage";
import PredictionPage from "./PredictionPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/predict" element={<PredictionPage />} />
      </Routes>
    </Router>
  );
}

export default App;
