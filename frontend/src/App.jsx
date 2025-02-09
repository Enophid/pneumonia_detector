import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    await detectPneumonia(formData);
  };

  async function detectPneumonia(formData) {
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData,
            cache: 'no-store' // Prevent caching
        });
        
        if (!response.ok) {
            throw new Error('Server error or offline');
        }
        
        const data = await response.json();
        setResult(data);
        
    } catch (error) {
        console.error('Connection error:', error);
        alert('Server is offline. Please start the backend server.');
    } finally {
        setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Pneumonia Detection System</h1>
      
      <div className="upload-section">
        <form onSubmit={handleSubmit}>
          <input 
            type="file" 
            accept="image/*"
            onChange={handleFileChange}
          />
          <button type="submit" disabled={!selectedFile || loading}>
            {loading ? 'Processing...' : 'Detect Pneumonia'}
          </button>
        </form>
      </div>

      {selectedFile && (
        <div className="preview-section">
          <h3>Selected Image:</h3>
          <img 
            src={URL.createObjectURL(selectedFile)} 
            alt="Preview" 
            className="image-preview"
          />
        </div>
      )}

      {result && result.success && (
        <div className="result-section">
          <h3>Results:</h3>
          <p>
            {result.has_pneumonia 
              ? `Pneumonia Detected (Confidence: ${(result.confidence * 100).toFixed(2)}%)`
              : 'No Pneumonia Detected'
            }
          </p>
          {result.has_pneumonia && (
            <div>
              <img 
                src={result.image_data} 
                alt="Result with detection"
                className="result-image"
              />
            </div>
          )}
        </div>
      )}

      {result && !result.success && (
        <div className="error-section">
          <p>Error processing image: {result.error}</p>
        </div>
      )}
    </div>
  );
}

export default App
