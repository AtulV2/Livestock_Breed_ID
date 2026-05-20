import { useState } from 'react';


function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0) return alert("Please select at least one image.");

    setLoading(true);
    setPredictions([]);

    const formData = new FormData();
    // Append all selected files to the form data
    selectedFiles.forEach(file => {
      formData.append("files", file);
    });

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();
      if (data.error) {
        setPredictions([{ filename: 'Server Error', error: data.error }]);
      } else {
        setPredictions(data.results || []);
      }
    } catch (error) {
      console.error("Error predicting:", error);
      alert("Failed to process images. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '700px', margin: '50px auto', fontFamily: 'sans-serif' }}>
      <h1 >Livestock Breed ID System</h1>
      <p>Upload one or multiple cattle images to identify their breed.</p>
      <h1></h1>

      <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'block', marginBottom: '15px' }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{ padding: '10px 20px', cursor: 'pointer' }}
        >
          {loading ? "Analyzing..." : "Identify Breeds"}
        </button>
      </form>

                {predictions.length > 0 && (
                  <div>
                    <h2>Results:</h2>
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                      {predictions.map((pred, index) => (
                        <li
                          key={index}
                          style={{
                            border: '1px solid #ccc',
                            padding: '15px',
                            marginBottom: '10px',
                            borderRadius: '8px',
                            backgroundColor: pred.confidence ? '#fff' : '#fff9f0' // Light orange tint if no cow
                          }}
                        >
                          <strong>File:</strong> {pred.filename} <br />
                          
                          {pred.error ? (
                            <span style={{ color: 'red' }}>Error: {pred.error}</span>
                          ) : (
                            <>
                              <strong>Result:</strong> 
                              <span style={{ 
                                color: pred.confidence ? 'green' : '#d97706', 
                                fontWeight: 'bold',
                                marginLeft: '5px' 
                              }}>
                                {pred.breed}
                              </span> 
                              <br />
                              
                              {/* ONLY show confidence if it is NOT null */}
                              {pred.confidence && (
                                <>
                                  <strong>Confidence:</strong> {pred.confidence}
                                </>
                              )}
                            </>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
    </div>
  );
}

export default App;