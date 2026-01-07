import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [mode, setMode] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setResponse(null);
    setError('');

    try {
      const data =
        mode === 'text'
          ? { transcript: textInput, question }
          : { video_url: videoUrl, question };

      const endpoint =
        mode === 'text'
          ? 'http://127.0.0.1:8080/process_text'
          : 'http://127.0.0.1:8080/process_video';

      const res = await axios.post(endpoint, data);

      if (res.data && res.data.answer) {
        setResponse(res.data);
      } else {
        setError('No valid response received.');
      }
    } catch (error) {
      setError(error.response?.data?.error || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = () => {
    setMode(mode === 'text' ? 'video' : 'text');
  };

  return (
    <div className="app-container">
      <div className="content">
        <h1 className="app-title">Transcriptify</h1>

        <div className="toggle-container">
          <div className="toggle-wrapper" onClick={handleToggle}>
            <div className={`toggle-slider ${mode === 'video' ? 'video-mode' : ''}`}></div>
            <div className="toggle-text left">Text</div>
            <div className="toggle-text right">Video</div>
          </div>
        </div>

        <div className="input-section">
          {mode === 'video' ? (
            <div className="input-group">
              <label>Video URL</label>
              <input
                type="text"
                className="custom-input"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
              />
            </div>
          ) : (
            <div className="input-group">
              <label>Text Input</label>
              <input
                type="text"
                className="custom-input"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
              />
            </div>
          )}

          <div className="input-group">
            <label>Question</label>
            <input
              type="text"
              className="custom-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>

          <button className="submit-btn" onClick={handleSubmit} disabled={loading}>
            {loading ? 'Processing...' : 'Submit'}
          </button>

          {error && <div className="error-message">{error}</div>}

          {response && (
            <div className="response">
              <h3>Response</h3>
              <p><strong>Answer:</strong> {response.answer}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
