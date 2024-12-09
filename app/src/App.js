import React, { useState } from 'react';
import './App.css';

function App() {
  const [mode, setMode] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');

  const handleToggle = () => {
    setMode(mode === 'text' ? 'video' : 'text');
  };

  const handleSubmit = async () => {
    if (textInput && question) {
      const response = await fetch('http://127.0.0.1:8080/answer_query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: `${textInput}\n${question}`,
        }),
      });

      const data = await response.json();
      setResponse(data.response);
    }
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
        {mode === 'video' ? (
          <div className="input-section">
            <div className="input-group">
              <label>Video Input</label>
              <input 
                type="text" 
                placeholder="Enter video URL or details" 
                className="custom-input"
              />
            </div>
            <div className="input-group">
              <label>Question</label>
              <input 
                type="text" 
                placeholder="Enter your question" 
                className="custom-input"
              />
            </div>
            <button className="submit-btn">Submit</button>
          </div>
        ) : (
          <div className="input-section">
            <div className="input-group">
              <label>Text Input</label>
              <input 
                type="text" 
                placeholder="Enter your text" 
                className="custom-input"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
              />
            </div>
            <div className="input-group">
              <label>Question</label>
              <input 
                type="text" 
                placeholder="Enter your question" 
                className="custom-input"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
            </div>
            <button className="submit-btn" onClick={handleSubmit}>Submit</button>
            {response && (
              <div className="response">
                <h3>Response:</h3>
                <p>{response}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
