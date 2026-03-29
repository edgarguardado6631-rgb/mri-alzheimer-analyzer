import React, { useState } from 'react';
import { Grid, Column, Tile, TextInput, Button, InlineLoading } from '@carbon/react';
import { Send, ImageSearch } from '@carbon/icons-react';
import API_URL from '../config';
import './Welcome.scss';

const Welcome = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<{ role: 'user' | 'bot'; content: string | React.ReactNode }[]>([
    { role: 'bot', content: "Hello! I am your Alzheimer's MRI Analysis Assistant. You can ask me to evaluate a scan or show you the latest data." },
  ]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleSend = async () => {
    if (!query.trim()) return;
    const userMsg = query;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setQuery('');
    setLoading(true);

    try {
      if (userMsg.toLowerCase().includes('review') || userMsg.toLowerCase().includes('scan')) {
        setMessages(prev => [...prev, { role: 'bot', content: 'To review a scan, please use the upload button (ghost icon).' }]);
      } else {
        try {
          const healthCheck = await fetch(`${API_URL}/`);
          const data = await healthCheck.json();
          setMessages(prev => [...prev, { role: 'bot', content: `Backend says: "${data.message}". I am ready.` }]);
        } catch {
          setMessages(prev => [...prev, { role: 'bot', content: "I couldn't reach the backend. Is it running on port 8000?" }]);
        }
      }
    } catch {
      setMessages(prev => [...prev, { role: 'bot', content: 'Error processing your request.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setMessages(prev => [...prev, { role: 'user', content: `Uploaded: ${file.name}` }]);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Network response was not ok');
      const result = await response.json();
      setMessages(prev => [
        ...prev,
        {
          role: 'bot',
          content: (
            <div>
              <strong>Analysis Result:</strong> {result.prediction}
              <br />
              <small>Confidence: {(result.confidence * 100).toFixed(1)}%</small>
              <p>{result.analysis}</p>
            </div>
          ),
        },
      ]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', content: `Error analyzing file: ${error}` }]);
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSend();
  };

  return (
    <Grid className="welcome-page">
      <Column lg={16} md={8} sm={4}>
        <div className="welcome-header">
          <p className="cds--label">AI-Powered Neuroimaging</p>
          <h1 className="cds--type-productive-heading-05">MRI Analysis Assistant</h1>
          <p className="cds--type-body-long-01 welcome-subtitle">
            Upload a NIfTI scan or ask a question about your patients' MRI data.
          </p>
        </div>
      </Column>

      <Column lg={{ span: 12, offset: 2 }} md={8} sm={4}>
        <Tile className="chat-window">
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.role}`}>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}
            {loading && (
              <div className="chat-message bot">
                <InlineLoading description="Analyzing..." status="active" />
              </div>
            )}
          </div>

          <div className="chat-input-area">
            <TextInput
              id="chat-input"
              labelText=""
              placeholder="Ask me something about patient 002_S_0413..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              size="lg"
            />
            <Button renderIcon={Send} onClick={handleSend} iconDescription="Send" disabled={loading}>
              Send
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              accept=".nii,.nii.gz"
            />
            <Button
              kind="ghost"
              renderIcon={ImageSearch}
              iconDescription="Upload MRI scan"
              hasIconOnly
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            />
          </div>
        </Tile>
      </Column>
    </Grid>
  );
};

export default Welcome;
