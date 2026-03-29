import React, { useState } from 'react';
import { Tile, TextInput, Button, Loading } from '@carbon/react';
import { Send, ImageSearch } from '@carbon/icons-react';
import './Welcome.scss'; // We'll create this for specific chat styles

const Welcome = () => {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState<{ role: 'user' | 'bot', content: string | React.ReactNode }[]>([
        { role: 'bot', content: 'Hello! I am your Alzheimer\'s MRI Analysis Assistant. You can ask me to evaluate a scan or show you the latest data.' }
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
            if (userMsg.toLowerCase().includes("review") || userMsg.toLowerCase().includes("scan")) {
                setMessages(prev => [...prev, { role: 'bot', content: "To review a scan, please use the upload button (ghost icon)." }]);
            } else {
                try {
                    const healthCheck = await fetch('http://localhost:8000/');
                    const data = await healthCheck.json();
                    setMessages(prev => [...prev, { role: 'bot', content: `Backend says: "${data.message}". I am ready.` }]);
                } catch (err) {
                    setMessages(prev => [...prev, { role: 'bot', content: "I couldn't reach the backend. Is it running on port 8000?" }]);
                }
            }
        } catch (err) {
            setMessages(prev => [...prev, { role: 'bot', content: "Error processing your request." }]);
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
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const result = await response.json();
            setMessages(prev => [...prev, {
                role: 'bot',
                content: (
                    <div>
                        <strong>Analysis Result:</strong> {result.prediction}<br />
                        <small>Confidence: {(result.confidence * 100).toFixed(1)}%</small>
                        <p>{result.analysis}</p>
                    </div>
                )
            }]);
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
        <div className="welcome-page cds--grid">
            <div className="cds--row">
                <div className="cds--col-lg-16">
                    <h2 style={{ margin: '2rem 0' }}>Welcome to MRI Analysis AI</h2>
                </div>
            </div>
            <div className="cds--row chat-container">
                <div className="cds--col-lg-12 cds--offset-lg-2">
                    <Tile className="chat-window">
                        <div className="chat-history">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`chat-message ${msg.role}`}>
                                    <div className="message-content">
                                        {msg.content}
                                    </div>
                                </div>
                            ))}
                            {loading && <Loading small withOverlay={false} />}
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
                            <Button renderIcon={Send} onClick={handleSend} iconDescription="Send">
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
                                iconDescription="Upload MRI"
                                onClick={() => fileInputRef.current?.click()}
                            />
                        </div>
                    </Tile>
                </div>
            </div>
        </div>
    );
};

export default Welcome;
