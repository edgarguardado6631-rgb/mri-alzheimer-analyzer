import React, { useState, useRef, useEffect } from 'react';
import { Button, TextArea } from '@carbon/react';
import {
  Send,
  Attachment,
  WatsonMachineLearning,
  UserAvatar,
  Upload,
  Analytics,
  Information,
  ChatBot,
} from '@carbon/icons-react';
import API_URL from '../config';
import './Welcome.scss';

// ── Types ──────────────────────────────────────────────────────────────────
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string | React.ReactNode;
  timestamp: Date;
}

// Structured for LLM integration: swap handleAssistantReply with an LLM API call
// that receives conversationHistory (array of {role, content}) and returns a string.
type ConversationEntry = { role: 'user' | 'assistant'; content: string };

// ── Suggested prompts ──────────────────────────────────────────────────────
const SUGGESTED_PROMPTS = [
  {
    icon: Upload,
    label: 'Analyze an MRI scan',
    description: 'Upload a .nii or .nii.gz file for CNN classification',
  },
  {
    icon: Analytics,
    label: 'Show patient statistics',
    description: 'Cohort size, scans processed, and model accuracy',
  },
  {
    icon: ChatBot,
    label: 'How does the model work?',
    description: "Explain the Alzheimer's CNN detection pipeline",
  },
  {
    icon: Information,
    label: 'What can you help me with?',
    description: 'Overview of available features and capabilities',
  },
];

// ── Typing indicator ───────────────────────────────────────────────────────
const TypingIndicator = () => (
  <div className="typing-indicator">
    <span className="typing-dot" />
    <span className="typing-dot" />
    <span className="typing-dot" />
  </div>
);

// ── Scan result card ───────────────────────────────────────────────────────
const ScanResultCard = ({ prediction, confidence, analysis }: { prediction: string; confidence: number; analysis: string }) => (
  <div className="scan-result-card">
    <p className="scan-result-label">CNN Analysis Complete</p>
    <p className="scan-result-prediction">{prediction}</p>
    <div className="scan-result-confidence">
      <div
        className="confidence-bar"
        style={{ width: `${(confidence * 100).toFixed(0)}%` }}
      />
      <span>{(confidence * 100).toFixed(1)}% confidence</span>
    </div>
    <p className="scan-result-analysis">{analysis}</p>
    <p className="scan-result-note">
      LLM-powered in-depth analysis coming soon — the assistant will explain findings in natural language.
    </p>
  </div>
);

// ── Main component ─────────────────────────────────────────────────────────
const Welcome = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<ConversationEntry[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const addMessage = (role: 'user' | 'assistant', content: string | React.ReactNode) => {
    setMessages(prev => [
      ...prev,
      { id: `${Date.now()}-${Math.random()}`, role, content, timestamp: new Date() },
    ]);
  };

  // ── Reply logic ── replace this function body with an LLM API call in the future
  const handleAssistantReply = async (userText: string, history: ConversationEntry[]) => {
    setIsTyping(true);
    try {
      if (userText.toLowerCase().includes('scan') || userText.toLowerCase().includes('upload') || userText.toLowerCase().includes('analyze')) {
        addMessage('assistant', 'To analyze a scan, click the attachment button below and upload a .nii or .nii.gz MRI file. The CNN model will classify it and provide a confidence score.');
      } else if (userText.toLowerCase().includes('statistic') || userText.toLowerCase().includes('patient') || userText.toLowerCase().includes('accuracy')) {
        const res = await fetch(`${API_URL}/data/stats`);
        const data = await res.json();
        addMessage('assistant', (
          <div>
            <p>Here is the current cohort summary:</p>
            <ul>
              <li><strong>Total patients:</strong> {data.total_patients}</li>
              <li><strong>Scans processed:</strong> {data.scans_processed}</li>
              <li><strong>Model accuracy:</strong> {(data.model_accuracy * 100).toFixed(1)}%</li>
            </ul>
          </div>
        ));
      } else if (userText.toLowerCase().includes('how') || userText.toLowerCase().includes('model') || userText.toLowerCase().includes('pipeline')) {
        addMessage('assistant', 'The system uses a Convolutional Neural Network (CNN) trained on the ADNI dataset. When you upload a NIfTI MRI scan, the model extracts the mid-axial slice, resizes it to 128×128, normalizes pixel values, and runs inference to classify the scan as either "Alzheimer\'s Detected" or "Normal Cognition". Future versions will pair this with an LLM for detailed narrative explanations.');
      } else if (userText.toLowerCase().includes('help') || userText.toLowerCase().includes('what can')) {
        addMessage('assistant', 'I can help you with:\n• Analyzing MRI scans (.nii / .nii.gz) using our CNN model\n• Viewing patient cohort statistics\n• Navigating to the MRI viewer (Data Visualization tab)\n• Explaining the model and detection pipeline\n\nIn a future update, I will be connected to an LLM for richer clinical-language responses.');
      } else {
        const healthRes = await fetch(`${API_URL}/`);
        const data = await healthRes.json();
        addMessage('assistant', `I'm connected to the backend (${data.message}). I can analyze MRI scans, show patient statistics, or explain how the model works. What would you like to do?`);
      }
    } catch {
      addMessage('assistant', "I couldn't reach the backend API. Please make sure the FastAPI server is running on port 8000.");
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = async (text?: string) => {
    const userText = (text ?? input).trim();
    if (!userText || isTyping) return;

    setInput('');
    addMessage('user', userText);

    const updatedHistory: ConversationEntry[] = [
      ...conversationHistory,
      { role: 'user', content: userText },
    ];
    setConversationHistory(updatedHistory);

    await handleAssistantReply(userText, updatedHistory);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    addMessage('user', `Uploading scan: ${file.name}`);
    setIsTyping(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Prediction request failed');
      const result = await response.json();
      addMessage('assistant', (
        <ScanResultCard
          prediction={result.prediction}
          confidence={result.confidence}
          analysis={result.analysis}
        />
      ));
    } catch (error) {
      addMessage('assistant', `Error analyzing the scan: ${error}. Please ensure the backend is running and the model is loaded.`);
    } finally {
      setIsTyping(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="chat-page">

      {/* ── Messages area ── */}
      <div className="chat-messages">
        {!hasMessages ? (
          <div className="chat-empty-state">
            <div className="chat-empty-icon">
              <WatsonMachineLearning size={48} />
            </div>
            <h2 className="cds--type-productive-heading-04">How can I help you today?</h2>
            <p className="cds--type-body-long-01 chat-empty-subtitle">
              Ask me about MRI analysis, upload a scan, or explore patient data.
            </p>
            <div className="suggested-prompts">
              {SUGGESTED_PROMPTS.map(p => (
                <button
                  key={p.label}
                  className="suggested-prompt-card"
                  onClick={() => handleSend(p.label)}
                >
                  <p.icon size={20} className="prompt-icon" />
                  <div>
                    <p className="prompt-label">{p.label}</p>
                    <p className="prompt-desc">{p.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map(msg => (
              <div key={msg.id} className={`message-row ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'assistant'
                    ? <WatsonMachineLearning size={20} />
                    : <UserAvatar size={20} />}
                </div>
                <div className="message-body">
                  <span className="message-role">
                    {msg.role === 'assistant' ? 'NeuroScan AI' : 'You'}
                  </span>
                  <div className="message-content">{msg.content}</div>
                  <time className="message-time">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </time>
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="message-row assistant">
                <div className="message-avatar">
                  <WatsonMachineLearning size={20} />
                </div>
                <div className="message-body">
                  <span className="message-role">NeuroScan AI</span>
                  <TypingIndicator />
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* ── Input area ── */}
      <div className="chat-input-wrapper">
        <div className="chat-input-box">
          <TextArea
            id="chat-input"
            labelText=""
            placeholder="Message NeuroScan AI…  (Shift + Enter for new line)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isTyping}
          />
          <div className="chat-input-actions">
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              accept=".nii,.nii.gz"
            />
            <Button
              kind="ghost"
              renderIcon={Attachment}
              iconDescription="Upload MRI scan (.nii / .nii.gz)"
              hasIconOnly
              size="md"
              onClick={() => fileInputRef.current?.click()}
              disabled={isTyping}
            />
            <Button
              renderIcon={Send}
              iconDescription="Send message"
              hasIconOnly
              size="md"
              onClick={() => handleSend()}
              disabled={!input.trim() || isTyping}
            />
          </div>
        </div>
        <p className="chat-disclaimer">
          NeuroScan AI may make mistakes. Always verify results with a qualified clinical professional.
        </p>
      </div>

    </div>
  );
};

export default Welcome;
