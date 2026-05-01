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

type ConversationEntry = { role: 'user' | 'assistant'; content: string };

interface ScanResult {
  filename: string;
  prediction: string;
  confidence: number;
  class_index: number;
  all_probabilities: number[];
}

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
const ScanResultCard = ({ prediction, confidence }: { prediction: string; confidence: number }) => (
  <div className="scan-result-card">
    <p className="scan-result-label">CNN Classification</p>
    <p className="scan-result-prediction">{prediction}</p>
    <div className="scan-result-confidence">
      <div
        className="confidence-bar"
        style={{ width: `${(confidence * 100).toFixed(0)}%` }}
      />
      <span>{(confidence * 100).toFixed(1)}% confidence</span>
    </div>
  </div>
);

// ── Main component ─────────────────────────────────────────────────────────
const Welcome = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<ConversationEntry[]>([]);
  const [scanContext, setScanContext] = useState<ScanResult | null>(null);

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

  // ── Claude API call ────────────────────────────────────────────────────────
  const callClaude = async (
    history: ConversationEntry[],
    context: ScanResult | null = null,
  ): Promise<string> => {
    const res = await fetch(`${API_URL}/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history,
        scan_context: context
          ? {
              filename: context.filename,
              prediction: context.prediction,
              confidence: context.confidence,
              class_index: context.class_index,
              all_probabilities: context.all_probabilities,
            }
          : null,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? 'Unknown error from /ai/chat');
    }
    const data = await res.json();
    return data.content as string;
  };

  const handleAssistantReply = async (_userText: string, history: ConversationEntry[]) => {
    setIsTyping(true);
    try {
      const reply = await callClaude(history, scanContext);
      addMessage('assistant', reply);
      setConversationHistory(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (err) {
      addMessage('assistant', `Error reaching the AI: ${err instanceof Error ? err.message : err}. Make sure the backend is running and ANTHROPIC_API_KEY is set.`);
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

    const userMsg = `Uploaded scan for analysis: ${file.name}`;
    addMessage('user', `Uploading scan: ${file.name}`);
    setIsTyping(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Prediction request failed');
      const result = await response.json();

      // Show compact CNN result card
      addMessage('assistant', (
        <ScanResultCard
          prediction={result.prediction}
          confidence={result.confidence}
        />
      ));

      // Store scan context for the conversation
      const ctx: ScanResult = {
        filename: file.name,
        prediction: result.prediction,
        confidence: result.confidence,
        class_index: result.class_index,
        all_probabilities: result.all_probabilities ?? [],
      };
      setScanContext(ctx);

      // Build history entry for this exchange and call Claude for enriched interpretation
      const updatedHistory: ConversationEntry[] = [
        ...conversationHistory,
        { role: 'user', content: userMsg },
        { role: 'assistant', content: `CNN result: ${result.prediction} (${(result.confidence * 100).toFixed(1)}% confidence)` },
        { role: 'user', content: 'Please provide a clinical interpretation of this result.' },
      ];
      setConversationHistory(updatedHistory);

      const reply = await callClaude(updatedHistory, ctx);
      addMessage('assistant', reply);
      setConversationHistory(prev => [...prev, { role: 'assistant', content: reply }]);

    } catch (error) {
      addMessage('assistant', `Error analyzing the scan: ${error instanceof Error ? error.message : error}. Please ensure the backend is running and the model is loaded.`);
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
              <img src="/neuroscan-icon.svg" alt="NeuroScan AI" width="80" height="80" />
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
