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
import { getJson, runPrediction, CLASS_COLORS } from '../api';
import type { DatasetStats } from '../api';
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
  class_labels?: string[];
}

const UPLOAD_PROMPT = 'Analyze an MRI scan';
const STATS_PROMPT = 'Show patient statistics';

// ── Suggested prompts ──────────────────────────────────────────────────────
const SUGGESTED_PROMPTS = [
  {
    icon: Upload,
    label: UPLOAD_PROMPT,
    description: 'Upload a T1 .nii or .nii.gz scan for CN vs AD classification',
  },
  {
    icon: Analytics,
    label: STATS_PROMPT,
    description: 'Cohort size, scans available, and validated model performance',
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
const ScanResultCard = ({ result }: { result: ScanResult }) => {
  const labels = result.class_labels ?? ['CN', 'AD'];
  return (
    <div className="scan-result-card">
      <p className="scan-result-label">Hippocampal ROI classifier</p>
      <p className="scan-result-prediction">{result.prediction}</p>
      {result.all_probabilities.map((p, i) => (
        <div key={labels[i]} className="scan-result-confidence">
          <span style={{ minWidth: '2rem', fontWeight: 600 }}>{labels[i]}</span>
          <div className="confidence-track">
            <div
              className="confidence-bar"
              style={{ width: `${(p * 100).toFixed(0)}%`, background: CLASS_COLORS[labels[i]] }}
            />
          </div>
          <span style={{ minWidth: '3rem', textAlign: 'right' }}>{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
      <p className="scan-result-note">Research prototype. Not a diagnosis.</p>
    </div>
  );
};

const StatsCard = ({ stats }: { stats: DatasetStats }) => (
  <div className="scan-result-card">
    <p className="scan-result-label">Dataset and model</p>
    <ul className="stats-list">
      <li><strong>{stats.total_patients.toLocaleString()}</strong> patients and <strong>{stats.total_scans.toLocaleString()}</strong> T1 scans available</li>
      <li><strong>{stats.training_subjects.toLocaleString()}</strong> subjects used to train and validate the model</li>
      <li>ROC AUC <strong>{stats.model_auc.toFixed(2)}</strong>, accuracy <strong>{(stats.model_accuracy * 100).toFixed(1)}%</strong></li>
    </ul>
    <p className="scan-result-note">See the Data page for the full report and the MRI viewer.</p>
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

  // Auto-scroll to the latest message. Skip the empty state: scrolling there
  // moved the whole window and exposed the background above the header.
  useEffect(() => {
    if (messages.length === 0 && !isTyping) return;
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    const data = await getJson<{ content: string }>('/ai/chat', {
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
    return data.content;
  };

  const showStats = async () => {
    addMessage('user', STATS_PROMPT);
    setIsTyping(true);
    try {
      const stats = await getJson<DatasetStats>('/data/stats');
      addMessage('assistant', <StatsCard stats={stats} />);
    } catch (err) {
      addMessage('assistant', `I could not load the statistics: ${err instanceof Error ? err.message : err}`);
    } finally {
      setIsTyping(false);
    }
  };

  const handleAssistantReply = async (_userText: string, history: ConversationEntry[]) => {
    setIsTyping(true);
    try {
      const reply = await callClaude(history, scanContext);
      addMessage('assistant', reply);
      setConversationHistory(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (err) {
      addMessage('assistant', `I could not reach the AI assistant: ${err instanceof Error ? err.message : err}`);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = async (text?: string) => {
    const userText = (text ?? input).trim();
    if (!userText || isTyping) return;

    if (userText === UPLOAD_PROMPT) {
      fileInputRef.current?.click();
      return;
    }
    if (userText === STATS_PROMPT) {
      await showStats();
      return;
    }

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
    addMessage('assistant', 'Aligning the scan to the MNI brain template and extracting the hippocampal region. This usually takes 20 to 60 seconds.');
    setIsTyping(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const result = await runPrediction<ScanResult & { processing_seconds?: number }>(formData);

      const ctx: ScanResult = {
        filename: file.name,
        prediction: result.prediction,
        confidence: result.confidence,
        class_index: result.class_index,
        all_probabilities: result.all_probabilities ?? [],
        class_labels: result.class_labels,
      };
      addMessage('assistant', <ScanResultCard result={ctx} />);
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
      addMessage('assistant', `I could not analyze the scan: ${error instanceof Error ? error.message : error}`);
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
              <img src="/neuroscan-icon.png" alt="NeuroScan AI" width="80" height="80" />
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

          {/* Left — file upload */}
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
            className="chat-input-attach"
          />

          {/* Centre — text */}
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

          {/* Right — send */}
          <Button
            renderIcon={Send}
            iconDescription="Send message"
            hasIconOnly
            size="md"
            onClick={() => handleSend()}
            disabled={!input.trim() || isTyping}
            className="chat-input-send"
          />

        </div>
        <p className="chat-disclaimer">
          NeuroScan AI may make mistakes. Always verify results with a qualified clinical professional.
        </p>
      </div>

    </div>
  );
};

export default Welcome;
