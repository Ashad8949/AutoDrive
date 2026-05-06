/**
 * AutoDrive AI Chat Widget — with Voice Support
 *
 * Usage:
 *   import { ChatWidget } from './ChatWidget';
 *   <ChatWidget apiUrl="https://autodrive-chatbot.azurewebsites.net" />
 *
 * Features:
 *  - Floating button bottom-right
 *  - Streaming AI responses
 *  - 🎤 Mic button — click to record, click again to send (Groq Whisper STT)
 *  - 🔊 AI responses spoken aloud automatically (browser TTS)
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

// ── Types ─────────────────────────────────────────────────────────────
interface Message {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

interface ChatWidgetProps {
  apiUrl?: string;
  title?: string;
  accentColor?: string;
  /** If true, AI responses are spoken aloud automatically */
  voiceEnabled?: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────────
function speak(text: string) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const clean = text.replace(/[*_`#]/g, '').replace(/\[ACTION:[^\]]+\]/g, '');
  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 1.05;
  utt.pitch = 1;
  window.speechSynthesis.speak(utt);
}

function stopSpeaking() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
}

// ── Styles ────────────────────────────────────────────────────────────
const S = {
  fab: (color: string): React.CSSProperties => ({
    position: 'fixed', bottom: 24, right: 24,
    width: 60, height: 60, borderRadius: '50%',
    backgroundColor: color, color: 'white', border: 'none',
    cursor: 'pointer', fontSize: 24,
    boxShadow: '0 4px 16px rgba(0,0,0,0.25)', zIndex: 9999,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'transform 0.15s',
  }),
  panel: (): React.CSSProperties => ({
    position: 'fixed', bottom: 96, right: 24,
    width: 380, height: 540,
    backgroundColor: '#fff', borderRadius: 16,
    boxShadow: '0 8px 40px rgba(0,0,0,0.18)',
    display: 'flex', flexDirection: 'column',
    zIndex: 9998, overflow: 'hidden',
    animation: 'slideUp 0.2s ease',
  }),
  header: (color: string): React.CSSProperties => ({
    backgroundColor: color, color: 'white',
    padding: '14px 18px', fontWeight: 600, fontSize: 15,
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  }),
  messages: (): React.CSSProperties => ({
    flex: 1, overflowY: 'auto', padding: 16,
    display: 'flex', flexDirection: 'column', gap: 10,
  }),
  bubble: (role: 'user' | 'assistant', color: string): React.CSSProperties => ({
    alignSelf: role === 'user' ? 'flex-end' : 'flex-start',
    maxWidth: '82%',
    backgroundColor: role === 'user' ? color : '#f3f4f6',
    color: role === 'user' ? 'white' : '#111827',
    padding: '10px 14px',
    borderRadius: role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
    fontSize: 14, lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  }),
  inputRow: (): React.CSSProperties => ({
    padding: '10px 12px', borderTop: '1px solid #e5e7eb',
    display: 'flex', gap: 6, alignItems: 'flex-end',
  }),
  textarea: (): React.CSSProperties => ({
    flex: 1, padding: '9px 13px', borderRadius: 20,
    border: '1px solid #d1d5db', outline: 'none',
    fontSize: 14, fontFamily: 'inherit', resize: 'none', lineHeight: 1.4,
  }),
  iconBtn: (bg: string, disabled = false): React.CSSProperties => ({
    width: 38, height: 38, flexShrink: 0,
    backgroundColor: bg, color: 'white', border: 'none',
    borderRadius: '50%', cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.45 : 1, display: 'flex',
    alignItems: 'center', justifyContent: 'center', fontSize: 16,
    transition: 'opacity 0.15s',
  }),
};

// ── Component ─────────────────────────────────────────────────────────
export const ChatWidget: React.FC<ChatWidgetProps> = ({
  apiUrl = 'http://localhost:8002',
  title = 'AutoDrive AI Assistant',
  accentColor = '#1a56db',
  voiceEnabled = true,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => { if (isOpen) inputRef.current?.focus(); }, [isOpen]);

  // ── Send text message ──────────────────────────────────────────────
  const sendMessage = useCallback(async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || isStreaming) return;
    setInput('');
    stopSpeaking();
    setMessages(prev => [...prev, { role: 'user', content: msg }]);
    setIsStreaming(true);
    setMessages(prev => [...prev, { role: 'assistant', content: '', isStreaming: true }]);

    try {
      const res = await fetch(`${apiUrl}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const payload = JSON.parse(raw);
            if (payload.token) {
              fullResponse += payload.token;
              setMessages(prev => {
                const u = [...prev];
                u[u.length - 1] = { ...u[u.length - 1], content: u[u.length - 1].content + payload.token };
                return u;
              });
            }
          } catch { /* skip malformed */ }
        }
      }

      // Speak the full response aloud
      if (voiceEnabled && fullResponse) {
        setIsSpeaking(true);
        speak(fullResponse);
        setTimeout(() => setIsSpeaking(false), fullResponse.length * 55);
      }
    } catch {
      setMessages(prev => {
        const u = [...prev];
        u[u.length - 1] = { ...u[u.length - 1], content: 'Sorry, something went wrong. Please try again.' };
        return u;
      });
    } finally {
      setIsStreaming(false);
      setMessages(prev => {
        const u = [...prev];
        if (u[u.length - 1]?.isStreaming) u[u.length - 1] = { ...u[u.length - 1], isStreaming: false };
        return u;
      });
    }
  }, [input, isStreaming, apiUrl, sessionId, voiceEnabled]);

  // ── Voice recording ────────────────────────────────────────────────
  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mr.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const form = new FormData();
        form.append('audio', blob, 'audio.webm');
        try {
          const res = await fetch(`${apiUrl}/voice/transcribe`, { method: 'POST', body: form });
          const { transcript } = await res.json();
          if (transcript) sendMessage(transcript);
        } catch {
          setMessages(prev => [...prev, { role: 'assistant', content: 'Could not transcribe audio. Please try again.' }]);
        }
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setIsRecording(true);
    } catch {
      alert('Microphone access denied. Please allow microphone permission.');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  return (
    <>
      <style>{`
        @keyframes slideUp { from { opacity:0; transform:translateY(16px) } to { opacity:1; transform:translateY(0) } }
        @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5) } 50% { box-shadow: 0 0 0 8px rgba(239,68,68,0) } }
      `}</style>

      {/* Floating toggle button */}
      <button style={S.fab(accentColor)} onClick={() => { setIsOpen(o => !o); stopSpeaking(); }} aria-label="Toggle chat">
        {isOpen ? '✕' : '💬'}
      </button>

      {isOpen && (
        <div style={S.panel()}>
          {/* Header */}
          <div style={S.header(accentColor)}>
            <span>🚗 {title}</span>
            {isSpeaking && (
              <button onClick={stopSpeaking} style={{ background: 'rgba(255,255,255,0.2)', border: 'none', borderRadius: 12, color: 'white', padding: '2px 10px', cursor: 'pointer', fontSize: 12 }}>
                🔊 Stop
              </button>
            )}
          </div>

          {/* Messages */}
          <div style={S.messages()}>
            {messages.length === 0 && (
              <p style={{ color: '#9ca3af', textAlign: 'center', marginTop: 48, fontSize: 14 }}>
                Hi! Ask me about our cars, prices, or book a test drive.<br />
                <span style={{ fontSize: 12 }}>🎤 Use the mic to speak</span>
              </p>
            )}
            {messages.map((msg, i) => (
              <div key={i} style={S.bubble(msg.role, accentColor)}>
                {msg.content || (msg.isStreaming ? '▋' : '')}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input row */}
          <div style={S.inputRow()}>
            {/* Mic button */}
            <button
              onClick={toggleRecording}
              style={{
                ...S.iconBtn(isRecording ? '#ef4444' : '#6b7280'),
                animation: isRecording ? 'pulse 1s infinite' : 'none',
              }}
              title={isRecording ? 'Stop recording' : 'Speak'}
            >
              🎤
            </button>

            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isRecording ? 'Listening…' : 'Ask about cars… (Enter to send)'}
              rows={1}
              disabled={isStreaming || isRecording}
              style={S.textarea()}
            />

            {/* Send button */}
            <button
              onClick={() => sendMessage()}
              disabled={isStreaming || !input.trim()}
              style={S.iconBtn(accentColor, isStreaming || !input.trim())}
              aria-label="Send"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatWidget;
