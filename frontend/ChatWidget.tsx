/**
 * AutoDrive AI Chat Widget
 * Styled to match autodriveai.duckdns.org — Tailwind blue/slate palette, light theme.
 *
 * Usage:
 *   import { ChatWidget } from './ChatWidget';
 *   <ChatWidget apiUrl="https://autodrive-chatbot.azurewebsites.net" />
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

interface ChatWidgetProps {
  apiUrl?: string;
  title?: string;
  voiceEnabled?: boolean;
  /** Called when the LLM mentions a car — receives its id for deep-linking */
  onCarMentioned?: (carId: string) => void;
}

// ── Design tokens (mirror the site's Tailwind palette) ────────────────
const C = {
  blue600:   '#2563eb',
  blue700:   '#1d4ed8',
  blue50:    '#eff6ff',
  blue100:   '#dbeafe',
  violet600: '#7c3aed',
  slate900:  '#0f172a',
  slate800:  '#1e293b',
  slate600:  '#475569',
  slate400:  '#94a3b8',
  slate200:  '#e2e8f0',
  slate100:  '#f1f5f9',
  slate50:   '#f8fafc',
  white:     '#ffffff',
  green600:  '#16a34a',
  green50:   '#f0fdf4',
};

// ── TTS helpers ───────────────────────────────────────────────────────
function speak(text: string) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const clean = text.replace(/[*_`#]/g, '').replace(/\[(?:ACTION|CAR_ID):[^\]]+\]/g, '').replace(/<[^>]+>/g, '');
  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 1.05;
  window.speechSynthesis.speak(utt);
}
function stopSpeaking() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
}

// ── Component ─────────────────────────────────────────────────────────
export const ChatWidget: React.FC<ChatWidgetProps> = ({
  apiUrl = 'http://localhost:8002',
  title = 'AutoDrive AI',
  voiceEnabled = true,
  onCarMentioned,
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
  useEffect(() => { if (isOpen) setTimeout(() => inputRef.current?.focus(), 100); }, [isOpen]);

  // ── Send message ────────────────────────────────────────────────────
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
            if (payload.car_id && onCarMentioned) onCarMentioned(payload.car_id);
          } catch { /* skip */ }
        }
      }

      if (voiceEnabled && fullResponse) {
        setIsSpeaking(true);
        speak(fullResponse);
        setTimeout(() => setIsSpeaking(false), Math.min(fullResponse.length * 55, 15000));
      }
    } catch {
      setMessages(prev => {
        const u = [...prev];
        u[u.length - 1] = { ...u[u.length - 1], content: '⚠️ Could not connect. Please try again.' };
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
  }, [input, isStreaming, apiUrl, sessionId, voiceEnabled, onCarMentioned]);

  // ── Voice recording ─────────────────────────────────────────────────
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
          setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Could not transcribe audio.' }]);
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

  // ── Suggestion chips ────────────────────────────────────────────────
  const suggestions = [
    'SUVs under ₹20 lakh',
    'Best electric cars?',
    'Compare Creta vs Seltos',
    'Lowest mileage cars',
  ];

  return (
    <>
      <style>{`
        @keyframes slideUp   { from { opacity:0; transform:translateY(16px) } to { opacity:1; transform:translateY(0) } }
        @keyframes micPulse  { 0%,100% { box-shadow:0 0 0 0 rgba(220,38,38,.4) } 50% { box-shadow:0 0 0 8px rgba(220,38,38,0) } }
        @keyframes fadeIn    { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:translateY(0) } }
        @keyframes spin { to { transform:rotate(360deg) } }
        .ad-msg { animation: fadeIn .25s ease; }
        .ad-textarea:focus { outline:none; border-color:${C.blue600}; box-shadow:0 0 0 3px rgba(37,99,235,.12); }
        .ad-textarea::placeholder { color:${C.slate400}; }
        .ad-fab:hover { transform:scale(1.08); box-shadow:0 8px 30px rgba(37,99,235,.4); }
        .ad-send:hover:not(:disabled) { background:${C.blue700}; transform:scale(1.04); }
        .ad-chip:hover { background:${C.blue50}; border-color:${C.blue600}; color:${C.blue600}; }
        .ad-scroll::-webkit-scrollbar { width:4px; }
        .ad-scroll::-webkit-scrollbar-track { background:transparent; }
        .ad-scroll::-webkit-scrollbar-thumb { background:${C.slate200}; border-radius:2px; }
      `}</style>

      {/* ── FAB ── */}
      <button
        className="ad-fab"
        onClick={() => { setIsOpen(o => !o); stopSpeaking(); }}
        aria-label="Toggle AutoDrive AI chat"
        style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
          width: 56, height: 56, borderRadius: '50%',
          background: `linear-gradient(135deg, ${C.blue600}, ${C.violet600})`,
          border: 'none', cursor: 'pointer', color: 'white',
          fontSize: 22, display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 20px rgba(37,99,235,.35)',
          transition: 'transform .2s, box-shadow .2s',
        }}
      >
        {isOpen ? '✕' : '🚗'}
      </button>

      {/* ── Chat panel ── */}
      {isOpen && (
        <div style={{
          position: 'fixed', bottom: 92, right: 24, zIndex: 9998,
          width: 390, height: 560, borderRadius: 20,
          background: C.white,
          boxShadow: '0 20px 60px rgba(15,23,42,.15), 0 4px 16px rgba(15,23,42,.08)',
          border: `1px solid ${C.slate200}`,
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
          animation: 'slideUp .25s ease',
        }}>

          {/* Header */}
          <div style={{
            background: `linear-gradient(135deg, ${C.blue600}, ${C.violet600})`,
            padding: '14px 18px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                background: 'rgba(255,255,255,.18)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16,
              }}>🤖</div>
              <div>
                <div style={{ color: 'white', fontWeight: 700, fontSize: 14, letterSpacing: '-.2px' }}>{title}</div>
                <div style={{ color: 'rgba(255,255,255,.75)', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#4ade80', display: 'inline-block' }} />
                  Online · Powered by LLaMA 3.3
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {isSpeaking && (
                <button onClick={stopSpeaking} style={{
                  background: 'rgba(255,255,255,.2)', border: 'none', borderRadius: 8,
                  color: 'white', padding: '3px 10px', cursor: 'pointer', fontSize: 12, fontWeight: 600,
                }}>🔊 Stop</button>
              )}
            </div>
          </div>

          {/* Messages */}
          <div className="ad-scroll" style={{
            flex: 1, overflowY: 'auto', padding: 16,
            display: 'flex', flexDirection: 'column', gap: 12,
            background: C.slate50,
          }}>
            {messages.length === 0 && (
              <div style={{ padding: '20px 0' }}>
                <div style={{ textAlign: 'center', marginBottom: 16 }}>
                  <div style={{ fontSize: 32, marginBottom: 8 }}>🚗</div>
                  <p style={{ color: C.slate600, fontSize: 13, lineHeight: 1.6 }}>
                    Hi! I know every car in our inventory.<br />Ask me anything or try a suggestion:
                  </p>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center' }}>
                  {suggestions.map(s => (
                    <button key={s} className="ad-chip" onClick={() => sendMessage(s)} style={{
                      padding: '5px 12px', borderRadius: 9999,
                      background: C.white, border: `1px solid ${C.slate200}`,
                      color: C.slate600, fontSize: 12, cursor: 'pointer',
                      transition: 'all .2s', fontFamily: 'inherit',
                    }}>{s}</button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className="ad-msg" style={{
                display: 'flex',
                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                gap: 8, alignItems: 'flex-end',
              }}>
                {msg.role === 'assistant' && (
                  <div style={{
                    width: 28, height: 28, borderRadius: 8, flexShrink: 0,
                    background: `linear-gradient(135deg, ${C.blue600}, ${C.violet600})`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 13,
                  }}>🤖</div>
                )}
                <div style={{
                  maxWidth: '78%',
                  padding: '10px 14px',
                  borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                  background: msg.role === 'user'
                    ? `linear-gradient(135deg, ${C.blue600}, ${C.violet600})`
                    : C.white,
                  color: msg.role === 'user' ? 'white' : C.slate800,
                  fontSize: 13.5, lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                  border: msg.role === 'assistant' ? `1px solid ${C.slate200}` : 'none',
                  boxShadow: msg.role === 'assistant' ? '0 1px 4px rgba(15,23,42,.06)' : 'none',
                }}>
                  {msg.content || (msg.isStreaming ? (
                    <span style={{ display: 'inline-flex', gap: 3, padding: '2px 0' }}>
                      {[0, 150, 300].map(d => (
                        <span key={d} style={{
                          width: 6, height: 6, borderRadius: '50%', background: C.slate400,
                          display: 'inline-block',
                          animation: `spin 1s linear ${d}ms infinite`,
                        }} />
                      ))}
                    </span>
                  ) : '')}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input row */}
          <div style={{
            padding: '10px 14px 12px',
            background: C.white,
            borderTop: `1px solid ${C.slate100}`,
          }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
              {/* Mic */}
              <button onClick={toggleRecording} style={{
                width: 36, height: 36, flexShrink: 0, borderRadius: 10,
                background: isRecording ? '#fee2e2' : C.slate100,
                border: `1px solid ${isRecording ? '#fca5a5' : C.slate200}`,
                cursor: 'pointer', fontSize: 15, display: 'flex',
                alignItems: 'center', justifyContent: 'center',
                animation: isRecording ? 'micPulse 1s infinite' : 'none',
                transition: 'all .2s',
              }} title={isRecording ? 'Stop recording' : 'Voice input'}>🎤</button>

              <textarea
                className="ad-textarea"
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={isRecording ? 'Listening…' : 'Ask about any car…'}
                rows={1}
                disabled={isStreaming || isRecording}
                style={{
                  flex: 1, padding: '9px 13px', borderRadius: 12,
                  border: `1.5px solid ${C.slate200}`, background: C.slate50,
                  fontSize: 13.5, fontFamily: 'inherit', resize: 'none',
                  lineHeight: 1.5, color: C.slate800, transition: 'border-color .2s, box-shadow .2s',
                }}
              />

              {/* Send */}
              <button
                className="ad-send"
                onClick={() => sendMessage()}
                disabled={isStreaming || !input.trim()}
                style={{
                  width: 36, height: 36, flexShrink: 0, borderRadius: 10,
                  background: isStreaming || !input.trim() ? C.slate200 : C.blue600,
                  border: 'none', cursor: isStreaming || !input.trim() ? 'not-allowed' : 'pointer',
                  color: isStreaming || !input.trim() ? C.slate400 : 'white',
                  fontSize: 15, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'all .2s',
                }}
                aria-label="Send message"
              >➤</button>
            </div>
            <p style={{ margin: '8px 0 0', fontSize: 11, color: C.slate400, textAlign: 'center' }}>
              Enter to send · Shift+Enter for new line · 🎤 voice input
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatWidget;
